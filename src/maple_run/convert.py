# Copyright © 2026 DeepGrove AI.
"""Pack Hugging Face Maple bf16 tensors into 2-bit codes + per-row alpha.

Port of ``docs/sources/mlx_lm_ternary.py`` from MLX to NumPy. Streams one
safetensors shard at a time so the ~40 GB dump is never fully resident.

Critical constants from DeepGrove:

- ``DEFAULT_THRESHOLD_SCALE = 0.7``
- ``GROUP_SIZE = 128`` for ternary packing
- Packing: 16 two-bit codes per uint32, LSB first, codes = sign(w)*mask + 1
  so {−1, 0, +1} → {0, 1, 2}
- Ternarize in float32; only the final alpha is cast back to the weight dtype
- Exclude ``.mlp.gate.weight`` (router stays float32)
- ``lm_head.weight`` and ``model.word_embeddings.weight`` are 4-bit RTN
  (group_size 64), not ternary
"""

from __future__ import annotations

import gc
import json
import mmap
import shutil
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from maple_run.pack import (
    DEFAULT_THRESHOLD_SCALE,
    GROUP_SIZE,
    HEAD_BITS,
    HEAD_GROUP_SIZE,
    quantize_rtn,
    ternarize,
)

SHARD_BYTES = 2 << 30

# Not ternarized: the router runs in float32. (The head and embeddings are
# claimed by RTN_KEYS before the ternary check runs; norms are 1D.)
TERNARY_EXCLUDE = (".mlp.gate.weight",)
RTN_KEYS = ("lm_head.weight", "model.word_embeddings.weight")
# Internal training artifacts that must not ship.
SKIP_AUX = (
    "modeling_maple.py",
    "configuration_maple.py",
    "fa3.py",
    "fa3_utils.py",
    "model.safetensors.index.json",
)

_ST_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "I64": np.dtype(np.int64),
    "U64": np.dtype(np.uint64),
    "I32": np.dtype(np.int32),
    "U32": np.dtype(np.uint32),
    "I16": np.dtype(np.int16),
    "U16": np.dtype(np.uint16),
    "I8": np.dtype(np.int8),
    "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}


def _bf16_to_f32(raw: bytes | memoryview, shape: tuple[int, ...]) -> np.ndarray:
    """Interpret IEEE-754 bfloat16 bytes as float32 (high 16 bits of f32)."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    expected = int(np.prod(shape, dtype=np.int64))
    if u16.size != expected:
        raise ValueError(f"BF16 payload size {u16.size} != {expected} for shape {shape}")
    u16 = np.array(u16, dtype=np.uint16, copy=True, order="C").reshape(shape)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _decode_tensor(dtype: str, shape: list[int], raw: bytes | memoryview) -> np.ndarray:
    shape_t = tuple(int(d) for d in shape)
    if dtype == "BF16":
        return _bf16_to_f32(raw, shape_t)
    try:
        np_dtype = _ST_DTYPES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported safetensors dtype {dtype!r}") from exc
    arr = np.frombuffer(raw, dtype=np_dtype)
    expected = int(np.prod(shape_t, dtype=np.int64))
    if arr.size != expected:
        raise ValueError(f"{dtype} payload size {arr.size} != {expected} for shape {shape_t}")
    arr = np.array(arr, copy=True, order="C").reshape(shape_t)
    return arr


def iter_safetensors(path: Path):
    """Yield ``(name, ndarray)`` one tensor at a time from a safetensors file.

    BF16 values are promoted to float32 (the ternarizer already requires that).
    """
    path = Path(path)
    with path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                start, stop = spec["data_offsets"]
                raw = mm[data_start + start : data_start + stop]
                yield key, _decode_tensor(spec["dtype"], spec["shape"], raw)
        finally:
            mm.close()


def _is_ternary_target(key: str, shape) -> bool:
    return (
        key.endswith(".weight")
        and len(shape) == 2
        and not any(pat in key for pat in TERNARY_EXCLUDE)
    )


class _ShardWriter:
    """Accumulates tensors and flushes ~2 GB safetensors shards."""

    def __init__(self, output: Path, shard_bytes: int = SHARD_BYTES):
        self.output = output
        self.shard_bytes = shard_bytes
        self.tensors: dict[str, np.ndarray] = {}
        self.nbytes = 0
        self.weight_map: dict[str, str] = {}
        self.shard_names: list[str] = []
        self.total_size = 0

    def add(self, key: str, value: np.ndarray) -> None:
        value = np.ascontiguousarray(value)
        self.tensors[key] = value
        self.nbytes += int(value.nbytes)
        if self.nbytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.tensors:
            return
        name = f"model-part-{len(self.shard_names):05d}.safetensors"
        path = self.output / name
        save_file(self.tensors, str(path))
        for k in self.tensors:
            self.weight_map[k] = name
        self.shard_names.append(name)
        self.total_size += path.stat().st_size
        self.tensors = {}
        self.nbytes = 0
        gc.collect()

    def finalize(self) -> None:
        self.flush()
        n = len(self.shard_names)
        if n == 0:
            raise RuntimeError("No tensors were written; source checkpoint was empty.")
        if n == 1:
            final = {self.shard_names[0]: "model.safetensors"}
        else:
            final = {
                name: f"model-{i + 1:05d}-of-{n:05d}.safetensors"
                for i, name in enumerate(self.shard_names)
            }
        for old, new in final.items():
            (self.output / old).rename(self.output / new)
        weight_map = {k: final[v] for k, v in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total_size}, "weight_map": weight_map}
        (self.output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2)
        )


class _Converter:
    def __init__(
        self,
        output: Path,
        config: dict,
        threshold_scale: float,
        shard_bytes: int = SHARD_BYTES,
    ):
        self.writer = _ShardWriter(output, shard_bytes=shard_bytes)
        self.threshold_scale = threshold_scale
        self.num_experts = int(config["num_experts"])
        self.tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        # model.layers.{l}.mlp.experts.{e}.{proj}.{param} accumulate here until
        # a full expert set is present, then are stacked into switch_mlp form.
        self.pending: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
        self.kinds: dict[str, str] = {}

    def _emit_rtn(self, prefix: str, packed, scales, biases) -> None:
        self.writer.add(f"{prefix}.weight", packed)
        self.writer.add(f"{prefix}.scales", np.ascontiguousarray(scales, dtype=np.float32))
        self.writer.add(f"{prefix}.biases", np.ascontiguousarray(biases, dtype=np.float32))
        self.kinds[prefix] = "rtn_4bit"

    def _emit_ternary(self, prefix: str, packed, row_alpha) -> None:
        self.writer.add(f"{prefix}.weight", packed)
        self.writer.add(
            f"{prefix}.row_alpha", np.ascontiguousarray(row_alpha, dtype=np.float32)
        )
        self.kinds[prefix] = "ternary"

    def _convert_expert(self, key: str, value: np.ndarray) -> None:
        # key: model.layers.{l}.mlp.experts.{e}.{proj}.{param}
        parts = key.split(".")
        layer, expert, proj, param = parts[2], int(parts[5]), parts[6], parts[7]
        slot = self.pending.setdefault((layer, proj, param), {})
        slot[expert] = value
        if len(slot) < self.num_experts:
            return
        stacked = np.stack([slot[e] for e in range(self.num_experts)], axis=0)
        del self.pending[(layer, proj, param)]
        prefix = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
        if param == "weight":
            packed, row_alpha = ternarize(stacked, self.threshold_scale)
            self._emit_ternary(prefix, packed, row_alpha)
        else:
            if np.issubdtype(stacked.dtype, np.floating):
                stacked = stacked.astype(np.float32, copy=False)
            self.writer.add(f"{prefix}.{param}", stacked)
            self.kinds[f"{prefix}.{param}"] = "float"

    def add(self, key: str, value: np.ndarray) -> None:
        if key.startswith("lm_head.") and self.tie_word_embeddings:
            return
        if ".mlp.experts." in key:
            self._convert_expert(key, value)
        elif key in RTN_KEYS:
            packed, scales, biases = quantize_rtn(
                value, group_size=HEAD_GROUP_SIZE, bits=HEAD_BITS
            )
            self._emit_rtn(key[: -len(".weight")], packed, scales, biases)
        elif _is_ternary_target(key, value.shape):
            packed, row_alpha = ternarize(value, self.threshold_scale)
            self._emit_ternary(key[: -len(".weight")], packed, row_alpha)
        else:
            if np.issubdtype(value.dtype, np.floating):
                value = value.astype(np.float32, copy=False)
            self.writer.add(key, value)
            self.kinds[key] = "float"

    def finalize(self) -> None:
        if self.pending:
            missing = sorted(self.pending)
            raise RuntimeError(f"Incomplete expert sets at end of stream: {missing}")
        self.writer.finalize()


def _write_config(
    source_config: dict,
    output: Path,
    threshold_scale: float,
    kinds: dict[str, str],
) -> None:
    config = dict(source_config)
    config["model_type"] = "maple"
    for stale in ("score_function", "n_group", "topk_group", "routed_scaling_factor"):
        config.pop(stale, None)
    config.pop("auto_map", None)
    config.pop("quantize", None)
    config.pop("quantization_config", None)
    quantization = {
        "group_size": GROUP_SIZE,
        "bits": 2,
        "mode": "affine",
        "lm_head": {"group_size": HEAD_GROUP_SIZE, "bits": HEAD_BITS},
        "model.word_embeddings": {"group_size": HEAD_GROUP_SIZE, "bits": HEAD_BITS},
    }
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    by_kind: dict[str, list[str]] = {"ternary": [], "rtn_4bit": [], "float": []}
    for name, kind in kinds.items():
        by_kind.setdefault(kind, []).append(name)
    for kind in by_kind:
        by_kind[kind].sort()
    config["maple_run"] = {
        "format": "packed-ternary-v1",
        "experts": "stacked_switch_mlp",
        "expert_template": "model.layers.{layer}.mlp.switch_mlp.{proj}.weight",
        "threshold_scale": threshold_scale,
        "ternary": {
            "bits": 2,
            "group_size": GROUP_SIZE,
            "pack": "uint32_lsb_first",
            "scale": "row_alpha",
        },
        "rtn_4bit": {
            "bits": HEAD_BITS,
            "group_size": HEAD_GROUP_SIZE,
            "mode": "affine",
            "pack": "uint32_lsb_first",
        },
        "keys": by_kind,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))


def _shard_names(index_path: Path) -> list[str]:
    weight_map = json.loads(index_path.read_text())["weight_map"]
    return sorted(set(weight_map.values()))


def _try_local_hub_snapshot(repo_id: str) -> Path | None:
    """Return a complete Hugging Face hub snapshot if it is already on disk.

    ``from_pretrained`` / Transformers is the wrong loader here: Maple's
    modeling file imports ``fa3.py``, and materializing the 40 GB dump as
    ``nn.Linear`` weights defeats shard streaming.
    """
    if "/" not in repo_id or Path(repo_id).is_dir():
        return None
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = snapshot_download(repo_id=repo_id, local_files_only=True)
    except LocalEntryNotFoundError:
        return None
    snap = Path(path)
    has_weights = (snap / "model.safetensors.index.json").exists() or any(
        snap.glob("*.safetensors")
    )
    if not (snap / "config.json").exists() or not has_weights:
        return None
    return snap


def _copy_sidecars(source: Path, output: Path) -> None:
    for f in source.iterdir():
        if f.is_dir() or f.suffix == ".safetensors" or f.name in SKIP_AUX:
            continue
        shutil.copy2(f, output / f.name)


def _stage_local(source: Path, output: Path) -> list[str]:
    index = source / "model.safetensors.index.json"
    if index.exists():
        shard_names = _shard_names(index)
    else:
        shard_names = sorted(p.name for p in source.glob("*.safetensors"))
        if not shard_names:
            raise FileNotFoundError(f"No .safetensors found in {source}")
    _copy_sidecars(source, output)
    return shard_names


def _stage_hf(repo_id: str, output: Path) -> list[str]:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id)
    for fn in files:
        if (
            fn.endswith(".safetensors")
            or fn in SKIP_AUX
            or fn.startswith(("__pycache__", "."))
        ):
            continue
        hf_hub_download(repo_id=repo_id, filename=fn, local_dir=str(output))
    if "model.safetensors.index.json" in files:
        idx = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            local_dir=str(output / "_src"),
        )
        return _shard_names(Path(idx))
    shard_names = sorted(f for f in files if f.endswith(".safetensors"))
    if not shard_names:
        raise FileNotFoundError(f"No .safetensors found in {repo_id}")
    return shard_names


def convert_checkpoint(
    source: str,
    output: str,
    threshold_scale: float = DEFAULT_THRESHOLD_SCALE,
    *,
    shard_bytes: int = SHARD_BYTES,
    keep_source: bool = False,
) -> Path:
    """Convert a local checkpoint directory or Hugging Face repo id.

    Writes packed safetensors plus ``config.json`` describing ternary / 4-bit /
    float tensors. Expert weights are stacked into ``mlp.switch_mlp.{proj}``.

    Repo ids such as ``deepgrove/maple-preview`` use the Hugging Face hub cache
    when the snapshot is already downloaded; shards are read in place (not
    copied, not loaded through Transformers).
    """
    output_dir = Path(output).expanduser()
    if output_dir.exists() and list(output_dir.glob("model*.safetensors")):
        raise FileExistsError(
            f"Output directory {output_dir} already contains model shards; "
            "remove them or choose a new directory (stale shards would be "
            "loaded together with the new ones)."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source).expanduser()
    local_dir = source_path if source_path.is_dir() else _try_local_hub_snapshot(source)
    is_local = local_dir is not None
    if is_local:
        source_dir = local_dir
        print(f"Using local checkpoint {source_dir}", flush=True)
        shard_names = _stage_local(source_dir, output_dir)
        get_shard = lambda name: source_dir / name  # noqa: E731
        drop_shard = lambda path: None  # noqa: E731
    else:
        from huggingface_hub import hf_hub_download

        shard_names = _stage_hf(source, output_dir)
        src_dir = output_dir / "_src"
        src_dir.mkdir(exist_ok=True)
        get_shard = lambda name: Path(  # noqa: E731
            hf_hub_download(repo_id=source, filename=name, local_dir=str(src_dir))
        )
        drop_shard = lambda path: None if keep_source else path.unlink()  # noqa: E731

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json after staging {source}")
    source_config = json.loads(config_path.read_text())
    converter = _Converter(
        output_dir, source_config, threshold_scale, shard_bytes=shard_bytes
    )

    for i, name in enumerate(shard_names, 1):
        print(f"[{i}/{len(shard_names)}] {name}", flush=True)
        path = get_shard(name)
        for key, tensor in iter_safetensors(path):
            converter.add(key, tensor)
            del tensor
        gc.collect()
        drop_shard(path)

    converter.finalize()
    _write_config(source_config, output_dir, threshold_scale, converter.kinds)
    if not is_local and not keep_source:
        shutil.rmtree(output_dir / "_src", ignore_errors=True)

    size = converter.writer.total_size
    print(f"Done: {size / 1e9:.2f} GB -> {output_dir}", flush=True)
    return output_dir
