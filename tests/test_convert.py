"""Streaming converter tests against a tiny fake Maple checkpoint."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from maple_run.cli import main
from maple_run.convert import (
    _try_local_hub_snapshot,
    convert_checkpoint,
    iter_safetensors,
)
from maple_run.pack import dequantize_rtn, ternarize, unpack_2bit


def _tiny_config(**overrides) -> dict:
    cfg = {
        "architectures": ["MapleForCausalLM"],
        "hidden_size": 128,
        "num_experts": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "moe_intermediate_size": 128,
        "vocab_size": 64,
        "tie_word_embeddings": False,
        "dtype": "float32",
        "rms_norm_eps": 1e-6,
        "quantize": True,
        "auto_map": {"AutoConfig": "configuration_maple.MapleConfig"},
    }
    cfg.update(overrides)
    return cfg


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _make_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    h, v, e = 128, 64, 2
    w = {
        "lm_head.weight": rng.standard_normal((v, h)).astype(np.float32),
        "model.word_embeddings.weight": rng.standard_normal((v, h)).astype(np.float32),
        "model.norm.weight": rng.standard_normal((h,)).astype(np.float32),
        "model.layers.0.input_layernorm.weight": rng.standard_normal((h,)).astype(np.float32),
        "model.layers.0.post_attention_layernorm.weight": rng.standard_normal((h,)).astype(
            np.float32
        ),
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((h, h)).astype(
            np.float32
        ),
        "model.layers.0.self_attn.k_proj.weight": rng.standard_normal((h, h)).astype(
            np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": rng.standard_normal((h, h)).astype(
            np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": rng.standard_normal((h, h)).astype(
            np.float32
        ),
        "model.layers.0.self_attn.q_norm.weight": rng.standard_normal((64,)).astype(
            np.float32
        ),
        "model.layers.0.self_attn.k_norm.weight": rng.standard_normal((64,)).astype(
            np.float32
        ),
        "model.layers.0.mlp.gate.weight": rng.standard_normal((e, h)).astype(np.float32),
    }
    for expert in range(e):
        w[f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"] = rng.standard_normal(
            (h, h)
        ).astype(np.float32)
        w[f"model.layers.0.mlp.experts.{expert}.up_proj.weight"] = rng.standard_normal(
            (h, h)
        ).astype(np.float32)
        w[f"model.layers.0.mlp.experts.{expert}.down_proj.weight"] = rng.standard_normal(
            (h, h)
        ).astype(np.float32)
    return w


def _write_bf16_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    blobs: list[bytes] = []
    header: dict[str, dict] = {}
    offset = 0
    for key, value in tensors.items():
        f32 = np.ascontiguousarray(value, dtype=np.float32)
        u16 = (f32.view(np.uint32) >> 16).astype(np.uint16, copy=False)
        data = u16.tobytes()
        header[key] = {
            "dtype": "BF16",
            "shape": list(f32.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes = header_bytes + b" " * pad
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for blob in blobs:
            fh.write(blob)


def test_convert_tiny_checkpoint_classifies_and_stacks(tmp_path: Path):
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir()
    weights = _make_weights(rng)
    save_file(weights, str(src / "model.safetensors"))
    _write_json(src / "config.json", _tiny_config())
    (src / "tokenizer.json").write_text("{}")
    (src / "modeling_maple.py").write_text("# should not be copied\n")

    convert_checkpoint(str(src), str(dst))

    assert (dst / "tokenizer.json").exists()
    assert not (dst / "modeling_maple.py").exists()
    packed = load_file(str(dst / "model.safetensors"))
    cfg = json.loads((dst / "config.json").read_text())
    kinds = cfg["maple_run"]["keys"]

    assert cfg["maple_run"]["experts"] == "stacked_switch_mlp"
    assert cfg["quantization"]["bits"] == 2
    assert cfg["quantization"]["lm_head"] == {"group_size": 64, "bits": 4}
    assert "auto_map" not in cfg
    assert "quantize" not in cfg

    assert "model.layers.0.self_attn.q_proj" in kinds["ternary"]
    assert "lm_head" in kinds["rtn_4bit"]
    assert "model.word_embeddings" in kinds["rtn_4bit"]
    assert "model.layers.0.mlp.gate.weight" in kinds["float"]
    assert "model.norm.weight" in kinds["float"]

    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in packed
    stacked = packed["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    alpha = packed["model.layers.0.mlp.switch_mlp.gate_proj.row_alpha"]
    assert stacked.dtype == np.uint32
    assert stacked.shape == (2, 128, 128 // 16)
    assert alpha.shape == (2, 128)

    src_stack = np.stack(
        [
            weights["model.layers.0.mlp.experts.0.gate_proj.weight"],
            weights["model.layers.0.mlp.experts.1.gate_proj.weight"],
        ],
        axis=0,
    )
    expect_packed, expect_alpha = ternarize(src_stack)
    np.testing.assert_array_equal(stacked, expect_packed)
    np.testing.assert_allclose(alpha, expect_alpha, rtol=1e-6, atol=1e-7)

    q_codes = unpack_2bit(packed["model.layers.0.self_attn.q_proj.weight"])
    expect_q, _ = ternarize(weights["model.layers.0.self_attn.q_proj.weight"])
    np.testing.assert_array_equal(q_codes, unpack_2bit(expect_q))

    recon = dequantize_rtn(
        packed["lm_head.weight"], packed["lm_head.scales"], packed["lm_head.biases"]
    )
    assert recon.shape == weights["lm_head.weight"].shape
    gate = packed["model.layers.0.mlp.gate.weight"]
    assert gate.dtype == np.float32
    np.testing.assert_allclose(gate, weights["model.layers.0.mlp.gate.weight"])


def test_convert_streams_experts_across_shards(tmp_path: Path):
    rng = np.random.default_rng(1)
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir()
    weights = _make_weights(rng)
    shard_a = {k: v for k, v in weights.items() if ".experts.1." not in k}
    shard_b = {k: v for k, v in weights.items() if ".experts.1." in k}
    save_file(shard_a, str(src / "model-00001-of-00002.safetensors"))
    save_file(shard_b, str(src / "model-00002-of-00002.safetensors"))
    _write_json(
        src / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 1},
            "weight_map": {
                **{k: "model-00001-of-00002.safetensors" for k in shard_a},
                **{k: "model-00002-of-00002.safetensors" for k in shard_b},
            },
        },
    )
    _write_json(src / "config.json", _tiny_config())

    convert_checkpoint(str(src), str(dst))
    packed = load_file(str(dst / "model.safetensors"))
    assert packed["model.layers.0.mlp.switch_mlp.down_proj.weight"].shape[0] == 2
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert "model.layers.0.mlp.switch_mlp.down_proj.weight" in index["weight_map"]


def test_convert_incomplete_experts_raises(tmp_path: Path):
    rng = np.random.default_rng(2)
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir()
    weights = _make_weights(rng)
    incomplete = {k: v for k, v in weights.items() if ".experts.1." not in k}
    save_file(incomplete, str(src / "model.safetensors"))
    _write_json(src / "config.json", _tiny_config())
    with pytest.raises(RuntimeError, match="Incomplete expert"):
        convert_checkpoint(str(src), str(dst))


def test_convert_refuses_existing_shards(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    (dst / "model.safetensors").write_bytes(b"stale")
    _write_json(src / "config.json", _tiny_config())
    save_file({"x": np.ones((2,), dtype=np.float32)}, str(src / "model.safetensors"))
    with pytest.raises(FileExistsError):
        convert_checkpoint(str(src), str(dst))


def test_iter_safetensors_promotes_bf16(tmp_path: Path):
    path = tmp_path / "bf16.safetensors"
    original = {"w": np.array([[1.0, -2.0], [0.5, 4.0]], dtype=np.float32)}
    _write_bf16_safetensors(path, original)
    loaded = dict(iter_safetensors(path))
    assert loaded["w"].dtype == np.float32
    # Truncating f32→bf16 drops the low 16 bits; stay within a few ulp of the
    # truncated value rather than the original f32.
    truncated = (original["w"].view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    np.testing.assert_allclose(loaded["w"], truncated, rtol=0, atol=0)


def test_repo_id_uses_local_hub_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(5)
    src = tmp_path / "hub-snap"
    dst = tmp_path / "out"
    src.mkdir()
    save_file(_make_weights(rng), str(src / "model.safetensors"))
    _write_json(src / "config.json", _tiny_config())

    monkeypatch.setattr(
        "maple_run.convert._try_local_hub_snapshot", lambda repo_id: src
    )
    convert_checkpoint("deepgrove/maple-preview", str(dst))
    assert (dst / "model.safetensors").exists()
    packed = load_file(str(dst / "model.safetensors"))
    assert "model.layers.0.mlp.switch_mlp.gate_proj.weight" in packed


def test_try_local_hub_snapshot_finds_maple_cache():
    snap = _try_local_hub_snapshot("deepgrove/maple-preview")
    if snap is None:
        pytest.skip("maple-preview hub snapshot not on disk")
    assert (snap / "config.json").exists()
    assert (snap / "model-00001-of-00009.safetensors").exists()


def test_cli_convert(tmp_path: Path):
    rng = np.random.default_rng(4)
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir()
    save_file(_make_weights(rng), str(src / "model.safetensors"))
    _write_json(src / "config.json", _tiny_config())
    assert main(["convert", str(src), "-o", str(dst)]) == 0
    assert (dst / "model.safetensors").exists()
    assert (dst / "config.json").exists()
