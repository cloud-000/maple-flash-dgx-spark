# Copyright © 2026 DeepGrove AI.
"""FlashHead: cluster the 4-bit lm_head and score only the top probes.

Port of ``generate_flash_head`` / ``FlashHead`` from
``docs/sources/mlx_lm_ternary.py`` and DeepGrove ``mlx_lm/models/maple.py``.

Phase one scores 4-bit cluster centroids. Phase two computes exact 4-bit
logits only for the tokens of the top ``n_probes`` clusters (plus forced
control tokens such as EOS). Prefill and the default generate path keep the
exact ``lm_head``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from maple_run.pack import HEAD_BITS, HEAD_GROUP_SIZE, dequantize_rtn, quantize_rtn

FLASHHEAD_SHARD = "model-flashhead.safetensors"
DEFAULT_N_CLUSTERS = 4748
DEFAULT_N_PROBES = 512
DEFAULT_KMEANS_ITERS = 60
_FORCE_TAGS = ("</think>", "<|im_end|>", "<|endoftext|>")


def _balanced_spherical_kmeans(
    W,
    n_clusters: int,
    n_iter: int = DEFAULT_KMEANS_ITERS,
    seed: int = 42,
):
    """Balanced spherical k-means over lm_head rows (equal-size clusters).

    ``W`` is ``[N, D]`` float32, already L2-normalized. Heavy matmuls stay on
    ``W.device``; overflow eviction is the same serial numpy pass as DeepGrove.
    """
    import torch

    N, D = W.shape
    cluster_size = N // n_clusters
    device = W.device
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    init = torch.randperm(N, generator=g, device=device)[:n_clusters]
    C = W.index_select(0, init)
    C = C / (C.norm(dim=1, keepdim=True) + 1e-8)

    best_obj = -1.0
    best_centroids = None
    best_labels = None
    for it in range(n_iter):
        labels = (W @ C.T).argmax(dim=1)
        labels_np = labels.detach().cpu().numpy().astype(np.int32, copy=True)

        counts = np.bincount(labels_np, minlength=n_clusters)
        over = np.where(counts > cluster_size)[0]
        if len(over):
            evict: list[int] = []
            for ci in over:
                members = np.where(labels_np == ci)[0]
                mem_t = torch.from_numpy(members.astype(np.int64)).to(device)
                sims = (W.index_select(0, mem_t) * C[int(ci)]).sum(dim=1)
                sims_np = sims.detach().cpu().numpy()
                keep = np.argpartition(sims_np, -cluster_size)[-cluster_size:]
                mask = np.ones(len(members), dtype=bool)
                mask[keep] = False
                dumped = members[mask]
                labels_np[dumped] = -1
                evict.extend(dumped.tolist())

            recount = np.bincount(
                labels_np[labels_np >= 0], minlength=n_clusters
            ).astype(np.int32)
            full = recount >= cluster_size
            evict_arr = np.array(evict, dtype=np.int64)
            for bs in range(0, len(evict_arr), 2048):
                pts = evict_arr[bs : bs + 2048]
                pts_t = torch.from_numpy(pts).to(device)
                sims_b = (W.index_select(0, pts_t) @ C.T).detach().cpu().numpy()
                sims_b[:, full] = -2.0
                for i, pt in enumerate(pts):
                    bc = int(np.argmax(sims_b[i]))
                    if sims_b[i, bc] <= -2.0:
                        bc = int(np.argmax(recount < cluster_size))
                    labels_np[int(pt)] = bc
                    recount[bc] += 1
                    if recount[bc] >= cluster_size:
                        full[bc] = True
                        sims_b[:, bc] = -2.0

        labels = torch.from_numpy(labels_np.astype(np.int64, copy=False)).to(device)
        counts_f = torch.bincount(labels, minlength=n_clusters).clamp(min=1).to(W.dtype)
        sums = torch.zeros((n_clusters, D), device=device, dtype=W.dtype)
        sums.index_add_(0, labels, W)
        C = sums / counts_f[:, None]
        C = C / (C.norm(dim=1, keepdim=True) + 1e-8)

        obj = float((W * C.index_select(0, labels)).sum().item() / N)
        if obj > best_obj:
            best_obj = obj
            best_centroids = C.detach().float().cpu().numpy().copy()
            best_labels = labels_np.copy()
        if (it + 1) % 5 == 0 or it < 3:
            print(f"  kmeans iter {it + 1}/{n_iter}: obj={obj:.6f}", flush=True)

    return best_centroids, best_labels


def _force_tokens(model_dir: Path, config: dict, vocab_size: int) -> list[int]:
    force: list[int] = []
    eos = config.get("eos_token_id")
    if eos is not None:
        force.extend(eos if isinstance(eos, list) else [eos])
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        for tag in _FORCE_TAGS:
            force.extend(tok.encode(tag, add_special_tokens=False))
    except Exception as exc:
        print(f"  tokenizer unavailable ({exc}); forcing EOS only", flush=True)
    return [int(t) for t in dict.fromkeys(force) if 0 <= int(t) < vocab_size]


def _load_lm_head(model_dir: Path, weight_map: dict[str, str]) -> dict[str, np.ndarray]:
    needed = ["lm_head.weight", "lm_head.scales", "lm_head.biases"]
    if any(k not in weight_map for k in needed):
        raise FileNotFoundError(
            "Checkpoint has no quantized lm_head (lm_head.weight/scales/biases); "
            "cannot attach FlashHead."
        )
    from safetensors import safe_open

    tensors: dict[str, np.ndarray] = {}
    for shard in sorted({weight_map[k] for k in needed}):
        with safe_open(str(model_dir / shard), framework="numpy") as fh:
            for k in needed:
                if k in fh.keys():
                    tensors[k] = fh.get_tensor(k)
    missing = [k for k in needed if k not in tensors]
    if missing:
        raise FileNotFoundError(f"lm_head tensors missing from shards: {missing}")
    return tensors


def generate_flash_head(
    model_dir: str,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    n_iter: int = DEFAULT_KMEANS_ITERS,
    n_probes: int = DEFAULT_N_PROBES,
    seed: int = 42,
    head_copy: bool = False,
) -> dict:
    """Cluster the lm_head of a packed checkpoint and attach FlashHead data.

    Writes ``lm_head_flash.*`` into ``model-flashhead.safetensors`` and records
    metadata (including forced control tokens) in ``config.json``. Does not
    rewrite the ternary body shards.
    """
    import torch

    model_dir = Path(model_dir).expanduser()
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("tie_word_embeddings"):
        raise SystemExit("FlashHead requires an untied lm_head.")
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"Missing {index_path}; convert before --flash-head-only.")
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    tensors = _load_lm_head(model_dir, weight_map)

    maple_run = config.get("maple_run") or {}
    rtn = maple_run.get("rtn_4bit") or (config.get("quantization") or {}).get("lm_head") or {}
    group_size = int(rtn.get("group_size", HEAD_GROUP_SIZE))
    bits = int(rtn.get("bits", HEAD_BITS))
    W = dequantize_rtn(
        tensors["lm_head.weight"],
        tensors["lm_head.scales"],
        tensors["lm_head.biases"],
        group_size=group_size,
        bits=bits,
    ).astype(np.float32, copy=False)
    row_norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    # Cluster directions, not magnitudes: high-frequency tokens have small
    # rows but large cosines; without row normalization their clusters are
    # directionally incoherent and the probe phase misses them.
    W_dir = W / row_norms
    vocab_size = int(W_dir.shape[0])
    if vocab_size % n_clusters:
        raise SystemExit(
            f"--clusters must divide the vocab size {vocab_size}; got {n_clusters}"
        )
    if n_probes > n_clusters:
        raise SystemExit(
            f"--probes ({n_probes}) must not exceed --clusters ({n_clusters})"
        )
    cluster_size = vocab_size // n_clusters
    if not torch.cuda.is_available() and vocab_size >= 10_000:
        raise SystemExit("FlashHead clustering of this vocab needs CUDA.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"FlashHead: clustering {vocab_size} x {W_dir.shape[1]} lm_head into "
        f"{n_clusters} clusters of {cluster_size}",
        flush=True,
    )
    W_t = torch.from_numpy(np.ascontiguousarray(W_dir)).to(device)
    centroids, labels = _balanced_spherical_kmeans(W_t, n_clusters, n_iter, seed)
    del W_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    token_map = np.zeros((n_clusters, cluster_size), dtype=np.int32)
    for c in range(n_clusters):
        members = list(np.where(labels == c)[0][:cluster_size])
        while len(members) < cluster_size:
            members.append(int(members[0]) if members else 0)
        token_map[c] = members

    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    cluster_scale = row_norms[:, 0][token_map].max(axis=1)
    scaled = centroids * cluster_scale[:, None]
    # Match MLX: quantize after rounding to bf16.
    scaled_bf16 = torch.from_numpy(scaled.astype(np.float32, copy=False)).to(
        torch.bfloat16
    ).float().numpy()
    cw, cscales, cbiases = quantize_rtn(scaled_bf16, group_size=HEAD_GROUP_SIZE, bits=HEAD_BITS)
    force = _force_tokens(model_dir, config, vocab_size)

    out: dict[str, np.ndarray] = {
        "lm_head_flash.centroids.weight": np.ascontiguousarray(cw),
        "lm_head_flash.centroids.scales": np.ascontiguousarray(cscales, dtype=np.float32),
        "lm_head_flash.centroids.biases": np.ascontiguousarray(cbiases, dtype=np.float32),
        "lm_head_flash.token_map": np.ascontiguousarray(token_map),
    }
    if head_copy:
        order = token_map.reshape(-1)
        for k in ("weight", "scales", "biases"):
            src = tensors[f"lm_head.{k}"]
            out[f"lm_head_flash.head.{k}"] = np.ascontiguousarray(
                src[order].reshape(n_clusters, cluster_size, *src.shape[1:])
            )

    shard_path = model_dir / FLASHHEAD_SHARD
    old_size = shard_path.stat().st_size if shard_path.exists() else 0
    save_file(out, str(shard_path))
    for k in [k for k, v in weight_map.items() if v == FLASHHEAD_SHARD and k not in out]:
        del weight_map[k]
    for k in out:
        weight_map[k] = FLASHHEAD_SHARD
    index["metadata"]["total_size"] = (
        int(index["metadata"].get("total_size", 0)) + shard_path.stat().st_size - old_size
    )
    index["weight_map"] = weight_map
    index_path.write_text(json.dumps(index, indent=2))

    meta = {
        "n_clusters": int(n_clusters),
        "cluster_size": int(cluster_size),
        "n_probes": int(n_probes),
        "group_size": HEAD_GROUP_SIZE,
        "bits": HEAD_BITS,
        "head_group_size": group_size,
        "head_bits": bits,
        "scaled_centroids": True,
        "force_tokens": force,
    }
    config["flash_head"] = meta
    maple_run = dict(config.get("maple_run") or {})
    maple_run["flash_head"] = meta
    config["maple_run"] = maple_run
    (model_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    print(f"FlashHead attached ({n_clusters} clusters, force={force})", flush=True)
    return meta


class FlashHead:
    """Two-phase approximate lm_head for single-stream decode.

    Greedy decoding is exact whenever the true argmax lies in a probed
    cluster. Prefill (seq_len != 1) stays on the exact head.
    """

    def __init__(
        self,
        *,
        centroids_weight,
        centroids_scales,
        centroids_biases,
        token_map,
        head_weight,
        head_scales,
        head_biases,
        force_ids,
        force_weight,
        force_scales,
        force_biases,
        n_probes: int,
        vocab_size: int,
        group_size: int = HEAD_GROUP_SIZE,
        dtype: "torch.dtype",
        device: "torch.device",
    ):
        import torch

        self.centroids_weight = centroids_weight.contiguous()
        self.centroids_scales = centroids_scales.contiguous()
        self.centroids_biases = centroids_biases.contiguous()
        self.token_map = token_map.contiguous()
        self.head_weight = head_weight.contiguous()
        self.head_scales = head_scales.contiguous()
        self.head_biases = head_biases.contiguous()
        self.n_clusters, self.cluster_size, _ = self.head_weight.shape
        self.n_probes = min(int(n_probes), self.n_clusters)
        self.group_size = int(group_size)
        self.vocab_size = int(vocab_size)
        self.force_ids = force_ids.contiguous()
        self.force_weight = force_weight
        self.force_scales = force_scales
        self.force_biases = force_biases
        self.n_force = int(self.force_ids.numel())
        self._full = torch.empty(1, 1, self.vocab_size, device=device, dtype=dtype)
        self._cluster_ids = torch.empty(self.n_probes, device=device, dtype=torch.int32)

    def __call__(self, hidden):
        import torch

        from maple_run.kernels.rtn4 import rtn4_gemv, rtn4_indexed_gemv

        hv = hidden[:, -1, :]
        scores = rtn4_gemv(
            hv,
            self.centroids_weight,
            self.centroids_scales,
            self.centroids_biases,
            group_size=self.group_size,
        )
        _, topi = torch.topk(scores, self.n_probes, dim=-1)
        self._cluster_ids.copy_(topi.reshape(-1).to(torch.int32))
        probe = rtn4_indexed_gemv(
            hv,
            self.head_weight,
            self.head_scales,
            self.head_biases,
            self._cluster_ids,
            group_size=self.group_size,
        ).reshape(-1)
        oids = self.token_map.index_select(0, self._cluster_ids.to(torch.int64)).reshape(
            -1
        )
        logits = probe
        if self.n_force:
            force = rtn4_gemv(
                hv,
                self.force_weight,
                self.force_scales,
                self.force_biases,
                group_size=self.group_size,
            ).reshape(-1)
            oids = torch.cat([oids, self.force_ids], dim=0)
            logits = torch.cat([logits, force], dim=0)
        self._full.fill_(float("-inf"))
        self._full.view(-1).index_put_((oids.long(),), logits, accumulate=False)
        return self._full


def build_flash_head(
    *,
    config: dict,
    lm_head,
    weights: dict,
    device,
    dtype,
) -> FlashHead:
    """Construct ``FlashHead`` from checkpoint tensors; permute lm_head at load."""
    import torch

    meta = config.get("flash_head") or (config.get("maple_run") or {}).get("flash_head")
    if not meta:
        raise ValueError("Checkpoint has no flash_head metadata; run --flash-head-only.")
    if not meta.get("scaled_centroids"):
        raise ValueError(
            "FlashHead metadata predates scaled centroids; regenerate with "
            "`maple-run convert --flash-head-only`."
        )
    n_clusters = int(meta["n_clusters"])
    cluster_size = int(meta["cluster_size"])
    n_probes = int(meta.get("n_probes", DEFAULT_N_PROBES))
    group_size = int(meta.get("head_group_size", meta.get("group_size", HEAD_GROUP_SIZE)))
    token_map = weights.pop("lm_head_flash.token_map").to(device=device, dtype=torch.int32)
    if tuple(token_map.shape) != (n_clusters, cluster_size):
        raise ValueError(
            f"token_map shape {tuple(token_map.shape)} != "
            f"{(n_clusters, cluster_size)} from flash_head metadata."
        )
    order = token_map.reshape(-1).to(torch.int64)
    if "lm_head_flash.head.weight" in weights:
        head_w = weights.pop("lm_head_flash.head.weight")
        head_s = weights.pop("lm_head_flash.head.scales")
        head_b = weights.pop("lm_head_flash.head.biases")
        if head_w.dtype != torch.uint32:
            head_w = head_w.view(torch.uint32)
        head_w = head_w.reshape(n_clusters, cluster_size, -1)
        head_s = head_s.reshape(n_clusters, cluster_size, -1)
        head_b = head_b.reshape(n_clusters, cluster_size, -1)
    else:
        packed = lm_head.packed_weight
        if packed.dtype != torch.uint32:
            packed = packed.view(torch.uint32)
        head_w = packed.view(torch.int32).index_select(0, order).view(torch.uint32)
        head_w = head_w.reshape(n_clusters, cluster_size, -1).contiguous()
        head_s = lm_head.scales.index_select(0, order).reshape(
            n_clusters, cluster_size, -1
        ).contiguous()
        head_b = lm_head.biases.index_select(0, order).reshape(
            n_clusters, cluster_size, -1
        ).contiguous()

    centroids_w = weights.pop("lm_head_flash.centroids.weight")
    if centroids_w.dtype != torch.uint32:
        centroids_w = centroids_w.view(torch.uint32)
    centroids_s = weights.pop("lm_head_flash.centroids.scales")
    centroids_b = weights.pop("lm_head_flash.centroids.biases")
    weights.pop("lm_head_flash.cluster_scale", None)

    force = [int(t) for t in meta.get("force_tokens", []) if 0 <= int(t) < int(config["vocab_size"])]
    if force:
        force_ids = torch.tensor(force, device=device, dtype=torch.int64)
        force_w = (
            lm_head.packed_weight.view(torch.int32)
            .index_select(0, force_ids)
            .view(torch.uint32)
            .contiguous()
        )
        force_s = lm_head.scales.index_select(0, force_ids).contiguous()
        force_b = lm_head.biases.index_select(0, force_ids).contiguous()
    else:
        force_ids = torch.empty(0, device=device, dtype=torch.int64)
        force_w = force_s = force_b = None

    return FlashHead(
        centroids_weight=centroids_w,
        centroids_scales=centroids_s,
        centroids_biases=centroids_b,
        token_map=token_map,
        head_weight=head_w,
        head_scales=head_s,
        head_biases=head_b,
        force_ids=force_ids,
        force_weight=force_w,
        force_scales=force_s,
        force_biases=force_b,
        n_probes=n_probes,
        vocab_size=int(config["vocab_size"]),
        group_size=group_size,
        dtype=dtype,
        device=device,
    )
