"""Fused top-k / temperature / nucleus sampling over the full vocab.

The torch chain (``topk`` -> ``softmax`` -> ``sort`` -> ``cumsum`` -> ``scatter``
-> ``multinomial``) is about ten launches per token, and unlike the decode
forward it runs outside the CUDA graph, so it pays CPU launch latency too:
~204 us/token measured by ablation on this host. This is two launches — a
per-block top-k over the 151936 logits, then one CTA that merges them,
softmaxes, applies the nucleus and inverts the CDF.

Randomness still comes from torch: the caller passes one uniform drawn from the
generator ``--seed`` selects, so seeded runs stay reproducible and the CUDA
graph replay check keeps comparing like with like.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_TOP_K = 64
_BLOCK = 4096

_NEG_INF = tl.constexpr(float("-inf"))


@triton.jit
def _topk_partial_kernel(
    logits_ptr,
    vals_ptr,
    idx_ptr,
    VOCAB,
    stride_l,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Top-K of one vocab block. Any global top-K token is in its block's top-K."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, BLOCK)
    x = tl.load(logits_ptr + offs * stride_l, mask=offs < VOCAB, other=_NEG_INF).to(
        tl.float32
    )
    for i in tl.static_range(0, K):
        j = tl.argmax(x, axis=0)
        tl.store(vals_ptr + pid * K + i, tl.max(x, axis=0))
        tl.store(idx_ptr + pid * K + i, pid * BLOCK + j)
        x = tl.where(lane == j, _NEG_INF, x)


@triton.jit
def _sample_final_kernel(
    vals_ptr,
    idx_ptr,
    uniform_ptr,
    out_ptr,
    n_cand,
    inv_temperature,
    top_p,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Merge block candidates, then temperature softmax + nucleus + CDF inverse."""
    offs = tl.arange(0, BLOCK_C)
    mask = offs < n_cand
    v = tl.load(vals_ptr + offs, mask=mask, other=_NEG_INF).to(tl.float32)
    ids = tl.load(idx_ptr + offs, mask=mask, other=0)

    karange = tl.arange(0, BLOCK_K)
    kv = tl.full((BLOCK_K,), _NEG_INF, dtype=tl.float32)
    ki = tl.zeros((BLOCK_K,), dtype=tl.int32)
    # Extracting the top-K by repeated argmax also leaves it sorted descending,
    # which is exactly the order the nucleus cutoff needs.
    for i in tl.static_range(0, K):
        j = tl.argmax(v, axis=0)
        kv = tl.where(karange == i, tl.max(v, axis=0), kv)
        ki = tl.where(karange == i, tl.sum(tl.where(offs == j, ids, 0), axis=0), ki)
        v = tl.where(offs == j, _NEG_INF, v)

    p = tl.exp((kv - tl.max(kv, axis=0)) * inv_temperature)
    p = tl.where(karange < K, p, 0.0)
    p = p / tl.sum(p, axis=0)

    # Nucleus: drop tokens whose exclusive prefix mass already reached top_p.
    cum = tl.cumsum(p, axis=0)
    p = tl.where(cum - p <= top_p, p, 0.0)

    total = tl.sum(p, axis=0)
    cum = tl.cumsum(p, axis=0)
    u = tl.load(uniform_ptr).to(tl.float32) * total
    # First index whose inclusive prefix exceeds u. Dropped tokens leave cum
    # flat, so they can never be the first to exceed it.
    pick = tl.sum(tl.where((cum <= u) & (karange < K), 1, 0), axis=0)
    pick = tl.minimum(pick, K - 1)
    tl.store(out_ptr, tl.sum(tl.where(karange == pick, ki, 0), axis=0).to(tl.int64))


def fused_sample(
    logits: torch.Tensor,
    uniform: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sample one token id from 1-D ``logits``. ``uniform`` is a scalar in [0, 1).

    Equivalent to ``sample_next``: top-k on logits, temperature softmax over
    those, nucleus ``top_p``, then draw. Returns a 0-D int64 CUDA tensor.
    """
    if logits.ndim != 1:
        raise ValueError(f"fused_sample expects 1-D logits, got {tuple(logits.shape)}.")
    if not 0 < top_k <= MAX_TOP_K:
        raise ValueError(f"fused_sample needs 0 < top_k <= {MAX_TOP_K}, got {top_k}.")
    if temperature <= 0:
        raise ValueError("fused_sample needs temperature > 0 (greedy is argmax).")

    vocab = int(logits.shape[0])
    n_blocks = triton.cdiv(vocab, _BLOCK)
    n_cand = n_blocks * top_k
    if workspace is None:
        vals = torch.empty(n_cand, device=logits.device, dtype=torch.float32)
        idx = torch.empty(n_cand, device=logits.device, dtype=torch.int32)
        out = torch.empty((), device=logits.device, dtype=torch.int64)
    else:
        vals, idx, out = workspace

    _topk_partial_kernel[(n_blocks,)](
        logits, vals, idx, vocab, logits.stride(0),
        K=top_k, BLOCK=_BLOCK, num_warps=4,
    )
    _sample_final_kernel[(1,)](
        vals, idx, uniform, out, n_cand,
        1.0 / float(temperature), float(top_p),
        K=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        BLOCK_C=triton.next_power_of_2(n_cand),
        num_warps=4,
    )
    return out


def sampler_workspace(vocab: int, top_k: int, device) -> tuple[torch.Tensor, ...]:
    """Preallocate ``fused_sample`` scratch so decode does not allocate per token."""
    n_cand = triton.cdiv(vocab, _BLOCK) * top_k
    return (
        torch.empty(n_cand, device=device, dtype=torch.float32),
        torch.empty(n_cand, device=device, dtype=torch.int32),
        torch.empty((), device=device, dtype=torch.int64),
    )
