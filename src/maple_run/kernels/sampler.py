"""Fused top-k / temperature / nucleus sampling over the full vocab.

The torch chain (``topk`` -> ``softmax`` -> ``sort`` -> ``cumsum`` -> ``scatter``
-> ``multinomial``) is about ten launches per token. Two Triton launches replace
that; generate captures them in the decode CUDA graph when replay matches eager.

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
    stride_lb,
    stride_l,
    stride_cb,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Top-K of one vocab block. Any global top-K token is in its block's top-K.

    The batch is a grid dimension: rows share no work, and one program per
    (block, row) keeps a batched draw the same arithmetic as ``B = 1``.
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, BLOCK)
    x = tl.load(
        logits_ptr + b * stride_lb + offs * stride_l, mask=offs < VOCAB, other=_NEG_INF
    ).to(tl.float32)
    base = b * stride_cb + pid * K
    for i in tl.static_range(0, K):
        j = tl.argmax(x, axis=0)
        tl.store(vals_ptr + base + i, tl.max(x, axis=0))
        tl.store(idx_ptr + base + i, pid * BLOCK + j)
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
    stride_cb,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Merge block candidates, then temperature softmax + nucleus + CDF inverse."""
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < n_cand
    v = tl.load(vals_ptr + b * stride_cb + offs, mask=mask, other=_NEG_INF).to(tl.float32)
    ids = tl.load(idx_ptr + b * stride_cb + offs, mask=mask, other=0)

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
    u = tl.load(uniform_ptr + b).to(tl.float32) * total
    # First index whose inclusive prefix exceeds u. Dropped tokens leave cum
    # flat, so they can never be the first to exceed it.
    pick = tl.sum(tl.where((cum <= u) & (karange < K), 1, 0), axis=0)
    pick = tl.minimum(pick, K - 1)
    tl.store(out_ptr + b, tl.sum(tl.where(karange == pick, ki, 0), axis=0).to(tl.int64))


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

    _topk_partial_kernel[(n_blocks, 1)](
        logits, vals, idx, vocab, 0, logits.stride(0), 0,
        K=top_k, BLOCK=_BLOCK, num_warps=4,
    )
    _sample_final_kernel[(1,)](
        vals, idx, uniform, out, n_cand,
        1.0 / float(temperature), float(top_p), 0,
        K=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        BLOCK_C=triton.next_power_of_2(n_cand),
        num_warps=4,
    )
    return out


def fused_sample_batched(
    logits: torch.Tensor,
    uniform: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sample one id per row of ``logits`` ``[B, vocab]``. ``uniform`` is ``[B]``.

    Same two launches as :func:`fused_sample`, with the batch on the grid, so
    row ``b`` here draws exactly what it would have drawn alone given the same
    uniform. Returns int64 ``[B]``.

    ``temperature``, ``top_p`` and ``top_k`` apply to the whole batch. Per-row
    settings would need per-row scalars (and, for ``top_k``, a recompile, since
    it is a ``tl.static_range`` bound); a scheduler mixing request settings
    wants to bucket rows by their sampling params rather than pay that.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"fused_sample_batched expects [B, vocab], got {tuple(logits.shape)}."
        )
    if not 0 < top_k <= MAX_TOP_K:
        raise ValueError(f"fused_sample needs 0 < top_k <= {MAX_TOP_K}, got {top_k}.")
    if temperature <= 0:
        raise ValueError("fused_sample needs temperature > 0 (greedy is argmax).")

    bsz, vocab = logits.shape
    if uniform.numel() != bsz:
        raise ValueError(f"uniform has {uniform.numel()} draws, expected {bsz}.")
    n_blocks = triton.cdiv(vocab, _BLOCK)
    n_cand = n_blocks * top_k
    if workspace is None:
        vals = torch.empty(bsz, n_cand, device=logits.device, dtype=torch.float32)
        idx = torch.empty(bsz, n_cand, device=logits.device, dtype=torch.int32)
        out = torch.empty(bsz, device=logits.device, dtype=torch.int64)
    else:
        vals, idx, out = workspace
        vals, idx, out = vals[:bsz], idx[:bsz], out[:bsz]

    _topk_partial_kernel[(n_blocks, bsz)](
        logits, vals, idx, vocab, logits.stride(0), logits.stride(1), vals.stride(0),
        K=top_k, BLOCK=_BLOCK, num_warps=4,
    )
    _sample_final_kernel[(bsz,)](
        vals, idx, uniform.reshape(-1), out, n_cand,
        1.0 / float(temperature), float(top_p), vals.stride(0),
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


def batched_sampler_workspace(
    vocab: int, top_k: int, batch: int, device
) -> tuple[torch.Tensor, ...]:
    """``fused_sample_batched`` scratch for up to ``batch`` rows.

    Sized once at the widest bucket; narrower replays slice off the front, so
    a bucketed decode graph never allocates.
    """
    n_cand = triton.cdiv(vocab, _BLOCK) * top_k
    return (
        torch.empty(batch, n_cand, device=device, dtype=torch.float32),
        torch.empty(batch, n_cand, device=device, dtype=torch.int32),
        torch.empty(batch, device=device, dtype=torch.int64),
    )
