"""Packed ternary GEMV: y = x @ W_packed, W never unpacked to bf16.

    Target: DGX Spark GB10 (sm_121). Triton reads uint32 codes in place, accumulates
    ``(code - 1) * x`` in float32, then scales by per-row ``α``. A kernel that
    dequantizes to a dense bf16 matrix and then matmuls has failed the design.

    Decode (batch=1) weight loads use ``evict_first`` so a layer's codes do not
    occupy L2 after the GEMV; activations stay ``evict_last``. Prefill reuses the
    same tiles across tokens, so those hints stay off. In-model QKV/O otherwise
    lands at ~131 GB/s against ~175 isolated because the next kernel's weights
    fight the last.

Packing (see ``maple_run.pack``): 16 LSB-first 2-bit codes per uint32, codes
``{0, 1, 2}`` = ``{−1, 0, +1} + 1``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_CODES_PER_WORD = 16


def _gemv_launch_meta(nwords: int, batch: int) -> tuple[int, int, int, int]:
    """BLOCK_N, BLOCK_K_WORDS, warps, stages. Decode (batch=1) needs many CTAs.

    K tiles wider than 64 words spill: a ``[BLOCK_N, BLOCK_K_WORDS*16]`` float32
    ternary tile at 128 words is 64 KB per CTA. The width was picked by timing
    full decode, not a microbench: cold-but-isolated the 4- and 8-row tiles look
    equal, in the model the 4-row tile is worth ~100 us/token (369 -> 384 tok/s)
    because 768 programs cover DRAM latency where 384 do not.
    """
    if batch == 1:
        if nwords >= 64:
            return 2, 64, 1, 3
        if nwords >= 32:
            return 16, 32, 4, 4
        return 16, 8, 4, 2
    return 32, 16, 4, 3


@triton.jit
def _ternary_gemv_kernel(
    x_ptr,
    packed_ptr,
    alpha_ptr,
    y_ptr,
    rms_w_ptr,
    N,
    nwords,
    stride_xb,
    stride_xk,
    stride_wn,
    stride_ww,
    stride_a,
    stride_yb,
    stride_yn,
    eps,
    HAS_RMS: tl.constexpr,
    STREAM_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_row = x_ptr + pid_b * stride_xb

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 16
    shifts = (tl.arange(0, 16) * 2).to(tl.uint32)
    k_total = nwords * 16

    # RMSNorm folded in without a pre-pass: rstd is a scalar, so
    # sum_k T[n,k] * (x[k] * rstd * w[k]) == rstd * sum_k T[n,k] * (x[k] * w[k]).
    # Accumulating sumsq alongside the dot product keeps the weight loads at the
    # top of the kernel; a separate reduction loop first stalled every CTA and
    # cost QKV ~30 GB/s (142 vs 173 with the norm removed entirely).
    sumsq = tl.zeros((), dtype=tl.float32)

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
            eviction_policy="evict_first" if STREAM_W else "",
        ).to(tl.uint32)
        # codes[n, w, i] = (word >> (2*i)) & 3  →  ternary = code - 1 ∈ {-1,0,+1}
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0x3
        ternary = tl.reshape(codes.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))

        offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
        # K == nwords * 16; mask invalid K from a partial word tile.
        mask_k = offs_k < k_total
        x_tile = tl.load(
            x_row + offs_k * stride_xk,
            mask=mask_k,
            other=0.0,
            eviction_policy="evict_last" if STREAM_W else "",
        ).to(tl.float32)
        if HAS_RMS:
            sumsq += tl.sum(x_tile * x_tile, axis=0)
            rms_w = tl.load(rms_w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            x_tile = x_tile * rms_w
        acc += tl.sum(ternary * x_tile[None, :], axis=1)

    if HAS_RMS:
        acc = acc * tl.rsqrt(sumsq / k_total + eps)

    alpha = tl.load(alpha_ptr + offs_n * stride_a, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(
        y_ptr + pid_b * stride_yb + offs_n * stride_yn,
        acc * alpha,
        mask=mask_n,
    )


def ternary_gemv(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_alpha: torch.Tensor,
    *,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    packed_kn: bool = False,
) -> torch.Tensor:
    """Decode GEMV ``y = x @ W`` with ``W`` kept as packed uint32 codes.

    Parameters
    ----------
    x:
        Activations ``[K]`` or ``[B, K]`` (any leading dims are flattened).
    packed_weight:
        ``uint32`` codes ``[N, K/16]`` (``packed_kn=False``) or ``[K/16, N]``
        (``packed_kn=True``, coalesced consecutive output rows).
    row_alpha:
        Per-output-row scale ``[N]``.
    rms_weight:
        If set, RMSNorm ``x`` in-kernel (fused input RMSNorm + GEMV).
    packed_kn:
        If True, packed codes are stored ``[nwords, N]``.
    """
    if not x.is_cuda or not packed_weight.is_cuda or not row_alpha.is_cuda:
        raise RuntimeError("ternary_gemv requires CUDA tensors (packed weights stay on GPU).")
    if packed_weight.dtype != torch.uint32:
        raise TypeError(
            f"packed_weight must be uint32 codes, got {packed_weight.dtype}; "
            "refusing to run a dense-weight path."
        )
    if packed_weight.ndim != 2:
        raise ValueError(
            f"packed_weight must be 2-D [N, K/16] or [K/16, N]; got shape "
            f"{tuple(packed_weight.shape)}. Stacked experts are a later fused kernel."
        )

    if packed_kn:
        nwords, n_out = packed_weight.shape
        stride_wn = packed_weight.stride(1)
        stride_ww = packed_weight.stride(0)
    else:
        n_out, nwords = packed_weight.shape
        stride_wn = packed_weight.stride(0)
        stride_ww = packed_weight.stride(1)
    k_in = nwords * _CODES_PER_WORD
    alpha = row_alpha.reshape(-1)
    if alpha.numel() != n_out:
        raise ValueError(
            f"row_alpha has {alpha.numel()} elements, expected one per output row ({n_out})."
        )
    if x.shape[-1] != k_in:
        raise ValueError(f"x last dim {x.shape[-1]} != packed K ({k_in}).")

    leading = x.shape[:-1]
    batch = int(x.numel() // k_in)
    x_mat = x.reshape(batch, k_in)
    y = torch.empty(batch, n_out, device=x.device, dtype=x.dtype)
    has_rms = rms_weight is not None
    rms_w = x_mat if rms_weight is None else rms_weight.reshape(-1)
    if has_rms and rms_w.numel() != k_in:
        raise ValueError(f"rms_weight has {rms_w.numel()} elements, expected {k_in}.")

    block_n, block_k_words, num_warps, num_stages = _gemv_launch_meta(nwords, batch)
    grid = (triton.cdiv(n_out, block_n), batch)
    _ternary_gemv_kernel[grid](
        x_mat,
        packed_weight,
        alpha,
        y,
        rms_w,
        n_out,
        nwords,
        x_mat.stride(0),
        x_mat.stride(1),
        stride_wn,
        stride_ww,
        alpha.stride(0),
        y.stride(0),
        y.stride(1),
        float(rms_eps),
        HAS_RMS=has_rms,
        STREAM_W=batch == 1,
        BLOCK_N=block_n,
        BLOCK_K_WORDS=block_k_words,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.reshape(*leading, n_out)
