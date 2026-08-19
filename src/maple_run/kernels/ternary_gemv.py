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


def _gemv_launch_meta(nwords: int, batch: int) -> tuple[int, int, int, int, int]:
    """BLOCK_N, BLOCK_K_WORDS, BLOCK_B, warps, stages.

    Decode (batch=1) needs many CTAs. K tiles wider than 64 words spill: a
    ``[BLOCK_N, BLOCK_K_WORDS*16]`` float32 ternary tile at 128 words is 64 KB
    per CTA. The width was picked by timing full decode, not a microbench:
    cold-but-isolated the 4- and 8-row tiles look equal, in the model the 4-row
    tile is worth ~100 us/token (369 -> 384 tok/s) because 768 programs cover
    DRAM latency where 384 do not.

    ``batch > 1`` runs ``_ternary_gemm_kernel``, where ``BLOCK_B`` rows share one
    load of the codes, so the shape of the tuning problem inverts: the batch-1
    kernel wants many thin CTAs (``BLOCK_N=2``, 1536 of them) to cover DRAM
    latency, the batched one wants fatter CTAs that keep each weight tile busy
    across ``BLOCK_B`` columns. Tuned against cold weights
    (``bench/tune_batched_gemv.py``: 768 MB of layer-sized replicas cycled
    round-robin so a tile is never resident in L2 when it is reused) -- hot-L2
    microbenchmarks pick a different and much worse config here, as they do for
    the batch-1 path.

    ``BLOCK_B`` grows with the batch only up to the point where the register
    tile starts costing occupancy. Decode batches (<= 32) stay narrow; prefill
    arrives here too, with batch = prompt length, and wants the wide tiles.

    Tuned with the fused RMSNorm on, because QKV -- the projection that matters
    -- always folds it in, and that path carries a second dot for the residual
    of ``x * rms_w`` (see the kernel). Tuning the bare GEMV picks configs that
    fall apart once that dot appears: ``BLOCK_N=64 / warps=8`` measured 17 us
    bare and 455 us fused. Against a config tuned for it the residual dot costs
    ~8% (25.1 vs 23.3 us at B=8), which is what it takes to land on the batch-1
    kernel's rounding exactly rather than merely inside a bfloat16 ulp.
    """
    if batch == 1:
        if nwords >= 64:
            return 2, 64, 1, 1, 3
        if nwords >= 32:
            return 16, 32, 1, 4, 4
        return 16, 8, 1, 4, 2
    if nwords < 16:
        block_b = 16 if batch <= 16 else 32
        return 32, 4, block_b, 4, 2
    if batch <= 16:                     # 25.1 us at B=8 on cold QKV
        return 32, 8, 16, 2, 3
    if batch <= 32:                     # 37.1 us; the residual dot is why the
        return 16, 2, 32, 2, 3          # K tile has to shrink this far at B=32
    if batch <= 64:                     # 50.6 us at B=64
        return 32, 8, 64, 4, 2
    return 64, 8, 128, 8, 3             # prefill: 127 us at B=256, 502 at B=1024


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


@triton.jit
def _ternary_gemm_kernel(
    x_ptr,
    packed_ptr,
    alpha_ptr,
    y_ptr,
    rms_w_ptr,
    N,
    nwords,
    B,
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
    IEEE: tl.constexpr,
    SPLIT_X: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """batch>1: one CTA holds ``[BLOCK_N, BLOCK_B]`` and reads each code tile once.

    The batch-1 kernel puts the batch on the grid, so B rows re-read the whole
    weight B times. Here B is a register tile and the codes are loaded once,
    which is the entire point: decode weight traffic stops scaling with B.

    Ternary values {-1, 0, +1} are exact in bfloat16 and the accumulator stays
    float32, so against the batch-1 kernel this is the same arithmetic in a
    different summation order, not a lower-precision path. float32 activations
    take ``input_precision="ieee"`` rather than being demoted to tf32.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    acc = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_B,), dtype=tl.float32)

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 16
    shifts = (tl.arange(0, 16) * 2).to(tl.uint32)
    k_total = nwords * 16

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
            eviction_policy="evict_first" if STREAM_W else "",
        ).to(tl.uint32)
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0x3

        offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_total
        x_tile = tl.load(
            x_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk,
            mask=mask_b[:, None] & mask_k[None, :],
            other=0.0,
            eviction_policy="evict_last" if STREAM_W else "",
        )
        xf = x_tile.to(tl.float32)
        # Same folded-RMSNorm identity as the batch-1 kernel, but rstd is now a
        # vector over the B rows instead of a scalar.
        if HAS_RMS:
            sumsq += tl.sum(xf * xf, axis=1)
            rms_w = tl.load(rms_w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            xf = xf * rms_w[None, :]

        if IEEE:
            ternary = tl.reshape(codes.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
            acc += tl.dot(ternary, tl.trans(xf), input_precision="ieee")
        else:
            ternary = tl.reshape(codes.to(tl.bfloat16) - 1.0, (BLOCK_N, BLOCK_K))
            x_hi = xf.to(tl.bfloat16)
            acc += tl.dot(ternary, tl.trans(x_hi))
            # x*rms_w is computed in float32 and would lose ~8 mantissa bits on
            # the way into a bf16 dot. The residual carries them: hi + lo holds
            # x*w to roughly float32, and ternary is exact, so the fused-norm
            # path keeps the batch-1 kernel's precision for one extra dot on
            # operands already in registers.
            if SPLIT_X:
                x_lo = (xf - x_hi.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(ternary, tl.trans(x_lo))

    if HAS_RMS:
        acc = acc * tl.rsqrt(sumsq / k_total + eps)[None, :]

    alpha = tl.load(alpha_ptr + offs_n * stride_a, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(
        y_ptr + offs_b[None, :] * stride_yb + offs_n[:, None] * stride_yn,
        acc * alpha[:, None],
        mask=mask_n[:, None] & mask_b[None, :],
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

    block_n, block_k_words, block_b, num_warps, num_stages = _gemv_launch_meta(
        nwords, batch
    )
    if batch == 1:
        grid = (triton.cdiv(n_out, block_n), 1)
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
            STREAM_W=True,
            BLOCK_N=block_n,
            BLOCK_K_WORDS=block_k_words,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y.reshape(*leading, n_out)

    # bfloat16 holds ternary exactly; every other dtype takes the float32 path.
    ieee = x_mat.dtype != torch.bfloat16
    b_tiles = triton.cdiv(batch, block_b)
    grid = (triton.cdiv(n_out, block_n), b_tiles)
    _ternary_gemm_kernel[grid](
        x_mat,
        packed_weight,
        alpha,
        y,
        rms_w,
        n_out,
        nwords,
        batch,
        x_mat.stride(0),
        x_mat.stride(1),
        stride_wn,
        stride_ww,
        alpha.stride(0),
        y.stride(0),
        y.stride(1),
        float(rms_eps),
        HAS_RMS=has_rms,
        # One B tile reads each code tile exactly once; more than one re-reads
        # them, so the codes are worth keeping in L2 instead of evicting.
        STREAM_W=b_tiles == 1,
        IEEE=ieee,
        SPLIT_X=(not ieee) and has_rms,
        BLOCK_N=block_n,
        BLOCK_K_WORDS=block_k_words,
        BLOCK_B=block_b,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.reshape(*leading, n_out)
