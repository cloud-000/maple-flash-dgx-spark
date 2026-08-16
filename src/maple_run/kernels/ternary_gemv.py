"""Packed ternary GEMV: y = x @ W_packed, W never unpacked to bf16.

Target: DGX Spark GB10 (sm_121). Triton reads uint32 codes in place, accumulates
``(code - 1) * x`` in float32, then scales by per-row ``α``. A kernel that
dequantizes to a dense bf16 matrix and then matmuls has failed the design.

Packing (see ``maple_run.pack``): 16 LSB-first 2-bit codes per uint32, codes
``{0, 1, 2}`` = ``{−1, 0, +1} + 1``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_CODES_PER_WORD = 16


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16, "BLOCK_K_WORDS": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K_WORDS": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K_WORDS": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K_WORDS": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K_WORDS": 32}, num_warps=4, num_stages=3),
    ],
    key=["N", "nwords"],
)
@triton.jit
def _ternary_gemv_kernel(
    x_ptr,
    packed_ptr,
    alpha_ptr,
    y_ptr,
    N,
    nwords,
    stride_xb,
    stride_xk,
    stride_wn,
    stride_ww,
    stride_a,
    stride_yb,
    stride_yn,
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

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
        ).to(tl.uint32)
        # codes[n, w, i] = (word >> (2*i)) & 3  →  ternary = code - 1 ∈ {-1,0,+1}
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0x3
        ternary = tl.reshape(codes.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))

        offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
        # K == nwords * 16; mask invalid K from a partial word tile.
        mask_k = offs_k < (nwords * 16)
        x_tile = tl.load(
            x_row + offs_k * stride_xk,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(ternary * x_tile[None, :], axis=1)

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
) -> torch.Tensor:
    """Decode GEMV ``y = x @ W`` with ``W`` kept as packed uint32 codes.

    Parameters
    ----------
    x:
        Activations ``[K]`` or ``[B, K]`` (any leading dims are flattened).
    packed_weight:
        ``uint32`` codes ``[N, K/16]``.
    row_alpha:
        Per-output-row scale ``[N]``.
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
            f"packed_weight must be 2-D [N, K/16]; got shape {tuple(packed_weight.shape)}. "
            "Stacked experts are a later fused kernel."
        )

    n_out, nwords = packed_weight.shape
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

    def _grid(meta):
        return (triton.cdiv(n_out, meta["BLOCK_N"]), batch)

    _ternary_gemv_kernel[_grid](
        x_mat,
        packed_weight,
        alpha,
        y,
        n_out,
        nwords,
        x_mat.stride(0),
        x_mat.stride(1),
        packed_weight.stride(0),
        packed_weight.stride(1),
        alpha.stride(0),
        y.stride(0),
        y.stride(1),
    )
    return y.reshape(*leading, n_out)
