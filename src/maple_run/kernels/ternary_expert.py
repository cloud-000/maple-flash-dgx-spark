"""Fused packed-expert GEMV: selected experts only, weights stay 3-D uint32.

Decode (and prefill) dispatch is one kernel over ``(token, top-k slot)`` pairs.
A Python loop over 256 experts, or 256 separate launches, is the failure mode
this exists to replace. Packed codes are never unpacked to a dense bf16 matrix.
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
        triton.Config({"BLOCK_N": 128, "BLOCK_K_WORDS": 8}, num_warps=8, num_stages=2),
    ],
    key=["N", "nwords"],
)
@triton.jit
def _ternary_expert_gemv_kernel(
    x_ptr,
    packed_ptr,
    alpha_ptr,
    ids_ptr,
    y_ptr,
    N,
    nwords,
    stride_xs,
    stride_xk,
    stride_we,
    stride_wn,
    stride_ww,
    stride_ae,
    stride_an,
    stride_ys,
    stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    expert_id = tl.load(ids_ptr + pid_s).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_row = x_ptr + pid_s * stride_xs

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 16
    shifts = (tl.arange(0, 16) * 2).to(tl.uint32)
    packed_e = packed_ptr + expert_id * stride_we
    alpha_e = alpha_ptr + expert_id * stride_ae

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        packed = tl.load(
            packed_e + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
        ).to(tl.uint32)
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0x3
        ternary = tl.reshape(codes.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))

        offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < (nwords * 16)
        x_tile = tl.load(
            x_row + offs_k * stride_xk,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(ternary * x_tile[None, :], axis=1)

    alpha = tl.load(alpha_e + offs_n * stride_an, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(
        y_ptr + pid_s * stride_ys + offs_n * stride_yn,
        acc * alpha,
        mask=mask_n,
    )


def ternary_expert_gemv(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_alpha: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    """Indexed packed GEMV for stacked experts.

    Parameters
    ----------
    x:
        Activations. Either one row per token (``[..., K]`` with
        ``expert_ids`` shaped ``[..., topk]``) or one row per slot
        (``x.shape[:-1] == expert_ids.shape``).
    packed_weight:
        ``uint32`` codes ``[E, N, K/16]``.
    row_alpha:
        Per-expert per-output-row scale ``[E, N]``.
    expert_ids:
        Expert indices (any integer dtype) selecting rows of ``packed_weight``.

    Returns
    -------
    Tensor of shape ``expert_ids.shape + (N,)``.
    """
    if packed_weight.ndim != 3:
        raise ValueError(
            f"packed_weight must be 3-D [E, N, K/16]; got shape {tuple(packed_weight.shape)}."
        )
    if packed_weight.dtype != torch.uint32:
        raise TypeError(
            f"packed_weight must be uint32 codes, got {packed_weight.dtype}; "
            "refusing to run a dense-weight path."
        )
    if not x.is_cuda or not packed_weight.is_cuda or not row_alpha.is_cuda:
        raise RuntimeError("ternary_expert_gemv requires CUDA tensors.")
    if not expert_ids.is_cuda:
        raise RuntimeError("expert_ids must be a CUDA tensor (no GPU→CPU routing).")

    n_exp, n_out, nwords = packed_weight.shape
    k_in = nwords * _CODES_PER_WORD
    if x.shape[-1] != k_in:
        raise ValueError(f"x last dim {x.shape[-1]} != packed K ({k_in}).")
    if tuple(row_alpha.shape) != (n_exp, n_out):
        raise ValueError(
            f"row_alpha shape {tuple(row_alpha.shape)} != {(n_exp, n_out)}."
        )

    id_shape = tuple(expert_ids.shape)
    if x.shape[:-1] == id_shape:
        slot_x = x.reshape(-1, k_in)
    elif x.shape[:-1] == id_shape[:-1]:
        topk = id_shape[-1]
        tokens = int(x.numel() // k_in)
        if tokens * topk != expert_ids.numel():
            raise ValueError(
                f"x leading {x.shape[:-1]} incompatible with expert_ids {id_shape}."
            )
        slot_x = (
            x.reshape(tokens, k_in)
            .unsqueeze(1)
            .expand(tokens, topk, k_in)
            .reshape(tokens * topk, k_in)
        )
    else:
        raise ValueError(
            f"x shape {tuple(x.shape)} incompatible with expert_ids {id_shape}."
        )

    slots = slot_x.shape[0]
    ids = expert_ids.reshape(slots).contiguous()
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.to(torch.int32)
    y = torch.empty(slots, n_out, device=x.device, dtype=x.dtype)

    def _grid(meta):
        return (triton.cdiv(n_out, meta["BLOCK_N"]), slots)

    _ternary_expert_gemv_kernel[_grid](
        slot_x,
        packed_weight,
        row_alpha,
        ids,
        y,
        n_out,
        nwords,
        slot_x.stride(0),
        slot_x.stride(1),
        packed_weight.stride(0),
        packed_weight.stride(1),
        packed_weight.stride(2),
        row_alpha.stride(0),
        row_alpha.stride(1),
        y.stride(0),
        y.stride(1),
    )
    return y.reshape(*id_shape, n_out)
