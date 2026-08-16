"""4-bit RTN embedding lookup and GEMV. Used for Maple embeddings / lm_head.

Dequant is ``code * scale + bias`` with group size 64, 8 LSB-first codes per
uint32. The full vocabulary table stays packed; only selected embedding rows
are dequantized, and the lm_head GEMV reads packed codes in-kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from maple_run.pack import HEAD_GROUP_SIZE

_CODES_PER_WORD = 8


def unpack_4bit_torch(packed: torch.Tensor) -> torch.Tensor:
    """Unpack LSB-first 4-bit codes; last axis ``nwords`` → ``nwords * 8``."""
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int64)
    words = packed.to(torch.int64)
    codes = (words.unsqueeze(-1) >> shifts) & 0xF
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * _CODES_PER_WORD)


def rtn4_embedding(
    input_ids: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = HEAD_GROUP_SIZE,
) -> torch.Tensor:
    """Gather packed rows for ``input_ids`` and dequant those rows only."""
    if packed_weight.dtype != torch.uint32:
        raise TypeError(
            f"packed_weight must be uint32 codes, got {packed_weight.dtype}."
        )
    # CUDA has no uint32 index_select; gather via int32 view, then restore.
    rows = packed_weight.view(torch.int32)[input_ids].view(torch.uint32)
    row_scales = scales[input_ids]
    row_biases = biases[input_ids]
    codes = unpack_4bit_torch(rows).to(torch.float32)
    k = codes.shape[-1]
    grouped = codes.reshape(*codes.shape[:-1], k // group_size, group_size)
    recon = grouped * row_scales.unsqueeze(-1).float() + row_biases.unsqueeze(-1).float()
    return recon.reshape(*codes.shape[:-1], k)


def _rtn4_launch_meta(nwords: int, batch: int) -> tuple[int, int, int, int]:
    """BLOCK_N, BLOCK_K_WORDS, warps, stages.

    One K-tile is one RTN group (8 uint32 = 64 codes) so scale/bias are scalars
    per output row, not a ``[BLOCK_N, BLOCK_K]`` load. Decode (batch=1) uses
    skinny ``BLOCK_N`` so GB10's 48 SMs stay busy on the 152k-row lm_head.
    """
    # 8 words × 8 codes/word = 64 = HEAD_GROUP_SIZE.
    block_k_words = 8
    if batch == 1:
        # lm_head N=152k: fat BLOCK_N still leaves thousands of CTAs; 2 warps
        # raises occupancy vs 4-warp tiles (1.36 ms / ~143 GB/s on GB10).
        return 64, block_k_words, 2, 2
    return 32, block_k_words, 4, 3


@triton.jit
def _rtn4_gemv_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    N,
    nwords,
    stride_xb,
    stride_xk,
    stride_wn,
    stride_ww,
    stride_sn,
    stride_sg,
    stride_bn,
    stride_bg,
    stride_yb,
    stride_yn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_row = x_ptr + pid_b * stride_xb

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 8
    shifts = (tl.arange(0, 8) * 4).to(tl.uint32)
    k_total = nwords * 8

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
        ).to(tl.uint32)
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0xF
        q = tl.reshape(codes.to(tl.float32), (BLOCK_N, BLOCK_K))

        offs_k = w0 * 8 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_total
        x_tile = tl.load(
            x_row + offs_k * stride_xk,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        # Tile is one RTN group: one scale/bias per output row, not per K.
        group = (w0 * 8) // GROUP_SIZE
        scale = tl.load(
            scale_ptr + offs_n * stride_sn + group * stride_sg,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + offs_n * stride_bn + group * stride_bg,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        acc += scale * tl.sum(q * x_tile[None, :], axis=1) + bias * tl.sum(x_tile, axis=0)

    tl.store(
        y_ptr + pid_b * stride_yb + offs_n * stride_yn,
        acc,
        mask=mask_n,
    )


def rtn4_gemv(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = HEAD_GROUP_SIZE,
    *,
    packed_kn: bool = False,
) -> torch.Tensor:
    """Decode GEMV ``y = x @ W`` with 4-bit packed ``W`` (affine RTN)."""
    if packed_weight.dtype != torch.uint32:
        raise TypeError(
            f"packed_weight must be uint32 codes, got {packed_weight.dtype}; "
            "refusing to run a dense-weight path."
        )
    if packed_weight.ndim != 2:
        raise ValueError(
            f"packed_weight must be 2-D [N, K/8] or [K/8, N]; got shape "
            f"{tuple(packed_weight.shape)}."
        )
    if not x.is_cuda or not packed_weight.is_cuda:
        raise RuntimeError("rtn4_gemv requires CUDA tensors.")

    if packed_kn:
        nwords, n_out = packed_weight.shape
        stride_wn = packed_weight.stride(1)
        stride_ww = packed_weight.stride(0)
        n_groups = nwords * _CODES_PER_WORD // group_size
        if scales.shape != (n_groups, n_out) or biases.shape != (n_groups, n_out):
            raise ValueError(
                f"packed_kn scales/biases expected {(n_groups, n_out)}, "
                f"got {tuple(scales.shape)} and {tuple(biases.shape)}."
            )
        stride_sn = scales.stride(1)
        stride_sg = scales.stride(0)
        stride_bn = biases.stride(1)
        stride_bg = biases.stride(0)
    else:
        n_out, nwords = packed_weight.shape
        stride_wn = packed_weight.stride(0)
        stride_ww = packed_weight.stride(1)
        n_groups = nwords * _CODES_PER_WORD // group_size
        if scales.shape != (n_out, n_groups) or biases.shape != (n_out, n_groups):
            raise ValueError(
                f"scales/biases expected {(n_out, n_groups)}, got {tuple(scales.shape)} "
                f"and {tuple(biases.shape)}."
            )
        stride_sn = scales.stride(0)
        stride_sg = scales.stride(1)
        stride_bn = biases.stride(0)
        stride_bg = biases.stride(1)

    k_in = nwords * _CODES_PER_WORD
    if x.shape[-1] != k_in:
        raise ValueError(f"x last dim {x.shape[-1]} != packed K ({k_in}).")

    leading = x.shape[:-1]
    batch = int(x.numel() // k_in)
    x_mat = x.reshape(batch, k_in)
    y = torch.empty(batch, n_out, device=x.device, dtype=x.dtype)

    block_n, block_k_words, num_warps, num_stages = _rtn4_launch_meta(nwords, batch)
    grid = (triton.cdiv(n_out, block_n), batch)
    _rtn4_gemv_kernel[grid](
        x_mat,
        packed_weight,
        scales,
        biases,
        y,
        n_out,
        nwords,
        x_mat.stride(0),
        x_mat.stride(1),
        stride_wn,
        stride_ww,
        stride_sn,
        stride_sg,
        stride_bn,
        stride_bg,
        y.stride(0),
        y.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_N=block_n,
        BLOCK_K_WORDS=block_k_words,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.reshape(*leading, n_out)
