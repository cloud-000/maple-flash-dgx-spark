"""Fused RMSNorm and residual+RMSNorm. One launch instead of a PyTorch op chain."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    eps,
    stride_x,
    stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_y + cols, x * rstd * w, mask=mask)


@triton.jit
def _add_rms_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    residual_ptr,
    norm_ptr,
    N,
    eps,
    stride_x,
    stride_y,
    stride_r,
    stride_n,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + y
    var = tl.sum(s * s, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(residual_ptr + row * stride_r + cols, s, mask=mask)
    tl.store(norm_ptr + row * stride_n + cols, s * rstd * w, mask=mask)


@triton.jit
def _moe_reduce_kernel(
    x_ptr,
    w_ptr,
    residual_ptr,
    y_ptr,
    N,
    stride_xt,
    stride_xs,
    stride_xn,
    stride_wt,
    stride_ws,
    stride_rt,
    stride_rn,
    stride_yt,
    stride_yn,
    HAS_RESIDUAL: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(0, TOPK):
        wt = tl.load(w_ptr + pid_t * stride_wt + s * stride_ws).to(tl.float32)
        xv = tl.load(
            x_ptr + pid_t * stride_xt + s * stride_xs + cols * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += wt * xv
    if HAS_RESIDUAL:
        acc += tl.load(
            residual_ptr + pid_t * stride_rt + cols * stride_rn, mask=mask, other=0.0
        ).to(tl.float32)
    tl.store(y_ptr + pid_t * stride_yt + cols * stride_yn, acc, mask=mask)


def _rows_and_n(x: torch.Tensor) -> tuple[int, int]:
    n = int(x.shape[-1])
    rows = int(x.numel() // n)
    return rows, n


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim; compute in fp32, store in ``x.dtype``."""
    if not x.is_cuda:
        raise RuntimeError("rms_norm requires CUDA tensors.")
    rows, n = _rows_and_n(x)
    x_mat = x.reshape(rows, n)
    y = torch.empty_like(x_mat)
    w = weight.reshape(-1)
    if w.numel() != n:
        raise ValueError(f"rms_norm weight has {w.numel()} elements, expected {n}.")
    _rms_norm_kernel[(rows,)](
        x_mat,
        w,
        y,
        n,
        float(eps),
        x_mat.stride(0),
        y.stride(0),
        BLOCK=triton.next_power_of_2(n),
    )
    return y.reshape(x.shape)


def add_rms_norm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual = x + y`` and ``norm = rms_norm(residual)`` in one launch."""
    if not x.is_cuda or not y.is_cuda:
        raise RuntimeError("add_rms_norm requires CUDA tensors.")
    if x.shape != y.shape:
        raise ValueError(f"add_rms_norm shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}.")
    rows, n = _rows_and_n(x)
    x_mat = x.reshape(rows, n)
    y_mat = y.reshape(rows, n)
    residual = torch.empty_like(x_mat)
    normed = torch.empty_like(x_mat)
    w = weight.reshape(-1)
    if w.numel() != n:
        raise ValueError(f"add_rms_norm weight has {w.numel()} elements, expected {n}.")
    _add_rms_norm_kernel[(rows,)](
        x_mat,
        y_mat,
        w,
        residual,
        normed,
        n,
        float(eps),
        x_mat.stride(0),
        y_mat.stride(0),
        residual.stride(0),
        normed.stride(0),
        BLOCK=triton.next_power_of_2(n),
    )
    return residual.reshape(x.shape), normed.reshape(x.shape)


def moe_reduce_add(
    expert_out: torch.Tensor,
    topk_weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = (expert_out * topk_weight).sum(topk) [+ residual]`` in one launch."""
    if expert_out.ndim < 2:
        raise ValueError("expert_out must be [..., topk, N].")
    topk = int(expert_out.shape[-2])
    n = int(expert_out.shape[-1])
    if tuple(topk_weight.shape) != tuple(expert_out.shape[:-1]):
        raise ValueError("topk_weight must match expert_out without the last dim.")
    tokens = int(expert_out.numel() // (topk * n))
    x = expert_out.reshape(tokens, topk, n)
    w = topk_weight.reshape(tokens, topk).float()
    y = torch.empty(tokens, n, device=expert_out.device, dtype=expert_out.dtype)
    has_res = residual is not None
    res = residual.reshape(tokens, n) if has_res else y
    block = 128 if n >= 128 else triton.next_power_of_2(n)
    _moe_reduce_kernel[(tokens, triton.cdiv(n, block))](
        x,
        w,
        res,
        y,
        n,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        res.stride(0),
        res.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_RESIDUAL=has_res,
        TOPK=topk,
        BLOCK=block,
    )
    return y.reshape(*expert_out.shape[:-2], n)
