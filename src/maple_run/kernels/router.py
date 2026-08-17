"""Fused MoE router: logits GEMV + softmax + top-k + renormalize.

The unfused chain (``F.linear(x.float(), w.float())`` → ``softmax`` → ``topk``
→ ``sum`` → ``div``) is eight launches per layer over a 256-wide vector; on
this host that cost more than the router weight traffic itself. Here it is two
launches: a split-K GEMV over the 256 experts, then one CTA per token that
reduces the split, softmaxes, extracts the top-k and renormalizes.

Router weights stay unquantized (``TERNARY_EXCLUDE``); they are read in the
checkpoint dtype and accumulated in float32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _router_gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    K,
    stride_xt,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_yt,
    stride_ys,
    stride_yn,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Partial router logits ``y[t, split, n] = sum_k x[t, k] * w[n, k]``."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_t = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    x_row = x_ptr + pid_t * stride_xt
    w_row = w_ptr + offs_n[:, None] * stride_wn

    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end
        x = tl.load(x_row + offs_k * stride_xk, mask=mask_k, other=0.0).to(tl.float32)
        w = tl.load(
            w_row + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(
        y_ptr + pid_t * stride_yt + pid_k * stride_ys + offs_n * stride_yn,
        acc,
        mask=mask_n,
    )


@triton.jit
def _router_topk_kernel(
    y_ptr,
    idx_ptr,
    wt_ptr,
    N,
    stride_yt,
    stride_ys,
    stride_yn,
    stride_it,
    stride_ik,
    stride_wt,
    stride_wk,
    SPLIT_K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce the split, softmax over experts, take top-k, renormalize."""
    pid_t = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in tl.static_range(0, SPLIT_K):
        logits += tl.load(
            y_ptr + pid_t * stride_yt + s * stride_ys + offs_n * stride_yn,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
    logits = tl.where(mask_n, logits, float("-inf"))

    probs = tl.exp(logits - tl.max(logits, axis=0))
    probs = probs / tl.sum(probs, axis=0)

    # Iterative extraction: TOPK is 8, so a sort would cost more than 8 passes.
    remaining = probs
    total = 0.0
    for i in tl.static_range(0, TOPK):
        j = tl.argmax(remaining, axis=0)
        p = tl.max(remaining, axis=0)
        total += p
        tl.store(idx_ptr + pid_t * stride_it + i * stride_ik, j)
        tl.store(wt_ptr + pid_t * stride_wt + i * stride_wk, p)
        remaining = tl.where(offs_n == j, float("-inf"), remaining)

    inv = 1.0 / (total + 1e-20)
    for i in tl.static_range(0, TOPK):
        p = tl.load(wt_ptr + pid_t * stride_wt + i * stride_wk)
        tl.store(wt_ptr + pid_t * stride_wt + i * stride_wk, p * inv)


def _gemv_meta(n: int, k: int, tokens: int) -> tuple[int, int, int, int]:
    """SPLIT_K, BLOCK_N, warps, stages. Decode wants CTAs; prefill wants reuse.

    256 experts x 2048 is only 1 MB, so one program per 8 experts leaves the
    machine idle. Splitting K 16 ways (1024 programs) runs the cold router at
    263 GB/s -- 96% of this host's 273 GB/s.
    """
    if tokens == 1:
        return 16, 4, 2, 3
    return 1, 32, 4, 2


def router_topk(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router logits → softmax → top-k → renormalized weights, in two launches.

    Parameters
    ----------
    x:
        Hidden states ``[..., K]`` (any float dtype; read without an fp32 copy).
    gate_weight:
        Router weight ``[N_experts, K]``.
    top_k:
        Experts per token.

    Returns
    -------
    ``(topk_idx int32 [..., top_k], topk_weight float32 [..., top_k])``
    """
    if not x.is_cuda or not gate_weight.is_cuda:
        raise RuntimeError("router_topk requires CUDA tensors.")
    if gate_weight.ndim != 2:
        raise ValueError(f"gate_weight must be 2-D [E, K], got {tuple(gate_weight.shape)}.")
    n_exp, k_in = gate_weight.shape
    if x.shape[-1] != k_in:
        raise ValueError(f"x last dim {x.shape[-1]} != router K ({k_in}).")
    if top_k > n_exp:
        raise ValueError(f"top_k={top_k} exceeds {n_exp} experts.")

    leading = x.shape[:-1]
    tokens = int(x.numel() // k_in)
    x_mat = x.reshape(tokens, k_in)

    split_k, block_n, num_warps, num_stages = _gemv_meta(n_exp, k_in, tokens)
    partial = torch.empty(tokens, split_k, n_exp, device=x.device, dtype=torch.float32)
    _router_gemv_kernel[(triton.cdiv(n_exp, block_n), split_k, tokens)](
        x_mat,
        gate_weight,
        partial,
        n_exp,
        k_in,
        x_mat.stride(0),
        x_mat.stride(1),
        gate_weight.stride(0),
        gate_weight.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        SPLIT_K=split_k,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    idx = torch.empty(tokens, top_k, device=x.device, dtype=torch.int32)
    wt = torch.empty(tokens, top_k, device=x.device, dtype=torch.float32)
    _router_topk_kernel[(tokens,)](
        partial,
        idx,
        wt,
        n_exp,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        idx.stride(0),
        idx.stride(1),
        wt.stride(0),
        wt.stride(1),
        SPLIT_K=split_k,
        TOPK=top_k,
        BLOCK_N=triton.next_power_of_2(n_exp),
        num_warps=1,
    )
    return idx.reshape(*leading, top_k), wt.reshape(*leading, top_k)
