"""Decode-step attention: QK RMSNorm + partial RoPE + GQA, KV stays preallocated.

Prefill still uses torch SDPA. This path is q_len=1 so the KV write index and
length can live in device tensors (CUDA-graph safe). The GQA loop walks only
``seqlen`` (and the sliding window), not the padded cache ``max_len``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_rope_store_q_k(
    qkv_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    USE_ROPE: tl.constexpr,
    eps,
    stride_out_h,
    stride_out_d,
):
    """RMSNorm (+ optional partial RoPE) one head from a packed QKV row into ``out``."""
    cols = tl.arange(0, HEAD_DIM)
    x = tl.load(qkv_ptr + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    rstd = tl.rsqrt(var + eps)
    x = x * rstd * w
    if USE_ROPE:
        half: tl.constexpr = ROTARY_DIM // 2
        pass_dim: tl.constexpr = HEAD_DIM - ROTARY_DIM
        offs_lo = tl.arange(0, half)
        x_lo = tl.load(qkv_ptr + offs_lo).to(tl.float32) * rstd * tl.load(w_ptr + offs_lo).to(
            tl.float32
        )
        x_hi = tl.load(qkv_ptr + half + offs_lo).to(tl.float32) * rstd * tl.load(
            w_ptr + half + offs_lo
        ).to(tl.float32)
        cos_lo = tl.load(cos_ptr + offs_lo).to(tl.float32)
        cos_hi = tl.load(cos_ptr + half + offs_lo).to(tl.float32)
        sin_lo = tl.load(sin_ptr + offs_lo).to(tl.float32)
        sin_hi = tl.load(sin_ptr + half + offs_lo).to(tl.float32)
        new_lo = x_lo * cos_lo + (-x_hi) * sin_lo
        new_hi = x_hi * cos_hi + x_lo * sin_hi
        tl.store(out_ptr + offs_lo * stride_out_d, new_lo)
        tl.store(out_ptr + (half + offs_lo) * stride_out_d, new_hi)
        if pass_dim > 0:
            offs_p = tl.arange(0, pass_dim)
            x_p = tl.load(qkv_ptr + ROTARY_DIM + offs_p).to(tl.float32) * rstd * tl.load(
                w_ptr + ROTARY_DIM + offs_p
            ).to(tl.float32)
            tl.store(out_ptr + (ROTARY_DIM + offs_p) * stride_out_d, x_p)
    else:
        tl.store(out_ptr + cols * stride_out_d, x)


@triton.jit
def _qk_norm_rope_store_kernel(
    qkv_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    seqlen_ptr,
    n_q,
    n_kv,
    stride_qh,
    stride_qd,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vh,
    stride_vt,
    stride_vd,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    USE_ROPE: tl.constexpr,
    eps,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, HEAD_DIM)
    pos = tl.load(seqlen_ptr)
    q_size = n_q * HEAD_DIM
    kv_size = n_kv * HEAD_DIM

    if pid < n_q:
        base = pid * HEAD_DIM
        _rms_rope_store_q_k(
            qkv_ptr + base,
            q_w_ptr,
            cos_ptr,
            sin_ptr,
            q_out_ptr + pid * stride_qh,
            HEAD_DIM,
            ROTARY_DIM,
            USE_ROPE,
            eps,
            stride_qh,
            stride_qd,
        )
    elif pid < n_q + n_kv:
        kv_h = pid - n_q
        base = q_size + kv_h * HEAD_DIM
        _rms_rope_store_q_k(
            qkv_ptr + base,
            k_w_ptr,
            cos_ptr,
            sin_ptr,
            k_cache_ptr + kv_h * stride_kh + pos * stride_kt,
            HEAD_DIM,
            ROTARY_DIM,
            USE_ROPE,
            eps,
            stride_kh,
            stride_kd,
        )
    else:
        kv_h = pid - n_q - n_kv
        base = q_size + kv_size + kv_h * HEAD_DIM
        x = tl.load(qkv_ptr + base + cols)
        tl.store(
            v_cache_ptr + kv_h * stride_vh + pos * stride_vt + cols * stride_vd,
            x,
        )


@triton.jit
def _decode_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seqlen_ptr,
    n_q_per_kv,
    scale,
    max_len,
    stride_qh,
    stride_qd,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_oh,
    stride_od,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    WINDOW: tl.constexpr,
):
    h = tl.program_id(0)
    kv_h = h // n_q_per_kv
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + h * stride_qh + offs_d * stride_qd).to(tl.float32)

    kv_len = tl.minimum(tl.load(seqlen_ptr) + 1, max_len)
    start = 0
    if WINDOW > 0:
        start = tl.maximum(0, kv_len - WINDOW)
    # Align the first tile so K/V loads stay coalesced; mask drops pad.
    start_aligned = (start // BLOCK_T) * BLOCK_T

    m_i = -1.0e9
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    neg = -1.0e9

    k_row = k_ptr + kv_h * stride_kh
    v_row = v_ptr + kv_h * stride_vh

    for t0 in range(start_aligned, kv_len, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = (offs_t >= start) & (offs_t < kv_len)
        k = tl.load(
            k_row + offs_t[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(mask_t, scores, neg)
        m_tile = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(mask_t, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(
            v_row + offs_t[:, None] * stride_vt + offs_d[None, :] * stride_vd,
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(out_ptr + h * stride_oh + offs_d * stride_od, acc / l_i)


def apply_qkv_decode(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    use_rope: bool,
    use_qk_norm: bool,
    eps: float,
) -> torch.Tensor:
    """QK RMSNorm + optional RoPE + store K/V at ``seqlen``. Returns Q ``[1, H, 1, D]``."""
    if qkv.ndim != 2 or qkv.shape[0] != 1:
        raise ValueError(f"apply_qkv_decode expects [1, qkv], got {tuple(qkv.shape)}.")
    if not use_qk_norm:
        raise ValueError("apply_qkv_decode requires QK RMSNorm.")
    q_out = torch.empty(1, num_heads, 1, head_dim, device=qkv.device, dtype=qkv.dtype)
    k_c = k_cache[0]
    v_c = v_cache[0]
    q2 = q_out[0, :, 0, :]
    _qk_norm_rope_store_kernel[(num_heads + 2 * num_kv_heads,)](
        qkv.reshape(-1),
        q_weight.reshape(-1),
        k_weight.reshape(-1),
        cos.reshape(-1),
        sin.reshape(-1),
        q2,
        k_c,
        v_c,
        seqlen,
        num_heads,
        num_kv_heads,
        q2.stride(0),
        q2.stride(1),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        v_c.stride(0),
        v_c.stride(1),
        v_c.stride(2),
        HEAD_DIM=head_dim,
        ROTARY_DIM=max(rotary_dim, 1),
        USE_ROPE=use_rope,
        eps=float(eps),
    )
    return q_out


def decode_gqa_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    scale: float,
    window: int | None,
) -> torch.Tensor:
    """GQA attention for ``q_len=1``. ``seqlen`` is the index just written (kv_len-1)."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
        raise ValueError(f"decode_gqa_attn expects q [1, H, 1, D], got {tuple(q.shape)}.")
    _, n_q, _, head_dim = q.shape
    n_kv = k_cache.shape[1]
    max_len = k_cache.shape[2]
    if n_q % n_kv != 0:
        raise ValueError(f"n_q={n_q} not divisible by n_kv={n_kv}.")
    out = torch.empty_like(q)
    q2 = q[0, :, 0, :]
    o2 = out[0, :, 0, :]
    win = 0 if window is None else int(window)
    _decode_gqa_kernel[(n_q,)](
        q2,
        k_cache[0],
        v_cache[0],
        o2,
        seqlen,
        n_q // n_kv,
        float(scale),
        max_len,
        q2.stride(0),
        q2.stride(1),
        k_cache[0].stride(0),
        k_cache[0].stride(1),
        k_cache[0].stride(2),
        v_cache[0].stride(0),
        v_cache[0].stride(1),
        v_cache[0].stride(2),
        o2.stride(0),
        o2.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_T=64,
        WINDOW=win,
        num_warps=4,
        num_stages=2,
    )
    return out
