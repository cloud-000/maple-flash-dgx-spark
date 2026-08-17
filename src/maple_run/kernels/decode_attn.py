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
    m_ptr,
    l_ptr,
    seqlen_ptr,
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
    stride_os,
    stride_oh,
    stride_od,
    stride_ms,
    stride_mh,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    WINDOW: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_G: tl.constexpr,
    SPLITS: tl.constexpr,
    IEEE: tl.constexpr,
):
    """One program per (KV head, sequence split).

    All ``GROUP`` query heads sharing a KV head are scored against the same K/V
    tile, so the cache is read once instead of once per query head. With
    ``SPLITS > 1`` each program covers a slice of the sequence and writes
    ``(acc, m, l)`` partials for ``_gqa_combine_kernel`` to merge.
    """
    kv_h = tl.program_id(0)
    sp = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < GROUP

    q = tl.load(
        q_ptr + (kv_h * GROUP + offs_g)[:, None] * stride_qh + offs_d[None, :] * stride_qd,
        mask=mask_g[:, None],
        other=0.0,
    )

    kv_len = tl.minimum(tl.load(seqlen_ptr) + 1, max_len)
    start = 0
    if WINDOW > 0:
        start = tl.maximum(0, kv_len - WINDOW)
    # Split the live range evenly, rounded up to whole tiles so every program's
    # K/V loads stay tile-aligned. Trailing splits may end up empty (l == 0).
    per = tl.cdiv(tl.cdiv(kv_len - start, SPLITS), BLOCK_T) * BLOCK_T
    lo = start + sp * per
    hi = tl.minimum(lo + per, kv_len)
    # Align the first tile so K/V loads stay coalesced; mask drops pad.
    lo_aligned = (lo // BLOCK_T) * BLOCK_T

    m_i = tl.full((BLOCK_G,), -1.0e9, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_G,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_G, HEAD_DIM), dtype=tl.float32)

    k_row = k_ptr + kv_h * stride_kh
    v_row = v_ptr + kv_h * stride_vh

    for t0 in range(lo_aligned, hi, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = (offs_t >= lo) & (offs_t < hi)
        k = tl.load(
            k_row + offs_t[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            mask=mask_t[:, None],
            other=0.0,
        )
        # Q and K are already bf16 in the decode cache, so the tensor-core dot
        # loses nothing over an fp32 reduction of the same values and still
        # accumulates in fp32. Broadcasting instead (a
        # [BLOCK_G, BLOCK_T, HEAD_DIM] tile) spills. An fp32 cache would
        # otherwise silently drop to tf32 here, so it takes the exact path.
        if IEEE:
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        else:
            scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(mask_t[None, :], scores, -1.0e9)
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_t[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            v_row + offs_t[:, None] * stride_vt + offs_d[None, :] * stride_vd,
            mask=mask_t[:, None],
            other=0.0,
        )
        # P rounds to the cache dtype for the second dot, as FlashAttention does;
        # the attention output is bf16 anyway, so this stays inside its rounding.
        if IEEE:
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if SPLITS == 1:
        tl.store(
            out_ptr + (kv_h * GROUP + offs_g)[:, None] * stride_oh
            + offs_d[None, :] * stride_od,
            acc / l_i[:, None],
            mask=mask_g[:, None],
        )
    else:
        tl.store(
            out_ptr
            + sp * stride_os
            + (kv_h * GROUP + offs_g)[:, None] * stride_oh
            + offs_d[None, :] * stride_od,
            acc,
            mask=mask_g[:, None],
        )
        base = sp * stride_ms + (kv_h * GROUP + offs_g) * stride_mh
        tl.store(m_ptr + base, m_i, mask=mask_g)
        tl.store(l_ptr + base, l_i, mask=mask_g)


@triton.jit
def _gqa_combine_kernel(
    part_ptr,
    m_ptr,
    l_ptr,
    out_ptr,
    stride_ps,
    stride_ph,
    stride_pd,
    stride_ms,
    stride_mh,
    stride_oh,
    stride_od,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Merge per-split ``(acc, m, l)`` partials for one query head."""
    h = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLITS)

    m = tl.load(m_ptr + offs_s * stride_ms + h * stride_mh)
    l = tl.load(l_ptr + offs_s * stride_ms + h * stride_mh)
    # Empty splits carry l == 0; drop them so they cannot move the max.
    m = tl.where(l > 0.0, m, -1.0e9)
    m_max = tl.max(m, axis=0)
    w = tl.where(l > 0.0, tl.exp(m - m_max), 0.0)

    part = tl.load(
        part_ptr + offs_s[:, None] * stride_ps + h * stride_ph + offs_d[None, :] * stride_pd
    ).to(tl.float32)
    acc = tl.sum(part * w[:, None], axis=0)
    denom = tl.sum(l * w, axis=0)
    tl.store(out_ptr + h * stride_oh + offs_d * stride_od, acc / denom)


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


_BLOCK_T = 64
_MAX_SPLITS = 8


def gqa_splits(max_len: int, window: int | None) -> int:
    """Sequence splits for the decode GQA grid.

    Fixed at cache-allocation time so the launch grid is a CUDA-graph constant.
    Only ``n_kv`` (4) programs would otherwise be resident; splitting restores
    occupancy without re-reading K/V per query head.
    """
    span = max_len if window is None else min(max_len, window)
    tiles = max(1, -(-span // _BLOCK_T))
    # Power of two: the combine kernel indexes splits with ``tl.arange``.
    # Rounded up -- a split with no tiles left to cover exits immediately and
    # the combine drops it, so overshooting costs far less than leaving the
    # machine at n_kv resident programs.
    splits = 1
    while splits < min(_MAX_SPLITS, tiles):
        splits *= 2
    return splits


def decode_gqa_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlen: torch.Tensor,
    *,
    scale: float,
    window: int | None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """GQA attention for ``q_len=1``. ``seqlen`` is the index just written (kv_len-1).

    ``workspace`` holds the per-split ``(acc, m, l)`` partials; pass a
    preallocated one so decode does not allocate inside a CUDA graph.
    """
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
        raise ValueError(f"decode_gqa_attn expects q [1, H, 1, D], got {tuple(q.shape)}.")
    _, n_q, _, head_dim = q.shape
    n_kv = k_cache.shape[1]
    max_len = k_cache.shape[2]
    if n_q % n_kv != 0:
        raise ValueError(f"n_q={n_q} not divisible by n_kv={n_kv}.")
    group = n_q // n_kv
    out = torch.empty_like(q)
    q2 = q[0, :, 0, :]
    o2 = out[0, :, 0, :]
    win = 0 if window is None else int(window)
    splits = gqa_splits(max_len, window)

    if splits == 1:
        part, m_buf, l_buf = o2, o2, o2
        stride_os = stride_ms = 0
    else:
        if workspace is None:
            part = torch.empty(splits, n_q, head_dim, device=q.device, dtype=torch.float32)
            m_buf = torch.empty(splits, n_q, device=q.device, dtype=torch.float32)
            l_buf = torch.empty(splits, n_q, device=q.device, dtype=torch.float32)
        else:
            part, m_buf, l_buf = workspace
        stride_os = part.stride(0)
        stride_ms = m_buf.stride(0)

    _decode_gqa_kernel[(n_kv, splits)](
        q2,
        k_cache[0],
        v_cache[0],
        part,
        m_buf,
        l_buf,
        seqlen,
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
        stride_os,
        part.stride(-2),
        part.stride(-1),
        stride_ms,
        m_buf.stride(-1),
        HEAD_DIM=head_dim,
        BLOCK_T=_BLOCK_T,
        WINDOW=win,
        GROUP=group,
        BLOCK_G=max(16, triton.next_power_of_2(group)),
        SPLITS=splits,
        IEEE=k_cache.dtype == torch.float32,
        num_warps=4,
        num_stages=2,
    )
    if splits > 1:
        _gqa_combine_kernel[(n_q,)](
            part,
            m_buf,
            l_buf,
            o2,
            part.stride(0),
            part.stride(1),
            part.stride(2),
            m_buf.stride(0),
            m_buf.stride(1),
            o2.stride(0),
            o2.stride(1),
            HEAD_DIM=head_dim,
            SPLITS=splits,
            num_warps=4,
        )
    return out
