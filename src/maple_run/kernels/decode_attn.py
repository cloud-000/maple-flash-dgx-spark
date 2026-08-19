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
    stride_xb,
    stride_cb,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    USE_ROPE: tl.constexpr,
    eps,
):
    """One program per (head slot, batch row).

    Each row has its own write position ``seqlen[b]`` and its own RoPE angles,
    and nothing here is shared across rows, so the batch is a grid dimension.
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    cols = tl.arange(0, HEAD_DIM)
    pos = tl.load(seqlen_ptr + b)
    q_size = n_q * HEAD_DIM
    kv_size = n_kv * HEAD_DIM
    row_ptr = qkv_ptr + b * stride_xb
    cos_b = cos_ptr + b * stride_cb
    sin_b = sin_ptr + b * stride_cb

    if pid < n_q:
        base = pid * HEAD_DIM
        _rms_rope_store_q_k(
            row_ptr + base,
            q_w_ptr,
            cos_b,
            sin_b,
            q_out_ptr + b * stride_qb + pid * stride_qh,
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
            row_ptr + base,
            k_w_ptr,
            cos_b,
            sin_b,
            k_cache_ptr + b * stride_kb + kv_h * stride_kh + pos * stride_kt,
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
        x = tl.load(row_ptr + base + cols)
        tl.store(
            v_cache_ptr
            + b * stride_vb
            + kv_h * stride_vh
            + pos * stride_vt
            + cols * stride_vd,
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
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    stride_mb,
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
    """One program per (KV head, sequence split, batch row).

    All ``GROUP`` query heads sharing a KV head are scored against the same K/V
    tile, so the cache is read once instead of once per query head. With
    ``SPLITS > 1`` each program covers a slice of the sequence and writes
    ``(acc, m, l)`` partials for ``_gqa_combine_kernel`` to merge.

    Batch rows have independent K/V and independent ``seqlen[b]``, so unlike
    the GEMV kernels there is no tile to amortise across the batch: the row is
    a grid dimension, which is also what keeps the ragged lengths free.
    """
    kv_h = tl.program_id(0)
    sp = tl.program_id(1)
    b = tl.program_id(2)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < GROUP

    q = tl.load(
        q_ptr
        + b * stride_qb
        + (kv_h * GROUP + offs_g)[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_g[:, None],
        other=0.0,
    )

    kv_len = tl.minimum(tl.load(seqlen_ptr + b) + 1, max_len)
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

    k_row = k_ptr + b * stride_kb + kv_h * stride_kh
    v_row = v_ptr + b * stride_vb + kv_h * stride_vh

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
            out_ptr
            + b * stride_ob
            + (kv_h * GROUP + offs_g)[:, None] * stride_oh
            + offs_d[None, :] * stride_od,
            acc / l_i[:, None],
            mask=mask_g[:, None],
        )
    else:
        tl.store(
            out_ptr
            + b * stride_ob
            + sp * stride_os
            + (kv_h * GROUP + offs_g)[:, None] * stride_oh
            + offs_d[None, :] * stride_od,
            acc,
            mask=mask_g[:, None],
        )
        base = b * stride_mb + sp * stride_ms + (kv_h * GROUP + offs_g) * stride_mh
        tl.store(m_ptr + base, m_i, mask=mask_g)
        tl.store(l_ptr + base, l_i, mask=mask_g)


@triton.jit
def _gqa_combine_kernel(
    part_ptr,
    m_ptr,
    l_ptr,
    out_ptr,
    stride_pb,
    stride_ps,
    stride_ph,
    stride_pd,
    stride_mb,
    stride_ms,
    stride_mh,
    stride_ob,
    stride_oh,
    stride_od,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Merge per-split ``(acc, m, l)`` partials for one (query head, row)."""
    h = tl.program_id(0)
    b = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLITS)

    m_base = m_ptr + b * stride_mb + offs_s * stride_ms + h * stride_mh
    l_base = l_ptr + b * stride_mb + offs_s * stride_ms + h * stride_mh
    m = tl.load(m_base)
    l = tl.load(l_base)
    # Empty splits carry l == 0; drop them so they cannot move the max.
    m = tl.where(l > 0.0, m, -1.0e9)
    m_max = tl.max(m, axis=0)
    w = tl.where(l > 0.0, tl.exp(m - m_max), 0.0)

    part = tl.load(
        part_ptr
        + b * stride_pb
        + offs_s[:, None] * stride_ps
        + h * stride_ph
        + offs_d[None, :] * stride_pd
    ).to(tl.float32)
    acc = tl.sum(part * w[:, None], axis=0)
    denom = tl.sum(l * w, axis=0)
    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od, acc / denom)


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
    """QK RMSNorm + optional RoPE + store K/V at ``seqlen``. Returns Q ``[B, H, 1, D]``.

    ``qkv`` is ``[B, qkv_dim]``, ``k_cache``/``v_cache`` are ``[B, n_kv, T, D]``
    and ``seqlen`` is one write index per row, so rows at different lengths can
    decode in the same launch.
    """
    if qkv.ndim != 2:
        raise ValueError(f"apply_qkv_decode expects [B, qkv], got {tuple(qkv.shape)}.")
    if not use_qk_norm:
        raise ValueError("apply_qkv_decode requires QK RMSNorm.")
    bsz = qkv.shape[0]
    if k_cache.shape[0] != bsz or v_cache.shape[0] != bsz:
        raise ValueError(
            f"KV cache batch {k_cache.shape[0]} != qkv batch {bsz}."
        )
    if seqlen.numel() != bsz:
        raise ValueError(f"seqlen has {seqlen.numel()} entries, expected {bsz}.")
    if qkv.stride(1) != 1:
        raise ValueError("apply_qkv_decode needs a row-contiguous qkv.")
    q_out = torch.empty(bsz, num_heads, 1, head_dim, device=qkv.device, dtype=qkv.dtype)
    q3 = q_out[:, :, 0, :]
    cos_2 = cos.reshape(bsz, -1)
    sin_2 = sin.reshape(bsz, -1)
    _qk_norm_rope_store_kernel[(num_heads + 2 * num_kv_heads, bsz)](
        qkv,
        q_weight.reshape(-1),
        k_weight.reshape(-1),
        cos_2,
        sin_2,
        q3,
        k_cache,
        v_cache,
        seqlen.reshape(-1),
        num_heads,
        num_kv_heads,
        qkv.stride(0),
        cos_2.stride(0),
        q3.stride(0),
        q3.stride(1),
        q3.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
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
    # Up to one SWA window of tiles, n_kv=4 programs already cover a 512-token
    # loop; splitting would only add a combine launch. Past that, split so the
    # grid is not stuck at 4 resident programs.
    if tiles <= 8:
        return 1
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
    """GQA attention for ``q_len=1``. ``seqlen[b]`` is the index just written (kv_len-1).

    ``q`` is ``[B, H, 1, D]`` and the caches are ``[B, n_kv, T, D]``; each row
    walks only its own live range, so a batch of sequences at different lengths
    costs no more than the longest one.

    ``workspace`` holds the per-split ``(acc, m, l)`` partials; pass a
    preallocated one so decode does not allocate inside a CUDA graph.
    """
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError(f"decode_gqa_attn expects q [B, H, 1, D], got {tuple(q.shape)}.")
    bsz, n_q, _, head_dim = q.shape
    if k_cache.shape[0] != bsz or v_cache.shape[0] != bsz:
        raise ValueError(f"KV cache batch {k_cache.shape[0]} != q batch {bsz}.")
    if seqlen.numel() != bsz:
        raise ValueError(f"seqlen has {seqlen.numel()} entries, expected {bsz}.")
    n_kv = k_cache.shape[1]
    max_len = k_cache.shape[2]
    if n_q % n_kv != 0:
        raise ValueError(f"n_q={n_q} not divisible by n_kv={n_kv}.")
    group = n_q // n_kv
    out = torch.empty_like(q)
    q3 = q[:, :, 0, :]
    o3 = out[:, :, 0, :]
    win = 0 if window is None else int(window)
    splits = gqa_splits(max_len, window)

    if splits == 1:
        part, m_buf, l_buf = o3, o3, o3
        stride_ob = o3.stride(0)
        stride_os = stride_mb = stride_ms = 0
    else:
        if workspace is None:
            part = torch.empty(
                bsz, splits, n_q, head_dim, device=q.device, dtype=torch.float32
            )
            m_buf = torch.empty(bsz, splits, n_q, device=q.device, dtype=torch.float32)
            l_buf = torch.empty(bsz, splits, n_q, device=q.device, dtype=torch.float32)
        else:
            part, m_buf, l_buf = workspace
            part = part[:bsz]
            m_buf = m_buf[:bsz]
            l_buf = l_buf[:bsz]
        stride_ob = part.stride(0)
        stride_os = part.stride(1)
        stride_mb = m_buf.stride(0)
        stride_ms = m_buf.stride(1)

    _decode_gqa_kernel[(n_kv, splits, bsz)](
        q3,
        k_cache,
        v_cache,
        part,
        m_buf,
        l_buf,
        seqlen.reshape(-1),
        float(scale),
        max_len,
        q3.stride(0),
        q3.stride(1),
        q3.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        stride_ob,
        stride_os,
        part.stride(-2),
        part.stride(-1),
        stride_mb,
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
        _gqa_combine_kernel[(n_q, bsz)](
            part,
            m_buf,
            l_buf,
            o3,
            part.stride(0),
            part.stride(1),
            part.stride(2),
            part.stride(3),
            m_buf.stride(0),
            m_buf.stride(1),
            m_buf.stride(2),
            o3.stride(0),
            o3.stride(1),
            o3.stride(2),
            HEAD_DIM=head_dim,
            SPLITS=splits,
            num_warps=4,
        )
    return out
