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


def _rtn4_launch_meta(nwords: int, batch: int) -> tuple[int, int, int, int, int]:
    """BLOCK_N, BLOCK_K_WORDS, BLOCK_B, warps, stages.

    ``BLOCK_K_WORDS`` is a multiple of 8 so each K-tile holds a whole number of
    RTN groups (8 uint32 = 64 codes). Decode (batch=1) uses fat K tiles; one
    group per tile was ~140 GB/s on the lm_head, eight groups with per-group
    scale/bias ~213 GB/s. With the codes untransposed a row's 32 groups are
    contiguous, so bf16 scales/biases also pay off here (820 vs 903 us).

    ``batch > 1`` runs ``_rtn4_gemm_kernel``, which keeps the exact per-group
    form ``s * dot(q, x) + b * sum(x)``. ``BLOCK_K_WORDS`` still buys a wide
    contiguous load; the tile is re-cut into groups only once it is in
    registers, for a batched dot. Tuned on cold weights via
    ``bench/tune_batched_gemv.py`` (175 MB lm_head, cycled so it is never
    L2-resident), against 240 GB/s for the tuned batch-1 kernel on the same
    harness.

    ``BLOCK_B`` stops at 32: the wide tile and the 32-column accumulator
    compete for the same registers, and past 16 columns the tile has to give
    way (``BLOCK_K_WORDS`` 16 -> 8, 188 -> 136 GB/s). Two ``BLOCK_B=16`` tiles
    instead would read the codes twice and measured far worse (94 GB/s), so a
    fat batch pays with tile width rather than with a second pass.
    """
    if batch == 1:
        if nwords >= 32:
            return 16, 64, 1, 2, 2
        if nwords >= 16:
            return 32, 16, 1, 2, 3
        return 64, 8, 1, 2, 2
    if batch <= 16:                     # 872 us / 201 GB/s at B=2, 931 at B=16
        return 32, 16, 16, 1, 2
    return 64, 8, 32, 2, 3              # 1288 us / 136 GB/s at B=32


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
    STREAM_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_row = x_ptr + pid_b * stride_xb

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 8
    N_GROUPS_TILE: tl.constexpr = BLOCK_K // GROUP_SIZE
    shifts = (tl.arange(0, 8) * 4).to(tl.uint32)
    k_total = nwords * 8
    n_groups = k_total // GROUP_SIZE

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
            eviction_policy="evict_first" if STREAM_W else "",
        ).to(tl.uint32)
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0xF
        q = tl.reshape(codes.to(tl.float32), (BLOCK_N, BLOCK_K))

        offs_k = w0 * 8 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_total
        x_tile = tl.load(
            x_row + offs_k * stride_xk,
            mask=mask_k,
            other=0.0,
            eviction_policy="evict_last" if STREAM_W else "",
        ).to(tl.float32)
        offs_g = (w0 * 8) // GROUP_SIZE + tl.arange(0, N_GROUPS_TILE)
        mask_g = offs_g < n_groups
        scale = tl.load(
            scale_ptr + offs_n[:, None] * stride_sn + offs_g[None, :] * stride_sg,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + offs_n[:, None] * stride_bn + offs_g[None, :] * stride_bg,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        # Broadcasting scale/bias across the tile beats reducing q*x per group
        # first (measured 196 vs 139 GB/s at N=151936, K=2048): the per-group
        # reduction needs a 3-D reshape whose register layout costs more than
        # the extra multiplies it saves.
        ones = tl.full((GROUP_SIZE,), 1.0, dtype=tl.float32)
        scale_k = tl.reshape(scale[:, :, None] * ones[None, None, :], (BLOCK_N, BLOCK_K))
        bias_k = tl.reshape(bias[:, :, None] * ones[None, None, :], (BLOCK_N, BLOCK_K))
        acc += tl.sum((scale_k * q + bias_k) * x_tile[None, :], axis=1)

    tl.store(
        y_ptr + pid_b * stride_yb + offs_n * stride_yn,
        acc,
        mask=mask_n,
    )


@triton.jit
def _rtn4_gemm_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    N,
    nwords,
    B,
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
    IEEE: tl.constexpr,
    STREAM_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """batch>1: one CTA holds ``[BLOCK_N, BLOCK_B]`` and reads each code tile once.

    The affine dequant is kept factored per group rather than materialized:

        sum_k (s_g * q[n,k] + b_g) * x[c,k]
            == s_g * dot(q, x)[n,c] + b_g * sum_k x[c,k]

    so the only value entering ``tl.dot`` is the raw 4-bit code, which is an
    integer 0..15 and therefore exact in bfloat16. Building a dequantized
    ``s*q + b`` tile and rounding *that* to bfloat16 would instead throw away
    ~8 mantissa bits of the scale before the tensor core ever sees it. The
    accumulator is float32 either way, so this matches the batch-1 kernel.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    acc = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 8
    GROUPS: tl.constexpr = BLOCK_K // GROUP_SIZE
    shifts = (tl.arange(0, 8) * 4).to(tl.uint32)
    k_total = nwords * 8
    n_groups = k_total // GROUP_SIZE

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        # One wide load per tile, as in the batch-1 kernel. Splitting the tile
        # into per-group loads instead (the obvious way to write the factored
        # dequant) drops a row's contiguous 256-byte read to eight 32-byte ones
        # and cost ~30% of DRAM bandwidth: 167 vs 240 GB/s on the lm_head.
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
            eviction_policy="evict_first" if STREAM_W else "",
        ).to(tl.uint32)
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0xF

        offs_k = w0 * 8 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_total
        x_tile = tl.load(
            x_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk,
            mask=mask_b[:, None] & mask_k[None, :],
            other=0.0,
            eviction_policy="evict_last" if STREAM_W else "",
        )
        offs_g = (w0 * 8) // GROUP_SIZE + tl.arange(0, GROUPS)
        mask_g = offs_g < n_groups
        scale = tl.load(
            scale_ptr + offs_n[:, None] * stride_sn + offs_g[None, :] * stride_sg,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + offs_n[:, None] * stride_bn + offs_g[None, :] * stride_bg,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)

        # The wide tile is then re-cut along K into its RTN groups and fed to a
        # batched dot, one matmul per group, so each group still meets its own
        # scale outside the tensor core.
        x_g = tl.reshape(x_tile, (BLOCK_B, GROUPS, GROUP_SIZE))
        sum_x = tl.trans(tl.sum(x_g.to(tl.float32), axis=2), 1, 0)
        q_g = tl.trans(tl.reshape(codes, (BLOCK_N, GROUPS, GROUP_SIZE)), 1, 0, 2)
        x_3 = tl.trans(x_g, 1, 2, 0)
        if IEEE:
            dot = tl.dot(
                q_g.to(tl.float32), x_3.to(tl.float32), input_precision="ieee"
            )
        else:
            dot = tl.dot(q_g.to(tl.bfloat16), x_3.to(tl.bfloat16))
        scale_g = tl.trans(scale, 1, 0)[:, :, None]
        bias_g = tl.trans(bias, 1, 0)[:, :, None]
        acc += tl.sum(scale_g * dot + bias_g * sum_x[:, None, :], axis=0)

    tl.store(
        y_ptr + offs_b[None, :] * stride_yb + offs_n[:, None] * stride_yn,
        acc,
        mask=mask_n[:, None] & mask_b[None, :],
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

    block_n, block_k_words, block_b, num_warps, num_stages = _rtn4_launch_meta(
        nwords, batch
    )
    if (block_k_words * _CODES_PER_WORD) % group_size != 0:
        raise ValueError(
            f"BLOCK_K_WORDS={block_k_words} is not a whole number of RTN groups "
            f"(group_size={group_size})."
        )
    if batch == 1:
        grid = (triton.cdiv(n_out, block_n), 1)
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
            STREAM_W=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y.reshape(*leading, n_out)

    b_tiles = triton.cdiv(batch, block_b)
    grid = (triton.cdiv(n_out, block_n), b_tiles)
    _rtn4_gemm_kernel[grid](
        x_mat,
        packed_weight,
        scales,
        biases,
        y,
        n_out,
        nwords,
        batch,
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
        # bfloat16 is the only dtype whose codes-times-activations product is
        # exact on the tensor core; anything else takes the float32 path.
        IEEE=x_mat.dtype != torch.bfloat16,
        STREAM_W=b_tiles == 1,
        BLOCK_N=block_n,
        BLOCK_K_WORDS=block_k_words,
        BLOCK_B=block_b,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.reshape(*leading, n_out)


@triton.jit
def _rtn4_indexed_gemv_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    bias_ptr,
    cluster_ids_ptr,
    y_ptr,
    n_out,
    cluster_size,
    nwords,
    stride_xb,
    stride_xk,
    stride_wc,
    stride_wn,
    stride_ww,
    stride_sc,
    stride_sn,
    stride_sg,
    stride_bc,
    stride_bn,
    stride_bg,
    stride_yb,
    stride_yn,
    GROUP_SIZE: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
    STREAM_W: tl.constexpr,
):
    """GEMV over flattened ``head[cluster_ids[n // C], n % C]`` rows."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_out
    probe = offs_n // CLUSTER_SIZE
    member = offs_n % CLUSTER_SIZE
    cid = tl.load(cluster_ids_ptr + probe, mask=mask_n, other=0).to(tl.int32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_row = x_ptr + pid_b * stride_xb

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 8
    N_GROUPS_TILE: tl.constexpr = BLOCK_K // GROUP_SIZE
    shifts = (tl.arange(0, 8) * 4).to(tl.uint32)
    k_total = nwords * 8
    n_groups = k_total // GROUP_SIZE

    packed_row = packed_ptr + cid * stride_wc + member * stride_wn
    scale_row = scale_ptr + cid * stride_sc + member * stride_sn
    bias_row = bias_ptr + cid * stride_bc + member * stride_bn

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        packed = tl.load(
            packed_row[:, None] + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
            eviction_policy="evict_first" if STREAM_W else "",
        ).to(tl.uint32)
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0xF
        q = tl.reshape(codes.to(tl.float32), (BLOCK_N, BLOCK_K))

        offs_k = w0 * 8 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_total
        x_tile = tl.load(
            x_row + offs_k * stride_xk,
            mask=mask_k,
            other=0.0,
            eviction_policy="evict_last" if STREAM_W else "",
        ).to(tl.float32)
        offs_g = (w0 * 8) // GROUP_SIZE + tl.arange(0, N_GROUPS_TILE)
        mask_g = offs_g < n_groups
        scale = tl.load(
            scale_row[:, None] + offs_g[None, :] * stride_sg,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(
            bias_row[:, None] + offs_g[None, :] * stride_bg,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        ones = tl.full((GROUP_SIZE,), 1.0, dtype=tl.float32)
        scale_k = tl.reshape(scale[:, :, None] * ones[None, None, :], (BLOCK_N, BLOCK_K))
        bias_k = tl.reshape(bias[:, :, None] * ones[None, None, :], (BLOCK_N, BLOCK_K))
        acc += tl.sum((scale_k * q + bias_k) * x_tile[None, :], axis=1)

    tl.store(
        y_ptr + pid_b * stride_yb + offs_n * stride_yn,
        acc,
        mask=mask_n,
    )


def rtn4_indexed_gemv(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    cluster_ids: torch.Tensor,
    group_size: int = HEAD_GROUP_SIZE,
) -> torch.Tensor:
    """Decode GEMV over selected FlashHead clusters.

    ``packed_weight`` is ``[n_clusters, cluster_size, nwords]`` — the lm_head
    rows permuted by ``token_map``. ``cluster_ids`` is ``[n_probes]``. Output
    logits are ``[n_probes * cluster_size]`` in cluster-major order, matching
    ``token_map[cluster_ids].reshape(-1)``.

    Tiles match decode ``rtn4_gemv`` (BLOCK_N=16, fat K) over the flattened
    16 k probed rows. A one-CTA-per-cluster launch under-occupied this GPU.
    """
    if packed_weight.dtype != torch.uint32:
        raise TypeError(
            f"packed_weight must be uint32 codes, got {packed_weight.dtype}; "
            "refusing to run a dense-weight path."
        )
    if packed_weight.ndim != 3:
        raise ValueError(
            f"packed_weight must be 3-D [n_clusters, cluster_size, nwords]; "
            f"got shape {tuple(packed_weight.shape)}."
        )
    if not x.is_cuda or not packed_weight.is_cuda:
        raise RuntimeError("rtn4_indexed_gemv requires CUDA tensors.")
    if cluster_ids.ndim != 1:
        raise ValueError(
            f"cluster_ids must be 1-D [n_probes]; got shape {tuple(cluster_ids.shape)}."
        )

    n_clusters, cluster_size, nwords = packed_weight.shape
    n_probes = int(cluster_ids.shape[0])
    n_groups = nwords * _CODES_PER_WORD // group_size
    expect_sb = (n_clusters, cluster_size, n_groups)
    if scales.shape != expect_sb or biases.shape != expect_sb:
        raise ValueError(
            f"scales/biases expected {expect_sb}, got {tuple(scales.shape)} "
            f"and {tuple(biases.shape)}."
        )

    k_in = nwords * _CODES_PER_WORD
    if x.shape[-1] != k_in:
        raise ValueError(f"x last dim {x.shape[-1]} != packed K ({k_in}).")

    leading = x.shape[:-1]
    batch = int(x.numel() // k_in)
    x_mat = x.reshape(batch, k_in)
    n_out = n_probes * cluster_size
    y = torch.empty(batch, n_out, device=x.device, dtype=x.dtype)

    # FlashHead probes one row at a time; the batch dim stays on the grid here,
    # so this path only wants the batch-1 tiling and ignores BLOCK_B.
    block_n, block_k_words, _block_b, num_warps, num_stages = _rtn4_launch_meta(
        nwords, 1
    )
    if (block_k_words * _CODES_PER_WORD) % group_size != 0:
        raise ValueError(
            f"BLOCK_K_WORDS={block_k_words} is not a whole number of RTN groups "
            f"(group_size={group_size})."
        )
    grid = (triton.cdiv(n_out, block_n), batch)
    _rtn4_indexed_gemv_kernel[grid](
        x_mat,
        packed_weight,
        scales,
        biases,
        cluster_ids,
        y,
        n_out,
        cluster_size,
        nwords,
        x_mat.stride(0),
        x_mat.stride(1),
        packed_weight.stride(0),
        packed_weight.stride(1),
        packed_weight.stride(2),
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),
        biases.stride(0),
        biases.stride(1),
        biases.stride(2),
        y.stride(0),
        y.stride(1),
        GROUP_SIZE=group_size,
        CLUSTER_SIZE=cluster_size,
        BLOCK_N=block_n,
        BLOCK_K_WORDS=block_k_words,
        STREAM_W=batch == 1,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.reshape(*leading, n_out)
