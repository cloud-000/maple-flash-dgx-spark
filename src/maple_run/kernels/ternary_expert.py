"""Fused packed-expert GEMV: selected experts only, weights stay 3-D uint32.

Batch-1 decode is one kernel over ``(token, top-k slot)`` pairs: eight selected
experts, each streamed once. A Python loop over 256 experts is the failure
mode that exists to replace.

Batched decode (and prefill) groups slots by expert so tokens that share an
expert share one load of that expert's packed codes -- the same inversion as
batch>1 ``ternary_gemv``. Launching one program per slot re-reads the same
row of ``switch_mlp`` once per hit and is why expert time grew linearly with
B. Packed codes are never unpacked to a dense bf16 matrix.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_CODES_PER_WORD = 16
# Grouped kernels win once many tokens share experts (prefill, decode B>=32).
# Below that, per-slot is faster: empty-expert CTAs and grouping overhead
# outweigh reuse of 1-2 hits per expert.
_GROUPED_MIN_TOKENS = int(os.environ.get("MAPLE_GROUPED_MIN_TOKENS", "32"))
GROUPED_MIN_TOKENS = _GROUPED_MIN_TOKENS


def _expert_nk(
    packed_weight: torch.Tensor, packed_kn: bool
) -> tuple[int, int, int, int, int, int]:
    """Return n_exp, n_out, nwords, stride_we, stride_wn, stride_ww."""
    if packed_kn:
        n_exp, nwords, n_out = packed_weight.shape
        return (
            n_exp,
            n_out,
            nwords,
            packed_weight.stride(0),
            packed_weight.stride(2),
            packed_weight.stride(1),
        )
    n_exp, n_out, nwords = packed_weight.shape
    return (
        n_exp,
        n_out,
        nwords,
        packed_weight.stride(0),
        packed_weight.stride(1),
        packed_weight.stride(2),
    )


def _expert_launch_meta(nwords: int) -> tuple[int, int, int, int]:
    """BLOCK_N, BLOCK_K_WORDS, warps, stages for one selected expert.

    Swept L2-cold at the Maple decode shapes (a single-buffer microbench keeps
    the selected experts resident and overstates this by ~1.7x). Narrow tiles
    win here: only 8 experts are selected, so ``BLOCK_N=2`` with one warp is
    what puts enough CTAs on the machine to cover DRAM latency -- up/gate
    ``K=2048`` 207 GB/s (against 178 at ``BLOCK_N=8``), down ``K=512`` 211.
    """
    if nwords >= 64:
        return 2, 64, 1, 3
    if nwords >= 32:
        return 4, 32, 1, 4
    return 16, 8, 4, 2


def _grouped_launch_meta(nwords: int, n_tokens: int) -> tuple[int, int, int, int, int]:
    """BLOCK_N, BLOCK_K_WORDS, BLOCK_B, warps, stages for grouped expert GEMM.

    Keep the batch-1 N-tiles (many thin CTAs hiding DRAM latency). ``BLOCK_B``
    is the tokens that share one code load; it stays small because top-8 of
    256 at decode B<=32 is typically 1-3 tokens per expert, and a fat B-tile
    just pads MMA. Prefill (many tokens) can take a wider B. Shape-based so
    CUDA graphs stay legal.
    """
    block_n, block_k_words, num_warps, num_stages = _expert_launch_meta(nwords)
    # Two tokens share a code load; a wider B-tile pads compute and was slower
    # than per-slot even when the extra columns were masked.
    block_b = 2 if n_tokens <= 32 else 4
    return block_n, block_k_words, block_b, num_warps, num_stages


def expert_slot_groups(
    expert_ids: torch.Tensor, n_exp: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Sort slots by expert id. Shapes depend only on ``expert_ids`` and ``n_exp``.

    Returns ``(order, offsets, unique_ids)``:

    - ``order``: int32 ``[slots]``, permutation that packs equal ids together
    - ``offsets``: int32 ``[n_exp + 1]``, prefix sums of per-expert counts
      into ``order``
    - ``unique_ids``: int32 ``[slots]`` compacted expert ids with ``-1``
      padding when ``slots < n_exp``; ``None`` when the grid can use
      ``program_id(1)`` as the expert id (``slots >= n_exp``)

    No GPU→CPU sync. Safe to capture inside a CUDA graph.
    """
    slots = int(expert_ids.numel())
    ids = expert_ids.reshape(slots)
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.to(torch.int32)
    order = torch.argsort(ids).to(torch.int32)
    counts = torch.zeros(n_exp, device=ids.device, dtype=torch.int32)
    counts.scatter_add_(
        0, ids.long(), torch.ones(slots, device=ids.device, dtype=torch.int32)
    )
    # pad/zeros stay on GPU — a Python ``offsets[0] = 0`` is a CPU→CUDA copy
    # and is illegal inside CUDA graph capture.
    offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))

    if slots >= n_exp:
        return order, offsets, None

    sorted_ids = ids.index_select(0, order.long())
    if sorted_ids.dtype != torch.int32:
        sorted_ids = sorted_ids.to(torch.int32)
    if slots == 1:
        unique_ids = sorted_ids
        return order, offsets, unique_ids

    starts = torch.cat(
        (
            torch.ones(1, device=ids.device, dtype=torch.int32),
            (sorted_ids[1:] != sorted_ids[:-1]).to(torch.int32),
        )
    )
    group_id = starts.cumsum(0) - 1
    unique_ids = torch.full((slots,), -1, device=ids.device, dtype=torch.int32)
    unique_ids.scatter_(0, group_id.long(), sorted_ids)
    return order, offsets, unique_ids


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
    STREAM_W: tl.constexpr,
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
            eviction_policy="evict_first" if STREAM_W else "",
        ).to(tl.uint32)
        codes = (packed[:, :, None] >> shifts[None, None, :]) & 0x3
        ternary = tl.reshape(codes.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))

        offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < (nwords * 16)
        x_tile = tl.load(
            x_row + offs_k * stride_xk,
            mask=mask_k,
            other=0.0,
            eviction_policy="evict_last" if STREAM_W else "",
        ).to(tl.float32)
        acc += tl.sum(ternary * x_tile[None, :], axis=1)

    alpha = tl.load(alpha_e + offs_n * stride_an, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(
        y_ptr + pid_s * stride_ys + offs_n * stride_yn,
        acc * alpha,
        mask=mask_n,
    )


@triton.jit
def _ternary_expert_gemv_grouped_kernel(
    x_ptr,
    packed_ptr,
    alpha_ptr,
    order_ptr,
    offsets_ptr,
    unique_ptr,
    y_ptr,
    N,
    nwords,
    stride_x,
    stride_xk,
    stride_we,
    stride_wn,
    stride_ww,
    stride_ae,
    stride_an,
    stride_ys,
    stride_yn,
    USE_COMPACT: tl.constexpr,
    PER_SLOT_X: tl.constexpr,
    TOPK: tl.constexpr,
    STREAM_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """One CTA = one expert's N-tile; tokens of that expert share the code load."""
    pid_n = tl.program_id(0)
    pid_e = tl.program_id(1)

    if USE_COMPACT:
        expert_id = tl.load(unique_ptr + pid_e)
        if expert_id < 0:
            return
    else:
        expert_id = pid_e
    expert_id = expert_id.to(tl.int64)

    start = tl.load(offsets_ptr + expert_id)
    count = tl.load(offsets_ptr + expert_id + 1) - start
    if count == 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    packed_e = packed_ptr + expert_id * stride_we
    alpha_e = alpha_ptr + expert_id * stride_ae
    alpha = tl.load(alpha_e + offs_n * stride_an, mask=mask_n, other=0.0).to(tl.float32)

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 16
    shifts = (tl.arange(0, 16) * 2).to(tl.uint32)
    k_total = nwords * 16

    if count == 1:
        slot = tl.load(order_ptr + start)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if PER_SLOT_X:
            x_idx = slot
        else:
            x_idx = slot // TOPK
        for w0 in range(0, nwords, BLOCK_K_WORDS):
            offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
            mask_w = offs_w < nwords
            packed = tl.load(
                packed_e + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
                mask=mask_n[:, None] & mask_w[None, :],
                other=0,
                eviction_policy="evict_first" if STREAM_W else "",
            ).to(tl.uint32)
            codes = (packed[:, :, None] >> shifts[None, None, :]) & 0x3
            ternary = tl.reshape(codes.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
            offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_total
            xb = tl.load(
                x_ptr + x_idx * stride_x + offs_k * stride_xk,
                mask=mask_k,
                other=0.0,
                eviction_policy="evict_last" if STREAM_W else "",
            ).to(tl.float32)
            acc += tl.sum(ternary * xb, axis=1)
        tl.store(
            y_ptr + slot * stride_ys + offs_n * stride_yn,
            acc * alpha,
            mask=mask_n,
        )
        return

    n_tiles = tl.cdiv(count, BLOCK_B)
    for tile in range(n_tiles):
        t0 = tile * BLOCK_B
        offs_b = t0 + tl.arange(0, BLOCK_B)
        mask_b = offs_b < count
        slots = tl.load(order_ptr + start + offs_b, mask=mask_b, other=0)
        acc = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)
        for w0 in range(0, nwords, BLOCK_K_WORDS):
            offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
            mask_w = offs_w < nwords
            packed = tl.load(
                packed_e + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
                mask=mask_n[:, None] & mask_w[None, :],
                other=0,
                eviction_policy="evict_first" if STREAM_W else "",
            ).to(tl.uint32)
            codes = (packed[:, :, None] >> shifts[None, None, :]) & 0x3
            ternary = tl.reshape(codes.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
            offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_total
            for b in tl.static_range(BLOCK_B):
                slot_b = tl.load(
                    order_ptr + start + t0 + b, mask=(t0 + b) < count, other=0
                )
                if PER_SLOT_X:
                    x_idx_b = slot_b
                else:
                    x_idx_b = slot_b // TOPK
                xb = tl.load(
                    x_ptr + x_idx_b * stride_x + offs_k * stride_xk,
                    mask=((t0 + b) < count) & mask_k,
                    other=0.0,
                    eviction_policy="evict_last" if STREAM_W else "",
                ).to(tl.float32)
                acc += tl.sum(ternary * xb, axis=1)[:, None] * (
                    tl.arange(0, BLOCK_B) == b
                ).to(tl.float32)[None, :]
        tl.store(
            y_ptr + slots[None, :] * stride_ys + offs_n[:, None] * stride_yn,
            acc * alpha[:, None],
            mask=mask_n[:, None] & mask_b[None, :],
        )


def ternary_expert_gemv(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    packed_kn: bool = False,
    slot_groups: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
    grouped: bool | None = None,
) -> torch.Tensor:
    """Indexed packed GEMV for stacked experts.

    Parameters
    ----------
    x:
        Activations. Either one row per token (``[..., K]`` with
        ``expert_ids`` shaped ``[..., topk]``) or one row per slot
        (``x.shape[:-1] == expert_ids.shape``).
    packed_weight:
        ``uint32`` codes ``[E, N, K/16]`` or ``[E, K/16, N]`` if ``packed_kn``.
    row_alpha:
        Per-expert per-output-row scale ``[E, N]``.
    expert_ids:
        Expert indices (any integer dtype) selecting rows of ``packed_weight``.
    slot_groups:
        Optional ``expert_slot_groups`` result, reused when up-gate and down
        share the same routing.

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

    n_exp, n_out, nwords, stride_we, stride_wn, stride_ww = _expert_nk(
        packed_weight, packed_kn
    )
    k_in = nwords * _CODES_PER_WORD
    if x.shape[-1] != k_in:
        raise ValueError(f"x last dim {x.shape[-1]} != packed K ({k_in}).")
    if tuple(row_alpha.shape) != (n_exp, n_out):
        raise ValueError(
            f"row_alpha shape {tuple(row_alpha.shape)} != {(n_exp, n_out)}."
        )

    id_shape = tuple(expert_ids.shape)
    per_slot = x.shape[:-1] == id_shape
    if per_slot:
        n_tokens = int(expert_ids.numel() // id_shape[-1]) if len(id_shape) > 1 else int(
            expert_ids.numel()
        )
        if len(id_shape) == 1:
            n_tokens = int(x.numel() // k_in)
        topk = 1
        x_mat = x.reshape(-1, k_in)
    elif x.shape[:-1] == id_shape[:-1]:
        topk = id_shape[-1]
        n_tokens = int(x.numel() // k_in)
        if n_tokens * topk != expert_ids.numel():
            raise ValueError(
                f"x leading {x.shape[:-1]} incompatible with expert_ids {id_shape}."
            )
        x_mat = x.reshape(n_tokens, k_in)
    else:
        raise ValueError(
            f"x shape {tuple(x.shape)} incompatible with expert_ids {id_shape}."
        )

    slots = int(expert_ids.numel())
    ids = expert_ids.reshape(slots)
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.to(torch.int32)
    y = torch.empty(slots, n_out, device=x.device, dtype=x.dtype)

    use_grouped = n_tokens >= _GROUPED_MIN_TOKENS if grouped is None else grouped
    if not use_grouped:
        if per_slot:
            slot_x = x_mat
        else:
            slot_x = (
                x_mat.unsqueeze(1).expand(n_tokens, topk, k_in).reshape(slots, k_in)
            )
        block_n, block_k_words, num_warps, num_stages = _expert_launch_meta(nwords)
        grid = (triton.cdiv(n_out, block_n), slots)
        _ternary_expert_gemv_kernel[grid](
            slot_x,
            packed_weight,
            row_alpha,
            ids.contiguous(),
            y,
            n_out,
            nwords,
            slot_x.stride(0),
            slot_x.stride(1),
            stride_we,
            stride_wn,
            stride_ww,
            row_alpha.stride(0),
            row_alpha.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_N=block_n,
            BLOCK_K_WORDS=block_k_words,
            STREAM_W=n_tokens == 1,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y.reshape(*id_shape, n_out)

    order, offsets, unique_ids = (
        slot_groups if slot_groups is not None else expert_slot_groups(ids, n_exp)
    )
    use_compact = unique_ids is not None
    n_launch = int(unique_ids.shape[0]) if use_compact else n_exp
    uniq = unique_ids if use_compact else order
    block_n, block_k_words, block_b, num_warps, num_stages = _grouped_launch_meta(
        nwords, n_tokens
    )
    grid = (triton.cdiv(n_out, block_n), n_launch)
    _ternary_expert_gemv_grouped_kernel[grid](
        x_mat,
        packed_weight,
        row_alpha,
        order,
        offsets,
        uniq,
        y,
        n_out,
        nwords,
        x_mat.stride(0),
        x_mat.stride(1),
        stride_we,
        stride_wn,
        stride_ww,
        row_alpha.stride(0),
        row_alpha.stride(1),
        y.stride(0),
        y.stride(1),
        USE_COMPACT=use_compact,
        PER_SLOT_X=per_slot,
        TOPK=max(topk, 1),
        STREAM_W=True,
        BLOCK_N=block_n,
        BLOCK_K_WORDS=block_k_words,
        BLOCK_B=block_b,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.reshape(*id_shape, n_out)


MLP_CLAMP = 7.0


@triton.jit
def _ternary_expert_swiglu_kernel(
    x_ptr,
    packed_ptr,
    alpha_ptr,
    ids_ptr,
    y_ptr,
    N,
    nwords,
    stride_xt,
    stride_xk,
    stride_we,
    stride_wn,
    stride_ww,
    stride_ae,
    stride_an,
    stride_ys,
    stride_yn,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
    STREAM_W: tl.constexpr,
):
    """Fused up+gate GEMV + SiLU/clamp. ``N`` is the intermediate size (half of packed rows)."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    token = pid_s // TOPK
    expert_id = tl.load(ids_ptr + pid_s).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_row = x_ptr + token * stride_xt

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 16
    shifts = (tl.arange(0, 16) * 2).to(tl.uint32)
    packed_e = packed_ptr + expert_id * stride_we
    alpha_e = alpha_ptr + expert_id * stride_ae

    for w0 in range(0, nwords, BLOCK_K_WORDS):
        offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
        mask_w = offs_w < nwords
        packed_up = tl.load(
            packed_e + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
            eviction_policy="evict_first" if STREAM_W else "",
        ).to(tl.uint32)
        packed_gate = tl.load(
            packed_e + (offs_n + N)[:, None] * stride_wn + offs_w[None, :] * stride_ww,
            mask=mask_n[:, None] & mask_w[None, :],
            other=0,
            eviction_policy="evict_first" if STREAM_W else "",
        ).to(tl.uint32)
        codes_up = (packed_up[:, :, None] >> shifts[None, None, :]) & 0x3
        codes_gate = (packed_gate[:, :, None] >> shifts[None, None, :]) & 0x3
        ternary_up = tl.reshape(codes_up.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
        ternary_gate = tl.reshape(codes_gate.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
        offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < (nwords * 16)
        x_tile = tl.load(
            x_row + offs_k * stride_xk,
            mask=mask_k,
            other=0.0,
            eviction_policy="evict_last" if STREAM_W else "",
        ).to(tl.float32)
        acc_up += tl.sum(ternary_up * x_tile[None, :], axis=1)
        acc_gate += tl.sum(ternary_gate * x_tile[None, :], axis=1)

    a_up = tl.load(alpha_e + offs_n * stride_an, mask=mask_n, other=0.0).to(tl.float32)
    a_gate = tl.load(alpha_e + (offs_n + N) * stride_an, mask=mask_n, other=0.0).to(
        tl.float32
    )
    up = acc_up * a_up
    gate = acc_gate * a_gate
    up = tl.minimum(tl.maximum(up, -7.0), 7.0)
    gate = tl.minimum(gate, 7.0)
    out = up * gate * tl.sigmoid(gate)
    tl.store(y_ptr + pid_s * stride_ys + offs_n * stride_yn, out, mask=mask_n)


@triton.jit
def _ternary_expert_swiglu_grouped_kernel(
    x_ptr,
    packed_ptr,
    alpha_ptr,
    order_ptr,
    offsets_ptr,
    unique_ptr,
    y_ptr,
    N,
    nwords,
    stride_xt,
    stride_xk,
    stride_we,
    stride_wn,
    stride_ww,
    stride_ae,
    stride_an,
    stride_ys,
    stride_yn,
    USE_COMPACT: tl.constexpr,
    TOPK: tl.constexpr,
    STREAM_W: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Grouped up+gate GEMV + SiLU/clamp. Tokens of one expert share both code loads."""
    pid_n = tl.program_id(0)
    pid_e = tl.program_id(1)

    if USE_COMPACT:
        expert_id = tl.load(unique_ptr + pid_e)
        if expert_id < 0:
            return
    else:
        expert_id = pid_e
    expert_id = expert_id.to(tl.int64)

    start = tl.load(offsets_ptr + expert_id)
    count = tl.load(offsets_ptr + expert_id + 1) - start
    if count == 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    packed_e = packed_ptr + expert_id * stride_we
    alpha_e = alpha_ptr + expert_id * stride_ae
    a_up = tl.load(alpha_e + offs_n * stride_an, mask=mask_n, other=0.0).to(tl.float32)
    a_gate = tl.load(alpha_e + (offs_n + N) * stride_an, mask=mask_n, other=0.0).to(
        tl.float32
    )

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 16
    shifts = (tl.arange(0, 16) * 2).to(tl.uint32)
    k_total = nwords * 16

    if count == 1:
        slot = tl.load(order_ptr + start)
        token = slot // TOPK
        acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
        x_row = x_ptr + token * stride_xt
        for w0 in range(0, nwords, BLOCK_K_WORDS):
            offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
            mask_w = offs_w < nwords
            packed_up = tl.load(
                packed_e + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
                mask=mask_n[:, None] & mask_w[None, :],
                other=0,
                eviction_policy="evict_first" if STREAM_W else "",
            ).to(tl.uint32)
            packed_gate = tl.load(
                packed_e + (offs_n + N)[:, None] * stride_wn + offs_w[None, :] * stride_ww,
                mask=mask_n[:, None] & mask_w[None, :],
                other=0,
                eviction_policy="evict_first" if STREAM_W else "",
            ).to(tl.uint32)
            codes_up = (packed_up[:, :, None] >> shifts[None, None, :]) & 0x3
            codes_gate = (packed_gate[:, :, None] >> shifts[None, None, :]) & 0x3
            ternary_up = tl.reshape(codes_up.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
            ternary_gate = tl.reshape(codes_gate.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
            offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_total
            x_tile = tl.load(
                x_row + offs_k * stride_xk,
                mask=mask_k,
                other=0.0,
                eviction_policy="evict_last" if STREAM_W else "",
            ).to(tl.float32)
            acc_up += tl.sum(ternary_up * x_tile[None, :], axis=1)
            acc_gate += tl.sum(ternary_gate * x_tile[None, :], axis=1)
        up = acc_up * a_up
        gate = acc_gate * a_gate
        up = tl.minimum(tl.maximum(up, -7.0), 7.0)
        gate = tl.minimum(gate, 7.0)
        out = up * gate * tl.sigmoid(gate)
        tl.store(y_ptr + slot * stride_ys + offs_n * stride_yn, out, mask=mask_n)
        return

    n_tiles = tl.cdiv(count, BLOCK_B)
    for tile in range(n_tiles):
        t0 = tile * BLOCK_B
        offs_b = t0 + tl.arange(0, BLOCK_B)
        mask_b = offs_b < count
        slots = tl.load(order_ptr + start + offs_b, mask=mask_b, other=0)
        acc_up = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)
        acc_gate = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)

        for w0 in range(0, nwords, BLOCK_K_WORDS):
            offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
            mask_w = offs_w < nwords
            packed_up = tl.load(
                packed_e + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
                mask=mask_n[:, None] & mask_w[None, :],
                other=0,
                eviction_policy="evict_first" if STREAM_W else "",
            ).to(tl.uint32)
            packed_gate = tl.load(
                packed_e + (offs_n + N)[:, None] * stride_wn + offs_w[None, :] * stride_ww,
                mask=mask_n[:, None] & mask_w[None, :],
                other=0,
                eviction_policy="evict_first" if STREAM_W else "",
            ).to(tl.uint32)
            codes_up = (packed_up[:, :, None] >> shifts[None, None, :]) & 0x3
            codes_gate = (packed_gate[:, :, None] >> shifts[None, None, :]) & 0x3
            ternary_up = tl.reshape(codes_up.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
            ternary_gate = tl.reshape(codes_gate.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))

            offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_total
            for b in tl.static_range(BLOCK_B):
                slot_b = tl.load(
                    order_ptr + start + t0 + b, mask=(t0 + b) < count, other=0
                )
                xb = tl.load(
                    x_ptr + (slot_b // TOPK) * stride_xt + offs_k * stride_xk,
                    mask=((t0 + b) < count) & mask_k,
                    other=0.0,
                    eviction_policy="evict_last" if STREAM_W else "",
                ).to(tl.float32)
                sel = (tl.arange(0, BLOCK_B) == b).to(tl.float32)[None, :]
                acc_up += tl.sum(ternary_up * xb, axis=1)[:, None] * sel
                acc_gate += tl.sum(ternary_gate * xb, axis=1)[:, None] * sel

        up = acc_up * a_up[:, None]
        gate = acc_gate * a_gate[:, None]
        up = tl.minimum(tl.maximum(up, -7.0), 7.0)
        gate = tl.minimum(gate, 7.0)
        out = up * gate * tl.sigmoid(gate)
        tl.store(
            y_ptr + slots[None, :] * stride_ys + offs_n[:, None] * stride_yn,
            out,
            mask=mask_n[:, None] & mask_b[None, :],
        )


@triton.jit
def _ternary_expert_down_sum_kernel(
    x_ptr,
    packed_ptr,
    alpha_ptr,
    ids_ptr,
    w_ptr,
    residual_ptr,
    y_ptr,
    N,
    nwords,
    stride_xt,
    stride_xs,
    stride_xk,
    stride_we,
    stride_wn,
    stride_ww,
    stride_ae,
    stride_an,
    stride_id_t,
    stride_id_s,
    stride_wt,
    stride_ws,
    stride_rt,
    stride_rn,
    stride_yt,
    stride_yn,
    HAS_RESIDUAL: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_WORDS: tl.constexpr,
    STREAM_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    BLOCK_K: tl.constexpr = BLOCK_K_WORDS * 16
    shifts = (tl.arange(0, 16) * 2).to(tl.uint32)

    for s in tl.static_range(0, TOPK):
        expert_id = tl.load(ids_ptr + pid_t * stride_id_t + s * stride_id_s).to(tl.int64)
        slot_w = tl.load(w_ptr + pid_t * stride_wt + s * stride_ws).to(tl.float32)
        x_row = x_ptr + pid_t * stride_xt + s * stride_xs
        packed_e = packed_ptr + expert_id * stride_we
        alpha_e = alpha_ptr + expert_id * stride_ae
        partial = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for w0 in range(0, nwords, BLOCK_K_WORDS):
            offs_w = w0 + tl.arange(0, BLOCK_K_WORDS)
            mask_w = offs_w < nwords
            packed = tl.load(
                packed_e + offs_n[:, None] * stride_wn + offs_w[None, :] * stride_ww,
                mask=mask_n[:, None] & mask_w[None, :],
                other=0,
                eviction_policy="evict_first" if STREAM_W else "",
            ).to(tl.uint32)
            codes = (packed[:, :, None] >> shifts[None, None, :]) & 0x3
            ternary = tl.reshape(codes.to(tl.float32) - 1.0, (BLOCK_N, BLOCK_K))
            offs_k = w0 * 16 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < (nwords * 16)
            x_tile = tl.load(
                x_row + offs_k * stride_xk,
                mask=mask_k,
                other=0.0,
                eviction_policy="evict_last" if STREAM_W else "",
            ).to(tl.float32)
            partial += tl.sum(ternary * x_tile[None, :], axis=1)
        alpha = tl.load(alpha_e + offs_n * stride_an, mask=mask_n, other=0.0).to(tl.float32)
        acc += slot_w * partial * alpha

    if HAS_RESIDUAL:
        acc += tl.load(
            residual_ptr + pid_t * stride_rt + offs_n * stride_rn,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
    tl.store(y_ptr + pid_t * stride_yt + offs_n * stride_yn, acc, mask=mask_n)


def ternary_expert_swiglu(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    packed_kn: bool = False,
    slot_groups: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
    grouped: bool | None = None,
) -> torch.Tensor:
    """Indexed up+gate GEMV with SiLU/clamp fused. Packed rows are ``[up; gate]``.

    ``x`` is per-token ``[..., K]``; ``expert_ids`` is ``[..., topk]``.
    Returns activated expert hidden ``[..., topk, N]`` with ``N = packed N / 2``.
    ``slot_groups`` is an optional ``expert_slot_groups`` result shared with down.
    """
    if packed_weight.ndim != 3 or packed_weight.dtype != torch.uint32:
        raise ValueError("packed_weight must be uint32 [E, 2N, K/16] or [E, K/16, 2N].")
    if not x.is_cuda or not packed_weight.is_cuda or not expert_ids.is_cuda:
        raise RuntimeError("ternary_expert_swiglu requires CUDA tensors.")
    n_exp, n_rows, nwords, stride_we, stride_wn, stride_ww = _expert_nk(
        packed_weight, packed_kn
    )
    if n_rows % 2 != 0:
        raise ValueError(f"packed up_gate rows must be even, got {n_rows}.")
    n_out = n_rows // 2
    k_in = nwords * _CODES_PER_WORD
    if x.shape[-1] != k_in:
        raise ValueError(f"x last dim {x.shape[-1]} != packed K ({k_in}).")
    if tuple(row_alpha.shape) != (n_exp, n_rows):
        raise ValueError(f"row_alpha shape {tuple(row_alpha.shape)} != {(n_exp, n_rows)}.")
    id_shape = tuple(expert_ids.shape)
    if x.shape[:-1] != id_shape[:-1]:
        raise ValueError(f"x shape {tuple(x.shape)} incompatible with expert_ids {id_shape}.")
    topk = id_shape[-1]
    tokens = int(x.numel() // k_in)
    x_mat = x.reshape(tokens, k_in)
    slots = tokens * topk
    ids = expert_ids.reshape(slots)
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.to(torch.int32)
    y = torch.empty(slots, n_out, device=x.device, dtype=x.dtype)

    use_grouped = tokens >= _GROUPED_MIN_TOKENS if grouped is None else grouped
    if not use_grouped:
        block_n, block_k_words, num_warps, num_stages = _expert_launch_meta(nwords)
        grid = (triton.cdiv(n_out, block_n), slots)
        _ternary_expert_swiglu_kernel[grid](
            x_mat,
            packed_weight,
            row_alpha,
            ids.contiguous(),
            y,
            n_out,
            nwords,
            x_mat.stride(0),
            x_mat.stride(1),
            stride_we,
            stride_wn,
            stride_ww,
            row_alpha.stride(0),
            row_alpha.stride(1),
            y.stride(0),
            y.stride(1),
            TOPK=topk,
            BLOCK_N=block_n,
            BLOCK_K_WORDS=block_k_words,
            STREAM_W=tokens == 1,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y.reshape(*id_shape, n_out)

    order, offsets, unique_ids = (
        slot_groups if slot_groups is not None else expert_slot_groups(ids, n_exp)
    )
    use_compact = unique_ids is not None
    n_launch = int(unique_ids.shape[0]) if use_compact else n_exp
    uniq = unique_ids if use_compact else order
    block_n, block_k_words, block_b, num_warps, num_stages = _grouped_launch_meta(
        nwords, tokens
    )
    grid = (triton.cdiv(n_out, block_n), n_launch)
    _ternary_expert_swiglu_grouped_kernel[grid](
        x_mat,
        packed_weight,
        row_alpha,
        order,
        offsets,
        uniq,
        y,
        n_out,
        nwords,
        x_mat.stride(0),
        x_mat.stride(1),
        stride_we,
        stride_wn,
        stride_ww,
        row_alpha.stride(0),
        row_alpha.stride(1),
        y.stride(0),
        y.stride(1),
        USE_COMPACT=use_compact,
        TOPK=topk,
        STREAM_W=True,
        BLOCK_N=block_n,
        BLOCK_K_WORDS=block_k_words,
        BLOCK_B=block_b,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.reshape(*id_shape, n_out)


def ternary_expert_down_sum(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    *,
    packed_kn: bool = False,
) -> torch.Tensor:
    """Down-proj GEMV, weighted sum over top-k, optional residual add. One launch.

    ``x`` and ``expert_ids`` / ``topk_weight`` share leading dims ``[..., topk]``.
    Returns ``[..., N]`` (top-k reduced).
    """
    if packed_weight.ndim != 3 or packed_weight.dtype != torch.uint32:
        raise ValueError("packed_weight must be uint32 [E, N, K/16] or [E, K/16, N].")
    if not x.is_cuda or not packed_weight.is_cuda or not expert_ids.is_cuda:
        raise RuntimeError("ternary_expert_down_sum requires CUDA tensors.")
    n_exp, n_out, nwords, stride_we, stride_wn, stride_ww = _expert_nk(
        packed_weight, packed_kn
    )
    k_in = nwords * _CODES_PER_WORD
    if x.shape[-1] != k_in:
        raise ValueError(f"x last dim {x.shape[-1]} != packed K ({k_in}).")
    if tuple(row_alpha.shape) != (n_exp, n_out):
        raise ValueError(f"row_alpha shape {tuple(row_alpha.shape)} != {(n_exp, n_out)}.")
    if expert_ids.shape != topk_weight.shape or x.shape[:-1] != tuple(expert_ids.shape):
        raise ValueError("x / expert_ids / topk_weight leading dims must match.")
    topk = int(expert_ids.shape[-1])
    tokens = int(expert_ids.numel() // topk)
    x_mat = x.reshape(tokens, topk, k_in)
    ids = expert_ids.reshape(tokens, topk)
    wts = topk_weight.reshape(tokens, topk).float()
    if ids.dtype not in (torch.int32, torch.int64):
        ids = ids.to(torch.int32)
    y = torch.empty(tokens, n_out, device=x.device, dtype=x.dtype)
    has_res = residual is not None
    if has_res:
        res = residual.reshape(tokens, n_out)
        if res.shape != y.shape:
            raise ValueError(f"residual shape {tuple(residual.shape)} != {tuple(y.shape)}.")
    else:
        res = y

    if tokens == 1:
        block_n, block_k_words, num_warps, num_stages = _expert_launch_meta(nwords)
    else:
        block_n, block_k_words, num_warps, num_stages = 32, 16, 4, 3
    grid = (triton.cdiv(n_out, block_n), tokens)
    _ternary_expert_down_sum_kernel[grid](
        x_mat,
        packed_weight,
        row_alpha,
        ids,
        wts,
        res,
        y,
        n_out,
        nwords,
        x_mat.stride(0),
        x_mat.stride(1),
        x_mat.stride(2),
        stride_we,
        stride_wn,
        stride_ww,
        row_alpha.stride(0),
        row_alpha.stride(1),
        ids.stride(0),
        ids.stride(1),
        wts.stride(0),
        wts.stride(1),
        res.stride(0),
        res.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_RESIDUAL=has_res,
        TOPK=topk,
        BLOCK_N=block_n,
        BLOCK_K_WORDS=block_k_words,
        STREAM_W=tokens == 1,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    leading = tuple(expert_ids.shape[:-1])
    return y.reshape(*leading, n_out)
