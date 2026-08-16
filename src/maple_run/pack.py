# Copyright © 2026 DeepGrove AI.
"""CPU reference for Maple's row-wise ternary pack.

Port of ``ternarize`` / ``_pack_2bit`` from ``docs/sources/mlx_lm_ternary.py``.
Keep this module free of MLX and of GPU kernels so packing can be tested on CPU.
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD_SCALE = 0.7
GROUP_SIZE = 128
HEAD_GROUP_SIZE = 64
HEAD_BITS = 4

# 16 two-bit codes per uint32, LSB first: bit positions 0, 2, ..., 30.
_SHIFTS_2 = np.arange(0, 32, 2, dtype=np.uint32)
# 8 four-bit codes per uint32, LSB first: bit positions 0, 4, ..., 28.
_SHIFTS_4 = np.arange(0, 32, 4, dtype=np.uint32)


def pack_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack 2-bit codes along the last axis, 16 per uint32, LSB first.

    This matches the packing ``mx.quantize(..., bits=2, mode="affine")``
    produces in DeepGrove's converter.
    """
    *lead, k = codes.shape
    if k % 16:
        raise ValueError(
            f"2-bit packing requires the last axis to be a multiple of 16; got {k}."
        )
    codes_u32 = codes.astype(np.uint32, copy=False).reshape(*lead, k // 16, 16)
    # Sum of non-overlapping shifted fields equals bitwise OR; accumulate in
    # uint64 so the numpy reduction cannot wrap, then narrow to uint32.
    packed = np.sum(codes_u32 << _SHIFTS_2, axis=-1, dtype=np.uint64)
    return packed.astype(np.uint32)


def unpack_2bit(packed: np.ndarray) -> np.ndarray:
    """Inverse of ``pack_2bit``: last axis ``nwords`` → ``nwords * 16`` codes."""
    packed_u32 = packed.astype(np.uint32, copy=False)
    codes = (packed_u32[..., np.newaxis] >> _SHIFTS_2) & np.uint32(3)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * 16)


def pack_4bit(codes: np.ndarray) -> np.ndarray:
    """Pack 4-bit codes along the last axis, 8 per uint32, LSB first."""
    *lead, k = codes.shape
    if k % 8:
        raise ValueError(
            f"4-bit packing requires the last axis to be a multiple of 8; got {k}."
        )
    codes_u32 = codes.astype(np.uint32, copy=False).reshape(*lead, k // 8, 8)
    packed = np.sum(codes_u32 << _SHIFTS_4, axis=-1, dtype=np.uint64)
    return packed.astype(np.uint32)


def unpack_4bit(packed: np.ndarray) -> np.ndarray:
    """Inverse of ``pack_4bit``: last axis ``nwords`` → ``nwords * 8`` codes."""
    packed_u32 = packed.astype(np.uint32, copy=False)
    codes = (packed_u32[..., np.newaxis] >> _SHIFTS_4) & np.uint32(0xF)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def quantize_rtn(
    weight: np.ndarray,
    group_size: int = HEAD_GROUP_SIZE,
    bits: int = HEAD_BITS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine round-to-nearest along the last axis.

    Used for ``lm_head`` / embeddings (DeepGrove: group 64, 4-bit). Returns
    ``(packed_uint32, scales, biases)`` where dequant is ``code * scale + bias``.
    Arithmetic is float32; scales and biases are stored float32.
    """
    if bits != HEAD_BITS:
        raise ValueError(f"Only {HEAD_BITS}-bit RTN is implemented; got {bits}.")
    k = weight.shape[-1]
    if k % group_size:
        raise ValueError(
            f"RTN conversion requires the last axis to be a multiple of "
            f"{group_size}; got {k}."
        )
    n_bins = np.float32((1 << bits) - 1)
    w = weight.astype(np.float32, copy=False)
    lead = w.shape[:-1]
    grouped = w.reshape(*lead, k // group_size, group_size)
    w_min = grouped.min(axis=-1, keepdims=True)
    w_max = grouped.max(axis=-1, keepdims=True)
    scale = (w_max - w_min) / n_bins
    zero = scale == 0
    safe_scale = np.where(zero, np.float32(1.0), scale)
    q = np.clip(np.rint((grouped - w_min) / safe_scale), 0, n_bins)
    q = np.where(zero, 0, q)
    packed = pack_4bit(q.reshape(*lead, k))
    scales = np.squeeze(scale, axis=-1).astype(np.float32, copy=False)
    biases = np.squeeze(w_min, axis=-1).astype(np.float32, copy=False)
    return packed, scales, biases


def dequantize_rtn(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int = HEAD_GROUP_SIZE,
    bits: int = HEAD_BITS,
) -> np.ndarray:
    """CPU inverse of ``quantize_rtn``."""
    if bits != HEAD_BITS:
        raise ValueError(f"Only {HEAD_BITS}-bit RTN is implemented; got {bits}.")
    codes = unpack_4bit(packed).astype(np.float32)
    k = codes.shape[-1]
    grouped = codes.reshape(*codes.shape[:-1], k // group_size, group_size)
    recon = grouped * scales[..., np.newaxis] + biases[..., np.newaxis]
    return recon.reshape(*codes.shape)


def ternarize(
    weight: np.ndarray,
    threshold_scale: float = DEFAULT_THRESHOLD_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Ternarize ``[..., N, K]`` weights row-wise; return ``(packed, row_alpha)``.

    ``K`` must be a multiple of ``GROUP_SIZE`` (128). Arithmetic runs in
    float32 regardless of ``weight.dtype``; only the final alpha is cast back
    to the weight dtype. Codes are ``{−1, 0, +1} + 1`` packed 16 per uint32.

    Default storage is one ``row_alpha`` per output row (not the repeated
    per-group scales used by generic MLX quantized readers).
    """
    if weight.shape[-1] % GROUP_SIZE:
        raise ValueError(
            f"Ternary conversion requires the reduction dim to be a multiple "
            f"of {GROUP_SIZE}; got {weight.shape[-1]}."
        )
    w = weight.astype(np.float32, copy=False)
    aw = np.abs(w)
    threshold = threshold_scale * np.mean(aw, axis=-1, keepdims=True)
    mask = (aw > threshold).astype(np.float32)
    alpha_num = np.sum(aw * mask, axis=-1, keepdims=True)
    alpha_den = np.maximum(np.sum(mask, axis=-1, keepdims=True), 1)
    alpha = (alpha_num / alpha_den).astype(weight.dtype, copy=False)
    row_alpha = np.squeeze(alpha, axis=-1)

    ternary = np.sign(w) * mask
    packed = pack_2bit(ternary + 1)
    return packed, row_alpha
