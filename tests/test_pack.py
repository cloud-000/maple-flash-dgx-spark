"""CPU packing tests against DeepGrove ternarize / _pack_2bit."""

import numpy as np
import pytest

from maple_run.pack import (
    DEFAULT_THRESHOLD_SCALE,
    dequantize_rtn,
    pack_2bit,
    pack_4bit,
    quantize_rtn,
    ternarize,
    unpack_2bit,
    unpack_4bit,
)


def _expected_ternary(weight: np.ndarray, threshold_scale: float) -> np.ndarray:
    w = weight.astype(np.float32, copy=False)
    aw = np.abs(w)
    threshold = threshold_scale * np.mean(aw, axis=-1, keepdims=True)
    mask = (aw > threshold).astype(np.float32)
    return np.sign(w) * mask


def test_pack_2bit_lsb_first_k128():
    """K=128 packs as 8 uint32 words, 16 codes/word, first code in bits 0–1."""
    codes = np.zeros((1, 128), dtype=np.uint32)
    codes[0, 0] = 2
    codes[0, 1] = 1
    codes[0, 15] = 2
    codes[0, 16] = 3  # first code of the second word

    packed = pack_2bit(codes)
    assert packed.dtype == np.uint32
    assert packed.shape == (1, 8)
    assert packed[0, 0] == np.uint32((2 << 0) | (1 << 2) | (2 << 30))
    assert packed[0, 1] == np.uint32(3)

    roundtrip = unpack_2bit(packed)
    np.testing.assert_array_equal(roundtrip, codes)


def test_pack_2bit_sixteen_codes_one_word():
    codes = (np.arange(16, dtype=np.uint32) % 3).reshape(1, 16)
    packed = pack_2bit(codes)
    expected = np.uint32(0)
    for i, c in enumerate(codes[0]):
        expected |= np.uint32(int(c) << (2 * i))
    assert packed.shape == (1, 1)
    assert packed[0, 0] == expected
    np.testing.assert_array_equal(unpack_2bit(packed), codes)


def test_ternarize_codes_match_sign_mask():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((32, 128)).astype(np.float32)
    packed, alpha = ternarize(weight)

    assert packed.dtype == np.uint32
    assert packed.shape == (32, 128 // 16)
    assert alpha.shape == (32,)
    assert alpha.dtype == weight.dtype

    codes = unpack_2bit(packed)
    assert codes.shape == weight.shape
    np.testing.assert_array_equal(np.unique(codes), np.array([0, 1, 2], dtype=codes.dtype))

    ternary = codes.astype(np.float32) - 1
    np.testing.assert_array_equal(ternary, _expected_ternary(weight, DEFAULT_THRESHOLD_SCALE))


def test_ternarize_row_alpha_is_mean_abs_of_survivors():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((8, 256)).astype(np.float32)
    packed, alpha = ternarize(weight)

    w = weight.astype(np.float32)
    aw = np.abs(w)
    threshold = DEFAULT_THRESHOLD_SCALE * np.mean(aw, axis=-1, keepdims=True)
    mask = aw > threshold
    expected = np.where(mask.any(axis=-1), np.sum(aw * mask, axis=-1) / np.sum(mask, axis=-1), 0.0)
    np.testing.assert_allclose(alpha, expected, rtol=1e-6, atol=1e-7)

    # Default storage is one alpha per output row, not K/128 group repeats.
    assert alpha.shape == (8,)
    assert packed.shape == (8, 256 // 16)


def test_ternarize_rejects_non_multiple_of_group_size():
    with pytest.raises(ValueError, match="multiple"):
        ternarize(np.ones((4, 64), dtype=np.float32))


def test_ternarize_all_zero_row():
    weight = np.zeros((2, 128), dtype=np.float32)
    packed, alpha = ternarize(weight)
    np.testing.assert_array_equal(unpack_2bit(packed), np.ones((2, 128), dtype=np.uint32))
    np.testing.assert_array_equal(alpha, np.zeros((2,), dtype=np.float32))


def test_pack_4bit_lsb_first():
    codes = np.zeros((1, 64), dtype=np.uint32)
    codes[0, 0] = 0xA
    codes[0, 1] = 0x3
    codes[0, 7] = 0xF
    packed = pack_4bit(codes)
    assert packed.dtype == np.uint32
    assert packed.shape == (1, 8)
    assert packed[0, 0] == np.uint32((0xA << 0) | (0x3 << 4) | (0xF << 28))
    np.testing.assert_array_equal(unpack_4bit(packed), codes)


def test_quantize_rtn_roundtrip_within_half_scale():
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((16, 64)).astype(np.float32)
    packed, scales, biases = quantize_rtn(weight)
    assert packed.dtype == np.uint32
    assert packed.shape == (16, 64 // 8)
    assert scales.shape == (16, 1)
    recon = dequantize_rtn(packed, scales, biases)
    codes = unpack_4bit(packed)
    assert int(codes.min()) >= 0
    assert int(codes.max()) <= 15
    bound = np.broadcast_to(0.5 * scales + 1e-5, weight.shape)
    assert np.all(np.abs(recon - weight) <= bound)


def test_ternarize_3d_expert_stack():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((4, 16, 128)).astype(np.float32)
    packed, alpha = ternarize(weight)
    assert packed.shape == (4, 16, 8)
    assert alpha.shape == (4, 16)
    ternary = unpack_2bit(packed).astype(np.float32) - 1
    np.testing.assert_array_equal(ternary, _expected_ternary(weight, DEFAULT_THRESHOLD_SCALE))
