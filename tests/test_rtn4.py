"""4-bit RTN embedding + GEMV vs CPU dequant."""

from __future__ import annotations

import numpy as np
import pytest

from maple_run.pack import dequantize_rtn, quantize_rtn

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_rtn4_embedding_gathers_and_dequants_rows():
    from maple_run.kernels.rtn4 import rtn4_embedding

    rng = np.random.default_rng(0)
    weight = rng.standard_normal((32, 128)).astype(np.float32)
    packed, scales, biases = quantize_rtn(weight)
    recon = dequantize_rtn(packed, scales, biases)

    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32)
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).cuda()
    biases_t = torch.from_numpy(np.ascontiguousarray(biases)).cuda()
    ids = torch.tensor([[3, 7, 0], [1, 1, 31]], device="cuda")
    y = rtn4_embedding(ids, packed_t, scales_t, biases_t)
    ref = torch.from_numpy(recon[ids.cpu().numpy()]).cuda()
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


@cuda
def test_rtn4_gemv_matches_dequantized_linear():
    from maple_run.kernels.rtn4 import rtn4_gemv

    rng = np.random.default_rng(1)
    weight = rng.standard_normal((64, 128)).astype(np.float32)
    packed, scales, biases = quantize_rtn(weight)
    recon = dequantize_rtn(packed, scales, biases)

    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32)
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).cuda()
    biases_t = torch.from_numpy(np.ascontiguousarray(biases)).cuda()
    x = torch.from_numpy(rng.standard_normal((2, 128)).astype(np.float32)).cuda()
    y = rtn4_gemv(x, packed_t, scales_t, biases_t)

    w_hat = torch.from_numpy(recon).cuda()
    y_ref = torch.nn.functional.linear(x, w_hat)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


@cuda
def test_rtn4_gemv_packed_kn_matches_nk():
    from maple_run.kernels.rtn4 import rtn4_gemv

    rng = np.random.default_rng(2)
    weight = rng.standard_normal((80, 128)).astype(np.float32)
    packed, scales, biases = quantize_rtn(weight)
    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32)
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).cuda()
    biases_t = torch.from_numpy(np.ascontiguousarray(biases)).cuda()
    x = torch.from_numpy(rng.standard_normal((1, 128)).astype(np.float32)).cuda()
    y_nk = rtn4_gemv(x, packed_t, scales_t, biases_t)
    y_kn = rtn4_gemv(
        x,
        packed_t.t().contiguous(),
        scales_t.t().contiguous(),
        biases_t.t().contiguous(),
        packed_kn=True,
    )
    torch.testing.assert_close(y_kn, y_nk, rtol=1e-4, atol=1e-4)


@cuda
def test_rtn4_gemv_fat_k_tile_matches_dequantized_linear():
    """Decode lm_head tiles 4 RTN groups (BLOCK_K_WORDS=32) when K>=256."""
    from maple_run.kernels.rtn4 import rtn4_gemv

    rng = np.random.default_rng(3)
    weight = rng.standard_normal((64, 256)).astype(np.float32)
    packed, scales, biases = quantize_rtn(weight)
    recon = dequantize_rtn(packed, scales, biases)

    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32)
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).cuda()
    biases_t = torch.from_numpy(np.ascontiguousarray(biases)).cuda()
    x = torch.from_numpy(rng.standard_normal((1, 256)).astype(np.float32)).cuda()
    y = rtn4_gemv(x, packed_t.t().contiguous(), scales_t.t().contiguous(),
                  biases_t.t().contiguous(), packed_kn=True)
    w_hat = torch.from_numpy(recon).cuda()
    y_ref = torch.nn.functional.linear(x, w_hat)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


@cuda
def test_rtn4_gemv_rejects_dense_weight():
    from maple_run.kernels.rtn4 import rtn4_gemv

    x = torch.randn(128, device="cuda")
    fake = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    scales = torch.ones(16, 2, device="cuda")
    with pytest.raises(TypeError, match="uint32"):
        rtn4_gemv(x, fake, scales, scales)
