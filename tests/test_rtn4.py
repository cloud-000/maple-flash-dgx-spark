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
def test_rtn4_indexed_gemv_matches_gathered_rows():
    from maple_run.kernels.rtn4 import rtn4_gemv, rtn4_indexed_gemv

    rng = np.random.default_rng(4)
    n_clusters, cluster_size, hidden = 4, 8, 128
    n_probes = 2
    weight = rng.standard_normal((n_clusters * cluster_size, hidden)).astype(np.float32)
    packed, scales, biases = quantize_rtn(weight)
    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32)
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).cuda()
    biases_t = torch.from_numpy(np.ascontiguousarray(biases)).cuda()
    head_w = packed_t.reshape(n_clusters, cluster_size, -1).contiguous()
    head_s = scales_t.reshape(n_clusters, cluster_size, -1).contiguous()
    head_b = biases_t.reshape(n_clusters, cluster_size, -1).contiguous()
    cluster_ids = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    x = torch.from_numpy(rng.standard_normal((2, hidden)).astype(np.float32)).cuda()
    y = rtn4_indexed_gemv(x, head_w, head_s, head_b, cluster_ids)
    order = cluster_ids.long()[:, None] * cluster_size + torch.arange(
        cluster_size, device="cuda"
    )
    order = order.reshape(-1)
    y_ref = rtn4_gemv(
        x,
        packed_t.view(torch.int32).index_select(0, order).view(torch.uint32),
        scales_t.index_select(0, order),
        biases_t.index_select(0, order),
    )
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)
    assert y.shape == (2, n_probes * cluster_size)


@cuda
def test_rtn4_gemv_rejects_dense_weight():
    from maple_run.kernels.rtn4 import rtn4_gemv

    x = torch.randn(128, device="cuda")
    fake = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    scales = torch.ones(16, 2, device="cuda")
    with pytest.raises(TypeError, match="uint32"):
        rtn4_gemv(x, fake, scales, scales)


def _rtn4_fixture(n_out, k_in, seed):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((n_out, k_in)).astype(np.float32)
    packed, scales, biases = quantize_rtn(weight)
    return (
        torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32),
        torch.from_numpy(np.ascontiguousarray(scales)).cuda(),
        torch.from_numpy(np.ascontiguousarray(biases)).cuda(),
        torch.from_numpy(dequantize_rtn(packed, scales, biases)).cuda(),
    )


@cuda
@pytest.mark.parametrize("batch", [2, 3, 8, 16, 17, 32])
def test_rtn4_batched_matches_batch1_rows(batch):
    """batch>1 takes _rtn4_gemm_kernel and must match the batch-1 kernel.

    It keeps the affine dequant factored -- s*dot(q,x) + b*sum(x) -- so only the
    raw 4-bit codes reach the tensor core, and those are exact in bfloat16.
    """
    from maple_run.kernels.rtn4 import rtn4_gemv

    packed_t, scales_t, biases_t, _ = _rtn4_fixture(512, 2048, 21)
    x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)

    y = rtn4_gemv(x, packed_t, scales_t, biases_t)
    y_ref = torch.cat(
        [rtn4_gemv(x[i : i + 1], packed_t, scales_t, biases_t) for i in range(batch)]
    )
    assert y.shape == (batch, 512)
    torch.testing.assert_close(y, y_ref, rtol=8e-3, atol=8e-3)


@cuda
@pytest.mark.parametrize("batch", [2, 8, 32])
def test_rtn4_batched_matches_dequantized_linear(batch):
    from maple_run.kernels.rtn4 import rtn4_gemv

    packed_t, scales_t, biases_t, recon = _rtn4_fixture(256, 1024, 22)
    x = torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)
    y = rtn4_gemv(x, packed_t, scales_t, biases_t).float()
    torch.testing.assert_close(y, x.float() @ recon.float().T, rtol=1e-2, atol=2e-2)


@cuda
@pytest.mark.parametrize("batch", [2, 8])
def test_rtn4_batched_packed_kn_matches_nk(batch):
    from maple_run.kernels.rtn4 import rtn4_gemv

    packed_t, scales_t, biases_t, _ = _rtn4_fixture(128, 512, 23)
    x = torch.randn(batch, 512, device="cuda", dtype=torch.bfloat16)
    y_nk = rtn4_gemv(x, packed_t, scales_t, biases_t)
    y_kn = rtn4_gemv(
        x,
        packed_t.t().contiguous(),
        scales_t.t().contiguous(),
        biases_t.t().contiguous(),
        packed_kn=True,
    )
    torch.testing.assert_close(y_kn, y_nk, rtol=8e-3, atol=8e-3)


@cuda
@pytest.mark.parametrize("batch", [2, 8])
def test_rtn4_batched_fp32_matches_dequantized_linear(batch):
    """float32 activations keep an ieee dot rather than dropping to tf32."""
    from maple_run.kernels.rtn4 import rtn4_gemv

    rng = np.random.default_rng(24)
    packed_t, scales_t, biases_t, recon = _rtn4_fixture(128, 512, 24)
    x = torch.from_numpy(rng.standard_normal((batch, 512)).astype(np.float32)).cuda()
    y = rtn4_gemv(x, packed_t, scales_t, biases_t)
    torch.testing.assert_close(
        y, torch.nn.functional.linear(x, recon), rtol=1e-4, atol=1e-4
    )
