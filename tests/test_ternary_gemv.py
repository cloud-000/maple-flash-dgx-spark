"""Packed GEMV vs dequantized F.linear. The kernel must not unpack W to dense bf16."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from maple_run.pack import pack_2bit, ternarize, unpack_2bit

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for packed GEMV"
)

REPO = Path(__file__).resolve().parents[1]
PACKED_CKPT = REPO / "checkpoints" / "maple-2bit"
Q_PROJ = "model.layers.0.self_attn.q_proj"


def _codes_to_weight(codes: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (codes.astype(np.float32) - 1.0) * alpha.reshape(-1, 1)


def _to_cuda_packed(packed: np.ndarray, alpha: np.ndarray):
    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).to(
        device="cuda", dtype=torch.uint32
    )
    alpha_t = torch.from_numpy(np.ascontiguousarray(alpha).reshape(-1)).to(
        device="cuda", dtype=torch.float32
    )
    return packed_t, alpha_t


@cuda
def test_matches_dequantized_linear_small_fp32():
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(0)
    weight = rng.standard_normal((64, 128)).astype(np.float32)
    packed, alpha = ternarize(weight)
    x_np = rng.standard_normal((1, 128)).astype(np.float32)

    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.from_numpy(x_np).to(device="cuda")
    y = ternary_gemv(x, packed_t, alpha_t)

    w_hat = torch.from_numpy(_codes_to_weight(unpack_2bit(packed), alpha)).to(
        device="cuda", dtype=torch.float32
    )
    y_ref = torch.nn.functional.linear(x, w_hat)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


@cuda
@pytest.mark.parametrize(
    "batch,n_out,k_in,dtype",
    [
        (1, 32, 128, torch.float32),
        (4, 64, 256, torch.float32),
        (1, 128, 128, torch.bfloat16),
        (2, 512, 2048, torch.bfloat16),
    ],
)
def test_matches_dequantized_linear_shapes(batch, n_out, k_in, dtype):
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(1)
    weight = rng.standard_normal((n_out, k_in)).astype(np.float32)
    packed, alpha = ternarize(weight)
    x_np = rng.standard_normal((batch, k_in)).astype(np.float32)

    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.from_numpy(x_np).to(device="cuda", dtype=dtype)
    y = ternary_gemv(x, packed_t, alpha_t)

    # Source of truth is dequantized F.linear in fp32. Comparing two bf16
    # matmuls on K=2048 disagrees by up to ~0.5 because tensor-core F.linear
    # accumulates in bf16; this kernel keeps a float32 acc.
    w_hat = torch.from_numpy(_codes_to_weight(unpack_2bit(packed), alpha)).to(
        device="cuda", dtype=torch.float32
    )
    y_ref = torch.nn.functional.linear(x.float(), w_hat)
    if dtype == torch.bfloat16:
        y_ref = y_ref.to(torch.bfloat16)
        rtol, atol = 1e-3, 1e-2
    else:
        rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)


@cuda
def test_1d_x_and_packed_linear_module():
    from maple_run.linear import PackedTernaryLinear

    rng = np.random.default_rng(2)
    weight = rng.standard_normal((48, 128)).astype(np.float32)
    packed, alpha = ternarize(weight)
    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.from_numpy(rng.standard_normal(128).astype(np.float32)).cuda()

    layer = PackedTernaryLinear(packed_t, alpha_t)
    y = layer(x)
    assert y.shape == (48,)

    w_hat = torch.from_numpy(_codes_to_weight(unpack_2bit(packed), alpha)).cuda()
    y_ref = torch.nn.functional.linear(x, w_hat)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


@cuda
def test_lsb_first_known_codes():
    """First code in bits 0–1: code 2 → +α, code 0 → −α, rest zero-weight (code 1)."""
    from maple_run.kernels.ternary_gemv import ternary_gemv

    codes = np.ones((2, 128), dtype=np.uint32)
    codes[0, 0] = 2
    codes[1, 1] = 0
    packed = pack_2bit(codes)
    alpha = np.array([0.5, 1.25], dtype=np.float32)
    x_np = np.arange(128, dtype=np.float32)
    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.from_numpy(x_np).cuda()
    y = ternary_gemv(x, packed_t, alpha_t)

    expected = torch.tensor(
        [0.5 * x_np[0], 1.25 * (-x_np[1])], device="cuda", dtype=torch.float32
    )
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


@cuda
def test_kernel_does_not_materialize_dense_bf16_w():
    import maple_run.kernels.ternary_gemv as gemv_mod
    from maple_run.kernels.ternary_gemv import ternary_gemv

    src = inspect.getsource(gemv_mod)
    assert "unpack_2bit" not in src
    assert "nn.functional.linear" not in src
    assert "torch.matmul" not in src
    assert "to(torch.bfloat16)" not in src
    assert "to(torch.float16)" not in src

    rng = np.random.default_rng(3)
    n_out, k_in = 2048, 2048
    packed, alpha = ternarize(rng.standard_normal((n_out, k_in)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.randn(1, k_in, device="cuda", dtype=torch.bfloat16)

    # Warmup so autotune / compile is not charged as a dense-W allocation.
    _ = ternary_gemv(x, packed_t, alpha_t)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    y = ternary_gemv(x, packed_t, alpha_t)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    dense_bf16 = n_out * k_in * 2
    assert extra < dense_bf16 / 4, (
        f"GEMV extra CUDA bytes {extra} looks like a dense unpack "
        f"(bf16 W would be {dense_bf16} bytes)"
    )
    assert y.shape == (1, n_out)
    assert packed_t.dtype == torch.uint32


@cuda
@pytest.mark.skipif(
    not (PACKED_CKPT / "model-00001-of-00003.safetensors").exists(),
    reason="packed checkpoint not present",
)
def test_real_q_proj_matches_dequantized_linear():
    from safetensors import safe_open

    from maple_run.kernels.ternary_gemv import ternary_gemv

    shard = PACKED_CKPT / "model-00001-of-00003.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as fh:
        packed = fh.get_tensor(f"{Q_PROJ}.weight")
        alpha = fh.get_tensor(f"{Q_PROJ}.row_alpha")

    packed_t = packed.to(device="cuda", dtype=torch.uint32)
    alpha_t = alpha.to(device="cuda", dtype=torch.float32)
    x = torch.randn(1, packed_t.shape[1] * 16, device="cuda", dtype=torch.bfloat16)
    y = ternary_gemv(x, packed_t, alpha_t)

    codes = unpack_2bit(packed.numpy())
    w_hat = torch.from_numpy(_codes_to_weight(codes, alpha.numpy())).to(
        device="cuda", dtype=torch.float32
    )
    y_ref = torch.nn.functional.linear(x.float(), w_hat).to(torch.bfloat16)
    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-2)


@cuda
def test_rejects_non_uint32_packed_weight():
    from maple_run.kernels.ternary_gemv import ternary_gemv

    x = torch.randn(128, device="cuda")
    fake = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    alpha = torch.ones(16, device="cuda")
    with pytest.raises(TypeError, match="uint32"):
        ternary_gemv(x, fake, alpha)


@cuda
def test_packed_kn_matches_nk_layout():
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(4)
    packed, alpha = ternarize(rng.standard_normal((96, 256)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.from_numpy(rng.standard_normal((1, 256)).astype(np.float32)).cuda()
    y_nk = ternary_gemv(x, packed_t, alpha_t)
    y_kn = ternary_gemv(x, packed_t.t().contiguous(), alpha_t, packed_kn=True)
    torch.testing.assert_close(y_kn, y_nk, rtol=1e-4, atol=1e-4)


def _batch1_rows(fn, x, *args, **kwargs):
    """Reference: the batch-1 kernel run once per row, stacked."""
    return torch.cat([fn(x[i : i + 1], *args, **kwargs) for i in range(x.shape[0])])


@cuda
@pytest.mark.parametrize("batch", [2, 3, 8, 16, 17, 32])
def test_batched_matches_batch1_rows(batch):
    """batch>1 takes _ternary_gemm_kernel; it must be the batch-1 kernel's answer.

    Both accumulate in float32 over the same exactly-representable products, so
    they differ only in summation order -- well inside one bfloat16 ulp of the
    shared output dtype.
    """
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(11)
    weight = rng.standard_normal((512, 2048)).astype(np.float32)
    packed, alpha = ternarize(weight)
    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)

    y = ternary_gemv(x, packed_t, alpha_t)
    y_ref = _batch1_rows(ternary_gemv, x, packed_t, alpha_t)
    assert y.shape == (batch, 512)
    torch.testing.assert_close(y, y_ref, rtol=8e-3, atol=8e-3)


@cuda
@pytest.mark.parametrize("batch", [2, 8, 32])
def test_batched_fused_rmsnorm_matches_batch1_rows(batch):
    """The folded RMSNorm keeps float32 precision across the batched dot.

    x*rms_w is a float32 product; feeding only its bfloat16 head to the tensor
    core would drop ~8 mantissa bits, so the kernel also dots the residual.
    """
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(12)
    weight = rng.standard_normal((512, 2048)).astype(np.float32)
    packed, alpha = ternarize(weight)
    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    rms_w = torch.rand(2048, device="cuda", dtype=torch.float32) + 0.5

    y = ternary_gemv(x, packed_t, alpha_t, rms_weight=rms_w)
    y_ref = _batch1_rows(ternary_gemv, x, packed_t, alpha_t, rms_weight=rms_w)
    torch.testing.assert_close(y, y_ref, rtol=8e-3, atol=8e-3)

    w_hat = torch.from_numpy(_codes_to_weight(unpack_2bit(packed), alpha)).cuda()
    xf = x.float()
    x_norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * rms_w
    torch.testing.assert_close(
        y.float(), x_norm @ w_hat.T, rtol=1e-2, atol=5e-2
    )


@cuda
@pytest.mark.parametrize("batch", [2, 8])
def test_batched_fp32_matches_batch1_rows(batch):
    """float32 activations must not be demoted to tf32 by the batched dot."""
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(13)
    weight = rng.standard_normal((128, 512)).astype(np.float32)
    packed, alpha = ternarize(weight)
    packed_t, alpha_t = _to_cuda_packed(packed, alpha)
    x = torch.from_numpy(
        rng.standard_normal((batch, 512)).astype(np.float32)
    ).cuda()

    y = ternary_gemv(x, packed_t, alpha_t)
    w_hat = torch.from_numpy(_codes_to_weight(unpack_2bit(packed), alpha)).cuda()
    torch.testing.assert_close(
        y, torch.nn.functional.linear(x, w_hat), rtol=1e-4, atol=1e-4
    )


@cuda
def test_batched_reads_each_code_tile_once_per_cta():
    """The point of the batched kernel: weight traffic stops scaling with B.

    Batch 32 through the batch-1 kernel re-reads the codes 32 times; through
    _ternary_gemm_kernel it reads them once per CTA, so wall time must stay far
    below 32x the batch-1 time on the same weights.
    """
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(14)
    packed, alpha = ternarize(rng.standard_normal((3072, 2048)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_packed(packed, alpha)

    def timed(batch):
        x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            ternary_gemv(x, packed_t, alpha_t)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(50):
            ternary_gemv(x, packed_t, alpha_t)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 50

    t1, t32 = timed(1), timed(32)
    assert t32 < 8 * t1, f"batch 32 took {t32/t1:.1f}x batch 1; weights not shared"
