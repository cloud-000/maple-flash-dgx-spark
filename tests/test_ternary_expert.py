"""Fused expert GEMV vs per-expert 2-D packed GEMV. No dense unpack."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from maple_run.pack import ternarize, unpack_2bit

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for packed expert GEMV"
)


def _codes_to_weight(codes: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (codes.astype(np.float32) - 1.0) * alpha.reshape(*alpha.shape, 1)


def _to_cuda_experts(packed: np.ndarray, alpha: np.ndarray):
    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).to(
        device="cuda", dtype=torch.uint32
    )
    alpha_t = torch.from_numpy(np.ascontiguousarray(alpha)).to(
        device="cuda", dtype=torch.float32
    )
    return packed_t, alpha_t


@cuda
def test_expert_gemv_matches_per_expert_2d():
    from maple_run.kernels.ternary_expert import ternary_expert_gemv
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(0)
    n_exp, n_out, k_in = 4, 64, 128
    packed, alpha = ternarize(rng.standard_normal((n_exp, n_out, k_in)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_experts(packed, alpha)
    x = torch.from_numpy(rng.standard_normal((3, k_in)).astype(np.float32)).cuda()
    ids = torch.tensor([[0, 3], [1, 1], [2, 0]], device="cuda", dtype=torch.int32)

    y = ternary_expert_gemv(x, packed_t, alpha_t, ids)
    assert y.shape == (3, 2, n_out)

    ref = torch.empty_like(y)
    for t in range(3):
        for s in range(2):
            e = int(ids[t, s])
            ref[t, s] = ternary_gemv(x[t], packed_t[e], alpha_t[e])
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@cuda
def test_expert_gemv_per_slot_x_for_down_proj():
    from maple_run.kernels.ternary_expert import ternary_expert_gemv
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(1)
    n_exp, n_out, k_in, topk = 4, 128, 128, 2
    packed, alpha = ternarize(rng.standard_normal((n_exp, n_out, k_in)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_experts(packed, alpha)
    x = torch.from_numpy(rng.standard_normal((3, topk, k_in)).astype(np.float32)).cuda()
    ids = torch.tensor([[0, 2], [3, 1], [1, 0]], device="cuda", dtype=torch.int32)

    y = ternary_expert_gemv(x, packed_t, alpha_t, ids)
    ref = torch.empty_like(y)
    for t in range(3):
        for s in range(topk):
            e = int(ids[t, s])
            ref[t, s] = ternary_gemv(x[t, s], packed_t[e], alpha_t[e])
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@cuda
def test_expert_gemv_matches_dequantized_linear():
    from maple_run.kernels.ternary_expert import ternary_expert_gemv

    rng = np.random.default_rng(2)
    n_exp, n_out, k_in = 4, 32, 128
    packed, alpha = ternarize(rng.standard_normal((n_exp, n_out, k_in)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_experts(packed, alpha)
    x = torch.from_numpy(rng.standard_normal((2, k_in)).astype(np.float32)).cuda()
    ids = torch.tensor([[1, 0], [3, 2]], device="cuda")

    y = ternary_expert_gemv(x, packed_t, alpha_t, ids)
    codes = unpack_2bit(packed)
    w_hat = torch.from_numpy(_codes_to_weight(codes, alpha)).cuda()
    ref = torch.stack(
        [
            torch.stack(
                [torch.nn.functional.linear(x[t], w_hat[int(ids[t, s])]) for s in range(2)]
            )
            for t in range(2)
        ]
    )
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@cuda
def test_expert_kernel_does_not_unpack_or_loop_experts():
    import maple_run.kernels.ternary_expert as mod

    src = inspect.getsource(mod)
    assert "unpack_2bit" not in src
    assert "nn.functional.linear" not in src
    assert "torch.matmul" not in src
    assert "for e in range" not in src
    assert "for expert" not in src


@cuda
def test_expert_rejects_2d_and_non_uint32():
    from maple_run.kernels.ternary_expert import ternary_expert_gemv

    x = torch.randn(2, 128, device="cuda")
    ids = torch.zeros(2, 2, device="cuda", dtype=torch.int32)
    packed2d = torch.zeros(16, 8, device="cuda", dtype=torch.uint32)
    alpha = torch.ones(16, device="cuda")
    with pytest.raises(ValueError, match="3-D"):
        ternary_expert_gemv(x, packed2d, alpha, ids)
    fake = torch.randn(4, 16, 8, device="cuda", dtype=torch.bfloat16)
    alpha3 = torch.ones(4, 16, device="cuda")
    with pytest.raises(TypeError, match="uint32"):
        ternary_expert_gemv(x, fake, alpha3, ids)
