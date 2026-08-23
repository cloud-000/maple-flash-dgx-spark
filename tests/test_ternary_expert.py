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
def test_expert_gemv_single_token_matches_2d():
    from maple_run.kernels.ternary_expert import ternary_expert_gemv
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(4)
    n_exp, n_out, k_in = 4, 64, 128
    packed, alpha = ternarize(rng.standard_normal((n_exp, n_out, k_in)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_experts(packed, alpha)
    x = torch.from_numpy(rng.standard_normal((1, k_in)).astype(np.float32)).cuda()
    ids = torch.tensor([[0, 3, 1, 2]], device="cuda", dtype=torch.int32)

    y = ternary_expert_gemv(x, packed_t, alpha_t, ids)
    ref = torch.empty_like(y)
    for s in range(ids.shape[-1]):
        ref[0, s] = ternary_gemv(x[0], packed_t[int(ids[0, s])], alpha_t[int(ids[0, s])])
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@cuda
def test_expert_gemv_groups_duplicate_ids():
    """Several tokens hitting the same expert must match per-slot GEMV."""
    from maple_run.kernels.ternary_expert import ternary_expert_gemv
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(5)
    n_exp, n_out, k_in, topk, n_tok = 4, 64, 128, 2, 8
    packed, alpha = ternarize(rng.standard_normal((n_exp, n_out, k_in)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_experts(packed, alpha)
    x = torch.from_numpy(rng.standard_normal((n_tok, k_in)).astype(np.float32)).cuda()
    # Force reuse: every token picks experts 0 and 1.
    ids = torch.tensor([[0, 1]] * n_tok, device="cuda", dtype=torch.int32)

    y = ternary_expert_gemv(x, packed_t, alpha_t, ids, grouped=True)
    ref = torch.empty_like(y)
    for t in range(n_tok):
        for s in range(topk):
            ref[t, s] = ternary_gemv(x[t], packed_t[int(ids[t, s])], alpha_t[int(ids[t, s])])
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@cuda
def test_expert_gemv_grouped_bf16_and_slot_groups_reuse():
    from maple_run.kernels.ternary_expert import expert_slot_groups, ternary_expert_gemv
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(6)
    n_exp, n_out, k_in, n_tok = 8, 128, 128, 16
    packed, alpha = ternarize(rng.standard_normal((n_exp, n_out, k_in)).astype(np.float32))
    packed_t, alpha_t = _to_cuda_experts(packed, alpha)
    x = torch.from_numpy(rng.standard_normal((n_tok, k_in)).astype(np.float32)).cuda().to(
        torch.bfloat16
    )
    ids = torch.randint(0, n_exp, (n_tok, 4), device="cuda", dtype=torch.int32)
    groups = expert_slot_groups(ids, n_exp)
    y = ternary_expert_gemv(x, packed_t, alpha_t, ids, slot_groups=groups, grouped=True)
    ref = torch.empty_like(y)
    for t in range(n_tok):
        for s in range(4):
            ref[t, s] = ternary_gemv(
                x[t], packed_t[int(ids[t, s])], alpha_t[int(ids[t, s])]
            )
    torch.testing.assert_close(y.float(), ref.float(), rtol=2e-2, atol=2e-2)


@cuda
def test_expert_slot_groups_offsets_cover_every_slot():
    from maple_run.kernels.ternary_expert import expert_slot_groups

    ids = torch.tensor(
        [[0, 3], [1, 1], [2, 0], [3, 3]], device="cuda", dtype=torch.int32
    )
    order, offsets, unique_ids = expert_slot_groups(ids, n_exp=4)
    assert offsets[0] == 0
    assert int(offsets[-1]) == ids.numel()
    counts = torch.bincount(ids.reshape(-1).long(), minlength=4)
    assert torch.equal(offsets[1:] - offsets[:-1], counts.to(offsets.dtype))
    # slots < n_exp is false here (8 >= 4), so the grid uses program_id as id.
    assert unique_ids is None
    restored = ids.reshape(-1)[order.long()].sort().values
    assert torch.equal(restored, ids.reshape(-1).sort().values)

    compact_ids = torch.tensor([[0, 5], [2, 5]], device="cuda", dtype=torch.int32)
    _order, _offsets, unique = expert_slot_groups(compact_ids, n_exp=16)
    assert unique is not None
    assert unique.shape[0] == compact_ids.numel()
    assert int((unique >= 0).sum()) == 3


@cuda
def test_expert_swiglu_grouped_matches_split_path():
    from maple_run.kernels.ternary_expert import ternary_expert_gemv, ternary_expert_swiglu
    from maple_run.model import MLP_CLAMP

    rng = np.random.default_rng(7)
    n_tok, hidden, n_exp, topk, inter = 12, 128, 6, 4, 64
    up_p, up_a = ternarize(rng.standard_normal((n_exp, inter, hidden)).astype(np.float32))
    gate_p, gate_a = ternarize(
        rng.standard_normal((n_exp, inter, hidden)).astype(np.float32)
    )

    def to_t(packed, alpha):
        return (
            torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32),
            torch.from_numpy(np.ascontiguousarray(alpha)).cuda(),
        )

    up_pt, up_at = to_t(up_p, up_a)
    gate_pt, gate_at = to_t(gate_p, gate_a)
    up_gate_p = torch.cat([up_pt, gate_pt], dim=1)
    up_gate_a = torch.cat([up_at, gate_at], dim=1)
    x = torch.from_numpy(rng.standard_normal((n_tok, hidden)).astype(np.float32)).cuda()
    ids = torch.randint(0, n_exp, (n_tok, topk), device="cuda", dtype=torch.int32)
    h = ternary_expert_swiglu(x, up_gate_p, up_gate_a, ids, grouped=True)
    up_gate = ternary_expert_gemv(x, up_gate_p, up_gate_a, ids, grouped=True)
    up, gate = up_gate.split(inter, dim=-1)
    href = torch.nn.functional.silu(gate.clamp(max=MLP_CLAMP)) * up.clamp(
        min=-MLP_CLAMP, max=MLP_CLAMP
    )
    torch.testing.assert_close(h, href, rtol=1e-4, atol=1e-4)


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
