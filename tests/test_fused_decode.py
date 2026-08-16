"""Fused RMSNorm, SwiGLU expert, and decode attention vs reference ops."""

from __future__ import annotations

import numpy as np
import pytest

from maple_run.pack import ternarize

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_rms_norm_matches_reference():
    from maple_run.kernels.fused_norm import rms_norm

    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((2, 3, 128)).astype(np.float32)).cuda()
    w = torch.from_numpy(rng.standard_normal(128).astype(np.float32)).cuda()
    y = rms_norm(x, w, 1e-6)
    var = x.float().pow(2).mean(-1, keepdim=True)
    ref = (w.float() * x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype)
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@cuda
def test_add_rms_norm_matches_reference():
    from maple_run.kernels.fused_norm import add_rms_norm, rms_norm

    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.standard_normal((4, 128)).astype(np.float32)).cuda()
    y = torch.from_numpy(rng.standard_normal((4, 128)).astype(np.float32)).cuda()
    w = torch.ones(128, device="cuda")
    residual, normed = add_rms_norm(x, y, w, 1e-6)
    torch.testing.assert_close(residual, x + y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(normed, rms_norm(x + y, w, 1e-6), rtol=1e-4, atol=1e-4)


@cuda
def test_gemv_fused_rms_matches_explicit_norm():
    from maple_run.kernels.fused_norm import rms_norm
    from maple_run.kernels.ternary_gemv import ternary_gemv

    rng = np.random.default_rng(2)
    packed, alpha = ternarize(rng.standard_normal((64, 128)).astype(np.float32))
    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32)
    alpha_t = torch.from_numpy(np.ascontiguousarray(alpha).reshape(-1)).cuda()
    x = torch.from_numpy(rng.standard_normal((1, 128)).astype(np.float32)).cuda()
    w = torch.from_numpy(rng.standard_normal(128).astype(np.float32)).cuda()
    y = ternary_gemv(x, packed_t, alpha_t, rms_weight=w, rms_eps=1e-6)
    y_ref = ternary_gemv(rms_norm(x, w, 1e-6), packed_t, alpha_t)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


@cuda
def test_expert_swiglu_and_down_sum_match_split_path():
    from maple_run.kernels.ternary_expert import (
        ternary_expert_down_sum,
        ternary_expert_gemv,
        ternary_expert_swiglu,
    )
    from maple_run.model import MLP_CLAMP

    rng = np.random.default_rng(3)
    n_tok, hidden, n_exp, topk, inter = 3, 128, 4, 2, 128
    up_p, up_a = ternarize(rng.standard_normal((n_exp, inter, hidden)).astype(np.float32))
    gate_p, gate_a = ternarize(rng.standard_normal((n_exp, inter, hidden)).astype(np.float32))
    down_p, down_a = ternarize(rng.standard_normal((n_exp, hidden, inter)).astype(np.float32))

    def to_t(packed, alpha):
        return (
            torch.from_numpy(np.ascontiguousarray(packed)).cuda().to(torch.uint32),
            torch.from_numpy(np.ascontiguousarray(alpha)).cuda(),
        )

    up_pt, up_at = to_t(up_p, up_a)
    gate_pt, gate_at = to_t(gate_p, gate_a)
    down_pt, down_at = to_t(down_p, down_a)
    up_gate_p = torch.cat([up_pt, gate_pt], dim=1)
    up_gate_a = torch.cat([up_at, gate_at], dim=1)
    x = torch.from_numpy(rng.standard_normal((n_tok, hidden)).astype(np.float32)).cuda()
    ids = torch.tensor([[0, 3], [1, 1], [2, 0]], device="cuda", dtype=torch.int32)
    wts = torch.tensor(
        [[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]], device="cuda", dtype=torch.float32
    )
    residual = torch.from_numpy(rng.standard_normal((n_tok, hidden)).astype(np.float32)).cuda()

    h = ternary_expert_swiglu(x, up_gate_p, up_gate_a, ids)
    y = ternary_expert_down_sum(h, down_pt, down_at, ids, wts, residual=residual)

    up_gate = ternary_expert_gemv(x, up_gate_p, up_gate_a, ids)
    up, gate = up_gate.split(inter, dim=-1)
    href = torch.nn.functional.silu(gate.clamp(max=MLP_CLAMP)) * up.clamp(
        min=-MLP_CLAMP, max=MLP_CLAMP
    )
    torch.testing.assert_close(h, href, rtol=1e-4, atol=1e-4)
    down = ternary_expert_gemv(href, down_pt, down_at, ids)
    ref = residual + (down.float() * wts.unsqueeze(-1)).sum(dim=-2)
    torch.testing.assert_close(y.float(), ref.float(), rtol=1e-4, atol=1e-4)


@cuda
def test_decode_attn_matches_sdpa():
    from maple_run.kernels.decode_attn import apply_qkv_decode, decode_gqa_attn
    from maple_run.kernels.fused_norm import rms_norm
    from maple_run.model import _apply_rotary_pos_emb

    bsz, n_q, n_kv, d, max_len = 1, 4, 2, 64, 128
    rotary_dim = 32
    q_w = torch.ones(d, device="cuda")
    k_w = torch.ones(d, device="cuda")
    qkv = torch.randn(1, (n_q + 2 * n_kv) * d, device="cuda", dtype=torch.float32)
    k_cache = torch.zeros(bsz, n_kv, max_len, d, device="cuda")
    v_cache = torch.zeros(bsz, n_kv, max_len, d, device="cuda")
    # Fill a few past tokens.
    k_cache[:, :, :7] = torch.randn(bsz, n_kv, 7, d, device="cuda")
    v_cache[:, :, :7] = torch.randn(bsz, n_kv, 7, d, device="cuda")
    seqlen = torch.tensor(7, device="cuda", dtype=torch.int64)
    pos = torch.tensor([[7]], device="cuda")
    # Fake cos/sin
    dummy = torch.linspace(0, 1, rotary_dim, device="cuda")
    cos = dummy.cos().view(1, 1, rotary_dim)
    sin = dummy.sin().view(1, 1, rotary_dim)

    q = apply_qkv_decode(
        qkv,
        q_w,
        k_w,
        cos,
        sin,
        k_cache,
        v_cache,
        seqlen,
        num_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=d,
        rotary_dim=rotary_dim,
        use_rope=True,
        use_qk_norm=True,
        eps=1e-6,
    )
    out = decode_gqa_attn(q, k_cache, v_cache, seqlen, scale=d**-0.5, window=32)

    q_s, k_s, v_s = qkv.split([n_q * d, n_kv * d, n_kv * d], dim=-1)
    q_s = q_s.view(1, 1, n_q, d).transpose(1, 2)
    k_s = k_s.view(1, 1, n_kv, d).transpose(1, 2)
    v_s = v_s.view(1, 1, n_kv, d).transpose(1, 2)
    q_s = rms_norm(q_s, q_w, 1e-6)
    k_s = rms_norm(k_s, k_w, 1e-6)
    q_s, k_s = _apply_rotary_pos_emb(q_s, k_s, cos, sin)
    k_ref = k_cache.clone()
    v_ref = v_cache.clone()
    k_ref[:, :, 7:8] = k_s
    v_ref[:, :, 7:8] = v_s
    torch.testing.assert_close(k_cache[:, :, 7], k_s[:, :, 0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_cache[:, :, 7], v_s[:, :, 0], rtol=1e-4, atol=1e-4)
    kv_k = k_ref[:, :, :8]
    kv_v = v_ref[:, :, :8]
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_s, kv_k, kv_v, scale=d**-0.5, enable_gqa=True
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    assert pos is not None
