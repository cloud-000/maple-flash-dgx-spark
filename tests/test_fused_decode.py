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


@cuda
def test_decode_attn_does_not_need_padded_tail():
    """GQA must match SDPA when the cache is much longer than seqlen."""
    from maple_run.kernels.decode_attn import apply_qkv_decode, decode_gqa_attn
    from maple_run.kernels.fused_norm import rms_norm
    from maple_run.model import _apply_rotary_pos_emb

    bsz, n_q, n_kv, d, max_len = 1, 4, 2, 64, 1024
    rotary_dim = 32
    q_w = torch.ones(d, device="cuda")
    k_w = torch.ones(d, device="cuda")
    qkv = torch.randn(1, (n_q + 2 * n_kv) * d, device="cuda", dtype=torch.float32)
    k_cache = torch.zeros(bsz, n_kv, max_len, d, device="cuda")
    v_cache = torch.zeros(bsz, n_kv, max_len, d, device="cuda")
    k_cache[:, :, :7] = torch.randn(bsz, n_kv, 7, d, device="cuda")
    v_cache[:, :, :7] = torch.randn(bsz, n_kv, 7, d, device="cuda")
    seqlen = torch.tensor(7, device="cuda", dtype=torch.int64)
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
    out = decode_gqa_attn(q, k_cache, v_cache, seqlen, scale=d**-0.5, window=None)

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
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_s, k_ref[:, :, :8], v_ref[:, :, :8], scale=d**-0.5, enable_gqa=True
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)



def _decode_attn_fixture(bsz, lens, *, n_q=8, n_kv=2, d=64, max_len=1024, rotary_dim=32):
    """Random qkv + a cache where row ``b`` already holds ``lens[b]`` tokens."""
    torch.manual_seed(7)
    q_w = torch.randn(d, device="cuda")
    k_w = torch.randn(d, device="cuda")
    qkv = torch.randn(bsz, (n_q + 2 * n_kv) * d, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros(bsz, n_kv, max_len, d, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros(bsz, n_kv, max_len, d, device="cuda", dtype=torch.bfloat16)
    for b, n in enumerate(lens):
        k_cache[b, :, :n] = torch.randn(n_kv, n, d, device="cuda", dtype=torch.bfloat16)
        v_cache[b, :, :n] = torch.randn(n_kv, n, d, device="cuda", dtype=torch.bfloat16)
    seqlen = torch.tensor(lens, device="cuda", dtype=torch.int64)
    cos = torch.randn(bsz, 1, rotary_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(bsz, 1, rotary_dim, device="cuda", dtype=torch.bfloat16)
    return q_w, k_w, qkv, k_cache, v_cache, seqlen, cos, sin


# Lengths that straddle a BLOCK_T (64) boundary and the split-K threshold, so a
# batch cannot accidentally pass by every row taking the same code path.
_RAGGED = [1, 63, 64, 65, 300, 511, 1000]


@cuda
@pytest.mark.parametrize("window", [None, 512])
def test_batched_decode_attn_matches_per_row_batch1(window):
    """Each row must be bit-identical to the same row run alone at B=1.

    The batch is a grid dimension here: rows share no arithmetic, so anything
    less than exact equality would mean a stride or a ``seqlen[b]`` is wrong.
    """
    from maple_run.kernels.decode_attn import apply_qkv_decode, decode_gqa_attn

    lens = _RAGGED
    bsz, n_q, n_kv, d, rot = len(lens), 8, 2, 64, 32
    q_w, k_w, qkv, k0, v0, seqlen, cos, sin = _decode_attn_fixture(bsz, lens)
    kb, vb = k0.clone(), v0.clone()
    kw = dict(
        num_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=d,
        rotary_dim=rot,
        use_rope=True,
        use_qk_norm=True,
        eps=1e-6,
    )
    q = apply_qkv_decode(qkv, q_w, k_w, cos, sin, kb, vb, seqlen, **kw)
    out = decode_gqa_attn(q, kb, vb, seqlen, scale=d**-0.5, window=window)

    for b in range(bsz):
        k1, v1 = k0[b : b + 1].clone(), v0[b : b + 1].clone()
        q1 = apply_qkv_decode(
            qkv[b : b + 1],
            q_w,
            k_w,
            cos[b : b + 1],
            sin[b : b + 1],
            k1,
            v1,
            seqlen[b : b + 1],
            **kw,
        )
        o1 = decode_gqa_attn(q1, k1, v1, seqlen[b : b + 1], scale=d**-0.5, window=window)
        assert torch.equal(q[b : b + 1], q1), f"row {b} q"
        assert torch.equal(kb[b : b + 1], k1), f"row {b} K write"
        assert torch.equal(vb[b : b + 1], v1), f"row {b} V write"
        assert torch.equal(out[b : b + 1], o1), f"row {b} attention out"


@cuda
def test_batched_decode_attn_matches_sdpa_per_row():
    from maple_run.kernels.decode_attn import apply_qkv_decode, decode_gqa_attn

    lens = _RAGGED
    bsz, n_q, n_kv, d, rot = len(lens), 8, 2, 64, 32
    q_w, k_w, qkv, k0, v0, seqlen, cos, sin = _decode_attn_fixture(bsz, lens)
    q = apply_qkv_decode(
        qkv,
        q_w,
        k_w,
        cos,
        sin,
        k0,
        v0,
        seqlen,
        num_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=d,
        rotary_dim=rot,
        use_rope=True,
        use_qk_norm=True,
        eps=1e-6,
    )
    out = decode_gqa_attn(q, k0, v0, seqlen, scale=d**-0.5, window=None)
    for b, n in enumerate(lens):
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[b : b + 1].float(),
            k0[b : b + 1, :, : n + 1].float(),
            v0[b : b + 1, :, : n + 1].float(),
            scale=d**-0.5,
            enable_gqa=True,
        )
        torch.testing.assert_close(
            out[b : b + 1].float(), ref, rtol=2e-2, atol=2e-2
        )


@cuda
def test_batched_decode_attn_ignores_other_rows_cache():
    """Row ``b``'s output must not move when another row's cache changes."""
    from maple_run.kernels.decode_attn import apply_qkv_decode, decode_gqa_attn

    lens = [5, 200, 900]
    bsz, n_q, n_kv, d, rot = len(lens), 8, 2, 64, 32
    q_w, k_w, qkv, k0, v0, seqlen, cos, sin = _decode_attn_fixture(bsz, lens)
    kw = dict(
        num_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=d,
        rotary_dim=rot,
        use_rope=True,
        use_qk_norm=True,
        eps=1e-6,
    )
    ka, va = k0.clone(), v0.clone()
    q = apply_qkv_decode(qkv, q_w, k_w, cos, sin, ka, va, seqlen, **kw)
    out_a = decode_gqa_attn(q, ka, va, seqlen, scale=d**-0.5, window=None)

    kb, vb = k0.clone(), v0.clone()
    kb[1] = torch.randn_like(kb[1])
    vb[1] = torch.randn_like(vb[1])
    q = apply_qkv_decode(qkv, q_w, k_w, cos, sin, kb, vb, seqlen, **kw)
    out_b = decode_gqa_attn(q, kb, vb, seqlen, scale=d**-0.5, window=None)

    for b in (0, 2):
        assert torch.equal(out_a[b], out_b[b]), f"row {b} moved with row 1's cache"
    assert not torch.equal(out_a[1], out_b[1])


@cuda
def test_decode_attn_row_map_matches_a_gathered_dense_cache():
    """``row_map`` must be pure indirection: same result, cache rows untouched."""
    from maple_run.kernels.decode_attn import apply_qkv_decode, decode_gqa_attn

    lens = [5, 100, 63, 300]
    bsz, n_q, n_kv, d, rot = len(lens), 8, 2, 64, 32
    q_w, k_w, qkv, _k, _v, seqlen, cos, sin = _decode_attn_fixture(bsz, lens)
    # A cache wider than the batch, with the sequences scattered across it.
    n_rows = 6
    torch.manual_seed(13)
    k0 = torch.randn(n_rows, n_kv, 1024, d, device="cuda", dtype=torch.bfloat16)
    v0 = torch.randn(n_rows, n_kv, 1024, d, device="cuda", dtype=torch.bfloat16)
    perm = torch.tensor([4, 1, 5, 0], device="cuda", dtype=torch.int32)
    kw = dict(
        num_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=d,
        rotary_dim=rot,
        use_rope=True,
        use_qk_norm=True,
        eps=1e-6,
    )

    ka, va = k0.clone(), v0.clone()
    q = apply_qkv_decode(qkv, q_w, k_w, cos, sin, ka, va, seqlen, row_map=perm, **kw)
    out = decode_gqa_attn(q, ka, va, seqlen, row_map=perm, scale=d**-0.5, window=None)

    idx = perm.long()
    kd, vd = k0[idx].clone(), v0[idx].clone()
    q_ref = apply_qkv_decode(qkv, q_w, k_w, cos, sin, kd, vd, seqlen, **kw)
    out_ref = decode_gqa_attn(q_ref, kd, vd, seqlen, scale=d**-0.5, window=None)

    assert torch.equal(q, q_ref)
    assert torch.equal(out, out_ref)
    assert torch.equal(ka[idx], kd) and torch.equal(va[idx], vd)
    untouched = [r for r in range(n_rows) if r not in perm.tolist()]
    assert torch.equal(ka[untouched], k0[untouched]), "wrote outside the mapped rows"
    assert torch.equal(va[untouched], v0[untouched]), "wrote outside the mapped rows"
