"""Maple packed forward: fused MoE, no 256-expert Python loop, decode traffic."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from maple_run.model import packed_decode_bytes
from maple_run.pack import quantize_rtn, ternarize

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

REPO = Path(__file__).resolve().parents[1]
PACKED_CKPT = REPO / "checkpoints" / "maple-2bit"


def _tiny_config(**overrides) -> dict:
    cfg = {
        "hidden_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "vocab_size": 64,
        "rms_norm_eps": 1e-6,
        "sliding_window": 32,
        "layer_types": ["sliding_attention"],
        "use_qk_norm": True,
        "nope_on_global_attention": True,
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
        "eos_token_id": 0,
        "maple_run": {
            "format": "packed-ternary-v1",
            "rtn_4bit": {"group_size": 64, "bits": 4},
        },
    }
    cfg.update(overrides)
    return cfg


def _tiny_weights(cfg: dict, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    h = cfg["hidden_size"]
    v = cfg["vocab_size"]
    e = cfg["num_experts"]
    inter = cfg["moe_intermediate_size"]
    n_q = cfg["num_attention_heads"] * cfg["head_dim"]
    n_kv = cfg["num_key_value_heads"] * cfg["head_dim"]
    tensors: dict[str, np.ndarray] = {}

    def put_rtn(prefix: str, w: np.ndarray) -> None:
        packed, scales, biases = quantize_rtn(w)
        tensors[f"{prefix}.weight"] = packed
        tensors[f"{prefix}.scales"] = scales
        tensors[f"{prefix}.biases"] = biases

    def put_ternary(prefix: str, w: np.ndarray) -> None:
        packed, alpha = ternarize(w)
        tensors[f"{prefix}.weight"] = packed
        tensors[f"{prefix}.row_alpha"] = np.ascontiguousarray(alpha, dtype=np.float32)

    put_rtn("lm_head", rng.standard_normal((v, h)).astype(np.float32))
    put_rtn("model.word_embeddings", rng.standard_normal((v, h)).astype(np.float32))
    tensors["model.norm.weight"] = np.ones((h,), dtype=np.float32)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones((h,), dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones((h,), dtype=np.float32)
        tensors[f"{p}.self_attn.q_norm.weight"] = np.ones((cfg["head_dim"],), dtype=np.float32)
        tensors[f"{p}.self_attn.k_norm.weight"] = np.ones((cfg["head_dim"],), dtype=np.float32)
        put_ternary(f"{p}.self_attn.q_proj", rng.standard_normal((n_q, h)).astype(np.float32))
        put_ternary(f"{p}.self_attn.k_proj", rng.standard_normal((n_kv, h)).astype(np.float32))
        put_ternary(f"{p}.self_attn.v_proj", rng.standard_normal((n_kv, h)).astype(np.float32))
        put_ternary(f"{p}.self_attn.o_proj", rng.standard_normal((h, n_q)).astype(np.float32))
        tensors[f"{p}.mlp.gate.weight"] = rng.standard_normal((e, h)).astype(np.float32)
        for proj, shape in (
            ("gate_proj", (e, inter, h)),
            ("up_proj", (e, inter, h)),
            ("down_proj", (e, h, inter)),
        ):
            put_ternary(f"{p}.mlp.switch_mlp.{proj}", rng.standard_normal(shape).astype(np.float32))

    out = {}
    for k, arr in tensors.items():
        if arr.dtype == np.uint32:
            out[k] = torch.from_numpy(np.ascontiguousarray(arr)).to(
                device="cuda", dtype=torch.uint32
            )
        else:
            out[k] = torch.from_numpy(np.ascontiguousarray(arr)).to(
                device="cuda", dtype=torch.float32
            )
    return out


def test_model_source_has_no_unfused_expert_loop():
    import maple_run.model as model_mod

    src = inspect.getsource(model_mod)
    assert "tokens_per_expert" not in src
    assert ".cpu().numpy()" not in src
    assert "for i, expert in" not in src
    assert "for i, num_tokens" not in src
    assert "range(self.num_experts)" not in src
    assert "range(256)" not in src
    assert "PackedTernaryExperts" in src
    assert "scaled_dot_product_attention" in src
    assert "import flash_attn" not in src
    assert "from flash_attn" not in src


def test_packed_decode_bytes_beats_unpacked_estimate():
    cfg = json.loads((REPO / "docs/sources/maple-preview-config.json").read_text())
    cfg["maple_run"] = {
        "format": "packed-ternary-v1",
        "rtn_4bit": {"group_size": 64, "bits": 4},
    }
    traffic = packed_decode_bytes(cfg)
    packed = traffic["packed_weight_bytes"]
    unpacked = traffic["unpacked_bf16_bytes"]
    assert packed < 0.7e9, packed
    assert unpacked > 2.0e9, unpacked
    assert unpacked / packed > 4


@cuda
def test_tiny_forward_and_generate_shapes():
    from maple_run.model import MapleForCausalLM

    cfg = _tiny_config()
    rng = np.random.default_rng(0)
    weights = _tiny_weights(cfg, rng)
    model = MapleForCausalLM.from_weight_dict(cfg, weights, device="cuda")
    ids = torch.randint(0, cfg["vocab_size"], (1, 5), device="cuda")
    logits = model.forward(ids, cache=model.make_cache(), logits_to_keep=1)
    assert logits.shape == (1, 1, cfg["vocab_size"])
    assert torch.isfinite(logits.float()).all()

    out = model.generate(ids, max_tokens=3)
    assert out.shape[0] == 1
    assert out.shape[1] == ids.shape[1] + 3
    assert out[:, : ids.shape[1]].eq(ids).all()


@cuda
def test_moe_matches_selected_expert_linears():
    from maple_run.kernels.ternary_gemv import ternary_gemv
    from maple_run.linear import PackedTernaryExperts
    from maple_run.model import MLP_CLAMP, MapleSparseMoeBlock

    rng = np.random.default_rng(3)
    n_tok, hidden, n_exp, topk, inter = 3, 128, 4, 2, 128
    gate_w = torch.from_numpy(rng.standard_normal((n_exp, hidden)).astype(np.float32)).cuda()
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
    up_gate = PackedTernaryExperts(
        torch.cat([up_pt, gate_pt], dim=1), torch.cat([up_at, gate_at], dim=1)
    )
    down = PackedTernaryExperts(down_pt, down_at)
    block = MapleSparseMoeBlock(gate_w, up_gate, down, top_k=topk, moe_intermediate=inter)

    x = torch.from_numpy(rng.standard_normal((1, n_tok, hidden)).astype(np.float32)).cuda()
    y = block(x)

    flat = x.reshape(n_tok, hidden)
    logits = torch.nn.functional.linear(flat.float(), gate_w.float())
    routing = torch.softmax(logits, dim=-1)
    scores, idx = torch.topk(routing, topk, dim=-1)
    wts = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    ref = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
    for t in range(n_tok):
        acc = torch.zeros(hidden, device="cuda", dtype=torch.float32)
        for s in range(topk):
            e = int(idx[t, s])
            up = ternary_gemv(flat[t], up_pt[e], up_at[e])
            gate = ternary_gemv(flat[t], gate_pt[e], gate_at[e])
            h = torch.nn.functional.silu(gate.clamp(max=MLP_CLAMP)) * up.clamp(
                min=-MLP_CLAMP, max=MLP_CLAMP
            )
            acc = acc + wts[t, s] * ternary_gemv(h, down_pt[e], down_at[e]).float()
        ref[t] = acc
    torch.testing.assert_close(y[0].float(), ref, rtol=1e-4, atol=1e-4)


@cuda
def test_global_layer_skips_rope():
    from maple_run.linear import PackedTernaryLinear
    from maple_run.model import MapleAttention, MapleRMSNorm, MapleRotaryEmbedding
    from maple_run.pack import ternarize

    rng = np.random.default_rng(4)
    h, n_q, n_kv, d = 128, 2, 2, 64
    qkv, a = ternarize(rng.standard_normal(((n_q + 2 * n_kv) * d, h)).astype(np.float32))
    o, oa = ternarize(rng.standard_normal((h, n_q * d)).astype(np.float32))
    qkv_t = torch.from_numpy(np.ascontiguousarray(qkv)).cuda().to(torch.uint32)
    a_t = torch.from_numpy(np.ascontiguousarray(a)).cuda()
    o_t = torch.from_numpy(np.ascontiguousarray(o)).cuda().to(torch.uint32)
    oa_t = torch.from_numpy(np.ascontiguousarray(oa)).cuda()
    ones = torch.ones(d, device="cuda")
    attn_rope = MapleAttention(
        PackedTernaryLinear(qkv_t, a_t),
        PackedTernaryLinear(o_t, oa_t),
        MapleRMSNorm(ones),
        MapleRMSNorm(ones),
        num_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=d,
        sliding_window=32,
        use_qk_norm=True,
        use_rope=True,
        layer_idx=0,
    )
    attn_nope = MapleAttention(
        PackedTernaryLinear(qkv_t, a_t),
        PackedTernaryLinear(o_t, oa_t),
        MapleRMSNorm(ones),
        MapleRMSNorm(ones),
        num_heads=n_q,
        num_kv_heads=n_kv,
        head_dim=d,
        sliding_window=None,
        use_qk_norm=True,
        use_rope=False,
        layer_idx=0,
    )
    x = torch.from_numpy(rng.standard_normal((1, 4, h)).astype(np.float32)).cuda()
    pos = torch.arange(4, device="cuda").unsqueeze(0)
    cos, sin = MapleRotaryEmbedding(d, 0.5, 10000.0, x.device)(pos, x.dtype)
    y_rope = attn_rope(x, pos, cos, sin, cache=None)
    y_nope = attn_nope(x, pos, cos, sin, cache=None)
    assert not torch.allclose(y_rope, y_nope, rtol=1e-3, atol=1e-3)


def test_from_packed_rejects_unpacked_config(tmp_path: Path):
    from maple_run.model import MapleForCausalLM

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "maple"}))
    with pytest.raises(ValueError, match="packed-ternary-v1"):
        MapleForCausalLM.from_packed(str(tmp_path))
