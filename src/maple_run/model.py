"""Maple forward using packed linears. Port structure from modeling_maple.py.

Local Hugging Face snapshot on this host (if present):

    ~/.cache/huggingface/hub/models--deepgrove--maple-preview/snapshots/ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07/

Do not keep the Python expert loop in ``moe_infer`` (it syncs GPU→CPU every
layer). Router stays float32. Attention Q/K/V/O projections are ternary;
activations still need an attention kernel (SDPA/FlashInfer/Triton), not Dao
``flash_attn`` inside the vLLM image.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from maple_run.linear import (
    PackedRTN4Embedding,
    PackedRTN4Linear,
    PackedTernaryExperts,
    PackedTernaryLinear,
)
from maple_run.pack import HEAD_GROUP_SIZE

MLP_CLAMP = 7.0
UNPACKED_BF16_BYTES_PER_TOKEN = 2.3e9


class MapleRMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * hidden_states).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class MapleRotaryEmbedding:
    """Partial RoPE matching Transformers default + Maple ``partial_rotary_factor``."""

    def __init__(self, head_dim: int, partial_rotary_factor: float, theta: float, device):
        rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / rotary_dim
            )
        )
        self.inv_freq = inv_freq
        self.rotary_dim = rotary_dim

    def __call__(self, position_ids: torch.Tensor, dtype: torch.dtype):
        # position_ids: [B, L]
        freqs = torch.einsum("bl,d->bld", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class KVCache:
    """Per-layer K/V. Sliding-window layers keep the last ``window`` tokens."""

    def __init__(self, layer_types: list[str], window: int):
        self.layer_types = layer_types
        self.window = window
        n = len(layer_types)
        self.k: list[torch.Tensor | None] = [None] * n
        self.v: list[torch.Tensor | None] = [None] * n
        self.seen = 0

    def update(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        k_prev, v_prev = self.k[layer_idx], self.v[layer_idx]
        if k_prev is None:
            k_all, v_all = key_states, value_states
        else:
            k_all = torch.cat([k_prev, key_states], dim=2)
            v_all = torch.cat([v_prev, value_states], dim=2)
        if self.layer_types[layer_idx] == "sliding_attention":
            k_all = k_all[:, :, -self.window :]
            v_all = v_all[:, :, -self.window :]
        self.k[layer_idx] = k_all
        self.v[layer_idx] = v_all
        return k_all, v_all


def _sliding_causal_mask(q_len: int, kv_len: int, window: int, device):
    """Boolean keep-mask for SDPA: causal and within ``window`` (query attends left)."""
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    return (k_pos <= q_pos) & ((q_pos - k_pos) < window)


class MapleAttention:
    def __init__(
        self,
        qkv: PackedTernaryLinear,
        o_proj: PackedTernaryLinear,
        q_norm: MapleRMSNorm,
        k_norm: MapleRMSNorm,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
        use_qk_norm: bool,
        use_rope: bool,
        layer_idx: int,
    ):
        self.qkv_proj = qkv
        self.o_proj = o_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.sliding_window = sliding_window
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.layer_idx = layer_idx
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        kv_len = k.shape[2]
        if q_len == 1:
            attn = F.scaled_dot_product_attention(
                q, k, v, scale=self.scale, enable_gqa=True
            )
        elif self.sliding_window is not None:
            mask = _sliding_causal_mask(q_len, kv_len, self.sliding_window, q.device)
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=self.scale, enable_gqa=True
            )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale, enable_gqa=True
            )
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn)


class MapleSparseMoeBlock:
    """Top-k softmax router + fused packed SwitchGLU. No 256-expert Python loop."""

    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_gate: PackedTernaryExperts,
        down: PackedTernaryExperts,
        top_k: int,
        moe_intermediate: int,
    ):
        self.gate_weight = gate_weight
        self.up_gate = up_gate
        self.down = down
        self.top_k = top_k
        self.moe_intermediate = moe_intermediate

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        x = hidden_states.reshape(-1, hidden)
        logits = F.linear(x.float(), self.gate_weight.float())
        routing = torch.softmax(logits, dim=-1, dtype=torch.float32)
        scores, topk_idx = torch.topk(routing, self.top_k, dim=-1)
        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)

        up_gate = self.up_gate(x, topk_idx)
        up, gate = up_gate.split(self.moe_intermediate, dim=-1)
        hidden_e = F.silu(gate.clamp(max=MLP_CLAMP)) * up.clamp(
            min=-MLP_CLAMP, max=MLP_CLAMP
        )
        expert_out = self.down(hidden_e, topk_idx)
        y = (expert_out.float() * topk_weight.unsqueeze(-1)).sum(dim=-2).to(
            hidden_states.dtype
        )
        return y.view(bsz, seq_len, hidden)


class MapleDecoderLayer:
    def __init__(
        self,
        self_attn: MapleAttention,
        mlp: MapleSparseMoeBlock,
        input_layernorm: MapleRMSNorm,
        post_attention_layernorm: MapleRMSNorm,
    ):
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def __call__(self, hidden_states, position_ids, cos, sin, cache):
        residual = hidden_states
        hidden_states = residual + self.self_attn(
            self.input_layernorm(hidden_states), position_ids, cos, sin, cache
        )
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


def _require_packed_format(config: dict) -> None:
    meta = config.get("maple_run") or {}
    if meta.get("format") != "packed-ternary-v1":
        raise ValueError(
            "Checkpoint is not maple-run packed-ternary-v1. Convert with "
            "`maple-run convert` rather than loading the Hugging Face bf16 dump."
        )


def _cat_qkv(weights: dict, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.cat(
        [
            weights.pop(f"{prefix}.self_attn.q_proj.weight"),
            weights.pop(f"{prefix}.self_attn.k_proj.weight"),
            weights.pop(f"{prefix}.self_attn.v_proj.weight"),
        ],
        dim=0,
    )
    alpha = torch.cat(
        [
            weights.pop(f"{prefix}.self_attn.q_proj.row_alpha"),
            weights.pop(f"{prefix}.self_attn.k_proj.row_alpha"),
            weights.pop(f"{prefix}.self_attn.v_proj.row_alpha"),
        ],
        dim=0,
    )
    return packed, alpha


def _cat_up_gate(weights: dict, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.cat(
        [
            weights.pop(f"{prefix}.mlp.switch_mlp.up_proj.weight"),
            weights.pop(f"{prefix}.mlp.switch_mlp.gate_proj.weight"),
        ],
        dim=1,
    )
    alpha = torch.cat(
        [
            weights.pop(f"{prefix}.mlp.switch_mlp.up_proj.row_alpha"),
            weights.pop(f"{prefix}.mlp.switch_mlp.gate_proj.row_alpha"),
        ],
        dim=1,
    )
    return packed, alpha


def _as_uint32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint32:
        return t
    return t.view(torch.uint32)


def packed_decode_bytes(config: dict) -> dict[str, int]:
    """HBM bytes touched per decode token (weights only; KV depends on context)."""
    n_layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    n_q = int(config["num_attention_heads"])
    n_kv = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    n_exp = int(config["num_experts"])
    topk = int(config["num_experts_per_tok"])
    inter = int(config["moe_intermediate_size"])
    vocab = int(config["vocab_size"])
    group = int(
        (config.get("maple_run") or {})
        .get("rtn_4bit", {})
        .get("group_size", HEAD_GROUP_SIZE)
    )

    qkv_n = (n_q + 2 * n_kv) * head_dim
    attn_codes = n_layers * (qkv_n + hidden) * hidden // 4  # 2-bit → 4 weights/byte
    attn_alpha = n_layers * (qkv_n + hidden) * 4
    router = n_layers * n_exp * hidden * 4
    expert_rows = topk * (2 * inter + hidden)
    expert_codes = n_layers * topk * (2 * inter * hidden + hidden * inter) * 2 // 8
    expert_alpha = n_layers * expert_rows * 4
    head_codes = vocab * hidden * 4 // 8
    head_scales = vocab * (hidden // group) * 4 * 2
    norms = n_layers * (2 * hidden + 2 * head_dim) * 4 + hidden * 4
    packed = (
        attn_codes
        + attn_alpha
        + router
        + expert_codes
        + expert_alpha
        + head_codes
        + head_scales
        + norms
    )
    unpacked = (
        n_layers * (qkv_n + hidden) * hidden * 2
        + router
        + n_layers * topk * (2 * inter * hidden + hidden * inter) * 2
        + vocab * hidden * 2
        + norms
    )
    return {
        "packed_weight_bytes": packed,
        "unpacked_bf16_bytes": unpacked,
        "unpacked_handoff_bytes": int(UNPACKED_BF16_BYTES_PER_TOKEN),
    }


class MapleForCausalLM:
    def __init__(self, config: dict, *, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.hidden_size = int(config["hidden_size"])
        self.vocab_size = int(config["vocab_size"])
        self.num_layers = int(config["num_hidden_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.num_experts = int(config["num_experts"])
        self.top_k = int(config["num_experts_per_tok"])
        self.moe_intermediate = int(config["moe_intermediate_size"])
        self.rms_eps = float(config.get("rms_norm_eps", 1e-6))
        self.sliding_window = int(config.get("sliding_window", 512))
        self.layer_types = list(config["layer_types"])
        self.use_qk_norm = bool(config.get("use_qk_norm", True))
        self.nope_on_global = bool(config.get("nope_on_global_attention", True))
        self.eos_token_id = config.get("eos_token_id")
        self.bos_token_id = config.get("bos_token_id")
        self.word_embeddings = None
        self.layers: list[MapleDecoderLayer] = []
        self.norm = None
        self.lm_head = None
        self.rotary = MapleRotaryEmbedding(
            self.head_dim,
            float(config.get("partial_rotary_factor", 0.5)),
            float(config.get("rope_theta", 10000)),
            device,
        )

    @classmethod
    def from_packed(
        cls,
        model_dir: str,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        model_dir = Path(model_dir).expanduser()
        config = json.loads((model_dir / "config.json").read_text())
        _require_packed_format(config)
        if isinstance(device, str):
            device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("MapleForCausalLM packed decode requires CUDA.")

        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            shards = sorted(set(weight_map.values()))
        else:
            shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
        weights: dict[str, torch.Tensor] = {}
        for name in shards:
            shard = load_file(str(model_dir / name), device=str(device))
            weights.update(shard)

        return cls.from_weight_dict(config, weights, device=device, dtype=dtype)

    @classmethod
    def from_weight_dict(
        cls,
        config: dict,
        weights: dict[str, torch.Tensor],
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        _require_packed_format(config)
        if isinstance(device, str):
            device = torch.device(device)
        model = cls(config, device=device, dtype=dtype)
        model._load_weights(weights)
        return model

    def _load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        rtn_group = int(
            (self.config.get("maple_run") or {})
            .get("rtn_4bit", {})
            .get("group_size", HEAD_GROUP_SIZE)
        )
        self.word_embeddings = PackedRTN4Embedding(
            _as_uint32(weights.pop("model.word_embeddings.weight")),
            weights.pop("model.word_embeddings.scales"),
            weights.pop("model.word_embeddings.biases"),
            group_size=rtn_group,
        )
        self.lm_head = PackedRTN4Linear(
            _as_uint32(weights.pop("lm_head.weight")),
            weights.pop("lm_head.scales"),
            weights.pop("lm_head.biases"),
            group_size=rtn_group,
        )
        self.norm = MapleRMSNorm(weights.pop("model.norm.weight"), self.rms_eps)

        layers = []
        for i in range(self.num_layers):
            prefix = f"model.layers.{i}"
            layer_type = self.layer_types[i]
            use_rope = not (self.nope_on_global and layer_type == "full_attention")
            sliding = self.sliding_window if layer_type == "sliding_attention" else None
            qkv_w, qkv_a = _cat_qkv(weights, prefix)
            up_w, up_a = _cat_up_gate(weights, prefix)
            attn = MapleAttention(
                PackedTernaryLinear(_as_uint32(qkv_w), qkv_a),
                PackedTernaryLinear(
                    _as_uint32(weights.pop(f"{prefix}.self_attn.o_proj.weight")),
                    weights.pop(f"{prefix}.self_attn.o_proj.row_alpha"),
                ),
                MapleRMSNorm(weights.pop(f"{prefix}.self_attn.q_norm.weight"), self.rms_eps),
                MapleRMSNorm(weights.pop(f"{prefix}.self_attn.k_norm.weight"), self.rms_eps),
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                sliding_window=sliding,
                use_qk_norm=self.use_qk_norm,
                use_rope=use_rope,
                layer_idx=i,
            )
            mlp = MapleSparseMoeBlock(
                weights.pop(f"{prefix}.mlp.gate.weight"),
                PackedTernaryExperts(_as_uint32(up_w), up_a),
                PackedTernaryExperts(
                    _as_uint32(weights.pop(f"{prefix}.mlp.switch_mlp.down_proj.weight")),
                    weights.pop(f"{prefix}.mlp.switch_mlp.down_proj.row_alpha"),
                ),
                top_k=self.top_k,
                moe_intermediate=self.moe_intermediate,
            )
            layers.append(
                MapleDecoderLayer(
                    attn,
                    mlp,
                    MapleRMSNorm(weights.pop(f"{prefix}.input_layernorm.weight"), self.rms_eps),
                    MapleRMSNorm(
                        weights.pop(f"{prefix}.post_attention_layernorm.weight"),
                        self.rms_eps,
                    ),
                )
            )
        self.layers = layers

    def make_cache(self) -> KVCache:
        return KVCache(self.layer_types, self.sliding_window)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        *,
        logits_to_keep: int = 1,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        hidden = self.word_embeddings(input_ids).to(self.dtype)
        bsz, seq_len, _ = hidden.shape
        past = 0 if cache is None else cache.seen
        position_ids = torch.arange(
            past, past + seq_len, device=input_ids.device
        ).unsqueeze(0).expand(bsz, -1)
        cos, sin = self.rotary(position_ids, hidden.dtype)

        for layer in self.layers:
            hidden = layer(hidden, position_ids, cos, sin, cache)
        if cache is not None:
            cache.seen = past + seq_len
        hidden = self.norm(hidden)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:, :]
        return self.lm_head(hidden)

    @torch.inference_mode()
    def generate(self, input_ids, max_tokens: int = 128):
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, device=self.device, dtype=torch.long)
        else:
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        cache = self.make_cache()
        logits = self.forward(input_ids, cache=cache, logits_to_keep=1)
        eos = self.eos_token_id
        out = [input_ids]
        for _ in range(max_tokens):
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out.append(next_id)
            if eos is not None and int(next_id[0, 0]) == int(eos):
                break
            logits = self.forward(next_id, cache=cache, logits_to_keep=1)
        return torch.cat(out, dim=-1)

    def decode_traffic(self) -> dict[str, int]:
        return packed_decode_bytes(self.config)
