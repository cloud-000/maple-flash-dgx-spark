"""Maple forward using packed linears. Port structure from modeling_maple.py.

Local Hugging Face snapshot on this host (if present):

    ~/.cache/huggingface/hub/models--deepgrove--maple-preview/snapshots/ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07/

Do not keep the Python expert loop in ``moe_infer`` (it syncs GPU→CPU every
layer). The router stays unquantized (never ternarized) and accumulates in
float32, though it is held in the model dtype. Attention Q/K/V/O projections are ternary;
activations still need an attention kernel (SDPA/FlashInfer/Triton), not Dao
``flash_attn`` inside the vLLM image.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from maple_run.kernels.decode_attn import apply_qkv_decode, decode_gqa_attn, gqa_splits
from maple_run.kernels.fused_norm import add_rms_norm, moe_reduce_add, rms_norm
from maple_run.kernels.router import router_topk
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
        return rms_norm(hidden_states, self.weight, self.eps)


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
        self.cos_table: torch.Tensor | None = None
        self.sin_table: torch.Tensor | None = None
        self.max_cached = 0
        self.ensure_length(4096)

    def ensure_length(self, n: int) -> None:
        if n <= self.max_cached:
            return
        n = max(int(n), 64)
        t = torch.arange(n, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_table = emb.cos()
        self.sin_table = emb.sin()
        self.max_cached = n

    def gather(self, position_ids: torch.Tensor, dtype: torch.dtype):
        idx = position_ids.long()
        if idx.numel() == 1:
            cos = self.cos_table.index_select(0, idx.reshape(1)).to(dtype)
            sin = self.sin_table.index_select(0, idx.reshape(1)).to(dtype)
            return cos.view(*idx.shape, self.rotary_dim), sin.view(
                *idx.shape, self.rotary_dim
            )
        return self.cos_table[idx].to(dtype), self.sin_table[idx].to(dtype)

    def __call__(self, position_ids: torch.Tensor, dtype: torch.dtype):
        self.ensure_length(int(position_ids.max().item()) + 1)
        return self.gather(position_ids, dtype)


class KVCache:
    """Preallocated K/V. Decode writes in place at ``seqlen`` (CUDA-graph safe)."""

    def __init__(
        self,
        layer_types: list[str],
        window: int,
        *,
        n_kv: int,
        head_dim: int,
        max_len: int,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
        n_heads: int | None = None,
    ):
        self.layer_types = layer_types
        self.window = window
        self.max_len = max_len
        n = len(layer_types)
        self.k = [
            torch.zeros(batch, n_kv, max_len, head_dim, dtype=dtype, device=device)
            for _ in range(n)
        ]
        self.v = [
            torch.zeros(batch, n_kv, max_len, head_dim, dtype=dtype, device=device)
            for _ in range(n)
        ]
        self.seqlen = torch.zeros((), dtype=torch.int64, device=device)
        self.seen = 0
        # Split-attention partials, shared by every layer (attention is
        # sequential) and preallocated so decode allocates nothing per step.
        n_heads = n_heads if n_heads is not None else n_kv
        splits = max(gqa_splits(max_len, window), gqa_splits(max_len, None))
        self.attn_ws = (
            torch.empty(splits, n_heads, head_dim, dtype=torch.float32, device=device),
            torch.empty(splits, n_heads, dtype=torch.float32, device=device),
            torch.empty(splits, n_heads, dtype=torch.float32, device=device),
        )

    def snapshot(self) -> dict:
        """CPU-side handle plus cloned K/V; used to restore after graph warmup."""
        return {
            "seqlen": self.seqlen.clone(),
            "seen": self.seen,
            "k": [t.clone() for t in self.k],
            "v": [t.clone() for t in self.v],
        }

    def restore(self, snap: dict) -> None:
        self.seqlen.copy_(snap["seqlen"])
        self.seen = snap["seen"]
        for dst, src in zip(self.k, snap["k"], strict=True):
            dst.copy_(src)
        for dst, src in zip(self.v, snap["v"], strict=True):
            dst.copy_(src)

    def update(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        start = self.seen
        end = start + key_states.shape[2]
        if end > self.max_len:
            raise RuntimeError(
                f"KV cache overflow: need {end} slots, max_len={self.max_len}."
            )
        self.k[layer_idx][:, :, start:end].copy_(key_states)
        self.v[layer_idx][:, :, start:end].copy_(value_states)
        k_all = self.k[layer_idx][:, :, :end]
        v_all = self.v[layer_idx][:, :, :end]
        if self.layer_types[layer_idx] == "sliding_attention":
            k_all = k_all[:, :, -self.window :]
            v_all = v_all[:, :, -self.window :]
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
        *,
        rms_weight: torch.Tensor | None = None,
        rms_eps: float = 1e-6,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states, rms_weight=rms_weight, rms_eps=rms_eps)

        if (
            q_len == 1
            and bsz == 1
            and cache is not None
            and self.use_qk_norm
        ):
            q = apply_qkv_decode(
                qkv.reshape(1, -1),
                self.q_norm.weight,
                self.k_norm.weight,
                cos,
                sin,
                cache.k[self.layer_idx],
                cache.v[self.layer_idx],
                cache.seqlen,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rotary_dim=cos.shape[-1],
                use_rope=self.use_rope,
                use_qk_norm=True,
                eps=self.q_norm.eps,
            )
            attn = decode_gqa_attn(
                q,
                cache.k[self.layer_idx],
                cache.v[self.layer_idx],
                cache.seqlen,
                scale=self.scale,
                window=self.sliding_window,
                workspace=cache.attn_ws,
            )
            attn = attn.transpose(1, 2).reshape(bsz, q_len, -1)
            return self.o_proj(attn)

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

    def __call__(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        x = hidden_states.reshape(-1, hidden)
        topk_idx, topk_weight = router_topk(x, self.gate_weight, self.top_k)

        hidden_e = self.up_gate.swiglu(x, topk_idx)
        expert_out = self.down(hidden_e, topk_idx)
        res = None if residual is None else residual.reshape(-1, hidden)
        y = moe_reduce_add(expert_out, topk_weight, residual=res)
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
        attn_out = self.self_attn(
            hidden_states,
            position_ids,
            cos,
            sin,
            cache,
            rms_weight=self.input_layernorm.weight,
            rms_eps=self.input_layernorm.eps,
        )
        hidden_states, mlp_in = add_rms_norm(
            hidden_states,
            attn_out,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        return self.mlp(mlp_in, residual=hidden_states)


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


def packed_decode_bytes(config: dict, side_bytes: int = 2) -> dict[str, int]:
    """HBM bytes touched per decode token (weights only; KV depends on context).

    ``side_bytes`` is the element size the router weight and the RTN
    scales/biases are held in at runtime; see
    ``MapleForCausalLM._cast_side_tensors``. ``row_alpha`` and the norms stay
    float32 as converted.
    """
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
    router = n_layers * n_exp * hidden * side_bytes
    expert_rows = topk * (2 * inter + hidden)
    expert_codes = n_layers * topk * (2 * inter * hidden + hidden * inter) * 2 // 8
    expert_alpha = n_layers * expert_rows * 4
    head_codes = vocab * hidden * 4 // 8
    head_scales = vocab * (hidden // group) * side_bytes * 2
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

    #: Unquantized tensors the MLX packer keeps in the source dtype. Everything
    #: else the converter promoted to float32 stays float32 (``row_alpha`` and
    #: the norms are read once per row and are not worth the churn).
    _BF16_SUFFIXES = (".mlp.gate.weight", ".scales", ".biases")

    def _cast_side_tensors(self, weights: dict[str, torch.Tensor]) -> None:
        """Hold the router and RTN scales in the model dtype, not fp32.

        ``maple_run.convert`` promotes every float tensor to float32 (the
        ternarizer needs float32 arithmetic). The DeepGrove MLX packer instead
        passes non-ternary weights through untouched and lets ``mx.quantize``
        return scales/biases in the source dtype, so a reference checkpoint has
        all of these in bf16. Together they are ~45 MB of the 462 MB read per
        decode token.

        This only pays off with the codes in their untransposed layout: when
        the head codes were stored ``[nwords, N]`` the tiles wanted
        ``BLOCK_N=32``, which turned bf16 scales into 64-byte loads and
        measured slower. Untransposed, each row's 32 groups are contiguous and
        bf16 wins (820 vs 903 us on the lm_head).
        """
        for key in list(weights):
            if weights[key].dtype == torch.float32 and key.endswith(self._BF16_SUFFIXES):
                weights[key] = weights[key].to(self.dtype)

    def _load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self._cast_side_tensors(weights)
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

    def make_cache(self, max_len: int = 2048, batch: int = 1) -> KVCache:
        self.rotary.ensure_length(max_len)
        return KVCache(
            self.layer_types,
            self.sliding_window,
            n_kv=self.num_kv_heads,
            head_dim=self.head_dim,
            max_len=max_len,
            batch=batch,
            dtype=self.dtype,
            device=self.device,
            n_heads=self.num_heads,
        )

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
        if cache is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device, dtype=torch.long
            ).unsqueeze(0).expand(bsz, -1)
        elif seq_len == 1:
            position_ids = cache.seqlen.view(1, 1).expand(bsz, 1)
        else:
            past = cache.seen
            position_ids = torch.arange(
                past, past + seq_len, device=input_ids.device, dtype=torch.long
            ).unsqueeze(0).expand(bsz, -1)
        cos, sin = self.rotary.gather(position_ids, hidden.dtype)

        for layer in self.layers:
            hidden = layer(hidden, position_ids, cos, sin, cache)
        if cache is not None:
            if seq_len == 1:
                cache.seqlen.add_(1)
            else:
                cache.seen = cache.seen + seq_len
                cache.seqlen.fill_(cache.seen)
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

        cache = self.make_cache(max_len=int(input_ids.shape[-1]) + int(max_tokens))
        logits = self.forward(input_ids, cache=cache, logits_to_keep=1)
        eos = self.eos_token_id
        out = [input_ids]
        for i in range(max_tokens):
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out.append(next_id)
            if eos is not None and int(next_id[0, 0]) == int(eos):
                break
            logits = self.forward(next_id, cache=cache, logits_to_keep=1)
        return torch.cat(out, dim=-1)

    def decode_traffic(self) -> dict[str, int]:
        return packed_decode_bytes(
            self.config, side_bytes=torch.empty(0, dtype=self.dtype).element_size()
        )
