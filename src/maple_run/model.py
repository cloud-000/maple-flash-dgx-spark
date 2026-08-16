"""Maple forward using packed linears. Port structure from modeling_maple.py.

Local Hugging Face snapshot on this host (if present):

    ~/.cache/huggingface/hub/models--deepgrove--maple-preview/snapshots/ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07/

Do not keep the Python expert loop in ``moe_infer`` (it syncs GPU→CPU every
layer). Router stays float32. Attention Q/K/V/O projections are ternary;
activations still need an attention kernel (SDPA/FlashInfer/Triton), not Dao
``flash_attn`` inside the vLLM image.
"""

from __future__ import annotations


class MapleForCausalLM:
    def __init__(self, config: dict):
        self.config = config

    @classmethod
    def from_packed(cls, model_dir: str):
        raise NotImplementedError("Phase 3: load packed checkpoint. See docs/HANDOFF.md.")

    def generate(self, input_ids, max_tokens: int = 128):
        raise NotImplementedError("Phase 3: decode loop. See docs/HANDOFF.md.")
