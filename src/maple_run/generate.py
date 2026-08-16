"""Text generation CLI helper."""

from __future__ import annotations


def generate(model_dir: str, prompt: str, max_tokens: int = 128) -> str:
    raise NotImplementedError(
        "Phase 3: load packed model + tokenizer, decode. See docs/HANDOFF.md."
    )
