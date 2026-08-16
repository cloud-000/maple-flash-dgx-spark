"""Packed decode speed. Greedy argmax is not the bench: it skips the sampler
and often EOS early (France ~38 tok), which inflates tok/s vs real use.

Run: ``uv run pytest tests/test_bench.py --bench -s``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maple_run.generate import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    generate,
)

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
PACKED_CKPT = REPO / "checkpoints" / "maple-2bit"

pytestmark = [
    pytest.mark.bench,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not (PACKED_CKPT / "config.json").exists(),
        reason="packed checkpoint missing",
    ),
]


def test_decode_speed_sampled():
    result = generate(
        str(PACKED_CKPT),
        "Write a haiku on groves",
        max_tokens=256,
        seed=0,
    )
    print(
        f"\nbench sampled T={DEFAULT_TEMPERATURE} top_p={DEFAULT_TOP_P} "
        f"top_k={DEFAULT_TOP_K}: {result.n_new} tok, "
        f"{result.decode_tok_s:.1f} tok/s",
        flush=True,
    )
    assert result.n_new >= 32, result.n_new
    assert result.decode_tok_s > 10.0, result.decode_tok_s
