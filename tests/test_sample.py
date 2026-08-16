"""GPU sampling: T=0/top-k=1 stay argmax; top-k then nucleus then multinomial."""

from __future__ import annotations

import inspect

import pytest

from maple_run.cli import main
from maple_run.generate import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    generate,
    is_greedy,
    sample_next,
)

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _gen(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def test_temperature_zero_matches_argmax():
    logits = torch.tensor([1.0, 4.0, 2.0, 3.0])
    assert int(sample_next(logits, temperature=0.0)) == 1
    assert int(sample_next(logits, temperature=0.0, top_p=0.5, top_k=3)) == 1


def test_topk_one_matches_argmax():
    logits = torch.tensor([1.0, 4.0, 2.0, 3.0])
    assert int(sample_next(logits, temperature=0.8, top_k=1, top_p=0.95)) == 1


def test_is_greedy_defaults():
    assert is_greedy(0.0, 0)
    assert is_greedy(0.0, 50)
    assert is_greedy(0.8, 1)
    assert not is_greedy(0.8, 0)
    assert not is_greedy(0.8, 50)


def test_topk_restricts_support():
    logits = torch.zeros(8)
    logits[1] = 8.0
    logits[5] = 7.0
    logits[2] = 1.0
    seen: set[int] = set()
    g = _gen(0)
    for _ in range(40):
        seen.add(int(sample_next(logits, temperature=1.0, top_k=2, generator=g)))
    assert seen <= {1, 5}
    assert seen == {1, 5}


def test_nucleus_keeps_head_mass():
    # After softmax the first two tokens hold almost all mass.
    logits = torch.tensor([5.0, 4.5, -20.0, -20.0])
    seen: set[int] = set()
    g = _gen(1)
    for _ in range(40):
        seen.add(
            int(sample_next(logits, temperature=1.0, top_p=0.8, top_k=0, generator=g))
        )
    assert seen <= {0, 1}


def test_seed_reproducible():
    logits = torch.linspace(-2.0, 3.0, 64)

    def draw(seed: int) -> list[int]:
        g = _gen(seed)
        return [
            int(
                sample_next(
                    logits, temperature=0.8, top_p=0.95, top_k=16, generator=g
                )
            )
            for _ in range(8)
        ]

    a = draw(7)
    b = draw(7)
    c = draw(8)
    assert a == b
    assert a != c


def test_cli_rejects_bad_sampling_args():
    with pytest.raises(SystemExit):
        main(["generate", "--model", "x", "--prompt", "y", "--temperature", "-1"])
    with pytest.raises(SystemExit):
        main(["generate", "--model", "x", "--prompt", "y", "--top-p", "1.5"])
    with pytest.raises(SystemExit):
        main(["generate", "--model", "x", "--prompt", "y", "--top-k", "-2"])


def test_generate_defaults_are_sampled():
    assert DEFAULT_TEMPERATURE == 1.0
    assert DEFAULT_TOP_P == 0.95
    assert DEFAULT_TOP_K == 20
    sig = inspect.signature(generate)
    assert sig.parameters["temperature"].default == DEFAULT_TEMPERATURE
    assert sig.parameters["top_p"].default == DEFAULT_TOP_P
    assert sig.parameters["top_k"].default == DEFAULT_TOP_K


def test_cli_help_lists_sampling_defaults(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "default 1.0" in out
    assert "default 0.95" in out
    assert "default 20" in out


@cuda
def test_sample_stays_on_cuda():
    device = torch.device("cuda")
    logits = torch.randn(151936, device=device)
    greedy = sample_next(logits, temperature=0.0)
    assert greedy.device.type == "cuda"
    assert greedy.dtype == torch.int64
    assert greedy.shape == ()
    sampled = sample_next(
        logits,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        generator=_gen(0, device),
    )
    assert sampled.device.type == "cuda"
    assert sampled.shape == ()
    assert 0 <= int(sampled) < 151936
