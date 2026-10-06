"""Sampling utilities and RNG state handling for Maple decode."""

from __future__ import annotations

import torch

from maple_run.kernels.sampler import MAX_TOP_K

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MAX_CONTEXT = 131072


def resolve_max_tokens(max_tokens: int, prompt_len: int, max_context: int) -> int:
    """Map a request cap onto remaining context. ``-1`` means no extra cap."""
    remaining = max(int(max_context) - int(prompt_len), 0)
    if max_tokens == -1:
        return remaining
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0 or -1")
    return min(int(max_tokens), remaining)


def is_greedy(temperature: float, top_k: int) -> bool:
    """True when the next token must be argmax (no multinomial)."""
    return temperature <= 0.0 or top_k == 1


def can_fuse_sampler(temperature: float, top_p: float, top_k: int) -> bool:
    """Whether the two-launch Triton sampler covers these settings."""
    return (
        not is_greedy(temperature, top_k)
        and 0 < top_k <= MAX_TOP_K
        and 0.0 < top_p <= 1.0
    )


def _nucleus(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero probability mass outside the nucleus. ``probs`` is 1-D."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    remove = cumsum > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0)
    return torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token from 1-D vocab logits. Stays on-device.

    Greedy ``argmax`` when ``temperature <= 0`` or ``top_k == 1``. Otherwise:
    top-k on logits, temperature softmax, nucleus top-p, then multinomial.
    Top-k before softmax skips a full-vocab softmax on the 152k lm_head.
    """
    if is_greedy(temperature, top_k):
        return logits.argmax()

    idx = None
    x = logits
    vocab = x.shape[-1]
    if top_k > 0 and top_k < vocab:
        x, idx = torch.topk(x, int(top_k), dim=-1)
    probs = torch.softmax(x.float().div(temperature), dim=-1)
    if 0.0 < top_p < 1.0:
        probs = _nucleus(probs, float(top_p))
        probs = probs / probs.sum().clamp_min(1e-12)
    nxt = torch.multinomial(probs, num_samples=1, generator=generator)
    if idx is not None:
        nxt = idx[nxt]
    return nxt.reshape(())


def _cuda_rng(device: torch.device) -> torch.Generator:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.default_generators[idx]


def _rng_snapshot(device: torch.device, generator: torch.Generator | None):
    default = _cuda_rng(device).get_state()
    extra = generator.get_state() if generator is not None else None
    return default, extra


def _rng_restore(
    device: torch.device,
    snap: tuple[torch.Tensor, torch.Tensor | None],
    generator: torch.Generator | None,
) -> None:
    default, extra = snap
    _cuda_rng(device).set_state(default)
    if generator is not None and extra is not None:
        generator.set_state(extra)
