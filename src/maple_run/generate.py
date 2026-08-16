"""Text generation CLI helper."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

# CLI / generate defaults. T=0 or top-k=1 is still greedy argmax.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20


@dataclass
class GenerateResult:
    text: str
    prompt_len: int
    n_new: int
    prefill_s: float
    decode_s: float

    @property
    def decode_tok_s(self) -> float:
        return self.n_new / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def prefill_tok_s(self) -> float:
        return self.prompt_len / self.prefill_s if self.prefill_s > 0 else 0.0


def is_greedy(temperature: float, top_k: int) -> bool:
    """True when the next token must be argmax (no multinomial)."""
    return temperature <= 0.0 or top_k == 1


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
    temperature softmax, then top-k, then nucleus top-p, then multinomial.
    """
    if is_greedy(temperature, top_k):
        return logits.argmax()

    probs = torch.softmax(logits.float().div(temperature), dim=-1)
    vocab = probs.shape[-1]
    if top_k > 0 and top_k < vocab:
        probs, idx = torch.topk(probs, int(top_k), dim=-1)
    else:
        idx = None
    if 0.0 < top_p < 1.0:
        probs = _nucleus(probs, float(top_p))
    probs = probs / probs.sum().clamp_min(1e-12)
    nxt = torch.multinomial(probs, num_samples=1, generator=generator)
    if idx is not None:
        nxt = idx[nxt]
    return nxt.reshape(())


def generate(
    model_dir: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = None,
) -> GenerateResult:
    """Load a packed checkpoint, decode, print text and tok/s."""
    from transformers import AutoTokenizer

    from maple_run.model import MapleForCausalLM

    model_dir = str(Path(model_dir).expanduser())
    print(f"Loading packed model from {model_dir}", flush=True)
    model = MapleForCausalLM.from_packed(model_dir)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using a model of type")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    else:
        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
    input_ids = input_ids.to(model.device).long()
    prompt_len = int(input_ids.shape[-1])

    traffic = model.decode_traffic()
    packed_mb = traffic["packed_weight_bytes"] / 1e6
    unpacked_mb = traffic["unpacked_bf16_bytes"] / 1e6
    print(
        f"Decode weight traffic: {packed_mb:.0f} MB/token packed vs "
        f"{unpacked_mb:.0f} MB/token unpacked bf16 "
        f"(handoff ~{traffic['unpacked_handoff_bytes'] / 1e9:.1f} GB/token)",
        flush=True,
    )

    with torch.inference_mode():
        # JIT decode kernels (q_len=1 path) before the timed run.
        model.forward(
            torch.zeros(1, 1, dtype=torch.long, device=model.device),
            cache=model.make_cache(max_len=8),
            logits_to_keep=1,
        )
        torch.cuda.synchronize()

    greedy = is_greedy(temperature, top_k)
    generator = None
    if seed is not None and not greedy:
        generator = torch.Generator(device=model.device)
        generator.manual_seed(int(seed))
    if not greedy:
        print(
            f"Sampling: temperature={temperature} top_p={top_p} "
            f"top_k={top_k} seed={seed}",
            flush=True,
        )

    cache = model.make_cache(max_len=prompt_len + max_tokens)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model.forward(input_ids, cache=cache, logits_to_keep=1)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        eos = model.eos_token_id
        new_tokens: list[int] = []
        pinned = torch.empty((), dtype=torch.long, pin_memory=True)
        eos_ready = torch.cuda.Event()
        for i in range(max_tokens):
            last = logits[0, -1]
            next_t = (
                last.argmax()
                if greedy
                else sample_next(
                    last,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=generator,
                )
            )
            pinned.copy_(next_t, non_blocking=True)
            eos_ready.record()
            if i + 1 < max_tokens:
                logits = model.forward(
                    next_t.view(1, 1), cache=cache, logits_to_keep=1
                )
            eos_ready.synchronize()
            next_id = int(pinned.item())
            new_tokens.append(next_id)
            if eos is not None and next_id == int(eos):
                break
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    prefill_s = t_prefill - t0
    decode_s = t_end - t_prefill
    n_new = len(new_tokens)
    if prompt_len and prefill_s > 0:
        print(
            f"Prefill: {prompt_len} tok in {prefill_s:.2f}s "
            f"({prompt_len / prefill_s:.1f} tok/s)",
            flush=True,
        )
    if n_new and decode_s > 0:
        print(
            f"Decode:  {n_new} tok in {decode_s:.2f}s "
            f"({n_new / decode_s:.1f} tok/s)",
            flush=True,
        )

    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(text, flush=True)
    return GenerateResult(
        text=text,
        prompt_len=prompt_len,
        n_new=n_new,
        prefill_s=prefill_s,
        decode_s=decode_s,
    )
