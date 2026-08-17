"""Text generation CLI helper."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from maple_run.kernels.sampler import MAX_TOP_K, fused_sample, sampler_workspace

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
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def decode_tok_s(self) -> float:
        return self.n_new / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def prefill_tok_s(self) -> float:
        return self.prompt_len / self.prefill_s if self.prefill_s > 0 else 0.0


def _log_print(msg: str) -> None:
    print(msg, flush=True)


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


_GRAPH_WARMUP = 3
_GRAPH_CHECK_TOKENS = 8


@dataclass
class _DecodeGraph:
    graph: torch.cuda.CUDAGraph
    token_ids: torch.Tensor
    logits: torch.Tensor


def _next_token(
    logits: torch.Tensor,
    *,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None,
    sampler_ws: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    last = logits[0, -1]
    if greedy:
        return last.argmax()
    if sampler_ws is not None:
        u = torch.rand((), device=last.device, generator=generator)
        return fused_sample(
            last,
            u,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            workspace=sampler_ws,
        )
    return sample_next(
        last,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
    )


def _capture_decode_graph(model, cache, token_ids: torch.Tensor) -> _DecodeGraph:
    """Capture ``forward(token_ids)`` on ``cache``. Caller restores KV afterwards."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    logits = None
    with torch.cuda.stream(stream):
        for _ in range(_GRAPH_WARMUP):
            logits = model.forward(token_ids, cache=cache, logits_to_keep=1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = model.forward(token_ids, cache=cache, logits_to_keep=1)
    torch.cuda.current_stream().wait_stream(stream)
    if logits is None:
        raise RuntimeError("CUDA graph capture did not produce logits.")
    return _DecodeGraph(graph=graph, token_ids=token_ids, logits=logits)


def _decode_n(
    *,
    n: int,
    logits: torch.Tensor,
    replay,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None,
    eos,
    sampler_ws: tuple[torch.Tensor, ...] | None = None,
) -> list[int]:
    ids: list[int] = []
    for i in range(n):
        next_t = _next_token(
            logits,
            greedy=greedy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
            sampler_ws=sampler_ws,
        )
        ids.append(int(next_t.item()))
        if eos is not None and ids[-1] == int(eos):
            break
        if i + 1 < n:
            logits = replay(next_t)
    return ids


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


def _try_decode_graph(
    model,
    cache,
    prefill_logits: torch.Tensor,
    *,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None,
    sampler_ws: tuple[torch.Tensor, ...] | None = None,
    log: Callable[[str], None] = _log_print,
) -> _DecodeGraph | None:
    """Capture a decode graph; keep it only if replay matches eager token ids."""
    snap = cache.snapshot()
    token_ids = torch.zeros(1, 1, dtype=torch.long, device=model.device)
    try:
        captured = _capture_decode_graph(model, cache, token_ids)
    except Exception as exc:
        cache.restore(snap)
        log(f"CUDA graph capture failed ({exc}); using eager decode")
        return None
    cache.restore(snap)

    n_check = _GRAPH_CHECK_TOKENS
    rng_state = _rng_snapshot(model.device, generator)
    eos = model.eos_token_id

    def eager_replay(next_t: torch.Tensor) -> torch.Tensor:
        return model.forward(next_t.view(1, 1), cache=cache, logits_to_keep=1)

    def graph_replay(next_t: torch.Tensor) -> torch.Tensor:
        captured.token_ids.copy_(next_t.view(1, 1))
        captured.graph.replay()
        return captured.logits

    eager_ids = _decode_n(
        n=n_check,
        logits=prefill_logits,
        replay=eager_replay,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
        eos=eos,
        sampler_ws=sampler_ws,
    )
    cache.restore(snap)
    _rng_restore(model.device, rng_state, generator)

    graph_ids = _decode_n(
        n=n_check,
        logits=prefill_logits,
        replay=graph_replay,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
        eos=eos,
        sampler_ws=sampler_ws,
    )
    cache.restore(snap)
    _rng_restore(model.device, rng_state, generator)

    if eager_ids != graph_ids:
        log(
            f"CUDA graph replay diverged (eager {eager_ids[:8]} vs graph "
            f"{graph_ids[:8]}); using eager decode"
        )
        return None
    log("CUDA graph decode: replay matches eager sampled ids")
    return captured


def load_packed(model_dir: str, flash_head: bool = False):
    """Load a packed Maple checkpoint and its tokenizer."""
    from transformers import AutoTokenizer

    from maple_run.model import MapleForCausalLM

    model_dir = str(Path(model_dir).expanduser())
    model = MapleForCausalLM.from_packed(model_dir, use_flash_head=flash_head)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using a model of type")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer


def message_text(content) -> str:
    """Flatten OpenAI message content (string or text parts) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type", "text") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                raise ValueError("Only text message content is supported.")
        return "".join(parts)
    raise ValueError("message content must be a string or a list of text parts.")


def _to_input_ids(encoded, device) -> torch.Tensor:
    input_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    return input_ids.to(device).long()


def encode_messages(tokenizer, messages: list[dict], device) -> torch.Tensor:
    """Apply the chat template, or concatenate roles if the tokenizer has none."""
    cleaned = [
        {"role": str(m["role"]), "content": message_text(m.get("content"))}
        for m in messages
    ]
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            cleaned,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return _to_input_ids(encoded, device)
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in cleaned)
    return _to_input_ids(
        tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt"),
        device,
    )


def encode_prompt(tokenizer, prompt: str, device) -> torch.Tensor:
    """Encode a raw completion prompt (no chat template)."""
    return _to_input_ids(
        tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt"),
        device,
    )


def encode_user_prompt(tokenizer, prompt: str, device) -> torch.Tensor:
    """Encode a CLI-style user string through the chat template."""
    return encode_messages(
        tokenizer, [{"role": "user", "content": prompt}], device
    )


def warmup_model(model, flash_head: bool = False) -> None:
    """JIT the decode (and exact-head prefill) kernels."""
    with torch.inference_mode():
        model.forward(
            torch.zeros(1, 1, dtype=torch.long, device=model.device),
            cache=model.make_cache(max_len=8),
            logits_to_keep=1,
        )
        if flash_head:
            model.forward(
                torch.zeros(1, 2, dtype=torch.long, device=model.device),
                cache=model.make_cache(max_len=8),
                logits_to_keep=1,
            )
        torch.cuda.synchronize()


def log_model_traffic(model, flash_head: bool = False, log: Callable[[str], None] = _log_print) -> None:
    traffic = model.decode_traffic()
    packed_mb = traffic["packed_weight_bytes"] / 1e6
    unpacked_mb = traffic["unpacked_bf16_bytes"] / 1e6
    log(
        f"Decode weight traffic: {packed_mb:.0f} MB/token packed vs "
        f"{unpacked_mb:.0f} MB/token unpacked bf16 "
        f"(handoff ~{traffic['unpacked_handoff_bytes'] / 1e9:.1f} GB/token)"
    )
    if flash_head:
        meta = model.config.get("flash_head") or {}
        log(
            f"FlashHead: {meta.get('n_clusters')} clusters, "
            f"{meta.get('n_probes')} probes, force={meta.get('force_tokens')}"
        )


def run_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    max_tokens: int = 128,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = None,
    log: Callable[[str], None] | None = _log_print,
    on_token: Callable[[int], None] | None = None,
) -> GenerateResult:
    """Decode from ``input_ids`` on an already-loaded packed model."""
    def emit(msg: str) -> None:
        if log is not None:
            log(msg)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(model.device).long()
    prompt_len = int(input_ids.shape[-1])
    greedy = is_greedy(temperature, top_k)
    generator = None
    if seed is not None and not greedy:
        generator = torch.Generator(device=model.device)
        generator.manual_seed(int(seed))
    sampler_ws = None
    if can_fuse_sampler(temperature, top_p, top_k):
        sampler_ws = sampler_workspace(model.vocab_size, top_k, model.device)
    if not greedy:
        emit(
            f"Sampling: temperature={temperature} top_p={top_p} "
            f"top_k={top_k} seed={seed} "
            f"({'fused' if sampler_ws is not None else 'torch'} sampler)"
        )

    cache = model.make_cache(max_len=prompt_len + max(max_tokens, 1))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_prefill = t0
    t_decode = t0
    t_end = t0
    new_tokens: list[int] = []
    hit_eos = False
    with torch.inference_mode():
        logits = model.forward(input_ids, cache=cache, logits_to_keep=1)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        graph = None
        if max_tokens > 1:
            graph = _try_decode_graph(
                model,
                cache,
                logits,
                greedy=greedy,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
                sampler_ws=sampler_ws,
                log=emit,
            )
            torch.cuda.synchronize()
        t_decode = time.perf_counter()
        eos = model.eos_token_id
        pinned = torch.empty((), dtype=torch.long, pin_memory=True)
        eos_ready = torch.cuda.Event()
        for i in range(max_tokens):
            next_t = _next_token(
                logits,
                greedy=greedy,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
                sampler_ws=sampler_ws,
            )
            pinned.copy_(next_t, non_blocking=True)
            eos_ready.record()
            if i + 1 < max_tokens:
                if graph is not None:
                    graph.token_ids.copy_(next_t.view(1, 1))
                    graph.graph.replay()
                    logits = graph.logits
                else:
                    logits = model.forward(
                        next_t.view(1, 1), cache=cache, logits_to_keep=1
                    )
            eos_ready.synchronize()
            next_id = int(pinned.item())
            new_tokens.append(next_id)
            if on_token is not None:
                on_token(next_id)
            if eos is not None and next_id == int(eos):
                hit_eos = True
                break
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    prefill_s = t_prefill - t0
    decode_s = t_end - t_decode
    n_new = len(new_tokens)
    if prompt_len and prefill_s > 0:
        emit(
            f"Prefill: {prompt_len} tok in {prefill_s:.2f}s "
            f"({prompt_len / prefill_s:.1f} tok/s)"
        )
    if n_new and decode_s > 0:
        emit(
            f"Decode:  {n_new} tok in {decode_s:.2f}s "
            f"({n_new / decode_s:.1f} tok/s)"
        )

    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    finish_reason = "stop" if hit_eos else "length"
    return GenerateResult(
        text=text,
        prompt_len=prompt_len,
        n_new=n_new,
        prefill_s=prefill_s,
        decode_s=decode_s,
        token_ids=new_tokens,
        finish_reason=finish_reason,
    )


def generate(
    model_dir: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = None,
    flash_head: bool = False,
) -> GenerateResult:
    """Load a packed checkpoint, decode, print text and tok/s."""
    model_dir = str(Path(model_dir).expanduser())
    print(f"Loading packed model from {model_dir}", flush=True)
    model, tokenizer = load_packed(model_dir, flash_head=flash_head)
    log_model_traffic(model, flash_head=flash_head)
    warmup_model(model, flash_head=flash_head)
    input_ids = encode_user_prompt(tokenizer, prompt, model.device)
    result = run_generate(
        model,
        tokenizer,
        input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    print(result.text, flush=True)
    return result
