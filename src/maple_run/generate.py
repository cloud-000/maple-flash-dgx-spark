"""Text generation CLI helper."""

from __future__ import annotations

import json
import re
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from maple_run.kernels.sampler import (
    MAX_TOP_K,
    batched_sampler_workspace,
    fused_sample,
    fused_sample_batched,
    sampler_workspace,
)

# CLI / generate defaults. T=0 or top-k=1 is still greedy argmax.
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
    return int(max_tokens)


@dataclass
class GenerateResult:
    text: str
    prompt_len: int
    n_new: int
    prefill_s: float
    decode_s: float
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)

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
    uniform: torch.Tensor | None = None
    next_id: torch.Tensor | None = None


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


def _capture_decode_graph(
    model,
    cache,
    token_ids: torch.Tensor,
    *,
    sample: tuple | None = None,
) -> _DecodeGraph:
    """Capture ``forward(token_ids)`` on ``cache``. Caller restores KV afterwards.

    ``sample`` is ``(workspace, temperature, top_p, top_k)`` to also capture
    ``fused_sample`` so the two sampler launches sit inside the replay instead
    of paying eager launch latency between tokens. ``uniform`` is a graph input
    copied each replay so ``--seed`` stays on the same generator as eager.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    logits = None
    uniform = None
    next_id = None
    with torch.cuda.stream(stream):
        for _ in range(_GRAPH_WARMUP):
            logits = model.forward(token_ids, cache=cache, logits_to_keep=1)
            if sample is not None:
                ws, temperature, top_p, top_k = sample
                fused_sample(
                    logits[0, -1],
                    torch.rand((), device=token_ids.device),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    workspace=ws,
                )
        graph = torch.cuda.CUDAGraph()
        if sample is not None:
            ws, temperature, top_p, top_k = sample
            uniform = torch.empty((), device=token_ids.device)
            uniform.uniform_()
            with torch.cuda.graph(graph):
                logits = model.forward(token_ids, cache=cache, logits_to_keep=1)
                next_id = fused_sample(
                    logits[0, -1],
                    uniform,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    workspace=ws,
                )
        else:
            with torch.cuda.graph(graph):
                logits = model.forward(token_ids, cache=cache, logits_to_keep=1)
    torch.cuda.current_stream().wait_stream(stream)
    if logits is None:
        raise RuntimeError("CUDA graph capture did not produce logits.")
    return _DecodeGraph(
        graph=graph,
        token_ids=token_ids,
        logits=logits,
        uniform=uniform,
        next_id=next_id,
    )


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
    stop_ids: set[int],
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
        if stop_ids and ids[-1] in stop_ids:
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
    stop_ids = stop_token_ids(model)

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
        stop_ids=stop_ids,
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
        stop_ids=stop_ids,
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
    if sampler_ws is not None and not greedy:
        try:
            captured = _capture_decode_graph(
                model,
                cache,
                token_ids,
                sample=(sampler_ws, temperature, top_p, top_k),
            )
            cache.restore(snap)
            log("CUDA graph decode: fused sampler captured")
        except Exception as exc:
            cache.restore(snap)
            log(
                f"CUDA graph sampler capture failed ({exc}); "
                "using forward-only graph"
            )
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


_REPLACEMENT = "\ufffd"
_THINK_START = "<think>"
_THINK_END = "</think>"
_TOOL_START = "<tool_call>"
_TOOL_END = "</tool_call>"
_TOOL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def _as_token_id_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value if v is not None}
    return {int(value)}


def stop_token_ids(model, tokenizer=None) -> set[int]:
    """Token ids that end a response (``<|im_end|>`` and pad/EOT)."""
    ids = _as_token_id_set(getattr(model, "eos_token_id", None))
    if tokenizer is not None:
        ids |= _as_token_id_set(getattr(tokenizer, "eos_token_id", None))
        ids |= _as_token_id_set(getattr(tokenizer, "pad_token_id", None))
    return ids


def think_token_ids(tokenizer) -> tuple[int | None, int | None]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    return vocab.get(_THINK_START), vocab.get(_THINK_END)


def skip_decode_ids(tokenizer, extra=None) -> set[int]:
    """Special / control tokens that must not appear in streamed text."""
    ids = _as_token_id_set(extra)
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        ids |= _as_token_id_set(getattr(tokenizer, attr, None))
    extra_special = getattr(tokenizer, "additional_special_tokens", None) or []
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for token in extra_special:
            tid = convert(token)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
    start, end = think_token_ids(tokenizer)
    ids |= {tid for tid in (start, end) if tid is not None}
    return ids


def prompt_in_think(tokenizer, input_ids: torch.Tensor) -> bool:
    """True when the chat template left the prompt inside an open ``<think>``."""
    start, end = think_token_ids(tokenizer)
    if start is None:
        return False
    last_s = last_e = -1
    for i, tid in enumerate(input_ids.reshape(-1).tolist()):
        if tid == start:
            last_s = i
        elif end is not None and tid == end:
            last_e = i
    return last_s > last_e


def decode_holding_incomplete(tokenizer, token_ids: list[int]) -> str:
    """Decode, but hold a trailing U+FFFD so incomplete UTF-8 is not emitted."""
    if not token_ids:
        return ""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if text.endswith(_REPLACEMENT):
        text = text[:-1]
    return text


def parse_tool_call_json(body: str) -> dict | None:
    """Parse Maple ``<tool_call>`` JSON into an OpenAI tool_calls item."""
    try:
        obj = json.loads(body.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not obj.get("name"):
        return None
    args = obj.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": str(obj["name"]), "arguments": args},
    }


def split_content_and_tools(text: str) -> tuple[str, list[dict], bool]:
    """Strip ``<tool_call>`` blocks from visible text. ``in_tool`` if one is open."""
    calls: list[dict] = []
    for match in _TOOL_RE.finditer(text):
        parsed = parse_tool_call_json(match.group(1))
        if parsed is not None:
            calls.append(parsed)
    last_start = text.rfind(_TOOL_START)
    last_end = text.rfind(_TOOL_END)
    in_tool = last_start >= 0 and last_start > last_end
    cut = last_start if in_tool else None
    visible = text if cut is None else text[:cut]
    visible = _TOOL_RE.sub("", visible)
    return visible.strip("\n"), calls, in_tool


class CompletionDecoder:
    """Split Maple ``<think>`` traces and hold incomplete UTF-8 across tokens.

    Maple's chat template always opens ``<think>`` on the generation prompt, so
    the completion starts mid-thought. mlx-lm puts that span in ``reasoning``
    and the answer in ``content``; dumping both into ``content`` (and emitting
    U+FFFD on partial Qwen bytes) is what agent harnesses see as garbage.
    """

    def __init__(
        self,
        tokenizer,
        *,
        in_think: bool,
        skip_ids: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.think_start, self.think_end = think_token_ids(tokenizer)
        self.skip_ids = skip_decode_ids(tokenizer) if skip_ids is None else set(skip_ids)
        self.state = "think" if in_think else "content"
        self._think_ids: list[int] = []
        self._content_ids: list[int] = []
        self._think_text = ""
        self._content_text = ""
        self.tool_calls: list[dict] = []
        self._emitted_tools = 0
        self.should_stop = False

    def take_new_tool_calls(self) -> list[dict]:
        new = self.tool_calls[self._emitted_tools :]
        self._emitted_tools = len(self.tool_calls)
        return new

    def push(self, token_id: int) -> tuple[str, str]:
        """Feed one token. Returns ``(reasoning_delta, content_delta)``."""
        tid = int(token_id)
        if self.think_start is not None and tid == self.think_start:
            if self.tool_calls:
                self.should_stop = True
                return "", ""
            self.state = "think"
            return "", ""
        if self.think_end is not None and tid == self.think_end:
            self.state = "content"
            return "", ""
        if tid in self.skip_ids:
            return "", ""
        if self.state == "think":
            self._think_ids.append(tid)
            text = decode_holding_incomplete(self.tokenizer, self._think_ids)
            delta = text[len(self._think_text) :]
            self._think_text = text
            return delta, ""
        self._content_ids.append(tid)
        raw = decode_holding_incomplete(self.tokenizer, self._content_ids)
        visible, calls, in_tool = split_content_and_tools(raw)
        if self.tool_calls and not in_tool and visible[len(self._content_text) :].strip():
            self.should_stop = True
        self.tool_calls = calls
        delta = visible[len(self._content_text) :]
        self._content_text = visible
        return "", delta

    def finalize(self) -> tuple[str, str]:
        return self._think_text, self._content_text


def _chat_message(message: dict) -> dict:
    """Keep tool_calls / reasoning / names; flatten multimodal content to text.

    Pi sends the system prompt as ``developer`` when the model is marked
    reasoning + ``supportsDeveloperRole``. Maple's jinja only reads ``system``.
    """
    out = dict(message)
    role = str(message["role"])
    if role == "developer":
        role = "system"
    out["role"] = role
    out["content"] = message_text(message.get("content"))
    return out


def _to_input_ids(encoded, device) -> torch.Tensor:
    input_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    return input_ids.to(device).long()


def encode_messages(
    tokenizer, messages: list[dict], device, *, tools: list | None = None
) -> torch.Tensor:
    """Apply the chat template, or concatenate roles if the tokenizer has none.

    Passes ``tools`` and keeps ``tool_calls`` so agent turns match mlx-lm.
    """
    cleaned = [_chat_message(m) for m in messages]
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if tools:
            kwargs["tools"] = tools
        encoded = tokenizer.apply_chat_template(cleaned, **kwargs)
        return _to_input_ids(encoded, device)
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in cleaned)
    return _to_input_ids(
        tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt"),
        device,
    )


def describe_chat_prompt(tokenizer, input_ids: torch.Tensor, tools: list | None) -> str:
    """One-line serve log: whether system/skills made it into the tokenized prompt."""
    ids = input_ids.reshape(-1)
    n = int(ids.numel())
    head = tokenizer.decode(ids[: min(n, 96)].tolist(), skip_special_tokens=False)
    head = " ".join(head.split())
    if len(head) > 160:
        head = head[:160] + "…"
    return f"Chat prompt {n} tok tools={len(tools or [])} head={head!r}"


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


_DECODE_BUCKETS = (1, 2, 4, 8, 16, 32)


def decode_buckets(max_batch: int, buckets=_DECODE_BUCKETS) -> tuple[int, ...]:
    """Graph widths to capture for a cache of ``max_batch`` rows.

    Powers of two up to the cache width, plus the cache width itself. A batch
    that shrinks as rows finish then always has a graph within 2x of it, which
    is the point of bucketing: one graph pinned at the starting width would
    keep paying for rows that have already stopped.
    """
    widths = [w for w in buckets if w < max_batch]
    widths.append(int(max_batch))
    return tuple(sorted(set(widths)))


class BucketedDecodeGraphs:
    """One captured decode graph per batch width, replayed at the narrowest fit.

    Every width shares the same input buffers, so a step is: write the live
    rows' tokens (and uniforms), replay, read the ids back. Rows past the live
    count are padding -- they run, and they write into the cache slots above
    the live ones, which is why the caller has to keep live sequences in the
    low slots (``KVCache.swap_slots``).

    Capture needs ``cache.remap`` on: the graph bakes in a kernel that reads
    the slot -> cache-row map, so compaction can permute slots between replays
    without re-capturing.
    """

    def __init__(
        self,
        model,
        cache,
        *,
        widths: tuple[int, ...],
        sample: tuple | None = None,
        log: Callable[[str], None] = _log_print,
    ):
        if not cache.remap:
            raise ValueError("BucketedDecodeGraphs needs cache.remap enabled.")
        device = model.device
        self.max_width = max(widths)
        self.widths = tuple(sorted(widths))
        self.sample = sample
        self.token_ids = torch.zeros(self.max_width, 1, dtype=torch.long, device=device)
        self.uniform = torch.rand(self.max_width, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.next_id: dict[int, torch.Tensor] = {}
        self.logits: dict[int, torch.Tensor] = {}

        snap = cache.snapshot()
        stream = torch.cuda.Stream()
        try:
            for w in self.widths:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(_GRAPH_WARMUP):
                        self._step(model, cache, w)
                torch.cuda.current_stream().wait_stream(stream)
                cache.restore(snap)
                # Each width keeps its own pool. Sharing one across the set
                # measured no smaller (208 vs 175 MB for six widths at B=16)
                # and would tie replay safety to replay order, which here is
                # whatever order the live count happens to take.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    logits, nxt = self._step(model, cache, w)
                cache.restore(snap)
                self.graphs[w] = graph
                self.logits[w] = logits
                self.next_id[w] = nxt
        finally:
            cache.restore(snap)
        log(
            f"Bucketed decode graphs: widths {list(self.widths)}"
            + (" with fused sampler" if sample is not None else " greedy")
        )

    def _step(self, model, cache, w: int):
        logits = model.forward(self.token_ids[:w], cache=cache, logits_to_keep=1)
        last = logits[:, -1, :]
        if self.sample is None:
            nxt = last.argmax(dim=-1)
        else:
            ws, temperature, top_p, top_k = self.sample
            nxt = fused_sample_batched(
                last,
                self.uniform[:w],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                workspace=ws,
            )
        return logits, nxt

    def width_for(self, live: int) -> int:
        for w in self.widths:
            if w >= live:
                return w
        raise ValueError(f"live batch {live} exceeds widest graph {self.max_width}.")

    def replay(
        self, tokens: torch.Tensor, uniform: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Step ``tokens`` ``[live, 1]``; returns the next ids ``[live]``."""
        live = int(tokens.shape[0])
        w = self.width_for(live)
        self.token_ids[:live].copy_(tokens.reshape(live, 1))
        if live < w:
            # Padding rows decode against the free slots above the live ones.
            self.token_ids[live:w].zero_()
        if uniform is not None:
            self.uniform[:live].copy_(uniform.reshape(-1)[:live])
        self.graphs[w].replay()
        return self.next_id[w][:live]


@dataclass
class BatchedResult:
    """Per-row token ids plus the timings for an aggregate tok/s."""

    token_ids: list[list[int]]
    prompt_lens: list[int]
    finish_reasons: list[str]
    prefill_s: float
    decode_s: float
    steps: int
    # Graph width replayed at each decode step (empty when decoding eager).
    widths: list[int] = field(default_factory=list)

    @property
    def n_new(self) -> int:
        return sum(len(t) for t in self.token_ids)

    @property
    def decode_tok_s(self) -> float:
        """Aggregate: every row's tokens over the shared wall clock."""
        return self.n_new / self.decode_s if self.decode_s > 0 else 0.0


@torch.inference_mode()
def batched_generate(
    model,
    prompts,
    *,
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    stop_ids: set[int] | None = None,
    cache=None,
    graphs: bool = False,
    buckets: tuple[int, ...] | None = None,
    runner: BucketedDecodeGraphs | None = None,
    log: Callable[[str], None] = _log_print,
) -> BatchedResult:
    """Decode several sequences in one batch.

    Prefill stays batch-1 -- each prompt is run into its own cache slot via
    :meth:`KVCache.prefill_slot` -- and only decode goes wide, which is where
    the packed weights get read once for the whole batch instead of once per
    row.

    Greedy by default (``temperature <= 0`` or ``top_k == 1``); otherwise the
    batched fused sampler draws one id per row from a per-row uniform.

    With ``graphs``, decode replays a captured graph chosen by the *live* row
    count rather than the starting batch. Finished rows are swapped out of the
    low slots so the live ones stay contiguous, so a batch of 32 that is down
    to 5 live rows replays the width-8 graph instead of paying for 27 rows
    that already stopped. Replay is checked against eager first and falls back
    if it disagrees, as the single-stream path does. ``buckets`` overrides the
    captured widths; the default is :func:`decode_buckets`.
    """
    device = model.device
    rows = [
        torch.as_tensor(p, device=device, dtype=torch.long).reshape(-1)
        for p in prompts
    ]
    if not rows:
        raise ValueError("batched_generate needs at least one prompt.")
    bsz = len(rows)
    prompt_lens = [int(t.numel()) for t in rows]
    if stop_ids is None:
        stop_ids = stop_token_ids(model)
    if cache is None:
        cache = model.make_cache(
            max_len=max(prompt_lens) + max(int(max_tokens), 1), batch=bsz
        )
    elif cache.batch < bsz:
        raise ValueError(f"cache batch {cache.batch} < {bsz} prompts.")
    elif cache.max_len < max(prompt_lens) + max(int(max_tokens), 1):
        # Retired rows keep stepping in their bucket, so every slot advances
        # for the whole run and the cache has to cover the longest one. The
        # decode K/V store indexes by seqlen without a bound check.
        raise ValueError(
            f"cache max_len {cache.max_len} < longest prompt "
            f"{max(prompt_lens)} + {max_tokens} tokens."
        )

    greedy = is_greedy(temperature, top_k)
    if not greedy and not can_fuse_sampler(temperature, top_p, top_k):
        raise ValueError(
            f"batched decode needs 0 < top_k <= {MAX_TOP_K} and 0 < top_p <= 1."
        )
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pending = [0] * bsz
    for b, row in enumerate(rows):
        with cache.prefill_slot(b):
            logits = model.forward(row.view(1, -1), cache=cache, logits_to_keep=1)
        pending[b] = int(_first_batched_id(logits, greedy, temperature, top_p, top_k, gen))
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # Slots get permuted as rows finish, so the kernels have to go through the
    # slot -> cache-row map from here on.
    cache.remap = True
    sample_ws = None
    if not greedy:
        # Sized at the cache width, not the live batch: the widest captured
        # graph runs cache.batch rows, and a narrower replay slices this.
        sample_ws = batched_sampler_workspace(
            int(model.config["vocab_size"]), int(top_k), cache.batch, device
        )
    if graphs and runner is None:
        runner = _try_batched_decode_graphs(
            model,
            cache,
            pending,
            sample=None if greedy else (sample_ws, temperature, top_p, top_k),
            buckets=buckets,
            log=log,
        )

    def step(tokens: torch.Tensor) -> torch.Tensor:
        live = int(tokens.shape[0])
        u = None if greedy else torch.rand(live, device=device, generator=gen)
        if runner is not None:
            return runner.replay(tokens, u)
        out_logits = model.forward(tokens, cache=cache, logits_to_keep=1)[:, -1, :]
        if greedy:
            return out_logits.argmax(dim=-1)
        return fused_sample_batched(
            out_logits,
            u,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            workspace=sample_ws,
        )

    out: list[list[int]] = [[] for _ in range(bsz)]
    reasons = ["length"] * bsz
    owner = list(range(bsz))  # slot -> output row
    live = bsz
    steps = 0
    widths: list[int] = []
    t0 = time.perf_counter()
    for i in range(int(max_tokens)):
        finished = []
        for slot in range(live):
            b = owner[slot]
            out[b].append(pending[slot])
            if pending[slot] in stop_ids:
                reasons[b] = "stop"
                finished.append(slot)
        if i + 1 >= int(max_tokens):
            break
        # Descending, so a slot pulled down from the top is never one we have
        # already retired.
        for slot in reversed(finished):
            last = live - 1
            if slot != last:
                cache.swap_slots(slot, last)
                owner[slot], owner[last] = owner[last], owner[slot]
                pending[slot], pending[last] = pending[last], pending[slot]
            live -= 1
        if live == 0:
            break
        tokens = torch.tensor(
            pending[:live], device=device, dtype=torch.long
        ).view(live, 1)
        nxt = step(tokens)
        pending[:live] = [int(x) for x in nxt.tolist()]
        steps += 1
        if runner is not None:
            width = runner.width_for(live)
            widths.append(width)
            cache.advance_host(1, width)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    return BatchedResult(
        token_ids=out,
        prompt_lens=prompt_lens,
        finish_reasons=reasons,
        prefill_s=prefill_s,
        decode_s=decode_s,
        steps=steps,
        widths=widths,
    )


def _first_batched_id(logits, greedy, temperature, top_p, top_k, gen):
    """Pick the token that follows a prompt, from its batch-1 prefill logits."""
    last = logits[0, -1]
    if greedy:
        return last.argmax()
    return fused_sample(
        last,
        torch.rand((), device=last.device, generator=gen),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def _try_batched_decode_graphs(
    model,
    cache,
    pending: list[int],
    *,
    sample: tuple | None,
    buckets: tuple[int, ...] | None = None,
    log: Callable[[str], None] = _log_print,
) -> BucketedDecodeGraphs | None:
    """Capture bucketed graphs and keep them only if replay matches eager."""
    live = len(pending)
    widths = decode_buckets(cache.batch) if buckets is None else tuple(sorted(buckets))
    try:
        runner = BucketedDecodeGraphs(
            model, cache, widths=widths, sample=sample, log=log
        )
    except Exception as exc:
        log(f"Bucketed graph capture failed ({exc}); using eager batched decode")
        return None

    # Same check the single-stream path makes: the same few steps, eager and
    # replayed, from the same cache state. The uniforms are fixed rather than
    # redrawn so a sampled comparison is about the graph, not RNG ordering.
    snap = cache.snapshot()
    n_check = min(_GRAPH_CHECK_TOKENS, 4)
    start_tok = torch.tensor(pending, device=model.device, dtype=torch.long).view(live, 1)
    fixed_u = [
        torch.rand(live, device=model.device) if sample is not None else None
        for _ in range(n_check)
    ]

    def eager_step(tokens, u):
        last = model.forward(tokens, cache=cache, logits_to_keep=1)[:, -1, :]
        if sample is None:
            return last.argmax(dim=-1)
        ws, temperature, top_p, top_k = sample
        return fused_sample_batched(
            last, u, temperature=temperature, top_p=top_p, top_k=top_k, workspace=ws
        )

    def run(step_fn):
        ids, tok = [], start_tok
        for u in fixed_u:
            tok = step_fn(tok, u).reshape(-1, 1)
            ids.append(tok.reshape(-1).tolist())
        return ids

    eager_ids = run(eager_step)
    cache.restore(snap)
    graph_ids = run(lambda tokens, u: runner.replay(tokens, u))
    cache.restore(snap)

    if eager_ids != graph_ids:
        log(
            f"Bucketed graph replay diverged (eager {eager_ids[0][:4]} vs graph "
            f"{graph_ids[0][:4]}); using eager batched decode"
        )
        return None
    log("Bucketed decode graphs: replay matches eager ids")
    return runner


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
    on_text: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    on_tool_calls: Callable[[list[dict]], None] | None = None,
) -> GenerateResult:
    """Decode from ``input_ids`` on an already-loaded packed model."""
    def emit(msg: str) -> None:
        if log is not None:
            log(msg)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(model.device).long()
    prompt_len = int(input_ids.shape[-1])
    max_context = int(
        (getattr(model, "config", None) or {}).get(
            "max_position_embeddings", DEFAULT_MAX_CONTEXT
        )
    )
    max_tokens = resolve_max_tokens(max_tokens, prompt_len, max_context)
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
    stop_ids = stop_token_ids(model, tokenizer)
    decoder = CompletionDecoder(
        tokenizer,
        in_think=prompt_in_think(tokenizer, input_ids),
        skip_ids=skip_decode_ids(tokenizer, extra=stop_ids),
    )
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
        pinned = torch.empty((), dtype=torch.long, pin_memory=True)
        eos_ready = torch.cuda.Event()
        sample_in_graph = graph is not None and graph.next_id is not None
        next_t: torch.Tensor | None = None
        for i in range(max_tokens):
            if next_t is None:
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
                    if sample_in_graph:
                        assert graph.uniform is not None
                        graph.uniform.copy_(
                            torch.rand((), device=model.device, generator=generator)
                        )
                    graph.graph.replay()
                    logits = graph.logits
                    if sample_in_graph:
                        next_t = graph.next_id
                    else:
                        next_t = None
                else:
                    logits = model.forward(
                        next_t.view(1, 1), cache=cache, logits_to_keep=1
                    )
                    next_t = None
            eos_ready.synchronize()
            next_id = int(pinned.item())
            new_tokens.append(next_id)
            if on_token is not None:
                on_token(next_id)
            r_delta, c_delta = decoder.push(next_id)
            if r_delta and on_reasoning is not None:
                on_reasoning(r_delta)
            if c_delta and on_text is not None:
                on_text(c_delta)
            new_calls = decoder.take_new_tool_calls()
            if new_calls and on_tool_calls is not None:
                on_tool_calls(new_calls)
            if stop_ids and next_id in stop_ids:
                hit_eos = True
                break
            if decoder.should_stop:
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

    reasoning, text = decoder.finalize()
    if decoder.tool_calls:
        finish_reason = "tool_calls"
    elif hit_eos:
        finish_reason = "stop"
    else:
        finish_reason = "length"
    return GenerateResult(
        text=text,
        prompt_len=prompt_len,
        n_new=n_new,
        prefill_s=prefill_s,
        decode_s=decode_s,
        token_ids=new_tokens,
        finish_reason=finish_reason,
        reasoning=reasoning,
        tool_calls=decoder.tool_calls,
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
    if result.reasoning:
        print(f"<think>\n{result.reasoning}\n</think>\n\n{result.text}", flush=True)
    else:
        print(result.text, flush=True)
    return result
