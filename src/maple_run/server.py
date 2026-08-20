"""OpenAI-compatible HTTP server for a packed Maple checkpoint.

Stdlib HTTP. A CUDA worker continuously batches requests into KV-cache slots.
Request body may override the CLI sampling defaults (temperature, top_p,
top_k, max_tokens, seed).
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch

from maple_run.generate import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    CompletionDecoder,
    GenerateResult,
    _first_batched_id,
    _try_batched_decode_graphs,
    can_fuse_sampler,
    describe_chat_prompt,
    encode_messages,
    encode_prompt,
    is_greedy,
    load_packed,
    log_model_traffic,
    prompt_in_think,
    resolve_max_tokens,
    sample_next,
    skip_decode_ids,
    stop_token_ids,
    warmup_model,
)
from maple_run.kernels.sampler import (
    MAX_TOP_K,
    batched_sampler_workspace,
    fused_sample_batched,
)

DEFAULT_MAX_BATCH = 8
DEFAULT_MAX_LEN = 8192

_MAX_BODY = 16 * 1024 * 1024


class RequestError(ValueError):
    """Invalid OpenAI-style request. ``status`` is the HTTP status to return."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ServerDefaults:
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_tokens: int = 128
    seed: int | None = None
    flash_head: bool = False
    model_id: str = "maple-2bit"
    max_batch: int = DEFAULT_MAX_BATCH
    max_len: int = DEFAULT_MAX_LEN


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int | None
    stream: bool
    model: str


def error_body(message: str, err_type: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": err_type, "code": None}}


def _as_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{name} must be a number")
    return float(value)


def _as_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise RequestError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise RequestError(f"{name} must be an integer")


def parse_sampling(body: dict, defaults: ServerDefaults) -> Sampling:
    """Merge an OpenAI request body with CLI sampling defaults."""
    if not isinstance(body, dict):
        raise RequestError("JSON body must be an object")
    n = body.get("n", 1)
    if n != 1:
        raise RequestError("only n=1 is supported")

    temperature = (
        _as_float("temperature", body["temperature"])
        if "temperature" in body
        else defaults.temperature
    )
    top_p = _as_float("top_p", body["top_p"]) if "top_p" in body else defaults.top_p
    top_k = _as_int("top_k", body["top_k"]) if "top_k" in body else defaults.top_k
    if "max_completion_tokens" in body:
        max_tokens = _as_int("max_completion_tokens", body["max_completion_tokens"])
    elif "max_tokens" in body:
        max_tokens = _as_int("max_tokens", body["max_tokens"])
    else:
        max_tokens = defaults.max_tokens
    seed = _as_int("seed", body["seed"]) if "seed" in body else defaults.seed
    stream = bool(body.get("stream", False))
    model = str(body.get("model") or defaults.model_id)

    if temperature < 0:
        raise RequestError("temperature must be >= 0")
    if top_p <= 0 or top_p > 1:
        raise RequestError("top_p must be in (0, 1]")
    if top_k < 0:
        raise RequestError("top_k must be >= 0")
    if max_tokens < -1:
        raise RequestError("max_tokens must be >= 0 or -1")
    return Sampling(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stream=stream,
        model=model,
    )


def parse_chat_messages(body: dict) -> list[dict]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    out: list[dict] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg:
            raise RequestError(f"messages[{i}] must be an object with a role")
        out.append(msg)
    return out


def parse_tools(body: dict) -> list | None:
    if "tools" not in body or body["tools"] is None:
        return None
    tools = body["tools"]
    if not isinstance(tools, list):
        raise RequestError("tools must be an array")
    return tools


def apply_system_prompt(body: dict, messages: list[dict]) -> list[dict]:
    """Honor a top-level ``system`` field if no system/developer message exists."""
    system = body.get("system")
    if not isinstance(system, str) or not system:
        return messages
    if any(str(m.get("role")) in ("system", "developer") for m in messages):
        return messages
    return [{"role": "system", "content": system}, *messages]


def parse_completion_prompt(body: dict) -> str:
    prompt = body.get("prompt")
    if prompt is None:
        raise RequestError("prompt is required")
    if isinstance(prompt, list):
        if not all(isinstance(p, str) for p in prompt):
            raise RequestError("prompt array must contain strings")
        return "".join(prompt)
    if not isinstance(prompt, str):
        raise RequestError("prompt must be a string")
    return prompt


def usage_dict(result: GenerateResult) -> dict[str, int]:
    return {
        "prompt_tokens": result.prompt_len,
        "completion_tokens": result.n_new,
        "total_tokens": result.prompt_len + result.n_new,
    }


def chat_completion_response(
    result: GenerateResult, *, model: str, created: int, req_id: str
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": result.text or None}
    if result.reasoning:
        message["reasoning"] = result.reasoning
        message["reasoning_content"] = result.reasoning
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": usage_dict(result),
    }


def text_completion_response(
    result: GenerateResult, *, model: str, created: int, req_id: str
) -> dict:
    return {
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": result.text,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": usage_dict(result),
    }


def chat_chunk(
    *,
    req_id: str,
    created: int,
    model: str,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[dict] | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
        delta["reasoning_content"] = reasoning
    if tool_calls:
        delta["tool_calls"] = tool_calls
    chunk: dict[str, Any] = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def text_chunk(
    *,
    req_id: str,
    created: int,
    model: str,
    text: str = "",
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict:
    chunk: dict[str, Any] = {
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def models_list(model_id: str) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "maple-run",
            }
        ],
    }


@dataclass
class _Job:
    input_ids: Any
    sampling: Sampling
    on_text: Callable[[str], None] | None = None
    on_reasoning: Callable[[str], None] | None = None
    on_tool_calls: Callable[[list[dict]], None] | None = None
    stop_ids: set[int] | None = None
    done: threading.Event = field(default_factory=threading.Event)
    result: GenerateResult | None = None
    error: BaseException | None = None


@dataclass
class _Slot:
    job: _Job
    pending: int
    max_tokens: int
    prompt_len: int
    prefill_s: float
    decode_t0: float
    token_ids: list[int] = field(default_factory=list)
    decoder: CompletionDecoder | None = None
    stop_ids: set[int] = field(default_factory=set)
    generator: Any = None
    finish_reason: str = "length"


def _log(msg: str) -> None:
    print(msg, flush=True)


class PackedEngine:
    """Load once; a CUDA worker admits requests into persistent cache slots."""

    def __init__(self, model, tokenizer, defaults: ServerDefaults):
        self.model = model
        self.tokenizer = tokenizer
        self.defaults = defaults
        self.max_batch = int(defaults.max_batch)
        self.max_len = int(defaults.max_len)
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._start_error: BaseException | None = None
        self._live = 0
        self._slots: list[_Slot | None] = [None] * self.max_batch
        self._cache = None
        self._runner = None
        self._sample_ws = None
        self._worker = threading.Thread(
            target=self._run, name="maple-batch", daemon=True
        )
        self._worker.start()

    def wait_ready(self) -> None:
        self._ready.wait()
        if self._start_error is not None:
            raise self._start_error

    def close(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=30)

    def chat(
        self,
        messages: list[dict],
        sampling: Sampling,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_tool_calls: Callable[[list[dict]], None] | None = None,
        tools: list | None = None,
    ) -> GenerateResult:
        if self.tokenizer is None:
            raise RuntimeError("PackedEngine.chat needs a tokenizer")
        input_ids = encode_messages(
            self.tokenizer, messages, self.model.device, tools=tools
        )
        print(describe_chat_prompt(self.tokenizer, input_ids, tools), flush=True)
        return self._submit(
            input_ids, sampling, on_text, on_reasoning, on_tool_calls
        )

    def complete(
        self,
        prompt: str,
        sampling: Sampling,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerateResult:
        if self.tokenizer is None:
            raise RuntimeError("PackedEngine.complete needs a tokenizer")
        input_ids = encode_prompt(self.tokenizer, prompt, self.model.device)
        return self._submit(input_ids, sampling, on_text)

    def run(
        self,
        input_ids,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        seed: int | None = None,
        stop_ids: set[int] | None = None,
    ) -> GenerateResult:
        """Decode ``input_ids`` on the worker. Used by tests; no tokenizer needed."""
        sampling = Sampling(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            stream=False,
            model=self.defaults.model_id,
        )
        return self._submit(input_ids, sampling, stop_ids=stop_ids)

    def _submit(
        self,
        input_ids,
        sampling: Sampling,
        on_text=None,
        on_reasoning=None,
        on_tool_calls=None,
        stop_ids: set[int] | None = None,
    ) -> GenerateResult:
        self.wait_ready()
        job = _Job(
            input_ids=input_ids,
            sampling=sampling,
            on_text=on_text,
            on_reasoning=on_reasoning,
            on_tool_calls=on_tool_calls,
            stop_ids=stop_ids,
        )
        self._queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return job.result

    def _run(self) -> None:
        try:
            self._setup()
        except Exception as exc:
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            with torch.inference_mode():
                while True:
                    if not self._admit():
                        break
                    if self._live == 0:
                        continue
                    self._emit_and_retire()
                    if self._live == 0:
                        continue
                    if self._stop.is_set():
                        self._drain_stop()
                        break
                    self._decode_step()
        except Exception as exc:
            self._fail_all(exc)

    def _setup(self) -> None:
        model = self.model
        device = getattr(model, "device", None)
        if (
            torch.cuda.is_available()
            and isinstance(device, torch.device)
            and device.type == "cuda"
            and device.index is not None
        ):
            torch.cuda.set_device(device)
        warmup_model(model, flash_head=False)
        flash = getattr(model, "lm_head_flash", None)
        if flash is not None:
            model.lm_head_flash = None
            _log("continuous batch uses exact lm_head (FlashHead is B=1 only)")
        self._cache = model.make_cache(max_len=self.max_len, batch=self.max_batch)
        dummy = torch.zeros(1, dtype=torch.long, device=model.device)
        pending: list[int] = []
        for b in range(self.max_batch):
            with self._cache.prefill_slot(b):
                logits = model.forward(
                    dummy.view(1, -1), cache=self._cache, logits_to_keep=1
                )
            pending.append(int(logits[0, -1].argmax()))
        self._cache.remap = True
        self._runner = _try_batched_decode_graphs(
            model, self._cache, pending, sample=None, log=_log
        )
        for b in range(self.max_batch):
            self._cache.reset_slot(b)
        self._sample_ws = batched_sampler_workspace(
            int(model.config["vocab_size"]),
            MAX_TOP_K,
            self.max_batch,
            model.device,
        )

    def _admit(self) -> bool:
        """Fill free slots from the queue. False means the worker should exit."""
        while self._live < self.max_batch:
            if self._stop.is_set():
                self._drain_stop()
                return False
            if self._live == 0:
                job = self._queue.get()
            else:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    return True
            if job is None:
                self._drain_stop()
                return False
            self._admit_job(job)
        return True

    def _drain_stop(self) -> None:
        while True:
            try:
                leftover = self._queue.get_nowait()
            except queue.Empty:
                break
            if leftover is not None:
                leftover.error = RuntimeError("engine stopped")
                leftover.done.set()
        for slot in range(self._live):
            row = self._slots[slot]
            if row is not None:
                self._finish_slot(slot, error=RuntimeError("engine stopped"))
        self._live = 0

    def _admit_job(self, job: _Job) -> None:
        ids = torch.as_tensor(
            job.input_ids, device=self.model.device, dtype=torch.long
        ).reshape(-1)
        prompt_len = int(ids.numel())
        if prompt_len >= self.max_len:
            job.error = RequestError(
                f"prompt length {prompt_len} exceeds max_len {self.max_len}"
            )
            job.done.set()
            return
        max_tokens = resolve_max_tokens(job.sampling.max_tokens, prompt_len, self.max_len)
        if max_tokens <= 0:
            job.result = GenerateResult(
                text="",
                prompt_len=prompt_len,
                n_new=0,
                prefill_s=0.0,
                decode_s=0.0,
                finish_reason="length",
            )
            job.done.set()
            return
        slot = self._live
        self._cache.reset_slot(slot)
        t0 = time.perf_counter()
        with self._cache.prefill_slot(slot):
            logits = self.model.forward(
                ids.view(1, -1), cache=self._cache, logits_to_keep=1
            )
        prefill_s = time.perf_counter() - t0
        sampling = job.sampling
        gen = None
        if sampling.seed is not None and not is_greedy(
            sampling.temperature, sampling.top_k
        ):
            gen = torch.Generator(device=self.model.device)
            gen.manual_seed(int(sampling.seed))
        pending = int(
            self._first_id(logits, sampling, gen)
        )
        stop = (
            job.stop_ids
            if job.stop_ids is not None
            else stop_token_ids(self.model, self.tokenizer)
        )
        decoder = None
        if self.tokenizer is not None:
            decoder = CompletionDecoder(
                self.tokenizer,
                in_think=prompt_in_think(self.tokenizer, ids),
                skip_ids=skip_decode_ids(self.tokenizer, extra=stop),
            )
        self._slots[slot] = _Slot(
            job=job,
            pending=pending,
            max_tokens=max_tokens,
            prompt_len=prompt_len,
            prefill_s=prefill_s,
            decode_t0=time.perf_counter(),
            decoder=decoder,
            stop_ids=stop,
            generator=gen,
        )
        self._live += 1

    def _first_id(self, logits, sampling: Sampling, gen):
        greedy = is_greedy(sampling.temperature, sampling.top_k)
        if greedy or can_fuse_sampler(
            sampling.temperature, sampling.top_p, sampling.top_k
        ):
            return _first_batched_id(
                logits,
                greedy,
                sampling.temperature,
                sampling.top_p,
                sampling.top_k,
                gen,
            )
        last = logits[0, -1]
        return sample_next(
            last,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            generator=gen,
        )

    def _emit_and_retire(self) -> None:
        finished: list[int] = []
        for slot in range(self._live):
            row = self._slots[slot]
            assert row is not None
            tok = row.pending
            row.token_ids.append(tok)
            try:
                if row.decoder is not None:
                    r_delta, c_delta = row.decoder.push(tok)
                    if r_delta and row.job.on_reasoning is not None:
                        row.job.on_reasoning(r_delta)
                    if c_delta and row.job.on_text is not None:
                        row.job.on_text(c_delta)
                    new_calls = row.decoder.take_new_tool_calls()
                    if new_calls and row.job.on_tool_calls is not None:
                        row.job.on_tool_calls(new_calls)
            except Exception as exc:
                row.job.error = exc
                finished.append(slot)
                continue
            if tok in row.stop_ids:
                row.finish_reason = "stop"
                finished.append(slot)
            elif row.decoder is not None and row.decoder.should_stop:
                row.finish_reason = (
                    "tool_calls" if row.decoder.tool_calls else "stop"
                )
                finished.append(slot)
            elif len(row.token_ids) >= row.max_tokens:
                row.finish_reason = "length"
                finished.append(slot)
            elif self._cache.seen[slot] >= self.max_len:
                # Last KV slot is full; another decode would index RoPE/KV OOB.
                row.finish_reason = "length"
                finished.append(slot)
        for slot in reversed(finished):
            last = self._live - 1
            if slot != last:
                self._cache.swap_slots(slot, last)
                self._slots[slot], self._slots[last] = (
                    self._slots[last],
                    self._slots[slot],
                )
            self._finish_slot(last)
            self._cache.reset_slot(last)
            self._slots[last] = None
            self._live -= 1

    def _finish_slot(
        self, slot: int, error: BaseException | None = None
    ) -> None:
        row = self._slots[slot]
        if row is None:
            return
        job = row.job
        if error is not None and job.error is None:
            job.error = error
        if job.result is None and job.error is None:
            reasoning = text = ""
            finish = row.finish_reason
            if row.decoder is not None:
                reasoning, text = row.decoder.finalize()
                if row.decoder.tool_calls:
                    finish = "tool_calls"
            decode_s = max(time.perf_counter() - row.decode_t0, 0.0)
            job.result = GenerateResult(
                text=text,
                prompt_len=row.prompt_len,
                n_new=len(row.token_ids),
                prefill_s=row.prefill_s,
                decode_s=decode_s,
                token_ids=list(row.token_ids),
                finish_reason=finish,
                reasoning=reasoning,
                tool_calls=list(row.decoder.tool_calls) if row.decoder else [],
            )
        job.done.set()

    def _decode_step(self) -> None:
        live = self._live
        device = self.model.device
        tokens = torch.tensor(
            [self._slots[s].pending for s in range(live)],
            device=device,
            dtype=torch.long,
        ).view(live, 1)
        if self._runner is not None:
            logits = self._runner.replay_logits(tokens)
            width = self._runner.width_for(live)
            self._cache.advance_host(1, live)
            for s in range(live, width):
                self._cache.reset_slot(s)
        else:
            logits = self.model.forward(
                tokens, cache=self._cache, logits_to_keep=1
            )[:, -1, :]
        nxt = self._sample_rows(logits)
        for s, tok in enumerate(nxt):
            self._slots[s].pending = int(tok)

    def _sample_rows(self, logits) -> list[int]:
        live = logits.shape[0]
        nxt = torch.empty(live, dtype=torch.long, device=logits.device)
        greedy_idx: list[int] = []
        groups: dict[tuple[float, float, int], list[int]] = {}
        for s in range(live):
            row = self._slots[s]
            assert row is not None
            sampling = row.job.sampling
            if is_greedy(sampling.temperature, sampling.top_k):
                greedy_idx.append(s)
            else:
                key = (sampling.temperature, sampling.top_p, sampling.top_k)
                groups.setdefault(key, []).append(s)
        if greedy_idx:
            idx = torch.tensor(greedy_idx, device=logits.device, dtype=torch.long)
            nxt[idx] = logits[idx].argmax(dim=-1)
        for (temperature, top_p, top_k), idxs in groups.items():
            idx = torch.tensor(idxs, device=logits.device, dtype=torch.long)
            rows = logits[idx]
            gens = [self._slots[i].generator for i in idxs]
            u = torch.empty(len(idxs), device=logits.device)
            for n, gen in enumerate(gens):
                u[n] = torch.rand((), device=logits.device, generator=gen)
            if can_fuse_sampler(temperature, top_p, top_k):
                sampled = fused_sample_batched(
                    rows,
                    u,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    workspace=self._sample_ws,
                )
            else:
                sampled = torch.stack(
                    [
                        sample_next(
                            rows[n],
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            generator=gens[n],
                        )
                        for n in range(len(idxs))
                    ]
                )
            nxt[idx] = sampled
        return [int(x) for x in nxt.tolist()]

    def _fail_all(self, exc: BaseException) -> None:
        for slot in range(self._live):
            self._finish_slot(slot, error=exc)
        self._live = 0
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job.error = exc
                job.done.set()


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def make_handler(engine: PackedEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
                return
            if path == "/v1/models":
                self._json(200, models_list(engine.defaults.model_id))
                return
            if path.startswith("/v1/models/"):
                model_id = path[len("/v1/models/") :]
                if model_id == engine.defaults.model_id:
                    self._json(200, models_list(model_id)["data"][0])
                    return
                self._json(404, error_body(f"model {model_id} not found", "not_found_error"))
                return
            self._json(404, error_body(f"unknown path {path}", "not_found_error"))

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                body = self._read_json()
                if path == "/v1/chat/completions":
                    self._chat(body)
                    return
                if path == "/v1/completions":
                    self._completions(body)
                    return
            except RequestError as exc:
                self._json(exc.status, error_body(str(exc)))
                return
            except Exception as exc:
                self._json(500, error_body(str(exc), "internal_error"))
                return
            self._json(404, error_body(f"unknown path {path}", "not_found_error"))

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY:
                raise RequestError("request body too large", 413)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestError(f"invalid JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise RequestError("JSON body must be an object")
            return body

        def _chat(self, body: dict) -> None:
            messages = apply_system_prompt(body, parse_chat_messages(body))
            sampling = parse_sampling(body, engine.defaults)
            tools = parse_tools(body)
            created = int(time.time())
            req_id = f"chatcmpl-{uuid.uuid4().hex}"
            if sampling.stream:
                self._stream_chat(messages, sampling, req_id, created, tools=tools)
                return
            result = engine.chat(messages, sampling, tools=tools)
            self._json(
                200,
                chat_completion_response(
                    result, model=sampling.model, created=created, req_id=req_id
                ),
            )

        def _completions(self, body: dict) -> None:
            prompt = parse_completion_prompt(body)
            sampling = parse_sampling(body, engine.defaults)
            created = int(time.time())
            req_id = f"cmpl-{uuid.uuid4().hex}"
            if sampling.stream:
                self._stream_text(prompt, sampling, req_id, created)
                return
            result = engine.complete(prompt, sampling)
            self._json(
                200,
                text_completion_response(
                    result, model=sampling.model, created=created, req_id=req_id
                ),
            )

        def _stream_headers(self) -> None:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

        def _write_sse(self, payload: dict) -> None:
            self.wfile.write(_sse(payload))
            self.wfile.flush()

        def _stream_chat(
            self,
            messages: list[dict],
            sampling: Sampling,
            req_id: str,
            created: int,
            tools: list | None = None,
        ) -> None:
            self._stream_headers()
            try:
                self._write_sse(
                    chat_chunk(
                        req_id=req_id,
                        created=created,
                        model=sampling.model,
                        role="assistant",
                    )
                )

                def on_text(delta: str) -> None:
                    self._write_sse(
                        chat_chunk(
                            req_id=req_id,
                            created=created,
                            model=sampling.model,
                            content=delta,
                        )
                    )

                def on_reasoning(delta: str) -> None:
                    self._write_sse(
                        chat_chunk(
                            req_id=req_id,
                            created=created,
                            model=sampling.model,
                            reasoning=delta,
                        )
                    )

                tool_index = 0

                def on_tool_calls(calls: list[dict]) -> None:
                    nonlocal tool_index
                    indexed = []
                    for call in calls:
                        item = dict(call)
                        item["index"] = tool_index
                        tool_index += 1
                        indexed.append(item)
                    self._write_sse(
                        chat_chunk(
                            req_id=req_id,
                            created=created,
                            model=sampling.model,
                            tool_calls=indexed,
                        )
                    )

                result = engine.chat(
                    messages,
                    sampling,
                    on_text=on_text,
                    on_reasoning=on_reasoning,
                    on_tool_calls=on_tool_calls,
                    tools=tools,
                )
                self._write_sse(
                    chat_chunk(
                        req_id=req_id,
                        created=created,
                        model=sampling.model,
                        finish_reason=result.finish_reason,
                        usage=usage_dict(result),
                    )
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except Exception as exc:
                try:
                    self._write_sse(error_body(str(exc), "internal_error"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

        def _stream_text(
            self, prompt: str, sampling: Sampling, req_id: str, created: int
        ) -> None:
            self._stream_headers()
            try:
                def on_text(delta: str) -> None:
                    self._write_sse(
                        text_chunk(
                            req_id=req_id,
                            created=created,
                            model=sampling.model,
                            text=delta,
                        )
                    )

                result = engine.complete(prompt, sampling, on_text=on_text)
                self._write_sse(
                    text_chunk(
                        req_id=req_id,
                        created=created,
                        model=sampling.model,
                        finish_reason=result.finish_reason,
                        usage=usage_dict(result),
                    )
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except Exception as exc:
                try:
                    self._write_sse(error_body(str(exc), "internal_error"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)

    return Handler


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(
    model_dir: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = 128,
    seed: int | None = None,
    flash_head: bool = False,
    max_batch: int = DEFAULT_MAX_BATCH,
    max_len: int = DEFAULT_MAX_LEN,
    engine: PackedEngine | None = None,
) -> None:
    """Load the packed model (unless ``engine`` is given) and serve forever."""
    model_dir = str(Path(model_dir).expanduser())
    defaults = ServerDefaults(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        flash_head=flash_head,
        model_id=Path(model_dir).name,
        max_batch=max_batch,
        max_len=max_len,
    )
    if engine is None:
        print(f"Loading packed model from {model_dir}", flush=True)
        model, tokenizer = load_packed(model_dir, flash_head=flash_head)
        log_model_traffic(model, flash_head=flash_head)
        engine = PackedEngine(model, tokenizer, defaults)
        engine.wait_ready()
    httpd = _Server((host, port), make_handler(engine))
    actual_host, actual_port = httpd.server_address[:2]
    print(
        f"OpenAI-compatible endpoint at http://{actual_host}:{actual_port}/v1 "
        f"(chat/completions, completions, models)",
        flush=True,
    )
    print(
        f"Defaults: temperature={defaults.temperature} top_p={defaults.top_p} "
        f"top_k={defaults.top_k} max_tokens={defaults.max_tokens} "
        f"seed={defaults.seed} flash_head={defaults.flash_head} "
        f"max_batch={defaults.max_batch} max_len={defaults.max_len}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down", flush=True)
    finally:
        httpd.server_close()
        if isinstance(engine, PackedEngine):
            engine.close()
