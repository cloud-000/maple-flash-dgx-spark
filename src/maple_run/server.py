"""OpenAI-compatible HTTP server for a packed Maple checkpoint.

Stdlib only. One CUDA generate at a time. Request body may override the
CLI sampling defaults (temperature, top_p, top_k, max_tokens, seed).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from maple_run.generate import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    GenerateResult,
    encode_messages,
    encode_prompt,
    load_packed,
    log_model_traffic,
    run_generate,
    warmup_model,
)

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
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
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
    role: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict:
    delta: dict[str, str] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
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


class PackedEngine:
    """Load once, generate under a lock. ``on_text`` receives decoded deltas."""

    def __init__(self, model, tokenizer, defaults: ServerDefaults):
        self.model = model
        self.tokenizer = tokenizer
        self.defaults = defaults
        self._lock = threading.Lock()
        self._warmed = False

    def chat(
        self,
        messages: list[dict],
        sampling: Sampling,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerateResult:
        input_ids = encode_messages(self.tokenizer, messages, self.model.device)
        return self._generate(input_ids, sampling, on_text)

    def complete(
        self,
        prompt: str,
        sampling: Sampling,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerateResult:
        input_ids = encode_prompt(self.tokenizer, prompt, self.model.device)
        return self._generate(input_ids, sampling, on_text)

    def _generate(self, input_ids, sampling: Sampling, on_text) -> GenerateResult:
        prev = ""
        acc: list[int] = []

        def on_token(tid: int) -> None:
            nonlocal prev
            acc.append(tid)
            if on_text is None:
                return
            text = self.tokenizer.decode(acc, skip_special_tokens=True)
            delta = text[len(prev) :]
            prev = text
            if delta:
                on_text(delta)

        with self._lock:
            if not self._warmed:
                warmup_model(self.model, flash_head=self.defaults.flash_head)
                self._warmed = True
            return run_generate(
                self.model,
                self.tokenizer,
                input_ids,
                max_tokens=sampling.max_tokens,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                seed=sampling.seed,
                on_token=on_token if on_text is not None else None,
            )


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
            messages = parse_chat_messages(body)
            sampling = parse_sampling(body, engine.defaults)
            created = int(time.time())
            req_id = f"chatcmpl-{uuid.uuid4().hex}"
            if sampling.stream:
                self._stream_chat(messages, sampling, req_id, created)
                return
            result = engine.chat(messages, sampling)
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
            self, messages: list[dict], sampling: Sampling, req_id: str, created: int
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

                result = engine.chat(messages, sampling, on_text=on_text)
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
    )
    if engine is None:
        print(f"Loading packed model from {model_dir}", flush=True)
        model, tokenizer = load_packed(model_dir, flash_head=flash_head)
        log_model_traffic(model, flash_head=flash_head)
        engine = PackedEngine(model, tokenizer, defaults)
        warmup_model(model, flash_head=flash_head)
        engine._warmed = True
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
        f"seed={defaults.seed} flash_head={defaults.flash_head}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down", flush=True)
    finally:
        httpd.server_close()
