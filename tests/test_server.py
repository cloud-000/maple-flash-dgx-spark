"""OpenAI-compatible HTTP server: request parsing, response shape, CLI flags."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from maple_run.cli import main
from maple_run.generate import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    GenerateResult,
    message_text,
)
from maple_run.server import (
    RequestError,
    ServerDefaults,
    apply_system_prompt,
    chat_completion_response,
    make_handler,
    parse_chat_messages,
    parse_completion_prompt,
    parse_sampling,
    parse_tools,
    text_completion_response,
)

torch = pytest.importorskip("torch")


def test_message_text_flattens_parts():
    assert message_text("hello") == "hello"
    assert message_text(None) == ""
    assert (
        message_text([{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}])
        == "abc"
    )
    with pytest.raises(ValueError, match="text"):
        message_text([{"type": "image_url", "image_url": {"url": "x"}}])


def test_parse_sampling_uses_defaults():
    defaults = ServerDefaults()
    s = parse_sampling({"messages": [{"role": "user", "content": "hi"}]}, defaults)
    assert s.temperature == DEFAULT_TEMPERATURE
    assert s.top_p == DEFAULT_TOP_P
    assert s.top_k == DEFAULT_TOP_K
    assert s.max_tokens == 128
    assert s.seed is None
    assert s.stream is False


def test_parse_sampling_overrides_and_max_completion_tokens():
    defaults = ServerDefaults(temperature=1.0, top_p=0.95, top_k=20, max_tokens=128)
    s = parse_sampling(
        {
            "temperature": 0,
            "top_p": 0.5,
            "top_k": 1,
            "max_completion_tokens": 32,
            "seed": 7,
            "stream": True,
            "model": "maple",
        },
        defaults,
    )
    assert s.temperature == 0.0
    assert s.top_p == 0.5
    assert s.top_k == 1
    assert s.max_tokens == 32
    assert s.seed == 7
    assert s.stream is True
    assert s.model == "maple"


def test_parse_sampling_rejects_bad_values():
    defaults = ServerDefaults()
    with pytest.raises(RequestError, match="temperature"):
        parse_sampling({"temperature": -1}, defaults)
    with pytest.raises(RequestError, match="top_p"):
        parse_sampling({"top_p": 1.5}, defaults)
    with pytest.raises(RequestError, match="top_k"):
        parse_sampling({"top_k": -2}, defaults)
    with pytest.raises(RequestError, match="n=1"):
        parse_sampling({"n": 2}, defaults)
    with pytest.raises(RequestError, match="max_tokens"):
        parse_sampling({"max_tokens": -2}, defaults)


def test_parse_sampling_accepts_max_tokens_minus_one():
    defaults = ServerDefaults()
    s = parse_sampling({"max_tokens": -1}, defaults)
    assert s.max_tokens == -1
    s = parse_sampling({"max_completion_tokens": -1}, defaults)
    assert s.max_tokens == -1


def test_parse_chat_and_prompt():
    messages = parse_chat_messages(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert messages[0]["role"] == "user"
    with pytest.raises(RequestError, match="messages"):
        parse_chat_messages({})
    assert parse_completion_prompt({"prompt": "abc"}) == "abc"
    assert parse_completion_prompt({"prompt": ["a", "b"]}) == "ab"
    with pytest.raises(RequestError, match="prompt"):
        parse_completion_prompt({})


def test_parse_tools():
    assert parse_tools({}) is None
    assert parse_tools({"tools": None}) is None
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    assert parse_tools({"tools": tools}) == tools
    with pytest.raises(RequestError, match="tools"):
        parse_tools({"tools": {"name": "x"}})


def test_apply_system_prompt_prepends_when_missing():
    messages = [{"role": "user", "content": "hi"}]
    out = apply_system_prompt({"system": "You are Bob"}, messages)
    assert out[0] == {"role": "system", "content": "You are Bob"}
    assert out[1]["role"] == "user"
    kept = apply_system_prompt(
        {"system": "ignored"},
        [{"role": "developer", "content": "already here"}, {"role": "user", "content": "hi"}],
    )
    assert kept[0]["content"] == "already here"


def test_openai_response_shape():
    result = GenerateResult(
        text="Paris",
        prompt_len=10,
        n_new=4,
        prefill_s=0.1,
        decode_s=0.2,
        token_ids=[1, 2, 3, 4],
        finish_reason="stop",
    )
    chat = chat_completion_response(
        result, model="maple-2bit", created=1, req_id="chatcmpl-x"
    )
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"]["content"] == "Paris"
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert "reasoning" not in chat["choices"][0]["message"]
    assert chat["usage"]["total_tokens"] == 14
    text = text_completion_response(
        result, model="maple-2bit", created=1, req_id="cmpl-x"
    )
    assert text["object"] == "text_completion"
    assert text["choices"][0]["text"] == "Paris"


def test_openai_response_includes_reasoning():
    result = GenerateResult(
        text="Paris",
        prompt_len=10,
        n_new=8,
        prefill_s=0.1,
        decode_s=0.2,
        token_ids=[1, 2, 3],
        finish_reason="stop",
        reasoning="the capital of France",
    )
    chat = chat_completion_response(
        result, model="maple-2bit", created=1, req_id="chatcmpl-x"
    )
    message = chat["choices"][0]["message"]
    assert message["content"] == "Paris"
    assert message["reasoning"] == "the capital of France"
    assert message["reasoning_content"] == "the capital of France"


def test_openai_response_includes_tool_calls():
    result = GenerateResult(
        text="",
        prompt_len=10,
        n_new=8,
        prefill_s=0.1,
        decode_s=0.2,
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": "{\"path\": \"x\"}"},
            }
        ],
    )
    chat = chat_completion_response(
        result, model="maple-2bit", created=1, req_id="chatcmpl-x"
    )
    message = chat["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "read"
    assert chat["choices"][0]["finish_reason"] == "tool_calls"


def test_cli_serve_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--host" in out
    assert "--port" in out
    assert "default 127.0.0.1" in out
    assert "default 8000" in out
    assert "default 1.0" in out
    assert "default 0.95" in out
    assert "default 20" in out
    assert "--flash-head" in out


def test_cli_serve_rejects_bad_sampling():
    with pytest.raises(SystemExit):
        main(["serve", "--model", "x", "--temperature", "-1"])
    with pytest.raises(SystemExit):
        main(["serve", "--model", "x", "--port", "99999"])


@dataclass
class FakeEngine:
    defaults: ServerDefaults = field(
        default_factory=lambda: ServerDefaults(model_id="maple-2bit")
    )
    text: str = "Hello from maple"
    reasoning: str = ""
    last_sampling: object | None = None
    last_tools: object | None = None
    last_messages: object | None = None

    tool_calls: list = field(default_factory=list)

    def chat(
        self,
        messages,
        sampling,
        on_text=None,
        on_reasoning=None,
        on_tool_calls=None,
        tools=None,
    ):
        self.last_sampling = sampling
        self.last_tools = tools
        self.last_messages = messages
        if on_reasoning is not None and self.reasoning:
            on_reasoning(self.reasoning)
        if on_text is not None:
            on_text(self.text)
        if on_tool_calls is not None and self.tool_calls:
            on_tool_calls(self.tool_calls)
        return GenerateResult(
            text=self.text,
            prompt_len=5,
            n_new=3,
            prefill_s=0.01,
            decode_s=0.02,
            token_ids=[1, 2, 3],
            finish_reason="tool_calls" if self.tool_calls else "stop",
            reasoning=self.reasoning,
            tool_calls=self.tool_calls,
        )

    def complete(self, prompt, sampling, on_text=None):
        self.last_sampling = sampling
        if on_text is not None:
            on_text(self.text)
        return GenerateResult(
            text=self.text,
            prompt_len=2,
            n_new=3,
            prefill_s=0.01,
            decode_s=0.02,
            token_ids=[1, 2, 3],
            finish_reason="length",
        )


@contextmanager
def _server(engine):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _post(port: int, path: str, body: dict) -> tuple[int, dict, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    raw = json.dumps(body).encode()
    conn.request(
        "POST",
        path,
        body=raw,
        headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
    )
    resp = conn.getresponse()
    payload = json.loads(resp.read().decode())
    headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, payload, headers


def _get(port: int, path: str) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    payload = json.loads(resp.read().decode())
    conn.close()
    return resp.status, payload


def test_http_chat_completions_and_models():
    engine = FakeEngine()
    with _server(engine) as port:
        status, body, headers = _post(
            port,
            "/v1/chat/completions",
            {
                "model": "maple-2bit",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 8,
                "max_tokens": 16,
            },
        )
        assert status == 200
        assert headers.get("access-control-allow-origin") == "*"
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello from maple"
        assert body["usage"]["completion_tokens"] == 3
        assert engine.last_sampling.temperature == 0.2
        assert engine.last_sampling.top_p == 0.8
        assert engine.last_sampling.top_k == 8
        assert engine.last_sampling.max_tokens == 16

        status, models = _get(port, "/v1/models")
        assert status == 200
        assert models["data"][0]["id"] == "maple-2bit"
        status, health = _get(port, "/health")
        assert status == 200
        assert health["status"] == "ok"


def test_http_completions_and_validation():
    engine = FakeEngine()
    with _server(engine) as port:
        status, body, _ = _post(
            port,
            "/v1/completions",
            {"prompt": "Once upon a time", "max_tokens": 8},
        )
        assert status == 200
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "Hello from maple"
        status, err, _ = _post(
            port,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "temperature": -1},
        )
        assert status == 400
        assert "temperature" in err["error"]["message"]
        status, body, _ = _post(
            port,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": -1,
            },
        )
        assert status == 200
        assert engine.last_sampling.max_tokens == -1
        status, err, _ = _post(
            port,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": -2,
            },
        )
        assert status == 400
        assert "max_tokens" in err["error"]["message"]


def test_http_chat_stream():
    engine = FakeEngine()
    with _server(engine) as port:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        raw = json.dumps(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        ).encode()
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(raw)),
            },
        )
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/event-stream" in resp.getheader("Content-Type")
        payload = resp.read().decode()
        conn.close()
        assert "Hello from maple" in payload
        assert "data: [DONE]" in payload
        assert "chat.completion.chunk" in payload


def test_http_chat_forwards_tools_and_keeps_tool_calls():
    engine = FakeEngine()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [
        {"role": "user", "content": "read foo"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\": \"foo\"}"},
                }
            ],
        },
        {"role": "tool", "content": "file contents"},
    ]
    with _server(engine) as port:
        status, body, _ = _post(
            port,
            "/v1/chat/completions",
            {"messages": messages, "tools": tools},
        )
        assert status == 200
        assert engine.last_tools == tools
        assert engine.last_messages[1]["tool_calls"][0]["function"]["name"] == "read_file"


def test_http_chat_stream_reasoning():
    engine = FakeEngine(text="Paris", reasoning="capital city")
    with _server(engine) as port:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        raw = json.dumps(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        ).encode()
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(raw)),
            },
        )
        resp = conn.getresponse()
        payload = resp.read().decode()
        conn.close()
        assert '"reasoning": "capital city"' in payload
        assert '"reasoning_content": "capital city"' in payload
        assert '"content": "Paris"' in payload


def test_http_chat_emits_openai_tool_calls():
    engine = FakeEngine(
        text="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": "{\"path\": \"x\"}"},
            }
        ],
    )
    with _server(engine) as port:
        status, body, _ = _post(
            port,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "read x"}]},
        )
        assert status == 200
        message = body["choices"][0]["message"]
        assert message["tool_calls"][0]["function"]["name"] == "read"
        assert body["choices"][0]["finish_reason"] == "tool_calls"

        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        raw = json.dumps(
            {
                "messages": [{"role": "user", "content": "read x"}],
                "stream": True,
            }
        ).encode()
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(raw)),
            },
        )
        payload = conn.getresponse().read().decode()
        conn.close()
        assert '"tool_calls"' in payload
        assert '"name": "read"' in payload
        assert '"index": 0' in payload
