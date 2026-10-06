"""Unit tests for System One (Jev) schemas, prompt formatting, and evaluator."""

from __future__ import annotations

import math
import pytest
import torch

from maple_run.system_one import (
    ChoiceQuestion,
    ChoiceResult,
    NoulQuestion,
    NoulResult,
    ScoreQuestion,
    ScoreResult,
    SystemOneError,
    SystemOneEvaluator,
    SystemOneRequest,
    SystemOneResponse,
    build_choice_prompt,
    build_noul_prompt,
    build_score_prompt,
    parse_question,
    parse_system_one_request,
)


class DummyTokenizer:
    """Minimal deterministic tokenizer for CPU unit testing."""

    def __init__(self):
        self.vocab = {
            "Yes": 101,
            "No": 102,
            "True": 103,
            "False": 104,
            "A": 201,
            "B": 202,
            "C": 203,
            "D": 204,
            "1": 301,
            "2": 302,
            "3": 303,
            "4": 304,
            "5": 305,
        }

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        if text in self.vocab:
            ids = [self.vocab[text]]
        else:
            ids = [hash(word) % 500 + 1 for word in text.split()]
            if not ids:
                ids = [1]
        if return_tensors == "pt":
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def get_vocab(self):
        return self.vocab


class DummyModel:
    """Mock model that returns custom logits at the final position."""

    def __init__(self, logits_map: dict[int, float] | None = None, vocab_size: int = 1000):
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.logits_map = logits_map or {}

    def forward(self, input_ids: torch.Tensor, cache=None, *, logits_to_keep: int = 1):
        logits = torch.zeros((1, 1, self.vocab_size), dtype=torch.float32)
        for tid, val in self.logits_map.items():
            if tid < self.vocab_size:
                logits[0, 0, tid] = val
        return logits


def test_schema_parsing_valid():
    raw = {
        "model": "maple-2bit",
        "state": {"ticket_id": "123", "text": "App keeps crashing on launch"},
        "questions": {
            "is_bug": {
                "type": "noul",
                "instructions": "Does this report a software bug?",
            },
            "team": {
                "type": "choice",
                "instructions": "Select handling team",
                "criteria": {
                    "eng": "Engineering issues",
                    "billing": "Payment inquiries",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "Rate severity",
                "criteria": ["low", "medium", "critical"],
            },
        },
    }
    req = parse_system_one_request(raw)
    assert req.model == "maple-2bit"
    assert isinstance(req.questions["is_bug"], NoulQuestion)
    assert isinstance(req.questions["team"], ChoiceQuestion)
    assert isinstance(req.questions["severity"], ScoreQuestion)
    assert req.questions["team"].option_keys() == ["eng", "billing"]


def test_schema_parsing_errors():
    with pytest.raises(SystemOneError, match="must be a JSON object"):
        parse_system_one_request(["not", "an", "object"])

    with pytest.raises(SystemOneError, match="'state' field is required"):
        parse_system_one_request({"questions": {"q": {"type": "noul", "instructions": "test"}}})

    with pytest.raises(SystemOneError, match="'questions' must be a non-empty object"):
        parse_system_one_request({"state": "hello", "questions": {}})

    with pytest.raises(SystemOneError, match="missing valid 'instructions'"):
        parse_question("q", {"type": "noul", "instructions": ""})

    with pytest.raises(SystemOneError, match="must be a list of 2 to 10"):
        parse_question("s", {"type": "score", "instructions": "test", "criteria": ["single"]})


def test_prompt_formatting():
    state = "User reports checkout button is disabled"
    q_noul = NoulQuestion(instructions="Is checkout broken?")
    noul_prompt = build_noul_prompt(state, q_noul)
    assert "Context:\nUser reports checkout button is disabled" in noul_prompt
    assert "Question: Is checkout broken?" in noul_prompt
    assert "Answer (Yes or No):" in noul_prompt

    q_choice = ChoiceQuestion(
        instructions="Pick category",
        criteria={"frontend": "UI issue", "backend": "API issue"},
    )
    choice_prompt, labels, key_map = build_choice_prompt(state, q_choice)
    assert labels == ["A", "B"]
    assert key_map == {"frontend": "A", "backend": "B"}
    assert "[A] frontend: UI issue" in choice_prompt
    assert "[B] backend: API issue" in choice_prompt
    assert "Selection:" in choice_prompt

    q_score = ScoreQuestion(
        instructions="Urgency level",
        criteria=["low", "medium", "high"],
    )
    score_prompt, score_labels = build_score_prompt(state, q_score)
    assert score_labels == ["1", "2", "3"]
    assert "1: low" in score_prompt
    assert "3: high" in score_prompt
    assert "Rating (1-3):" in score_prompt


def test_response_to_dict():
    resp = SystemOneResponse(
        model="maple-system1",
        nouls={"urgent": NoulResult(noul=0.95)},
        choices={
            "team": ChoiceResult(
                choice="eng",
                confidence=0.9,
                probabilities={"eng": 0.9, "ops": 0.1},
            )
        },
        scores={
            "level": ScoreResult(
                score=2.8,
                confidence=0.85,
                probabilities=[0.05, 0.1, 0.85],
                legend=["low", "med", "high"],
            )
        },
        usage={"input_tokens": 42, "output_tokens": 0},
    )
    d = resp.to_dict()
    assert d["model"] == "maple-system1"
    assert d["nouls"]["urgent"]["noul"] == 0.95
    assert d["choices"]["team"]["choice"] == "eng"
    assert d["scores"]["level"]["score"] == 2.8
    assert d["usage"]["output_tokens"] == 0


def test_evaluator_noul():
    # Setup logits: "Yes" (101) = 5.0, "No" (102) = 0.0
    model = DummyModel(logits_map={101: 5.0, 102: 0.0})
    tokenizer = DummyTokenizer()
    evaluator = SystemOneEvaluator(model, tokenizer)

    q = NoulQuestion(instructions="Is this positive?")
    res, toks = evaluator.evaluate_noul("state", q)
    assert toks > 0
    # exp(5) / (exp(5) + exp(0)) = 148.4 / 149.4 ~ 0.993
    assert res.noul > 0.95


def test_evaluator_choice():
    # Choice labels map to A (201), B (202)
    # A = 1.0, B = 4.0 -> B should win
    model = DummyModel(logits_map={201: 1.0, 202: 4.0})
    tokenizer = DummyTokenizer()
    evaluator = SystemOneEvaluator(model, tokenizer)

    q = ChoiceQuestion(
        instructions="Pick option",
        criteria={"first": "first option", "second": "second option"},
    )
    res, toks = evaluator.evaluate_choice("state", q)
    assert res.choice == "second"
    assert res.confidence > 0.9
    assert res.probabilities["second"] > res.probabilities["first"]


def test_evaluator_score():
    # Rubric: 1 (301), 2 (302), 3 (303)
    # Give high logit to 3
    model = DummyModel(logits_map={301: 0.0, 302: 0.0, 303: 10.0})
    tokenizer = DummyTokenizer()
    evaluator = SystemOneEvaluator(model, tokenizer)

    q = ScoreQuestion(instructions="Rate this", criteria=["low", "mid", "high"])
    res, toks = evaluator.evaluate_score("state", q)
    assert math.isclose(res.score, 3.0, abs_tol=0.05)
    assert res.confidence > 0.95
    assert len(res.probabilities) == 3


def test_evaluator_full_request():
    model = DummyModel(logits_map={101: 4.0, 102: 0.0, 201: 5.0, 202: 1.0, 301: 0.0, 302: 5.0})
    tokenizer = DummyTokenizer()
    evaluator = SystemOneEvaluator(model, tokenizer)

    req = SystemOneRequest(
        model="maple-test",
        state="Customer is happy with speed",
        questions={
            "satisfied": NoulQuestion(instructions="Are they satisfied?"),
            "category": ChoiceQuestion(
                instructions="Category",
                criteria={"feedback": "Product feedback", "support": "Support inquiry"},
            ),
            "tier": ScoreQuestion(instructions="Tier", criteria=["basic", "pro"]),
        },
    )
    resp = evaluator.evaluate_request(req)
    assert resp.model == "maple-test"
    assert resp.nouls["satisfied"].noul > 0.9
    assert resp.choices["category"].choice == "feedback"
    assert resp.scores["tier"].score > 1.8
    assert resp.usage["input_tokens"] > 0
    assert resp.usage["output_tokens"] == 0


from maple_run.server import ServerDefaults, make_handler
from maple_run.system_one.adapter import (
    is_system_one_request,
    openai_to_system_one,
    system_one_to_openai_chat_completion,
)
from http.server import ThreadingHTTPServer
from http.client import HTTPConnection
from contextlib import contextmanager
import json
import threading


def test_is_system_one_request():
    # 1. Model suffix
    assert is_system_one_request({"model": "maple-preview-system1"})
    assert is_system_one_request({"model": "maple-2bit-system1"})
    # 2. Header
    assert is_system_one_request({"model": "maple-preview"}, headers={"x-decision-mode": "system1"})
    assert is_system_one_request({"model": "maple-preview"}, headers={"X-Decision-Mode": "system1"})
    # 3. Explicit system_one field
    assert is_system_one_request({"model": "maple-preview", "system_one": True})
    # 4. Questions field
    assert is_system_one_request({"model": "maple-preview", "questions": {"q": {"type": "noul", "instructions": "test"}}})
    # 5. Standard chat request
    assert not is_system_one_request({"model": "maple-preview", "messages": [{"role": "user", "content": "hello"}]})


def test_openai_to_system_one_json_schema():
    body = {
        "model": "maple-2bit-system1",
        "messages": [
            {"role": "system", "content": "You are a decision classifier."},
            {"role": "user", "content": "Customer cannot log in to their dashboard."},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_account_issue": {
                            "type": "boolean",
                            "description": "Is this an account login issue?",
                        },
                        "dept": {
                            "type": "string",
                            "enum": ["auth", "billing", "general"],
                            "description": "Responsible department",
                        },
                        "urgency": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 3,
                            "description": "Urgency rating",
                        },
                    },
                    "required": ["is_account_issue", "dept"],
                }
            },
        },
    }
    req = openai_to_system_one(body)
    assert req.model == "maple-2bit"
    assert "Customer cannot log in to their dashboard." in req.state
    assert isinstance(req.questions["is_account_issue"], NoulQuestion)
    assert isinstance(req.questions["dept"], ChoiceQuestion)
    assert isinstance(req.questions["urgency"], ScoreQuestion)
    assert req.questions["dept"].option_keys() == ["auth", "billing", "general"]


def test_system_one_to_openai_chat_completion():
    resp = SystemOneResponse(
        model="maple-2bit",
        nouls={"is_bug": NoulResult(noul=0.88)},
        choices={"team": ChoiceResult(choice="eng", confidence=0.92, probabilities={"eng": 0.92, "ops": 0.08})},
        scores={"severity": ScoreResult(score=3.0, confidence=0.95, probabilities=[0.05, 0.0, 0.95], legend=["1", "2", "3"])},
        usage={"input_tokens": 35, "output_tokens": 0},
    )
    chat_resp = system_one_to_openai_chat_completion(resp, req_id="chatcmpl-test", created=1234567890, model="maple-2bit-system1")
    assert chat_resp["id"] == "chatcmpl-test"
    assert chat_resp["model"] == "maple-2bit-system1"
    assert chat_resp["object"] == "chat.completion"
    assert chat_resp["usage"]["completion_tokens"] == 0
    assert chat_resp["choices"][0]["finish_reason"] == "stop"
    
    parsed_content = json.loads(chat_resp["choices"][0]["message"]["content"])
    assert parsed_content["is_bug"] is True
    assert chat_resp["system_one"]["nouls"]["is_bug"]["noul"] == 0.88
    assert parsed_content["team"] == "eng"
    assert parsed_content["severity"] == 3.0


class SystemOneMockEngine:
    def __init__(self, model_id: str = "maple-2bit"):
        self.defaults = ServerDefaults(model_id=model_id)
        # 101: Yes, 102: No, 201: A, 202: B, 301: 1, 302: 2
        self.evaluator = SystemOneEvaluator(
            DummyModel(logits_map={101: 5.0, 102: 0.0, 201: 4.0, 202: 1.0, 301: 0.0, 302: 5.0}),
            DummyTokenizer(),
        )

    def evaluate_system_one(self, req: SystemOneRequest) -> SystemOneResponse:
        return self.evaluator.evaluate_request(req)


@contextmanager
def _test_server(engine):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _post_json(port: int, path: str, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    raw = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", "Content-Length": str(len(raw))}
    if headers:
        hdrs.update(headers)
    conn.request("POST", path, body=raw, headers=hdrs)
    resp = conn.getresponse()
    payload = json.loads(resp.read().decode())
    conn.close()
    return resp.status, payload


def _get_json(port: int, path: str) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    payload = json.loads(resp.read().decode())
    conn.close()
    return resp.status, payload


def test_server_system_one_routes():
    engine = SystemOneMockEngine()
    req_body = {
        "state": "Checkout page is returning 500 status code",
        "questions": {
            "is_bug": {"type": "noul", "instructions": "Is this an error bug?"},
            "component": {
                "type": "choice",
                "instructions": "Component responsible",
                "criteria": {"checkout": "Checkout service", "auth": "Auth service"},
            },
        },
    }
    with _test_server(engine) as port:
        # 1. /v1/systemone
        s1, p1 = _post_json(port, "/v1/systemone", req_body)
        assert s1 == 200
        assert p1["nouls"]["is_bug"]["noul"] > 0.9
        assert p1["choices"]["component"]["choice"] == "checkout"

        # 2. /v1/evaluate alias
        s2, p2 = _post_json(port, "/v1/evaluate", req_body)
        assert s2 == 200
        assert p2["choices"]["component"]["choice"] == "checkout"

        # 3. /v1/decisions alias
        s3, p3 = _post_json(port, "/v1/decisions", req_body)
        assert s3 == 200
        assert p3["nouls"]["is_bug"]["noul"] > 0.9

        # 4. /v1/models list and item
        sm, pm = _get_json(port, "/v1/models")
        assert sm == 200
        model_ids = [m["id"] for m in pm["data"]]
        assert "maple-2bit" in model_ids
        assert "maple-2bit-system1" in model_ids

        sm_item, pm_item = _get_json(port, "/v1/models/maple-2bit-system1")
        assert sm_item == 200
        assert pm_item["id"] == "maple-2bit-system1"


def test_server_chat_completions_interception():
    engine = SystemOneMockEngine()
    chat_body = {
        "model": "maple-2bit-system1",
        "messages": [{"role": "user", "content": "System is throwing 500 error on purchase."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_error": {"type": "boolean", "description": "Is this an error?"},
                        "tier": {"type": "string", "enum": ["high", "low"]},
                    },
                }
            },
        },
    }
    with _test_server(engine) as port:
        status, payload = _post_json(port, "/v1/chat/completions", chat_body)
        assert status == 200
        assert payload["object"] == "chat.completion"
        assert payload["usage"]["completion_tokens"] == 0
        content = json.loads(payload["choices"][0]["message"]["content"])
        assert "is_error" in content
        assert "tier" in content


def test_cli_evaluate(monkeypatch, capsys):
    from maple_run.cli import main

    def fake_load(model_path, **kwargs):
        return (
            DummyModel(logits_map={101: 5.0, 102: 0.0}),
            DummyTokenizer(),
        )

    monkeypatch.setattr("maple_run.generate.load_packed", fake_load)

    # Invoke evaluate with inline state and question
    q_json = json.dumps({"type": "noul", "instructions": "Is this valid?"})
    exit_code = main([
        "evaluate",
        "--model", "fake-model",
        "--state", "Test context",
        "--question", q_json,
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert "nouls" in res
    assert res["nouls"]["eval"]["noul"] > 0.9
