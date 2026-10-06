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
