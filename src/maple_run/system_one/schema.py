"""Data schemas and request/response parsing for TypeSafe System One (Jev)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


class SystemOneError(ValueError):
    """Raised when a System One request or schema is invalid."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str
    criteria: dict[str, str] | None = None
    type: Literal["noul"] = "noul"


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: dict[str, str] | list[str]
    type: Literal["choice"] = "choice"

    def option_keys(self) -> list[str]:
        if isinstance(self.criteria, dict):
            return list(self.criteria.keys())
        return list(self.criteria)

    def option_descriptions(self) -> dict[str, str]:
        if isinstance(self.criteria, dict):
            return {str(k): str(v or "") for k, v in self.criteria.items()}
        return {str(k): "" for k in self.criteria}


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    criteria: list[str]
    type: Literal["score"] = "score"


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class SystemOneRequest:
    model: str
    state: Any
    questions: dict[str, Question]
    temperature: float = 1.0


@dataclass(frozen=True)
class NoulResult:
    noul: float

    def to_dict(self) -> dict[str, float]:
        return {"noul": round(float(self.noul), 4)}


@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
        }


@dataclass(frozen=True)
class ScoreResult:
    score: float
    confidence: float
    probabilities: list[float]
    legend: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "probabilities": [round(float(p), 4) for p in self.probabilities],
            "legend": list(self.legend),
        }


@dataclass
class SystemOneResponse:
    model: str
    nouls: dict[str, NoulResult] = field(default_factory=dict)
    choices: dict[str, ChoiceResult] = field(default_factory=dict)
    scores: dict[str, ScoreResult] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"model": self.model}
        if self.nouls:
            out["nouls"] = {k: v.to_dict() for k, v in self.nouls.items()}
        if self.choices:
            out["choices"] = {k: v.to_dict() for k, v in self.choices.items()}
        if self.scores:
            out["scores"] = {k: v.to_dict() for k, v in self.scores.items()}
        out["usage"] = self.usage or {"input_tokens": 0, "output_tokens": 0}
        return out


def parse_question(name: str, data: dict[str, Any]) -> Question:
    if not isinstance(data, dict):
        raise SystemOneError(f"question '{name}' must be an object")
    q_type = str(data.get("type", "")).lower()
    instructions = data.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise SystemOneError(f"question '{name}' missing valid 'instructions' string")

    if q_type == "noul":
        crit = data.get("criteria")
        if crit is not None and not isinstance(crit, dict):
            raise SystemOneError(f"noul question '{name}' criteria must be a dict or null")
        return NoulQuestion(instructions=instructions.strip(), criteria=crit)

    if q_type == "choice":
        # Support options or criteria key
        crit = data.get("criteria")
        if crit is None and "options" in data:
            crit = data["options"]
        if not crit or (not isinstance(crit, dict) and not isinstance(crit, list)):
            raise SystemOneError(f"choice question '{name}' requires 'criteria' (dict or list)")
        if isinstance(crit, list) and not all(isinstance(x, str) for x in crit):
            raise SystemOneError(f"choice question '{name}' criteria list must contain strings")
        return ChoiceQuestion(instructions=instructions.strip(), criteria=crit)

    if q_type == "score":
        crit = data.get("criteria")
        if crit is None and "rubric" in data:
            crit = data["rubric"]
        if not isinstance(crit, list) or len(crit) < 2 or len(crit) > 10:
            raise SystemOneError(
                f"score question '{name}' criteria must be a list of 2 to 10 rubric levels"
            )
        if not all(isinstance(x, str) for x in crit):
            raise SystemOneError(f"score question '{name}' criteria levels must be strings")
        return ScoreQuestion(instructions=instructions.strip(), criteria=crit)

    raise SystemOneError(
        f"unknown question type '{q_type}' for '{name}'. Expected 'noul', 'choice', or 'score'."
    )


def parse_system_one_request(
    body: dict[str, Any], default_model: str = "maple-2bit"
) -> SystemOneRequest:
    if not isinstance(body, dict):
        raise SystemOneError("request body must be a JSON object")
    if "state" not in body:
        raise SystemOneError("'state' field is required in System One request")
    raw_questions = body.get("questions")
    if not isinstance(raw_questions, dict) or not raw_questions:
        raise SystemOneError("'questions' must be a non-empty object mapping question names to definitions")

    questions: dict[str, Question] = {}
    for name, q_data in raw_questions.items():
        if not isinstance(name, str) or not name.strip():
            raise SystemOneError("question key must be a non-empty string")
        questions[name] = parse_question(name, q_data)

    model = str(body.get("model") or default_model)
    temp = float(body.get("temperature", 1.0))
    if temp <= 0:
        temp = 1.0

    return SystemOneRequest(
        model=model,
        state=body["state"],
        questions=questions,
        temperature=temp,
    )
