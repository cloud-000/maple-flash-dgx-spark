"""OpenAI-compatible adapter for System One (Jev) decision requests."""

from __future__ import annotations

import json
from typing import Any

from maple_run.system_one.schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneError,
    SystemOneRequest,
    SystemOneResponse,
)


def is_system_one_request(body: dict[str, Any], headers: dict[str, str] | None = None) -> bool:
    """Check if an incoming request is intended for System One processing."""
    if not isinstance(body, dict):
        return False

    # Check header
    if headers:
        for k, v in headers.items():
            if k.lower() == "x-decision-mode" and v.lower() in ("systemone", "system_one", "system1", "system-1", "jev"):
                return True

    # Check explicit body flag
    if body.get("system_one") is True or body.get("mode") in ("system_one", "systemone", "jev"):
        return True

    # Check model alias suffix or name
    model = str(body.get("model", "")).lower()
    if any(suffix in model for suffix in ("system1", "system-1", "systemone", "jev")):
        return True

    # Check if native System One questions dictionary is directly provided
    if "questions" in body and isinstance(body["questions"], dict) and body["questions"]:
        return True

    return False


def _extract_state_from_messages(messages: list[dict[str, Any]]) -> Any:
    """Extract context state from OpenAI chat messages."""
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0].get("content", "")

    # Combine messages into a clean conversation state
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role}: {content.strip()}")
        elif isinstance(content, (dict, list)):
            parts.append(f"{role}: {json.dumps(content, ensure_ascii=False)}")
    return "\n\n".join(parts)


def _questions_from_json_schema(schema: dict[str, Any]) -> dict[str, Question]:
    """Parse JSON schema properties into System One question primitives."""
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise SystemOneError("response_format json_schema must contain 'properties'")

    questions: dict[str, Question] = {}
    for prop_name, prop_spec in properties.items():
        if not isinstance(prop_spec, dict):
            continue
        desc = str(prop_spec.get("description") or f"Evaluate {prop_name}")
        prop_type = str(prop_spec.get("type", "")).lower()

        # Boolean -> Noul
        if prop_type == "boolean":
            questions[prop_name] = NoulQuestion(instructions=desc)

        # Enum string -> Choice
        elif prop_type == "string" and "enum" in prop_spec:
            enum_vals = prop_spec["enum"]
            if isinstance(enum_vals, list) and enum_vals:
                questions[prop_name] = ChoiceQuestion(
                    instructions=desc, criteria=[str(x) for x in enum_vals]
                )

        # Integer/Number with rubric or min/max -> Score
        elif prop_type in ("integer", "number"):
            if "rubric" in prop_spec and isinstance(prop_spec["rubric"], list):
                rubric = [str(x) for x in prop_spec["rubric"]]
            elif "minimum" in prop_spec and "maximum" in prop_spec:
                mn = int(prop_spec["minimum"])
                mx = int(prop_spec["maximum"])
                rubric = [f"Level {i}" for i in range(mn, mx + 1)]
            else:
                # Default 1 to 5 scale
                rubric = ["Very Low", "Low", "Moderate", "High", "Critical"]

            if 2 <= len(rubric) <= 10:
                questions[prop_name] = ScoreQuestion(instructions=desc, criteria=rubric)

    if not questions:
        raise SystemOneError(
            "Could not map any properties from json_schema to System One primitives"
        )
    return questions


def openai_to_system_one(
    body: dict[str, Any], default_model: str = "maple-2bit"
) -> SystemOneRequest:
    """Translate an OpenAI /v1/chat/completions request to a SystemOneRequest."""
    if not isinstance(body, dict):
        raise SystemOneError("Request body must be a JSON object")

    model = str(body.get("model") or default_model)
    for suffix in ("-system1", "-system-1", "-systemone", "-jev"):
        if model.endswith(suffix):
            model = model[: -len(suffix)]
            break
    temp = float(body.get("temperature", 1.0))
    if temp <= 0:
        temp = 1.0

    # 1. State resolution
    if "state" in body:
        state = body["state"]
    elif "messages" in body and isinstance(body["messages"], list):
        state = _extract_state_from_messages(body["messages"])
    elif "prompt" in body:
        state = body["prompt"]
    else:
        raise SystemOneError("System One request requires 'messages', 'prompt', or 'state'")

    # 2. Questions resolution
    questions: dict[str, Question] = {}

    # Check for direct 'questions' in body (TypeSafe format passed via chat endpoint)
    if "questions" in body and isinstance(body["questions"], dict) and body["questions"]:
        from maple_run.system_one.schema import parse_question

        for name, q_data in body["questions"].items():
            questions[name] = parse_question(name, q_data)

    # Check for OpenAI response_format json_schema
    elif "response_format" in body and isinstance(body["response_format"], dict):
        rf = body["response_format"]
        schema = None
        if rf.get("type") == "json_schema" and "json_schema" in rf:
            schema = rf["json_schema"].get("schema")
        elif rf.get("type") == "json_object" and "schema" in rf:
            schema = rf["schema"]

        if isinstance(schema, dict):
            questions = _questions_from_json_schema(schema)

    if not questions:
        raise SystemOneError(
            "System One requires 'questions' or a valid 'response_format' json_schema"
        )

    return SystemOneRequest(
        model=model,
        state=state,
        questions=questions,
        temperature=temp,
    )


def system_one_to_openai_chat_completion(
    response: SystemOneResponse,
    *,
    req_id: str,
    created: int,
    model: str,
) -> dict[str, Any]:
    """Format a SystemOneResponse as an OpenAI-compatible chat completion payload."""
    structured_content: dict[str, Any] = {}

    for k, v in response.nouls.items():
        structured_content[k] = v.noul >= 0.5

    for k, v in response.choices.items():
        structured_content[k] = v.choice

    for k, v in response.scores.items():
        structured_content[k] = round(v.score, 2)

    content_str = json.dumps(structured_content, ensure_ascii=False)

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_str,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.get("input_tokens", 0),
            "completion_tokens": 0,
            "total_tokens": response.usage.get("input_tokens", 0),
        },
        "system_one": response.to_dict(),
    }
