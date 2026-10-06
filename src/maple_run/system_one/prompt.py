"""Prompt formatting and token label resolution for System One questions."""

from __future__ import annotations

import json
import string
from typing import Any

from maple_run.system_one.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion


def format_state(state: Any) -> str:
    """Format input state into a string suitable for prompt context."""
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, (dict, list)):
        return json.dumps(state, indent=2, ensure_ascii=False)
    return str(state)


def build_noul_prompt(state: Any, question: NoulQuestion) -> str:
    """Build discriminative prompt for a binary/noul question."""
    st = format_state(state)
    lines = [
        f"Context:\n{st}\n",
        f"Question: {question.instructions}",
    ]
    if question.criteria:
        crit_lines = []
        if "true" in question.criteria:
            crit_lines.append(f"- Yes: {question.criteria['true']}")
        if "false" in question.criteria:
            crit_lines.append(f"- No: {question.criteria['false']}")
        for k, v in question.criteria.items():
            if k not in ("true", "false"):
                crit_lines.append(f"- {k}: {v}")
        if crit_lines:
            lines.append("Criteria:\n" + "\n".join(crit_lines))
    lines.append("Answer (Yes or No):")
    return "\n\n".join(lines)


def build_choice_prompt(
    state: Any, question: ChoiceQuestion
) -> tuple[str, list[str], dict[str, str]]:
    """Build discriminative prompt for a choice question.

    Returns:
        (prompt_text, candidate_labels, key_to_label)
        where candidate_labels are single tokens like ['A', 'B', 'C'].
    """
    st = format_state(state)
    keys = question.option_keys()
    descs = question.option_descriptions()

    # Use letters A, B, C... as discrete single-token choice labels
    letters = list(string.ascii_uppercase)
    if len(keys) > len(letters):
        raise ValueError(f"Too many choice options ({len(keys)}). Maximum supported is {len(letters)}.")

    labels: list[str] = []
    key_to_label: dict[str, str] = {}
    options_text: list[str] = []

    for idx, key in enumerate(keys):
        label = letters[idx]
        labels.append(label)
        key_to_label[key] = label
        desc = descs.get(key, "").strip()
        if desc:
            options_text.append(f"[{label}] {key}: {desc}")
        else:
            options_text.append(f"[{label}] {key}")

    lines = [
        f"Context:\n{st}\n",
        f"Task: {question.instructions}",
        "Options:\n" + "\n".join(options_text),
        "Selection:",
    ]
    prompt_text = "\n\n".join(lines)
    return prompt_text, labels, key_to_label


def build_score_prompt(state: Any, question: ScoreQuestion) -> tuple[str, list[str]]:
    """Build discriminative prompt for an ordinal score question.

    Returns:
        (prompt_text, candidate_labels)
        where candidate_labels are ['1', '2', ..., 'K'].
    """
    st = format_state(state)
    k = len(question.criteria)
    labels = [str(i + 1) for i in range(k)]
    rubric_lines = [f"{i + 1}: {desc}" for i, desc in enumerate(question.criteria)]

    lines = [
        f"Context:\n{st}\n",
        f"Evaluation: {question.instructions}",
        f"Rubric (1 to {k}):\n" + "\n".join(rubric_lines),
        f"Rating (1-{k}):",
    ]
    prompt_text = "\n\n".join(lines)
    return prompt_text, labels
