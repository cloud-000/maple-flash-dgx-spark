"""Answer extraction used by EGGROLL reasoning rewards.

Quality-eval grading (GPQA / LCB / HMMT) lives in the ``bench`` project.
"""

from __future__ import annotations

import re

_BOX_CMDS = ("boxed", "fbox")
_TEXT_WRAP = re.compile(r"\\text\s*\{([^{}]*)\}")
_FRAC = re.compile(r"\\(?:dfrac|frac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_INT = re.compile(r"-?\d+")


def extract_last_boxed(text: str) -> str | None:
    """Return the body of the last ``\\boxed{...}`` / ``\\fbox{...}``."""
    last: str | None = None
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "\\":
            i += 1
            continue
        matched = None
        for cmd in _BOX_CMDS:
            if text.startswith(cmd, i + 1):
                matched = cmd
                break
        if matched is None:
            i += 1
            continue
        j = i + 1 + len(matched)
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != "{":
            i += 1
            continue
        body = _brace_body(text, j)
        if body is not None:
            last = body
            i = j + len(body) + 2
        else:
            i += 1
    return last


def _brace_body(text: str, open_idx: int) -> str | None:
    depth = 0
    for k in range(open_idx, len(text)):
        ch = text[k]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : k]
    return None


def normalize_math(s: str) -> str:
    """Strip common LaTeX wrappers so integer / fraction compares work."""
    s = s.strip()
    s = s.replace("\u200b", "")
    s = _TEXT_WRAP.sub(r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = s.replace("\\ ", " ")
    s = s.replace("$", "")
    s = s.replace("{,}", "")
    while True:
        nxt = _FRAC.sub(r"(\1)/(\2)", s)
        if nxt == s:
            break
        s = nxt
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = s.replace("\\%", "%")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace(" ", "")
    return s.strip()


def extract_aime_answer(text: str) -> str | None:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    ints = _INT.findall(text.replace(",", ""))
    return ints[-1] if ints else None


def aime_correct(pred: str | None, gold) -> bool:
    if pred is None:
        return False
    try:
        return int(normalize_math(pred).replace(",", "")) == int(gold)
    except (TypeError, ValueError):
        return False
