"""Answer extraction and grading for Maple-Preview quality evals.

Math extraction follows MathArena's last-``\\boxed`` rule (brace matching
instead of the ``regex`` recursive pattern). GPQA matches OpenAI simple-evals.
"""

from __future__ import annotations

import re
from fractions import Fraction

_BOX_CMDS = ("boxed", "fbox")
_TEXT_WRAP = re.compile(r"\\text\s*\{([^{}]*)\}")
_FRAC = re.compile(r"\\(?:dfrac|frac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_ANSWER_LETTER = re.compile(r"(?i)answer[ \t]*:[ \t]*\$?([A-D])\$?")
_LETTER_ONLY = re.compile(r"^\s*\(?([A-D])\)?\s*\.?\s*$")
_INT = re.compile(r"-?\d+")
_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


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


def _as_fraction(s: str) -> Fraction | None:
    s = normalize_math(s)
    if not s:
        return None
    s = s.replace(",", "").replace("(", "").replace(")", "")
    try:
        if "/" in s:
            num, _, den = s.partition("/")
            return Fraction(num) / Fraction(den)
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def math_correct(pred: str | None, gold) -> bool:
    """Grade a short math answer: int / fraction / sympy, then string."""
    if pred is None:
        return False
    gold_s = str(gold).strip()
    pn, gn = normalize_math(pred), normalize_math(gold_s)
    if pn == gn:
        return True
    pf, gf = _as_fraction(pred), _as_fraction(gold_s)
    if pf is not None and gf is not None and pf == gf:
        return True
    try:
        import sympy

        diff = sympy.simplify(sympy.sympify(pn) - sympy.sympify(gn))
        return diff == 0
    except Exception:
        return False


def extract_gpqa_letter(text: str) -> str | None:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        m = _LETTER_ONLY.match(boxed.strip())
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([A-D])\b", boxed)
        if m:
            return m.group(1).upper()
    matches = list(_ANSWER_LETTER.finditer(text))
    if matches:
        return matches[-1].group(1).upper()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        m = _LETTER_ONLY.match(lines[-1])
        if m:
            return m.group(1).upper()
    return None


def gpqa_correct(pred: str | None, gold: str) -> bool:
    return pred is not None and pred.upper() == str(gold).strip().upper()


def extract_python(text: str) -> str:
    """Last fenced block, matching LiveCodeBench's last-pair-of-backticks rule."""
    fences = list(_FENCE.finditer(text))
    if fences:
        return fences[-1].group(1).strip()
    lines = text.splitlines()
    ticks = [i for i, line in enumerate(lines) if "```" in line]
    if len(ticks) >= 2:
        return "\n".join(lines[ticks[-2] + 1 : ticks[-1]]).strip()
    return text.strip()
