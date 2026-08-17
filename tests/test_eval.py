"""Quality-eval scoring, CLI flags, and a fake-engine resume loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from maple_run.cli import main
from maple_run.eval import (
    DEEPGROVE_SCORES,
    LOADERS,
    Problem,
    parse_benches,
    run_eval,
)
from maple_run.eval_lcb import grade_payload
from maple_run.eval_score import (
    aime_correct,
    extract_aime_answer,
    extract_gpqa_letter,
    extract_last_boxed,
    extract_python,
    gpqa_correct,
    math_correct,
)


def test_parse_benches_all_and_aliases():
    assert parse_benches("all") == ["aime2026", "hmmt2026", "gpqa", "lcbv6"]
    assert parse_benches("aime,gpqa") == ["aime2026", "gpqa"]
    assert parse_benches("lcbv6,lcb") == ["lcbv6"]
    with pytest.raises(ValueError, match="unknown"):
        parse_benches("mmlu")


def test_deepgrove_mean_matches_blog():
    mean = sum(DEEPGROVE_SCORES.values()) / 4
    assert abs(mean - 78.725) < 1e-9


def test_extract_last_boxed_nested_and_last_wins():
    text = r"first \boxed{1} then \boxed{\frac{2}{3}}"
    assert extract_last_boxed(text) == r"\frac{2}{3}"
    assert extract_last_boxed("no box") is None
    assert extract_last_boxed(r"\boxed{090}") == "090"


def test_aime_int_and_leading_zeros():
    blob = "reason\n" + r"\boxed{090}"
    pred = extract_aime_answer(blob)
    assert aime_correct(pred, 90)
    assert aime_correct(extract_aime_answer("the answer is 12"), 12)
    assert not aime_correct(extract_aime_answer(r"\boxed{7}"), 8)


def test_math_fraction_equivalence():
    assert math_correct(r"\frac{1}{2}", "1/2")
    assert math_correct(r"\dfrac{2}{4}", "1/2")
    assert math_correct("42", 42)
    assert not math_correct("1/3", "1/2")


def test_gpqa_answer_letter():
    assert extract_gpqa_letter("foo\nAnswer: B\n") == "B"
    assert extract_gpqa_letter("Answer: $C$") == "C"
    assert extract_gpqa_letter(r"so \boxed{A}") == "A"
    assert extract_gpqa_letter("the choice is\n(D)") == "D"
    assert gpqa_correct("b", "B")
    assert not gpqa_correct("A", "B")


def test_extract_python_last_fence():
    text = "```python\nprint(1)\n```\nthen\n```\nprint(2)\n```"
    assert extract_python(text) == "print(2)"


def test_lcb_stdio_and_call_grade():
    stdin_ok = grade_payload(
        {
            "code": "print(sum(map(int, input().split())))\n",
            "tests": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
            "fn_name": None,
            "timeout": 2,
        }
    )
    assert stdin_ok
    stdin_bad = grade_payload(
        {
            "code": "print(0)\n",
            "tests": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
            "fn_name": None,
            "timeout": 2,
        }
    )
    assert not stdin_bad
    call_ok = grade_payload(
        {
            "code": "def add(a, b):\n    return a + b\n",
            "tests": [{"input": "1\n2", "output": "3", "testtype": "functional"}],
            "fn_name": "add",
            "timeout": 2,
        }
    )
    assert call_ok


def test_cli_eval_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "default 64000" in out
    assert "default 4" in out
    assert "dense head" in out
    assert "aime2026" in out


def test_cli_eval_rejects_bad_args():
    with pytest.raises(SystemExit):
        main(["eval", "--model", "x", "--n-samples", "0"])
    with pytest.raises(SystemExit):
        main(["eval", "--model", "x", "--bench", "nope"])
    with pytest.raises(SystemExit):
        main(["eval", "--model", "x", "--temperature", "-1"])


def test_run_eval_resume(tmp_path: Path, monkeypatch):
    problems = [
        Problem(bench="aime2026", problem_id="1", prompt="q", gold="7"),
    ]
    monkeypatch.setitem(LOADERS, "aime2026", lambda limit=None: problems)
    calls = {"n": 0}

    def generate_one(problem, prompt, extra, seed):
        calls["n"] += 1
        return SimpleNamespace(
            text=r"\boxed{7}",
            reasoning="",
            n_new=4,
            decode_tok_s=10.0,
            finish_reason="stop",
        )

    run_eval(
        "unused",
        ["aime2026"],
        output_dir=tmp_path,
        n_samples=2,
        generate_one=generate_one,
        seed=0,
        max_tokens=32,
    )
    assert calls["n"] == 2
    summary = (tmp_path / "aime2026" / "summary.json").read_text()
    assert '"n_correct": 2' in summary
    run_eval(
        "unused",
        ["aime2026"],
        output_dir=tmp_path,
        n_samples=2,
        generate_one=generate_one,
        seed=0,
        max_tokens=32,
    )
    assert calls["n"] == 2
