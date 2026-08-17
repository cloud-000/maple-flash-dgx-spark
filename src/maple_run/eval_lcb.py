"""LiveCodeBench v6 loading and isolated code grading.

Private tests are stored the way LiveCodeBench ships them (JSON, or pickled
zlib+base64). Generated programs run in a fresh Python subprocess so a crash
cannot take down the CUDA eval process.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import zlib
from base64 import b64decode
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from maple_run.eval_score import extract_python

LCB_REPO = "livecodebench/code_generation_lite"
LCB_CONFIG = "v6"
LCB_TIMEOUT = 6


@dataclass
class LcbProblem:
    question_id: str
    title: str
    question: str
    starter_code: str
    difficulty: str
    tests: list[dict]
    fn_name: str | None


def decode_lcb_tests(raw) -> list[dict]:
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = json.loads(
            pickle.loads(zlib.decompress(b64decode(raw.encode("utf-8"))))
        )
    if isinstance(parsed, dict):
        parsed = [parsed]
    out = []
    for item in parsed:
        out.append(
            {
                "input": item["input"],
                "output": item["output"],
                "testtype": item.get("testtype", "stdin"),
            }
        )
    return out


def load_lcb_v6(limit: int | None = None) -> list[LcbProblem]:
    from datasets import load_dataset

    ds = load_dataset(LCB_REPO, LCB_CONFIG, split="test")
    problems: list[LcbProblem] = []
    for row in ds:
        if limit is not None and len(problems) >= limit:
            break
        try:
            public = decode_lcb_tests(row["public_test_cases"])
            private = decode_lcb_tests(row["private_test_cases"])
        except Exception as exc:
            raise RuntimeError(
                f"failed to decode tests for {row.get('question_id')}: {exc}"
            ) from exc
        meta = row.get("metadata") or "{}"
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}
        fn_name = meta.get("func_name")
        problems.append(
            LcbProblem(
                question_id=str(row["question_id"]),
                title=str(row.get("question_title") or row["question_id"]),
                question=str(row["question_content"]),
                starter_code=str(row.get("starter_code") or ""),
                difficulty=str(row.get("difficulty") or ""),
                tests=public + private,
                fn_name=fn_name,
            )
        )
    return problems


def lcb_user_prompt(problem: LcbProblem) -> str:
    body = f"### Question:\n{problem.question}\n\n"
    if problem.starter_code.strip():
        body += (
            "### Format: You will use the following starter code to write "
            "the solution to the problem and enclose your code within delimiters.\n"
            f"```python\n{problem.starter_code}\n```\n\n"
        )
    else:
        body += (
            "### Format: Read the inputs from stdin solve the problem and write "
            "the answer to stdout (do not directly test on the sample inputs). "
            "Enclose your code within delimiters as follows. Ensure that when "
            "the python program runs, it reads the inputs, runs the algorithm "
            "and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    body += "### Answer: (use the provided format with backticks)\n"
    return body


LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)


def grade_lcb_code(code: str, problem: LcbProblem, timeout: int = LCB_TIMEOUT) -> bool:
    """Run ``code`` against ``problem.tests`` in a child interpreter."""
    payload = {
        "code": extract_python(code),
        "tests": problem.tests,
        "fn_name": problem.fn_name,
        "timeout": int(timeout),
    }
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CUDA") and k not in {"NVIDIA_VISIBLE_DEVICES"}
    }
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    with tempfile.TemporaryDirectory(prefix="maple-lcb-") as td:
        payload_path = Path(td) / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        wall = timeout * max(len(problem.tests), 1) + 15
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "maple_run.eval_lcb",
                    str(payload_path),
                ],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=wall,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    if proc.returncode != 0:
        return False
    try:
        return bool(json.loads(proc.stdout.strip().splitlines()[-1])["pass"])
    except (json.JSONDecodeError, KeyError, IndexError):
        return False


_IMPORTS = textwrap.dedent(
    """
    from string import *
    from re import *
    from datetime import *
    from collections import *
    from heapq import *
    from bisect import *
    from copy import *
    from math import *
    from random import *
    from statistics import *
    from itertools import *
    from functools import *
    from operator import *
    from io import *
    from sys import *
    from json import *
    from builtins import *
    from typing import *
    import string, re, datetime, collections, heapq, bisect, copy, math
    import random, statistics, itertools, functools, operator, io, sys, json
    sys.setrecursionlimit(50000)
    """
).strip()


def _strip_main_guard(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, ast.If):
        return code
    try:
        cond = ast.unparse(last.test).strip()
    except Exception:
        return code
    if cond not in {"__name__ == '__main__'", '__name__ == "__main__"'}:
        return code
    tree.body = tree.body[:-1] + list(last.body)
    return ast.unparse(tree)


def _call_args(raw_input: str) -> list:
    lines = raw_input.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        lines = [raw_input]
    return [json.loads(line) for line in lines]


def _outputs_match(pred: str, gold: str) -> bool:
    pred_lines = [ln.strip() for ln in pred.strip().splitlines()]
    gold_lines = [ln.strip() for ln in gold.strip().splitlines()]
    if pred_lines == gold_lines:
        return True
    if len(pred_lines) != len(gold_lines):
        return False
    for a, b in zip(pred_lines, gold_lines):
        if a == b:
            continue
        try:
            da = [Decimal(x) for x in a.split()]
            db = [Decimal(x) for x in b.split()]
        except Exception:
            return False
        if da != db:
            return False
    return True


def grade_payload(payload: dict) -> bool:
    code = payload["code"]
    tests = payload["tests"]
    fn_name = payload.get("fn_name")
    timeout = int(payload.get("timeout") or LCB_TIMEOUT)
    if not code.strip():
        return False
    if fn_name:
        return _grade_call(code, tests, fn_name, timeout)
    return _grade_stdio(code, tests, timeout)


def _grade_call(code: str, tests: list[dict], fn_name: str, timeout: int) -> bool:
    ns: dict = {}
    exec(_IMPORTS + "\n" + code, ns, ns)
    if "Solution" in ns and isinstance(ns["Solution"], type):
        method = getattr(ns["Solution"](), fn_name, None)
    else:
        method = ns.get(fn_name)
    if method is None:
        return False
    for test in tests:
        args = _call_args(test["input"])
        expected = json.loads(test["output"])
        pred = method(*args)
        if isinstance(pred, tuple):
            pred = list(pred)
        if pred != expected:
            return False
    return True


def _grade_stdio(code: str, tests: list[dict], timeout: int) -> bool:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUDA")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    with tempfile.TemporaryDirectory(prefix="maple-lcb-stdio-") as td:
        path = Path(td) / "sol.py"
        path.write_text(
            _IMPORTS + "\n" + _strip_main_guard(code) + "\n", encoding="utf-8"
        )
        for test in tests:
            stdin = test["input"]
            if not stdin.endswith("\n"):
                stdin += "\n"
            try:
                proc = subprocess.run(
                    [sys.executable, str(path)],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=int(timeout) + 1,
                    env=env,
                    check=False,
                    cwd=td,
                )
            except subprocess.TimeoutExpired:
                return False
            if proc.returncode != 0:
                return False
            if not _outputs_match(proc.stdout, test["output"]):
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m maple_run.eval_lcb PAYLOAD.json", file=sys.stderr)
        return 2
    payload = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    passed = False
    try:
        passed = grade_payload(payload)
    except Exception:
        passed = False
    print(json.dumps({"pass": bool(passed)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
