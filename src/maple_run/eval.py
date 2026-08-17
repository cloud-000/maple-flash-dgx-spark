"""Quality evals matching DeepGrove Maple-Preview's published table.

Protocol (dense / exact ``lm_head``, not FlashHead):

- Sampling: Maple generate defaults (T=1.0, top_p=0.95, top_k=20).
- ``n=4`` completions per problem, mean accuracy (MathArena and OpenAI
  simple-evals GPQA both use four repeats).
- ``max_tokens=64000`` (MathArena cap).
- AIME 2026 / HMMT Feb 2026: MathArena prompts and last-``\\boxed`` grading.
- GPQA-Diamond: OpenAI simple-evals shuffle + ``Answer: LETTER``.
- LiveCodeBench v6: official 175-problem ``code_generation_lite`` split,
  pass@1 as mean of binary test-suite results.

Published DeepGrove numbers (dense head) are recorded as ``DEEPGROVE_SCORES``
for the summary table. They did not release a harness; this is the public
protocol those scores line up with (AIME 87.5% = 105/120 at n=4, HMMT 78.8%
= 104/132 at n=4).
"""

from __future__ import annotations

import csv
import io
import json
import random
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from maple_run.eval_score import (
    aime_correct,
    extract_aime_answer,
    extract_gpqa_letter,
    extract_last_boxed,
    gpqa_correct,
    math_correct,
)
from maple_run.generate import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    encode_messages,
    load_packed,
    log_model_traffic,
    run_generate,
    warmup_model,
)

AIME_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}.\n"
    "The answer is an integer between 0 and 999 inclusive."
)
HMMT_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
GPQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering.\n"
    "\n"
    "{question}\n"
    "\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}"
)
GPQA_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
)

DEFAULT_N_SAMPLES = 4
DEFAULT_MAX_TOKENS = 64000
DEFAULT_SEED = 0

DEEPGROVE_SCORES = {
    "lcbv6": 75.1,
    "aime2026": 87.5,
    "hmmt2026": 78.8,
    "gpqa": 73.5,
}

BENCH_ALIASES = {
    "aime": "aime2026",
    "aime2026": "aime2026",
    "hmmt": "hmmt2026",
    "hmmt2026": "hmmt2026",
    "gpqa": "gpqa",
    "gpqa-d": "gpqa",
    "gpqa_diamond": "gpqa",
    "lcb": "lcbv6",
    "lcbv6": "lcbv6",
    "livecodebench": "lcbv6",
}

ALL_BENCHES = ("aime2026", "hmmt2026", "gpqa", "lcbv6")


@dataclass
class Problem:
    bench: str
    problem_id: str
    prompt: str
    gold: str
    system: str | None = None
    extra: dict = field(default_factory=dict)


def parse_benches(spec: str) -> list[str]:
    spec = spec.strip().lower()
    if spec in {"all", "*"}:
        return list(ALL_BENCHES)
    out: list[str] = []
    for part in spec.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in BENCH_ALIASES:
            known = ", ".join(ALL_BENCHES)
            raise ValueError(f"unknown bench {part!r}; choose from {known} or all")
        resolved = BENCH_ALIASES[name]
        if resolved not in out:
            out.append(resolved)
    if not out:
        raise ValueError("no benches selected")
    return out


def _load_hf_rows(repo: str, split: str = "train"):
    from datasets import load_dataset

    return load_dataset(repo, split=split)


def load_aime2026(limit: int | None = None) -> list[Problem]:
    rows = _load_hf_rows("MathArena/aime_2026", "train")
    problems = []
    for row in rows:
        if limit is not None and len(problems) >= limit:
            break
        pid = str(row["problem_idx"])
        prompt = f"{row['problem'].rstrip()}\n\n{AIME_INSTRUCTION}"
        problems.append(
            Problem(
                bench="aime2026",
                problem_id=pid,
                prompt=prompt,
                gold=str(row["answer"]),
            )
        )
    return problems


def load_hmmt2026(limit: int | None = None) -> list[Problem]:
    rows = _load_hf_rows("MathArena/hmmt_feb_2026", "train")
    problems = []
    for row in rows:
        if limit is not None and len(problems) >= limit:
            break
        pid = str(row["problem_idx"])
        prompt = f"{row['problem'].rstrip()}\n\n{HMMT_INSTRUCTION}"
        problems.append(
            Problem(
                bench="hmmt2026",
                problem_id=pid,
                prompt=prompt,
                gold=str(row["answer"]),
            )
        )
    return problems


def _gpqa_csv_text() -> str:
    with urllib.request.urlopen(GPQA_CSV_URL, timeout=60) as resp:
        return resp.read().decode("utf-8")


def load_gpqa(limit: int | None = None) -> list[Problem]:
    reader = csv.DictReader(io.StringIO(_gpqa_csv_text()))
    problems = []
    for i, row in enumerate(reader):
        if limit is not None and len(problems) >= limit:
            break
        problems.append(
            Problem(
                bench="gpqa",
                problem_id=str(i),
                prompt="",
                gold=row["Correct Answer"],
                extra={
                    "question": row["Question"],
                    "choices": [
                        row["Correct Answer"],
                        row["Incorrect Answer 1"],
                        row["Incorrect Answer 2"],
                        row["Incorrect Answer 3"],
                    ],
                },
            )
        )
    return problems


def load_lcbv6(limit: int | None = None) -> list[Problem]:
    from maple_run.eval_lcb import LCB_SYSTEM, lcb_user_prompt, load_lcb_v6

    problems = []
    for item in load_lcb_v6(limit=limit):
        problems.append(
            Problem(
                bench="lcbv6",
                problem_id=item.question_id,
                prompt=lcb_user_prompt(item),
                gold="",
                system=LCB_SYSTEM,
                extra={"lcb": item},
            )
        )
    return problems


LOADERS = {
    "aime2026": load_aime2026,
    "hmmt2026": load_hmmt2026,
    "gpqa": load_gpqa,
    "lcbv6": load_lcbv6,
}


def gpqa_prompt(problem: Problem, sample_idx: int, seed: int) -> tuple[str, str]:
    rng = random.Random(f"{seed}:{problem.problem_id}:{sample_idx}:gpqa")
    perm = rng.sample(range(4), 4)
    choices = [problem.extra["choices"][i] for i in perm]
    letter = "ABCD"[choices.index(problem.gold)]
    prompt = GPQA_TEMPLATE.format(
        question=problem.extra["question"],
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
    )
    return prompt, letter


def full_text(reasoning: str, text: str) -> str:
    if reasoning and text:
        return f"{reasoning}\n{text}"
    return reasoning or text


def score_sample(problem: Problem, text: str, extra: dict) -> tuple[bool, str | None]:
    if problem.bench == "aime2026":
        pred = extract_aime_answer(text)
        return aime_correct(pred, problem.gold), pred
    if problem.bench == "hmmt2026":
        pred = extract_last_boxed(text) or extract_aime_answer(text)
        return math_correct(pred, problem.gold), pred
    if problem.bench == "gpqa":
        pred = extract_gpqa_letter(text)
        return gpqa_correct(pred, extra["gold_letter"]), pred
    if problem.bench == "lcbv6":
        from maple_run.eval_lcb import grade_lcb_code

        pred = text
        ok = grade_lcb_code(text, problem.extra["lcb"])
        return ok, "pass" if ok else "fail"
    raise ValueError(problem.bench)


def _done_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        keys.add((str(rec["problem_id"]), int(rec["sample_idx"])))
    return keys


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    n_ok = sum(1 for r in records if r.get("correct"))
    acc = 100.0 * n_ok / n if n else 0.0
    by_problem: dict[str, list[bool]] = {}
    for rec in records:
        by_problem.setdefault(rec["problem_id"], []).append(bool(rec.get("correct")))
    return {
        "n_samples": n,
        "n_problems": len(by_problem),
        "n_correct": n_ok,
        "accuracy": acc,
        "pass1": acc,
    }


def run_eval(
    model_dir: str,
    benches: Iterable[str],
    *,
    output_dir: str | Path = "evals",
    n_samples: int = DEFAULT_N_SAMPLES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = DEFAULT_SEED,
    flash_head: bool = False,
    limit: int | None = None,
    generate_one: Callable | None = None,
) -> dict:
    """Run one or more benches. ``generate_one`` is for tests (skips CUDA)."""
    benches = list(benches)
    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    model = tokenizer = None
    if generate_one is None:
        print(f"Loading packed model from {model_dir}", flush=True)
        if flash_head:
            print(
                "Warning: DeepGrove reported the dense head; --flash-head is approximate",
                flush=True,
            )
        model, tokenizer = load_packed(model_dir, flash_head=flash_head)
        log_model_traffic(model, flash_head=flash_head)
        warmup_model(model, flash_head=flash_head)

    summaries: dict[str, dict] = {}
    for bench in benches:
        problems = LOADERS[bench](limit=limit)
        bench_dir = out_root / bench
        jsonl = bench_dir / "results.jsonl"
        done = _done_keys(jsonl)
        print(
            f"{bench}: {len(problems)} problems x {n_samples} samples "
            f"({len(done)} already on disk)",
            flush=True,
        )
        for p_i, problem in enumerate(problems):
            for s_i in range(n_samples):
                if (problem.problem_id, s_i) in done:
                    continue
                prompt = problem.prompt
                extra: dict = {}
                if bench == "gpqa":
                    prompt, letter = gpqa_prompt(problem, s_i, seed or 0)
                    extra["gold_letter"] = letter
                sample_seed = None
                if seed is not None:
                    sample_seed = int(seed) + p_i * n_samples + s_i
                t0 = time.perf_counter()
                if generate_one is not None:
                    result = generate_one(problem, prompt, extra, sample_seed)
                    reasoning = getattr(result, "reasoning", "")
                    text = result.text if hasattr(result, "text") else str(result)
                    n_new = getattr(result, "n_new", 0)
                    decode_tok_s = getattr(result, "decode_tok_s", 0.0)
                else:
                    messages = []
                    if problem.system:
                        messages.append({"role": "system", "content": problem.system})
                    messages.append({"role": "user", "content": prompt})
                    input_ids = encode_messages(tokenizer, messages, model.device)
                    result = run_generate(
                        model,
                        tokenizer,
                        input_ids,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=sample_seed,
                        log=None,
                    )
                    reasoning = result.reasoning
                    text = result.text
                    n_new = result.n_new
                    decode_tok_s = result.decode_tok_s
                blob = full_text(reasoning, text)
                correct, pred = score_sample(problem, blob, extra)
                rec = {
                    "bench": bench,
                    "problem_id": problem.problem_id,
                    "sample_idx": s_i,
                    "correct": bool(correct),
                    "pred": pred,
                    "gold": extra.get("gold_letter", problem.gold),
                    "n_new": n_new,
                    "decode_tok_s": decode_tok_s,
                    "elapsed_s": time.perf_counter() - t0,
                    "seed": sample_seed,
                    "finish_reason": getattr(result, "finish_reason", None),
                    "text": text,
                    "reasoning": reasoning,
                }
                _append_jsonl(jsonl, rec)
                done.add((problem.problem_id, s_i))
                mark = "ok" if correct else "miss"
                print(
                    f"  [{bench}] {problem.problem_id} sample {s_i} {mark} "
                    f"pred={pred!r} {n_new} tok {decode_tok_s:.1f} tok/s",
                    flush=True,
                )

        records = []
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        summary = _summarize(records)
        summary["bench"] = bench
        summary["deepgrove"] = DEEPGROVE_SCORES.get(bench)
        summary["protocol"] = {
            "n_samples": n_samples,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "flash_head": flash_head,
            "head": "flash" if flash_head else "dense",
        }
        (bench_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries[bench] = summary
        ref = summary["deepgrove"]
        ref_s = f"  DeepGrove {ref:.1f}%" if ref is not None else ""
        print(
            f"{bench}: {summary['accuracy']:.1f}% "
            f"({summary['n_correct']}/{summary['n_samples']})"
            f"{ref_s}",
            flush=True,
        )

    scores = [summaries[b]["accuracy"] for b in benches if b in summaries]
    overall = {
        "benches": summaries,
        "mean": sum(scores) / len(scores) if scores else 0.0,
        "deepgrove_mean": sum(DEEPGROVE_SCORES[b] for b in ALL_BENCHES) / 4,
    }
    (out_root / "summary.json").write_text(
        json.dumps(overall, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Mean over {len(benches)} bench(es): {overall['mean']:.1f}% "
        f"(DeepGrove 78.7%)",
        flush=True,
    )
    return overall
