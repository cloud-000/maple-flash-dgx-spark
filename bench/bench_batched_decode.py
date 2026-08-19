"""Batched-decode gate and throughput report.

    uv run --extra cuda python bench/bench_batched_decode.py gate
    uv run --extra cuda python bench/bench_batched_decode.py report

Two gates, because "same as B=1" mixes two different claims.

``companion`` is the plumbing gate and the strict one: a row's tokens must not
depend on which other rows share its batch. Both sides run the batched
kernels, so any difference is the batch dimension itself -- ``seqlen[b]``,
``stride_kb``/``stride_vb``, the grid -- and the tolerance is exact equality.
Prompt lengths are staggered so no two rows share a seqlen.

``solo`` is the literal gate: batched greedy vs the same prompt decoded alone
at B=1. B=1 dispatches to the tuned batch-1 GEMV and B>1 to the batched GEMM,
which sum the same exact products in a different order, so this one can
legitimately differ by a bf16 ULP that the MoE router's top-8 then amplifies.
It is reported as a match rate, not asserted.

``report`` compares aggregate tok/s for batched eager decode against B=1
eager decode. Both sides are eager on purpose: the CUDA-graph number is not
comparable until the graphs are bucketed by batch size.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, "/home/cloud/Code/maple-run")

from maple_run.generate import batched_generate, load_packed, stop_token_ids

MODEL_DIR = "checkpoints/maple-2bit"
SEED = 1234

_TEXT = (
    "The history of numerical computing on small devices is a history of "
    "trading precision for locality. Explain, step by step and with concrete "
    "examples, why moving a weight from memory costs more than multiplying "
    "it, and what that implies for the design of quantized inference kernels. "
) * 24


def _prompts(tokenizer, device, lengths):
    """One prompt per requested token length, all from the same text."""
    ids = tokenizer.encode(_TEXT)
    out = []
    for n in lengths:
        if n > len(ids):
            raise ValueError(f"prompt text has only {len(ids)} tokens, need {n}")
        out.append(torch.tensor(ids[:n], device=device, dtype=torch.long))
    return out


def _lengths(bsz: int) -> list[int]:
    """Staggered lengths so no two rows share a seqlen."""
    return [16 + 37 * i for i in range(bsz)]


def _decode(model, prompts, max_tokens, stop_ids):
    torch.manual_seed(SEED)
    return batched_generate(
        model, prompts, max_tokens=max_tokens, stop_ids=stop_ids
    ).token_ids


def _first_diff(got, want):
    n = min(len(got), len(want))
    for i in range(n):
        if got[i] != want[i]:
            return i
    return None if len(got) == len(want) else n


def gate_companion(model, tokenizer, max_tokens: int) -> int:
    """A row's tokens must not depend on its companions. Exact equality."""
    stop_ids = stop_token_ids(model, tokenizer)
    bad = 0
    for bsz in (1, 2, 3, 5, 8, 11, 16):
        prompts = _prompts(tokenizer, model.device, _lengths(bsz))
        batched = _decode(model, prompts, max_tokens, stop_ids)
        # Reference: the same row paired only with a copy of itself, so it is
        # still on the batched kernel but shares its batch with nobody else.
        alone = [_decode(model, [p, p], max_tokens, stop_ids)[0] for p in prompts]
        rows = sum(1 for g, w in zip(batched, alone, strict=True) if g != w)
        bad += rows
        lens = ",".join(str(len(p)) for p in prompts)
        print(
            f"  B={bsz:<3} lens[{lens}]  "
            f"{'all rows identical' if rows == 0 else f'{rows} ROWS DIFFER'}"
        )
        for b, (g, w) in enumerate(zip(batched, alone, strict=True)):
            if g != w:
                i = _first_diff(g, w)
                print(f"      row {b} diverges at token {i}: {g[i:i+4]} vs {w[i:i+4]}")
    return bad


def gate_solo(model, tokenizer, max_tokens: int) -> tuple[int, int]:
    """Batched vs true B=1. Reported, not asserted -- see the module docstring."""
    stop_ids = stop_token_ids(model, tokenizer)
    rows = matched = 0
    for bsz in (2, 3, 5, 8, 11, 16):
        prompts = _prompts(tokenizer, model.device, _lengths(bsz))
        batched = _decode(model, prompts, max_tokens, stop_ids)
        solo = [_decode(model, [p], max_tokens, stop_ids)[0] for p in prompts]
        same = sum(1 for g, w in zip(batched, solo, strict=True) if g == w)
        rows += bsz
        matched += same
        detail = ""
        for b, (g, w) in enumerate(zip(batched, solo, strict=True)):
            if g != w:
                detail += f"  [row {b} len {len(prompts[b])} @ tok {_first_diff(g, w)}]"
        print(f"  B={bsz:<3} {same}/{bsz} rows token-identical{detail}")
    return matched, rows


def _table(model, tokenizer, max_tokens, lengths_for, title) -> None:
    print(f"\n  {title}")
    print(
        f"  {'B':>3}  {'prompt lens':>13}  {'steps':>5}  {'decode s':>8}  "
        f"{'tok/s':>7}  {'per-row':>7}  {'vs B=1':>7}"
    )
    base = None
    for bsz in (1, 2, 4, 8, 16, 32):
        lengths = lengths_for(bsz)
        prompts = _prompts(tokenizer, model.device, lengths)
        batched_generate(model, prompts, max_tokens=4, stop_ids=set())  # warm
        r = batched_generate(model, prompts, max_tokens=max_tokens, stop_ids=set())
        tok_s = r.decode_tok_s
        if base is None:
            base = tok_s
        span = (
            f"{lengths[0]}"
            if len(set(lengths)) == 1
            else f"{min(lengths)}..{max(lengths)}"
        )
        print(
            f"  {bsz:>3}  {span:>13}  {r.steps:>5}  {r.decode_s:>8.3f}  "
            f"{tok_s:>7.1f}  {tok_s / bsz:>7.1f}  {tok_s / base:>6.2f}x"
        )


def report(model, tokenizer, max_tokens: int) -> None:
    """Aggregate tok/s vs B, greedy eager, no graphs and no fused sampler.

    Two tables. The uniform one is the scaling measurement: every row has the
    same prompt, so the only thing changing with B is the batch. The ragged one
    is the realistic mix, and it scales worse for a reason that is not the
    batching -- the lengths grow with B, so total KV read grows with B^2.

    What stops this short of linear is the MoE, not the projections. Per step
    at B=32 the packed QKV/O GEMM costs 1243 us against 887 at B=16 and the
    4-bit head 1246 against 941 -- both nearly flat, which is the step-1
    batched kernels doing their job -- while the expert kernels go 9273 us at
    B=16 to 17722 at B=32, near-linear, because top-8 of 256 experts means a
    wider batch touches more distinct expert rows rather than reusing the same
    ones. Grouping tokens by expert is the next lever, not more GEMV tuning.
    """
    print(f"\n  greedy eager decode, {max_tokens} tokens/row, no graphs")
    _table(
        model,
        tokenizer,
        max_tokens,
        lambda b: [256] * b,
        "uniform 256-token prompts (isolates batch scaling)",
    )
    _table(
        model,
        tokenizer,
        max_tokens,
        _lengths,
        "staggered lengths (realistic mix; KV work grows with B too)",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "report", "all"], default="all", nargs="?")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tokenizer = load_packed(args.model)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    rc = 0
    if args.mode in ("gate", "all"):
        print("\n=== gate: row output is independent of its batch companions ===")
        rc = gate_companion(model, tokenizer, args.max_tokens)
        print("  PASS" if rc == 0 else f"  FAIL: {rc} row(s) changed with companions")
        print("\n=== batched vs true B=1 (different kernel, reported) ===")
        matched, rows = gate_solo(model, tokenizer, args.max_tokens)
        print(f"  {matched}/{rows} rows token-identical to a B=1 run")
    if args.mode in ("report", "all"):
        print("\n=== aggregate tok/s: batched eager vs B=1 eager ===")
        report(model, tokenizer, args.max_tokens)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
