"""Batched-decode gates and throughput report.

    uv run --extra cuda python bench/bench_batched_decode.py gate
    uv run --extra cuda python bench/bench_batched_decode.py report

Three gates, because "same as B=1" mixes several different claims.

``companion`` is the plumbing gate and the strict one: a row's tokens must not
depend on which other rows share its batch. Both sides run the batched
kernels, so any difference is the batch dimension itself -- ``seqlen[b]``,
``stride_kb``/``stride_vb``, the slot -> cache-row map, the grid -- and the
tolerance is exact equality. Prompt lengths are staggered so no two rows share
a seqlen.

``graph`` is the same claim for the captured decode graphs: replaying a
bucketed graph must give the ids eager gives, including after rows retire and
the replays drop to a narrower bucket.

``solo`` is the literal gate: batched greedy vs the same prompt decoded alone
at B=1. B=1 dispatches to the tuned batch-1 GEMV and B>1 to the batched GEMM,
which sum the same exact products in a different order, so this one can
legitimately differ by a bf16 ULP that the MoE router's top-8 then amplifies.
It is reported as a match rate, not asserted.

``report`` compares aggregate tok/s for batched decode, eager against
bucketed-graph, at each batch size.
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


def _quiet(_msg: str) -> None:
    """Swallow per-run graph capture chatter; the tables are the output."""


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


def gate_graph(model, tokenizer, max_tokens: int) -> int:
    """Bucketed graph replay must reproduce eager ids, narrowing included."""
    bad = 0
    for bsz in (1, 2, 5, 8, 16):
        prompts = _prompts(tokenizer, model.device, _lengths(bsz))
        eager = batched_generate(
            model, prompts, max_tokens=max_tokens, stop_ids=set(), log=_quiet
        )
        graph = batched_generate(
            model, prompts, max_tokens=max_tokens, stop_ids=set(), graphs=True, log=_quiet
        )
        ok = graph.token_ids == eager.token_ids
        bad += 0 if ok else 1
        used = sorted(set(graph.widths))
        print(f"  B={bsz:<3} widths {used}  {'match' if ok else 'MISMATCH vs eager'}")

    # Now retire rows mid-run so the replays have to narrow. Real EOS is too
    # rare in a short window to exercise it, so the stop set is taken from the
    # rows themselves, at staggered positions.
    bsz = 16
    prompts = _prompts(tokenizer, model.device, _lengths(bsz))
    free = batched_generate(
        model, prompts, max_tokens=max_tokens, stop_ids=set(), log=_quiet
    )
    # Pick, for each row, a token whose first appearance anywhere in the batch
    # is in that row at a staggered position -- otherwise a common token
    # ("the") retires the whole batch on the same step.
    first_at: dict[int, tuple[int, int]] = {}
    for b, toks in enumerate(free.token_ids):
        for i, t in enumerate(toks):
            if t not in first_at or i < first_at[t][0]:
                first_at[t] = (i, b)
    stop_ids = set()
    for b in range(bsz):
        want = 4 + 4 * b
        for i, t in enumerate(free.token_ids[b]):
            if i >= want and first_at.get(t) == (i, b) and t not in stop_ids:
                stop_ids.add(t)
                break
    eager = batched_generate(
        model, prompts, max_tokens=max_tokens, stop_ids=stop_ids, log=_quiet
    )
    graph = batched_generate(
        model, prompts, max_tokens=max_tokens, stop_ids=stop_ids, graphs=True, log=_quiet
    )
    ok = graph.token_ids == eager.token_ids
    bad += 0 if ok else 1
    hist: dict[int, int] = {}
    for w in graph.widths:
        hist[w] = hist.get(w, 0) + 1
    alive = [len(t) for t in graph.token_ids]
    print(
        f"  B={bsz} with staggered stops: "
        f"{'match' if ok else 'MISMATCH vs eager'}; "
        f"rows ran {min(alive)}..{max(alive)} tokens; "
        f"replayed widths {dict(sorted(hist.items()))}"
    )
    if len(hist) == 1:
        print("    (no row retired in this window, so nothing narrowed)")
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


def _run(model, tokenizer, bsz, lengths, max_tokens, graphs):
    prompts = _prompts(tokenizer, model.device, lengths)
    batched_generate(
        model, prompts, max_tokens=4, stop_ids=set(), graphs=graphs, log=_quiet
    )
    return batched_generate(
        model, prompts, max_tokens=max_tokens, stop_ids=set(), graphs=graphs, log=_quiet
    )


def _table(model, tokenizer, max_tokens, lengths_for, title) -> None:
    print(f"\n  {title}")
    print(
        f"  {'B':>3}  {'prompt lens':>12}  {'eager tok/s':>11}  {'graph tok/s':>11}  "
        f"{'graph/eager':>11}  {'graph per-row':>13}  {'vs B=1 graph':>12}"
    )
    base = None
    for bsz in (1, 2, 4, 8, 16, 32):
        lengths = lengths_for(bsz)
        eager = _run(model, tokenizer, bsz, lengths, max_tokens, False)
        graph = _run(model, tokenizer, bsz, lengths, max_tokens, True)
        e, g = eager.decode_tok_s, graph.decode_tok_s
        if base is None:
            base = g
        span = (
            f"{lengths[0]}"
            if len(set(lengths)) == 1
            else f"{min(lengths)}..{max(lengths)}"
        )
        print(
            f"  {bsz:>3}  {span:>12}  {e:>11.1f}  {g:>11.1f}  {g / e:>10.2f}x  "
            f"{g / bsz:>13.1f}  {g / base:>11.2f}x"
        )


def report(model, tokenizer, max_tokens: int) -> None:
    """Aggregate tok/s vs B, greedy, eager against bucketed decode graphs.

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
    print(f"\n  greedy decode, {max_tokens} tokens/row")
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


def _prefilled_runner(model, tokenizer, bsz, prompt_len, max_tokens):
    """A cache prefilled with ``bsz`` prompts plus its bucketed graphs."""
    from maple_run.generate import _try_batched_decode_graphs

    prompts = _prompts(tokenizer, model.device, [prompt_len] * bsz)
    cache = model.make_cache(max_len=prompt_len + max_tokens + 8, batch=bsz)
    pending = []
    with torch.inference_mode():
        for b, p in enumerate(prompts):
            with cache.prefill_slot(b):
                lg = model.forward(p.view(1, -1), cache=cache, logits_to_keep=1)
            pending.append(int(lg[0, -1].argmax()))
    cache.remap = True
    runner = _try_batched_decode_graphs(
        model, cache, pending, sample=None, log=lambda _m: None
    )
    return cache, runner, pending


def narrowing(model, tokenizer, max_tokens: int) -> None:
    """What bucketing buys once rows retire.

    Measured: the replay cost of each captured width. Then projected onto a
    retirement trajectory, because a real trajectory is hard to stage here --
    under greedy decode the rows share most of their vocabulary, so any stop
    id retires most of the batch on the same step rather than staggering.
    That the narrowing is *correct* is the graph gate's job; this is only
    about what it saves.
    """
    bsz, prompt_len = 32, 256
    cache, runner, pending = _prefilled_runner(model, tokenizer, bsz, prompt_len, 64)
    print(f"\n  replay cost per captured width (B={bsz} cache, {prompt_len}-token rows)")
    print(f"  {'width':>6}  {'ms/step':>8}  {'vs width 1':>10}")
    cost: dict[int, float] = {}
    with torch.inference_mode():
        for w in runner.widths:
            tok = torch.tensor(
                pending[:w], device=model.device, dtype=torch.long
            ).view(w, 1)
            for _ in range(3):
                runner.replay(tok)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                runner.replay(tok)
            torch.cuda.synchronize()
            cost[w] = (time.perf_counter() - t0) / 20 * 1000
            print(f"  {w:>6}  {cost[w]:>8.2f}  {cost[w] / cost[1]:>9.2f}x")

    # Rows retire evenly across the window, so the live count walks 32 -> 1.
    steps = int(max_tokens)
    pinned = bucketed = 0.0
    for i in range(steps):
        live = max(1, bsz - (bsz * i) // steps)
        pinned += cost[bsz]
        bucketed += cost[runner.width_for(live)]
    print(
        f"\n  projected over {steps} steps with rows retiring evenly "
        f"({bsz} -> 1 live):"
    )
    print(f"    pinned at {bsz}: {pinned:7.0f} ms")
    print(f"    bucketed:     {bucketed:7.0f} ms   {pinned / bucketed:.2f}x faster")
    del cache, runner


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
        print("\n=== gate: bucketed graph replay == eager ===")
        bad = gate_graph(model, tokenizer, args.max_tokens)
        rc += bad
        print("  PASS" if bad == 0 else f"  FAIL: {bad} configuration(s) diverged")
        print("\n=== batched vs true B=1 (different kernel, reported) ===")
        matched, rows = gate_solo(model, tokenizer, args.max_tokens)
        print(f"  {matched}/{rows} rows token-identical to a B=1 run")
    if args.mode in ("report", "all"):
        print("\n=== aggregate tok/s: eager vs bucketed decode graphs ===")
        report(model, tokenizer, args.max_tokens)
        print("\n=== what bucketing buys as rows retire ===")
        narrowing(model, tokenizer, args.max_tokens)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
