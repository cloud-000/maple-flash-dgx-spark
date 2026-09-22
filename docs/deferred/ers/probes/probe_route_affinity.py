"""Phase 0.5: Route-Affinity vs Linguistic Plausibility Probe.

For each consecutive token pair (t, t+1) observed during greedy generation,
compute the average per-layer expert overlap.  Group pairs into overlap buckets
and report example token texts — so we can see empirically whether high-overlap
token pairs are semantically related plausible continuations or arbitrary.

Key question: does high expert overlap correlate with linguistic plausibility?
If yes, route-affinity candidate selection preserves α.
If no, route-affinity trades α for HBM savings and ERS needs rethinking.

Usage:
    uv run python bench/probe_route_affinity.py [--limit N] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from maple_run.generate import load_packed, run_generate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/maple-2bit")
    parser.add_argument("--prompts", default="tmp/prompts.jsonl")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of prompts to run")
    parser.add_argument("--max-tokens", type=int, default=120,
                        help="Decode steps per prompt")
    parser.add_argument("--output", default="runs/route_affinity_probe.json")
    parser.add_argument("--examples-per-bucket", type=int, default=12,
                        help="Token pairs to show per overlap bucket")
    return parser.parse_args()


def load_prompts(path: str, limit: int) -> list[str]:
    p = Path(path)
    defaults = [
        "Explain how a compiler optimizes abstract syntax trees into machine code.",
        "Write a Python implementation of a balanced BST with insertion and search.",
        "Discuss the historical causes of the French Revolution.",
        "What are the primary differences between TCP and UDP?",
        "Solve: if 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
        "Summarize the key ideas in Keynesian economics.",
        "Describe the lifecycle of a star from nebula to remnant.",
    ]
    if not p.exists():
        return defaults[:limit]
    prompts = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("prompt") or obj.get("text") or obj.get("instruction", "")
            if text:
                prompts.append(text)
            if len(prompts) >= limit:
                break
    return prompts[:limit]


def mean_overlap(expert_sets_t: list[set], expert_sets_t1: list[set]) -> float:
    """Average |E_l,t ∩ E_l,t+1| across layers."""
    assert len(expert_sets_t) == len(expert_sets_t1)
    return sum(len(a & b) for a, b in zip(expert_sets_t, expert_sets_t1)) / len(expert_sets_t)


def bucket(overlap: float, top_k: int) -> str:
    frac = overlap / top_k
    if frac < 0.25:
        return "0–2 (0–25%)"
    elif frac < 0.50:
        return "2–4 (25–50%)"
    elif frac < 0.625:
        return "4–5 (50–62%)"
    elif frac < 0.75:
        return "5–6 (62–75%)"
    elif frac < 0.875:
        return "6–7 (75–87%)"
    else:
        return "7–8 (87–100%)"


def main():
    args = parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_packed(args.model, flash_head=False, dense_head_source=None)
    top_k = int(model.config["num_experts_per_tok"])
    n_layers = int(model.config["num_hidden_layers"])
    print(f"  {n_layers} layers, top_k={top_k}, N={model.config['num_experts']}")

    prompts = load_prompts(args.prompts, args.limit)
    print(f"Running on {len(prompts)} prompts × {args.max_tokens} tokens each\n")

    # Collect all (prev_token_id, next_token_id, mean_overlap, [expert_sets per layer for both])
    # Structure: list of dicts
    all_pairs: list[dict] = []

    for p_idx, prompt in enumerate(prompts):
        print(f"Prompt {p_idx+1}/{len(prompts)}: {prompt[:70]}...")

        input_ids = torch.tensor(
            tokenizer.encode(prompt), device=model.device, dtype=torch.long
        )

        # trace[layer][step] = set of expert ids
        trace: list[list[set[int]]] = [[] for _ in range(n_layers)]
        # generated token ids in decode order
        generated_ids: list[int] = []

        def make_hook(layer_idx: int):
            def hook(topk_idx: torch.Tensor):
                if topk_idx.shape[0] == 1:
                    trace[layer_idx].append(set(topk_idx[0].cpu().tolist()))
            return hook

        for i, layer in enumerate(model.layers):
            layer.mlp.router_hook = make_hook(i)

        # Monkey-patch to capture generated token ids.
        # run_generate returns a GenerateResult with .ids (full sequence).
        res = run_generate(
            model,
            tokenizer,
            input_ids,
            max_tokens=args.max_tokens,
            temperature=0.0,
            use_graph=False,
            log=None,
        )
        model.set_router_hook(None)

        # res.token_ids is a list of generated token ints (excluding the prompt).
        # trace[l] has one entry per decode step, aligned with token_ids.
        n_steps = len(trace[0])
        if n_steps < 2:
            print(f"  Only {n_steps} decode steps, skipping.")
            continue

        decode_ids = res.token_ids  # list[int], length == n_steps

        # Align: decode_ids[i] is the token emitted at step i, whose expert trace is trace[l][i].
        # Consecutive pair: (decode_ids[i], decode_ids[i+1]) with overlap between trace[l][i] and trace[l][i+1].
        for i in range(min(len(decode_ids) - 1, n_steps - 1)):
            prev_id = decode_ids[i]
            next_id = decode_ids[i + 1]
            experts_prev = [trace[l][i] for l in range(n_layers)]
            experts_next = [trace[l][i + 1] for l in range(n_layers)]
            ov = mean_overlap(experts_prev, experts_next)
            prev_tok = tokenizer.decode([prev_id])
            next_tok = tokenizer.decode([next_id])
            all_pairs.append({
                "prev_id": prev_id,
                "next_id": next_id,
                "prev_tok": prev_tok,
                "next_tok": next_tok,
                "mean_overlap": ov,
                "bucket": bucket(ov, top_k),
            })

        print(f"  Collected {n_steps - 1} consecutive pairs.")

    if not all_pairs:
        print("No pairs collected.")
        return

    # ── Analysis ──────────────────────────────────────────────────────────────

    print(f"\nTotal pairs collected: {len(all_pairs)}")
    print(f"Mean overlap across all pairs: {sum(p['mean_overlap'] for p in all_pairs)/len(all_pairs):.3f} / {top_k}")

    # Bucket distribution
    bucket_groups: dict[str, list[dict]] = defaultdict(list)
    for pair in all_pairs:
        bucket_groups[pair["bucket"]].append(pair)

    bucket_order = [
        "0–2 (0–25%)",
        "2–4 (25–50%)",
        "4–5 (50–62%)",
        "5–6 (62–75%)",
        "6–7 (75–87%)",
        "7–8 (87–100%)",
    ]

    print("\n" + "=" * 70)
    print("OVERLAP BUCKET DISTRIBUTION & EXAMPLE TOKEN PAIRS")
    print("=" * 70)
    print(f"\n{'Bucket':<22} {'Count':>6} {'Pct':>6}  Examples (prev → next)")
    print("-" * 70)

    results = {}
    for bkt in bucket_order:
        pairs_in_bkt = bucket_groups.get(bkt, [])
        count = len(pairs_in_bkt)
        pct = 100.0 * count / len(all_pairs) if all_pairs else 0
        # Sample examples
        sample = random.sample(pairs_in_bkt, min(args.examples_per_bucket, len(pairs_in_bkt)))
        examples = [(p["prev_tok"], p["next_tok"], round(p["mean_overlap"], 2)) for p in sample]

        print(f"\n{bkt:<22} {count:>6} ({pct:4.1f}%)")
        for prev_tok, next_tok, ov in examples[:6]:
            print(f"    overlap={ov:.2f}  {repr(prev_tok):>18} → {repr(next_tok)}")

        results[bkt] = {
            "count": count,
            "pct": round(pct, 2),
            "examples": [
                {"prev": p["prev_tok"], "next": p["next_tok"], "overlap": round(p["mean_overlap"], 2)}
                for p in sample
            ],
        }

    # ── ERS economics per bucket ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ERS ECONOMICS: WHAT OVERLAP BUCKET DO WE NEED TO WIN?")
    print("=" * 70)
    print(f"\n{'Bucket':<22} {'S_size':>7} {'Verify cost':>12} {'Break-even α':>14} {'Viable?':>8}")
    print("-" * 70)

    for bkt in bucket_order:
        pairs_in_bkt = bucket_groups.get(bkt, [])
        if not pairs_in_bkt:
            continue
        mean_ov = sum(p["mean_overlap"] for p in pairs_in_bkt) / len(pairs_in_bkt)
        s_size = top_k + (top_k - mean_ov)   # base experts + new experts from candidate
        verify_cost = s_size / top_k
        breakeven_alpha = verify_cost           # need α > verify_cost to beat AR
        # Realistic α with decent drafter: 1 frontier + p per candidate
        # Assume p=0.45 per-token acceptance for a decent small drafter
        p_accept = 0.45
        realistic_alpha = 1 + p_accept  # 1 candidate scenario
        viable = realistic_alpha > breakeven_alpha
        pct_in_bucket = 100.0 * len(pairs_in_bkt) / len(all_pairs)
        print(
            f"{bkt:<22} {s_size:>7.1f} {verify_cost:>11.2f}× {breakeven_alpha:>13.2f} "
            f"{'✅' if viable else '🔴':>8}  ({pct_in_bucket:.1f}% of pairs)"
        )

    print(f"\n  Assuming drafter p_accept=0.45 per token → realistic α = 1.45 for 1 candidate.")
    print(f"  Bucket needs verify_cost < 1.45 → S_size < {1.45 * top_k:.1f} → mean_overlap > {top_k - (1.45*top_k - top_k):.1f}/8")

    # Save
    out = {
        "n_pairs": len(all_pairs),
        "top_k": top_k,
        "n_layers": n_layers,
        "mean_overlap_overall": sum(p["mean_overlap"] for p in all_pairs) / len(all_pairs),
        "bucket_results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
