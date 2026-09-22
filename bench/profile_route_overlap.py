"""Profile empirical route locality and expert overlap across layers in Maple.

Phase 0 deliverable for Expert-Reuse Speculation (ERS):
Measures expert reuse, union expansion, and prefix survival depth across
draft horizons M in [1, 16] and budgets delta in {0, 1, 2, 4}.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from maple_run.generate import load_packed, run_generate


def parse_args():
    parser = argparse.ArgumentParser(description="Profile MoE route locality for ERS")
    parser.add_argument(
        "--model",
        default="checkpoints/maple-2bit",
        help="Path to packed checkpoint directory",
    )
    parser.add_argument(
        "--prompts",
        default="tmp/prompts.jsonl",
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of prompts to evaluate",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Number of new decode tokens to generate per prompt",
    )
    parser.add_argument(
        "--max-draft-len",
        type=int,
        default=16,
        help="Maximum speculative draft window to evaluate (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy argmax)",
    )
    parser.add_argument(
        "--output",
        default="runs/route_overlap_analysis.json",
        help="Output JSON summary path",
    )
    return parser.parse_args()


def load_prompts(prompts_path: str, limit: int) -> list[str]:
    path = Path(prompts_path)
    if not path.exists():
        # Fallback default prompts if file missing
        return [
            "Explain how a compiler optimizes abstract syntax trees into machine code.",
            "Write a Python implementation of a LRU cache with O(1) operations.",
            "Discuss the historical causes and economic ramifications of the industrial revolution.",
            "What are the primary differences between TCP and UDP networking protocols?",
            "Solve the following mathematical puzzle: If 5 machines make 5 widgets in 5 minutes...",
        ][:limit]

    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = data.get("prompt") or data.get("text") or data.get("instruction")
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= limit:
                break
    return prompts


def analyze_route_traces(
    all_traces: list[list[list[set[int]]]],
    top_k: int = 8,
    max_draft: int = 16,
    deltas: tuple[int, ...] = (0, 1, 2, 4),
) -> dict:
    """Analyze router traces across prompts.

    all_traces: [prompt_idx][layer_idx][token_idx] -> set of expert IDs
    """
    n_prompts = len(all_traces)
    n_layers = len(all_traces[0])

    # 1. Pairwise subset probability Pr(E_{l, t+j} subseteq E_{l, t})
    subset_hits = defaultdict(int)
    subset_total = defaultdict(int)

    # 2. Average union size U(l, m) for draft length m in [1, max_draft]
    union_sizes_by_m = defaultdict(list)
    union_sizes_by_layer_m = defaultdict(lambda: defaultdict(list))

    # 3. Bounded union satisfaction Pr(U(l, m) <= top_k + delta)
    single_layer_surv = defaultdict(lambda: defaultdict(int))
    single_layer_total = defaultdict(lambda: defaultdict(int))

    # 4. Multi-layer joint survival Pr(forall l: U(l, m) <= top_k + delta)
    multi_layer_surv = defaultdict(lambda: defaultdict(int))
    multi_layer_total = defaultdict(int)

    # 5. Longest prefix survival depth per starting token
    survival_depths = {delta: [] for delta in deltas}

    # 6. Consecutive token expert intersection |E_{l, t} \cap E_{l, t+1}|
    intersections = []

    # 7. Cumulative budget across all layers: sum_l (|U(l, m)| - top_k) <= B
    cum_budgets = (0, 12, 24, 48, 72, 96)
    cum_survival_depths = {b: [] for b in cum_budgets}
    cum_expansion_by_m = defaultdict(list)

    for p_trace in all_traces:
        # p_trace is [layer_idx][token_idx] -> set of expert IDs
        T = len(p_trace[0])
        if T < 2:
            continue

        for t in range(T - 1):
            avail_m = min(max_draft, T - t)
            multi_layer_total[avail_m] += 1

            # Track consecutive token intersection
            for l in range(n_layers):
                intersections.append(len(p_trace[l][t].intersection(p_trace[l][t + 1])))

            # Check survival depths for each per-layer delta
            for delta in deltas:
                max_valid_m = 1
                for m in range(2, avail_m + 1):
                    all_layers_ok = True
                    for l in range(n_layers):
                        u_set = set()
                        for i in range(m):
                            u_set.update(p_trace[l][t + i])
                        if len(u_set) > top_k + delta:
                            all_layers_ok = False
                            break
                    if all_layers_ok:
                        max_valid_m = m
                    else:
                        break
                survival_depths[delta].append(max_valid_m)

            # Check cumulative budget survival depths
            for b in cum_budgets:
                max_valid_m = 1
                for m in range(2, avail_m + 1):
                    tot_extra = 0
                    for l in range(n_layers):
                        u_set = set()
                        for i in range(m):
                            u_set.update(p_trace[l][t + i])
                        tot_extra += (len(u_set) - top_k)
                    if tot_extra <= b:
                        max_valid_m = m
                    else:
                        break
                cum_survival_depths[b].append(max_valid_m)

            # Check per-m statistics
            for m in range(1, avail_m + 1):
                all_layers_ok_for_m = {delta: True for delta in deltas}
                tot_extra_for_m = 0

                for l in range(n_layers):
                    u_set = set()
                    for i in range(m):
                        u_set.update(p_trace[l][t + i])
                    u_size = len(u_set)
                    tot_extra_for_m += (u_size - top_k)

                    union_sizes_by_m[m].append(u_size)
                    union_sizes_by_layer_m[l][m].append(u_size)

                    for delta in deltas:
                        if u_size <= top_k + delta:
                            single_layer_surv[delta][m] += 1
                        else:
                            all_layers_ok_for_m[delta] = False
                        single_layer_total[delta][m] += 1

                cum_expansion_by_m[m].append(tot_extra_for_m)
                for delta in deltas:
                    if all_layers_ok_for_m[delta]:
                        multi_layer_surv[delta][m] += 1

            # Pairwise overlap Pr(E_{t+j} subseteq E_t)
            for j in range(1, min(max_draft, T - t)):
                for l in range(n_layers):
                    subset_total[j] += 1
                    if p_trace[l][t + j].issubset(p_trace[l][t]):
                        subset_hits[j] += 1

    # Aggregate summaries
    mean_union_size = {
        m: float(sum(vals) / len(vals)) for m, vals in sorted(union_sizes_by_m.items())
    }

    pairwise_subset_prob = {
        j: float(subset_hits[j] / subset_total[j]) if subset_total[j] > 0 else 0.0
        for j in sorted(subset_total.keys())
    }

    single_layer_rates = {
        f"delta_{delta}": {
            m: float(single_layer_surv[delta][m] / single_layer_total[delta][m])
            if single_layer_total[delta][m] > 0
            else 0.0
            for m in sorted(single_layer_total[delta].keys())
        }
        for delta in deltas
    }

    # Normalize multi-layer survival by all evaluation opportunities where horizon >= m
    multi_layer_rates = {}
    for delta in deltas:
        multi_layer_rates[f"delta_{delta}"] = {}
        for m in sorted(union_sizes_by_m.keys()):
            tot = sum(1 for d in survival_depths[delta] if True)
            # count how many tokens had survival depth >= m
            surv = sum(1 for d in survival_depths[delta] if d >= m)
            multi_layer_rates[f"delta_{delta}"][m] = float(surv / tot) if tot > 0 else 0.0

    mean_survival_depth = {
        f"delta_{delta}": float(sum(depths) / len(depths)) if depths else 1.0
        for delta, depths in survival_depths.items()
    }

    cum_mean_survival_depth = {
        f"budget_{b}": float(sum(depths) / len(depths)) if depths else 1.0
        for b, depths in cum_survival_depths.items()
    }

    cum_mean_expansion_by_m = {
        m: float(sum(vals) / len(vals)) for m, vals in sorted(cum_expansion_by_m.items())
    }

    mean_intersection = float(sum(intersections) / len(intersections)) if intersections else 0.0
    intersection_hist = {
        k: float(intersections.count(k) / len(intersections)) if intersections else 0.0
        for k in range(top_k + 1)
    }

    layer_breakdown = {}
    for l in range(n_layers):
        layer_breakdown[f"layer_{l}"] = {
            f"mean_union_m{m}": float(sum(union_sizes_by_layer_m[l][m]) / len(union_sizes_by_layer_m[l][m]))
            for m in [2, 4, 8, 16]
            if m in union_sizes_by_layer_m[l]
        }

    return {
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "top_k": top_k,
        "mean_intersection_consecutive": mean_intersection,
        "intersection_histogram": intersection_hist,
        "mean_union_size_by_m": mean_union_size,
        "cum_mean_expansion_by_m": cum_mean_expansion_by_m,
        "pairwise_subset_prob": pairwise_subset_prob,
        "single_layer_survival_rates": single_layer_rates,
        "multi_layer_survival_rates": multi_layer_rates,
        "mean_survival_depth": mean_survival_depth,
        "cum_mean_survival_depth": cum_mean_survival_depth,
        "layer_breakdown": layer_breakdown,
    }


def main():
    args = parse_args()
    print(f"Loading packed model from {args.model}...")
    model, tokenizer = load_packed(
        args.model,
        flash_head=False,
        dense_head_source=None,
    )
    top_k = int(model.config["num_experts_per_tok"])
    n_layers = int(model.config["num_hidden_layers"])
    print(f"Model loaded: {n_layers} layers, top_k={top_k}, num_experts={model.config['num_experts']}")

    prompts = load_prompts(args.prompts, args.limit)
    print(f"Profiling route locality over {len(prompts)} prompts (max {args.max_tokens} new tokens each)...")

    all_traces = []

    for p_idx, prompt in enumerate(prompts):
        print(f"\n--- Prompt {p_idx + 1}/{len(prompts)} ---")
        print(f"Prompt snippet: {prompt[:80]}...")

        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens, device=model.device, dtype=torch.long)

        # Tracing buffer: [layer][decode_step] -> set(expert_ids)
        current_trace = [[] for _ in range(n_layers)]

        def make_hook(layer_idx: int):
            def hook(topk_idx: torch.Tensor):
                # topk_idx is [batch*seq_len, top_k]
                # For decode step, batch*seq_len is 1
                if topk_idx.shape[0] == 1:
                    experts = set(topk_idx[0].cpu().tolist())
                    current_trace[layer_idx].append(experts)
            return hook

        # Attach hook to each layer
        for i, layer in enumerate(model.layers):
            layer.mlp.router_hook = make_hook(i)

        # Run eager generation (use_graph=False so router hooks fire on each step)
        res = run_generate(
            model,
            tokenizer,
            input_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_graph=False,
            log=None,
        )

        # Detach hooks
        model.set_router_hook(None)

        n_steps = len(current_trace[0])
        print(f"Generated {n_steps} decode steps (reasoning: {len(res.reasoning)} chars, text: {len(res.text)} chars)")
        if n_steps > 1:
            all_traces.append(current_trace)

    if not all_traces:
        print("No decode traces collected.")
        return

    print("\n" + "=" * 70)
    print("ANALYZING ROUTE OVERLAP & SPECULATIVE SURVIVAL METRICS")
    print("=" * 70)

    analysis = analyze_route_traces(
        all_traces,
        top_k=top_k,
        max_draft=args.max_draft_len,
        deltas=(0, 1, 2, 4),
    )

    # Print summary tables
    print(f"\n[0] Consecutive Token Expert Overlap: {analysis['mean_intersection_consecutive']:.2f} / {top_k} experts in common (Random: {top_k * top_k / int(model.config['num_experts']):.2f})")
    print("  Intersection histogram P(|E_t ∩ E_{t+1}| = i):")
    hist_str = "  " + " | ".join(f"i={k}: {v*100:4.1f}%" for k, v in analysis["intersection_histogram"].items() if k <= 6)
    print(hist_str)

    print("\n[1] Average Expert Working Set Union Size |U_l(m)| across draft length m:")
    print("m (draft len) | Avg Unique Experts | Amplification vs AR (8)")
    print("-" * 55)
    for m in [1, 2, 3, 4, 6, 8, 12, 16]:
        if m in analysis["mean_union_size_by_m"]:
            u = analysis["mean_union_size_by_m"][m]
            amp = u / top_k
            print(f"{m:<13} | {u:<18.2f} | {amp:<.2f}x")

    print("\n[2] Single-Layer vs Multi-Layer Joint Survival Rate at depth m (Strict Per-Layer Bound):")
    print("m   | Delta=0 (Single / Multi) | Delta=1 (Single / Multi) | Delta=2 (Single / Multi) | Delta=4 (Single / Multi)")
    print("-" * 98)
    for m in [2, 3, 4, 6, 8, 12, 16]:
        row = f"{m:<3} | "
        for delta in [0, 1, 2, 4]:
            s_rate = analysis["single_layer_survival_rates"][f"delta_{delta}"].get(m, 0.0) * 100
            m_rate = analysis["multi_layer_survival_rates"][f"delta_{delta}"].get(m, 0.0) * 100
            row += f"{s_rate:4.1f}% / {m_rate:4.1f}%          | "
        print(row)

    print("\n[3] Mean Speculative Prefix Survival Depth (Strict Per-Layer Bound, max draft = 16):")
    for delta, depth in analysis["mean_survival_depth"].items():
        print(f"  {delta}: {depth:.2f} tokens")

    print("\n[4] Cumulative Multi-Layer Budget Analysis (Total Extra Expert Loads Across All 24 Layers):")
    print("Budget B (Extra Loads) | Weight Amp Bound | Mean Survival Depth (Tokens)")
    print("-" * 65)
    base_loads = 24 * top_k
    for b_str, depth in analysis["cum_mean_survival_depth"].items():
        b_val = int(b_str.replace("budget_", ""))
        amp = (base_loads + b_val) / base_loads
        print(f"B={b_val:<20} | {amp:<16.3f}x | {depth:<.2f} tokens")

    # Save output JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved detailed analysis report to {out_path}")


if __name__ == "__main__":
    main()
