"""Probe expert popularity entropy, reuse distance, and cacheability in Maple.

Answers the question: Does Maple have cache locality without token-level route locality?
Measures:
1. Pairwise overlap |E_t ∩ E_{t+1}| (re-verifying ~2.6/8)
2. Expert popularity entropy H_l = -sum p_e log p_e and top-h cumulative mass H_l(h)
3. Reuse distance distribution (intervening token steps until expert e is re-used)
4. Cache hit rate under Static Top-C and Online LRU caches (C in {8, 16, 24, 32, 48, 64})
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from maple_run.generate import load_packed, run_generate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/maple-2bit")
    parser.add_argument("--prompts", default="tmp/prompts.jsonl")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of prompts to evaluate")
    parser.add_argument("--max-tokens", type=int, default=120,
                        help="Decode steps per prompt")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--output", default="runs/expert_cacheability_probe.json")
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
        "How does public key cryptography work under RSA?",
        "Explain the theory of general relativity and spacetime curvature.",
        "Write a comprehensive summary of how transformers use self-attention.",
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
    if len(prompts) < limit:
        prompts.extend(defaults[:limit - len(prompts)])
    return prompts[:limit]


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: deque[int] = deque()
        self.hits = 0
        self.misses = 0

    def access(self, item: int) -> bool:
        if item in self.cache:
            self.hits += 1
            self.cache.remove(item)
            self.cache.append(item)
            return True
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self.cache.popleft()
            self.cache.append(item)
            return False


def main():
    args = parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_packed(args.model, flash_head=False, dense_head_source=None)
    top_k = int(model.config["num_experts_per_tok"])
    n_layers = int(model.config["num_hidden_layers"])
    num_experts = int(model.config["num_experts"])
    max_entropy = math.log(num_experts)

    print(f"Config: {n_layers} layers, top_k={top_k}, N={num_experts} (max entropy = {max_entropy:.3f})")

    prompts = load_prompts(args.prompts, args.limit)
    print(f"Running on {len(prompts)} prompts × {args.max_tokens} tokens (temp={args.temperature})\n")

    # Storage per layer
    # layer_traces[l][prompt_idx] = list of expert sets (one per decode step)
    layer_traces: list[list[list[set[int]]]] = [[] for _ in range(n_layers)]

    for p_idx, prompt in enumerate(prompts):
        print(f"Prompt {p_idx+1}/{len(prompts)}: {prompt[:65]}...")
        input_ids = torch.tensor(
            tokenizer.encode(prompt), device=model.device, dtype=torch.long
        )

        trace: list[list[set[int]]] = [[] for _ in range(n_layers)]

        def make_hook(layer_idx: int):
            def hook(topk_idx: torch.Tensor):
                if topk_idx.shape[0] == 1:
                    trace[layer_idx].append(set(topk_idx[0].cpu().tolist()))
            return hook

        for i, layer in enumerate(model.layers):
            layer.mlp.router_hook = make_hook(i)

        res = run_generate(
            model,
            tokenizer,
            input_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_graph=False,
            log=None,
        )
        model.set_router_hook(None)

        n_steps = len(trace[0])
        if n_steps < 2:
            print(f"  Only {n_steps} steps, skipping.")
            continue

        for l in range(n_layers):
            layer_traces[l].append(trace[l])

    print("\n" + "="*80)
    print("ANALYSIS: POPULARITY ENTROPY, REUSE DISTANCE, AND CACHEABILITY")
    print("="*80)

    results_per_layer = []
    capacities = [8, 16, 24, 32, 48, 64]

    # Global collectors for summary
    all_layer_entropies = []
    all_layer_overlaps = []
    all_layer_h16 = []
    all_layer_h32 = []
    all_layer_h64 = []
    all_layer_median_rd = []
    all_layer_lru_hits = {c: [] for c in capacities}

    for l in range(n_layers):
        traces_l = layer_traces[l] # list of prompt sequences

        # 1. Pairwise overlap
        overlaps = []
        for seq in traces_l:
            for t in range(len(seq) - 1):
                overlaps.append(len(seq[t] & seq[t+1]))
        mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0

        # 2. Expert popularity counts
        expert_counts = np.zeros(num_experts, dtype=np.int64)
        total_accesses = 0
        for seq in traces_l:
            for s in seq:
                for e in s:
                    expert_counts[e] += 1
                    total_accesses += 1

        probs = expert_counts / total_accesses if total_accesses > 0 else np.zeros(num_experts)
        nonzero_p = probs[probs > 0]
        entropy = float(-np.sum(nonzero_p * np.log(nonzero_p))) if len(nonzero_p) > 0 else 0.0
        effective_experts = float(np.exp(entropy))

        # Top-h mass
        sorted_probs = np.sort(probs)[::-1]
        h_mass = {
            h: float(np.sum(sorted_probs[:h])) for h in [8, 16, 32, 64]
        }

        # 3. Reuse distance
        reuse_distances = []
        for seq in traces_l:
            last_seen = {}
            for t, s in enumerate(seq):
                for e in s:
                    if e in last_seen:
                        reuse_distances.append(t - last_seen[e])
                    last_seen[e] = t

        rd_mean = float(np.mean(reuse_distances)) if reuse_distances else float("inf")
        rd_median = float(np.median(reuse_distances)) if reuse_distances else float("inf")
        rd_p25 = float(np.percentile(reuse_distances, 25)) if reuse_distances else float("inf")
        rd_p75 = float(np.percentile(reuse_distances, 75)) if reuse_distances else float("inf")
        rd_le5 = float(np.mean([d <= 5 for d in reuse_distances])) if reuse_distances else 0.0
        rd_le10 = float(np.mean([d <= 10 for d in reuse_distances])) if reuse_distances else 0.0
        rd_le20 = float(np.mean([d <= 20 for d in reuse_distances])) if reuse_distances else 0.0

        # 4. LRU Cache Simulation
        lru_hit_rates = {}
        for c in capacities:
            cache = LRUCache(c)
            for seq in traces_l:
                # Reset cache per sequence/prompt to measure within-sequence cacheability
                cache.cache.clear()
                for s in seq:
                    for e in s:
                        cache.access(e)
            total_reqs = cache.hits + cache.misses
            rate = cache.hits / total_reqs if total_reqs > 0 else 0.0
            lru_hit_rates[c] = rate
            all_layer_lru_hits[c].append(rate)

        # 5. Static Top-C Hit Rate (Oracle static cache of top-C hottest experts in this layer)
        static_hit_rates = {
            c: float(np.sum(sorted_probs[:c])) for c in capacities
        }

        layer_data = {
            "layer": l,
            "mean_overlap": mean_overlap,
            "entropy": entropy,
            "effective_experts": effective_experts,
            "entropy_ratio": entropy / max_entropy,
            "top_8_mass": h_mass[8],
            "top_16_mass": h_mass[16],
            "top_32_mass": h_mass[32],
            "top_64_mass": h_mass[64],
            "rd_mean": rd_mean,
            "rd_median": rd_median,
            "rd_p25": rd_p25,
            "rd_p75": rd_p75,
            "rd_le5": rd_le5,
            "rd_le10": rd_le10,
            "rd_le20": rd_le20,
            "lru_hit_rates": lru_hit_rates,
            "static_hit_rates": static_hit_rates,
        }
        results_per_layer.append(layer_data)

        all_layer_entropies.append(entropy)
        all_layer_overlaps.append(mean_overlap)
        all_layer_h16.append(h_mass[16])
        all_layer_h32.append(h_mass[32])
        all_layer_h64.append(h_mass[64])
        all_layer_median_rd.append(rd_median)

    # Print Table
    print(f"{'Layer':<6} | {'Overlap':<7} | {'Entropy':<7} | {'Eff Exp':<7} | {'Top-16':<7} | {'Top-32':<7} | {'Med RD':<7} | {'LRU-16':<7} | {'LRU-32':<7} | {'LRU-64':<7}")
    print("-" * 88)
    for row in results_per_layer:
        print(
            f"{row['layer']:<6} | "
            f"{row['mean_overlap']:<7.2f} | "
            f"{row['entropy']:<7.2f} | "
            f"{row['effective_experts']:<7.1f} | "
            f"{row['top_16_mass']*100:<6.1f}% | "
            f"{row['top_32_mass']*100:<6.1f}% | "
            f"{row['rd_median']:<7.1f} | "
            f"{row['lru_hit_rates'][16]*100:<6.1f}% | "
            f"{row['lru_hit_rates'][32]*100:<6.1f}% | "
            f"{row['lru_hit_rates'][64]*100:<6.1f}%"
        )
    print("-" * 88)
    print(
        f"{'MEAN':<6} | "
        f"{np.mean(all_layer_overlaps):<7.2f} | "
        f"{np.mean(all_layer_entropies):<7.2f} | "
        f"{np.mean([r['effective_experts'] for r in results_per_layer]):<7.1f} | "
        f"{np.mean(all_layer_h16)*100:<6.1f}% | "
        f"{np.mean(all_layer_h32)*100:<6.1f}% | "
        f"{np.mean(all_layer_median_rd):<7.1f} | "
        f"{np.mean(all_layer_lru_hits[16])*100:<6.1f}% | "
        f"{np.mean(all_layer_lru_hits[32])*100:<6.1f}% | "
        f"{np.mean(all_layer_lru_hits[64])*100:<6.1f}%"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "n_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "max_entropy": max_entropy,
            "mean_overlap": float(np.mean(all_layer_overlaps)),
            "mean_entropy": float(np.mean(all_layer_entropies)),
            "mean_effective_experts": float(np.mean([r["effective_experts"] for r in results_per_layer])),
            "mean_top16_mass": float(np.mean(all_layer_h16)),
            "mean_top32_mass": float(np.mean(all_layer_h32)),
            "mean_top64_mass": float(np.mean(all_layer_h64)),
            "mean_lru_hit_rates": {c: float(np.mean(all_layer_lru_hits[c])) for c in capacities},
            "layers": results_per_layer,
        }, f, indent=2)
    print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()
