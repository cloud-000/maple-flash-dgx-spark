"""Weight-Space Redundancy, Expert Co-occurrence, and Batch Collision Probe.

Answers:
1. Weight-space compressibility: Are expert weights geometrically redundant, or orthogonal?
   - Computes relative variance ||W_e - W_mean||_F^2 / ||W_e||_F^2
   - Pairwise cosine similarity distribution across experts
   - SVD singular value spectrum of stacked expert matrices (gate_up, down)
2. Intervening Byte Traffic D_bytes:
   - Physical bytes streamed between consecutive expert invocations in single-sequence AR
3. Expert Co-occurrence:
   - Evaluates pair correlation C_l(e_i, e_j) and lift = P(e_i, e_j) / (P(e_i) * P(e_j))
4. Multi-sequence Batch Collision Economics:
   - Computes unique experts E[|U_l(B)|] vs B * k for concurrency B in {1, 2, 4, 8, 16, 32}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from maple_run.generate import load_packed, run_generate
from maple_run.pack import unpack_2bit


def analyze_weight_redundancy(checkpoint_dir: str, target_layers: list[int] = [5, 6, 11, 12]):
    print("\n" + "="*80)
    print("1. WEIGHT-SPACE REDUNDANCY PROBE (Testing Basis + Delta Hypothesis)")
    print("="*80)

    index_path = Path(checkpoint_dir) / "model.safetensors.index.json"
    with index_path.open() as f:
        index = json.load(f)["weight_map"]

    results = {}

    for l in target_layers:
        print(f"\nAnalyzing Layer {l} expert weights...")
        layer_results = {}
        projs = ["gate_proj", "up_proj", "down_proj"]
        for proj in projs:
            weight_key = f"model.layers.{l}.mlp.switch_mlp.{proj}.weight"
            alpha_key = f"model.layers.{l}.mlp.switch_mlp.{proj}.row_alpha"
            if weight_key not in index:
                continue
            fname = index[weight_key]
            fpath = Path(checkpoint_dir) / fname
            with safe_open(fpath, framework="pt", device="cpu") as sf:
                packed = sf.get_tensor(weight_key).numpy() # [E, N, nwords]
                alpha = sf.get_tensor(alpha_key).numpy() # [E, N]

            E, N, nwords = packed.shape
            codes = unpack_2bit(packed) # [E, N, K]
            K = codes.shape[-1]
            ternary = (codes.astype(np.float32) - 1.0) * alpha[..., np.newaxis]
            W_flat = ternary.reshape(E, N * K)

            norms = np.linalg.norm(W_flat, axis=1) # [E]
            mean_norm_sq = float(np.mean(norms ** 2))

            W_mean = np.mean(W_flat, axis=0, keepdims=True)
            mean_dist_sq = float(np.mean(np.linalg.norm(W_flat - W_mean, axis=1) ** 2))
            rel_residual = mean_dist_sq / mean_norm_sq

            # Pairwise cosine similarity among experts (64 sampled)
            np.random.seed(42)
            sub_idx = np.random.choice(E, min(64, E), replace=False)
            W_sub = W_flat[sub_idx]
            W_normed = W_sub / (np.linalg.norm(W_sub, axis=1, keepdims=True) + 1e-9)
            cos_matrix = W_normed @ W_normed.T
            iu = np.triu_indices(len(sub_idx), k=1)
            cos_vals = cos_matrix[iu]
            mean_cos = float(np.mean(cos_vals))
            median_cos = float(np.median(cos_vals))
            p90_cos = float(np.percentile(cos_vals, 90))

            # SVD singular value decay
            s = np.linalg.svd(W_flat, compute_uv=False)
            total_energy = float(np.sum(s ** 2))
            top1_energy = float(s[0] ** 2 / total_energy)
            top4_energy = float(np.sum(s[:4] ** 2) / total_energy)
            top16_energy = float(np.sum(s[:16] ** 2) / total_energy)
            top32_energy = float(np.sum(s[:32] ** 2) / total_energy)

            proj_name = proj
            print(f"  {proj_name:<18} [E={E}, N={N}, K={K}]:")
            print(f"    Residual variance around mean (||W_e - W_mean||^2 / ||W_e||^2): {rel_residual*100:.2f}%")
            print(f"    Pairwise cosine similarity: mean={mean_cos:.4f}, median={median_cos:.4f}, p90={p90_cos:.4f}")
            print(f"    SVD Energy captured:        Rank-1={top1_energy*100:.1f}%, Rank-4={top4_energy*100:.1f}%, Rank-16={top16_energy*100:.1f}%, Rank-32={top32_energy*100:.1f}%")

            layer_results[proj_name] = {
                "rel_residual": rel_residual,
                "mean_cos": mean_cos,
                "median_cos": median_cos,
                "top1_energy": top1_energy,
                "top4_energy": top4_energy,
                "top16_energy": top16_energy,
                "top32_energy": top32_energy,
            }
        results[f"layer_{l}"] = layer_results
    return results


def analyze_cooccurrence_and_concurrency(model_path: str = "checkpoints/maple-2bit", n_prompts: int = 6, max_tokens: int = 80):
    print("\n" + "="*80)
    print("2. CO-OCCURRENCE, INTERVENING BYTES & MULTI-SEQUENCE CONCURRENCY")
    print("="*80)

    model, tokenizer = load_packed(model_path, flash_head=False, dense_head_source=None)
    top_k = int(model.config["num_experts_per_tok"])
    n_layers = int(model.config["num_hidden_layers"])
    num_experts = int(model.config["num_experts"])

    # Sample prompts
    prompts = [
        "Explain how a compiler optimizes abstract syntax trees into machine code.",
        "Write a Python implementation of a balanced BST with insertion and search.",
        "Discuss the historical causes of the French Revolution.",
        "What are the primary differences between TCP and UDP?",
        "Describe the lifecycle of a star from nebula to remnant.",
        "Write a comprehensive summary of how transformers use self-attention.",
    ][:n_prompts]

    # traces[layer][prompt_idx] = list of expert sets
    traces: list[list[list[set[int]]]] = [[] for _ in range(n_layers)]

    for p_idx, p in enumerate(prompts):
        input_ids = torch.tensor(tokenizer.encode(p), device=model.device, dtype=torch.long)
        p_trace: list[list[set[int]]] = [[] for _ in range(n_layers)]

        def make_hook(layer_idx: int):
            def hook(topk_idx: torch.Tensor):
                if topk_idx.shape[0] == 1:
                    p_trace[layer_idx].append(set(topk_idx[0].cpu().tolist()))
            return hook

        for i, layer in enumerate(model.layers):
            layer.mlp.router_hook = make_hook(i)

        run_generate(model, tokenizer, input_ids, max_tokens=max_tokens, temperature=0.0, use_graph=False)
        model.set_router_hook(None)

        for l in range(n_layers):
            traces[l].append(p_trace[l])

    # 1. Byte Traffic per Token Step
    # Each token step streams:
    #   QKV (24 layers * 1.5MB) + O (24 layers * 1.0MB) = 60 MB
    #   Experts: 24 layers * 8 experts * 640 KB = 122.88 MB
    #   Router + Norms + Head + Embed ~ 4 MB
    BYTES_PER_TOKEN = (60.0 + 122.88 + 4.0) * (1024 * 1024) # ~196 MB
    L2_BYTES = 24.0 * (1024 * 1024) # 24 MB

    # Intervening bytes between consecutive uses of expert e
    all_d_bytes = []
    for l in range(n_layers):
        for p_trace in traces[l]:
            last_seen = {}
            for t, s in enumerate(p_trace):
                for e in s:
                    if e in last_seen:
                        token_dist = t - last_seen[e]
                        # Bytes streamed between uses of this expert in layer l
                        # includes intervening full forward passes
                        intervening_bytes = token_dist * BYTES_PER_TOKEN
                        all_d_bytes.append(intervening_bytes)
                    last_seen[e] = t

    median_bytes_mb = np.median(all_d_bytes) / (1024 * 1024)
    p25_bytes_mb = np.percentile(all_d_bytes, 25) / (1024 * 1024)
    p75_bytes_mb = np.percentile(all_d_bytes, 75) / (1024 * 1024)
    pct_fit_l2 = np.mean([b <= L2_BYTES for b in all_d_bytes]) * 100

    print(f"\nA. PHYSICAL INTERVENING BYTES (D_bytes vs 24 MB L2 Cache):")
    print(f"  Streaming traffic per token step: {BYTES_PER_TOKEN / (1024*1024):.1f} MB")
    print(f"  Median intervening bytes D_bytes: {median_bytes_mb:.1f} MB")
    print(f"  25th percentile D_bytes:         {p25_bytes_mb:.1f} MB")
    print(f"  75th percentile D_bytes:         {p75_bytes_mb:.1f} MB")
    print(f"  % of reuse events with D_bytes <= 24 MB L2: {pct_fit_l2:.2f}%")

    # 2. Co-occurrence Analysis (Pair correlation & Lift)
    print(f"\nB. EXPERT CO-OCCURRENCE WITHIN SINGLE TOKEN (top-k=8):")
    pair_counts = defaultdict(int)
    single_counts = defaultdict(int)
    total_tokens = 0

    for l in range(n_layers):
        for p_trace in traces[l]:
            for s in p_trace:
                total_tokens += 1
                s_list = sorted(list(s))
                for e in s_list:
                    single_counts[e] += 1
                for i in range(len(s_list)):
                    for j in range(i + 1, len(s_list)):
                        pair_counts[(s_list[i], s_list[j])] += 1

    # Compute lifts: P(e_i, e_j) / (P(e_i) * P(e_j))
    lifts = []
    top_pairs = []
    for (e1, e2), count in pair_counts.items():
        if count < 5: # filter very rare
            continue
        p_pair = count / total_tokens
        p1 = single_counts[e1] / total_tokens
        p2 = single_counts[e2] / total_tokens
        lift = p_pair / (p1 * p2)
        lifts.append(lift)
        top_pairs.append(((e1, e2), count, lift, p_pair))

    top_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"  Analyzed {len(pair_counts)} co-occurring pairs across {total_tokens} token-layer steps")
    print(f"  Mean Pairwise Lift:   {np.mean(lifts):.2f}x (1.0 = independent routing)")
    print(f"  Median Pairwise Lift: {np.median(lifts):.2f}x")
    print(f"  90th percentile Lift: {np.percentile(lifts, 90):.2f}x")
    print(f"  Top Co-occurring Expert Bundles (highest lift):")
    for (e1, e2), count, lift, p_pair in top_pairs[:5]:
        print(f"    Experts ({e1:>3}, {e2:>3}): Lift = {lift:.2f}x (co-occurred {count} times, P={p_pair*100:.2f}%)")

    # 3. Multi-sequence Batch Concurrency Economics
    print(f"\nC. MULTI-SEQUENCE BATCH COLLISION ECONOMICS (Layer-Major Batching):")
    print(f"  {'Batch B':<8} | {'Total Assignments (B*k)':<25} | {'Unique Experts E[|U|]':<22} | {'Reuse Factor (Total/Unique)':<28} | {'Unique / Total':<15}")
    print("  " + "-" * 105)

    # Flatten all observed token steps per layer into an empirical pool
    concurrencies = [1, 2, 4, 8, 16, 32, 64]
    for B in concurrencies:
        unique_counts = []
        for l in range(n_layers):
            all_sets = [s for p_trace in traces[l] for s in p_trace]
            if len(all_sets) < B:
                continue
            # Simulate 200 random batches of size B
            for _ in range(200):
                sampled_indices = np.random.choice(len(all_sets), B, replace=True)
                u = set()
                for idx in sampled_indices:
                    u.update(all_sets[idx])
                unique_counts.append(len(u))

        mean_u = float(np.mean(unique_counts))
        total_assign = B * top_k
        reuse_factor = total_assign / mean_u
        fraction = mean_u / total_assign
        print(f"  {B:<8} | {total_assign:<25} | {mean_u:<22.1f} | {reuse_factor:<28.2f}x | {fraction*100:<14.1f}%")


if __name__ == "__main__":
    analyze_weight_redundancy("checkpoints/maple-2bit")
    analyze_cooccurrence_and_concurrency("checkpoints/maple-2bit")
