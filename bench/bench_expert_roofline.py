"""Microbenchmark expert GEMV / SwitchGLU latency vs. vector batch r_e on GB10.

Phase 1 deliverable for Expert-Reuse Speculation (ERS):
Measures the arithmetic intensity roofline knee for PackedTernaryExperts.
Directly compares:
1. AR Decode Baseline: 1 token, 8 experts
2. ERS Grouped Experts: m tokens sharing the exact 8 base experts (r_e = m)
3. Standard Speculative Decoding: m tokens routing to U distinct experts (empirical expansion)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton

from maple_run.kernels.ternary_expert import (
    _expert_launch_meta,
    _grouped_launch_meta,
    _ternary_expert_gemv_kernel,
    _ternary_expert_swiglu_kernel,
    expert_slot_groups,
    ternary_expert_gemv,
    ternary_expert_swiglu,
)

E = 256
TOPK = 8
N_REPLICAS = 4  # 4 replicas * 134 MB = 536 MB, ensuring L2 eviction between iterations


def _pack(shape):
    return (
        torch.randint(0, 2**31, shape, device="cuda", dtype=torch.int64)
        .to(torch.uint32)
        .contiguous()
    )


def _timed_loop(fns: list, warmup: int = 4, iters: int = 24) -> float:
    """Cycle through replica functions to measure cold-L2 latency without flush noise."""
    n_rep = len(fns)
    for i in range(warmup):
        fns[i % n_rep]()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for i in range(iters):
        fn = fns[i % n_rep]
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms to us

    times.sort()
    return times[len(times) // 2]


def construct_standard_sd_ids(m_tokens: int, u_distinct: int, device: torch.device) -> torch.Tensor:
    """Construct (m_tokens, TOPK) IDs with realistic overlap matching Phase 0 union size."""
    if m_tokens == 1:
        return torch.arange(TOPK, device=device, dtype=torch.int64).unsqueeze(0)

    # Token 0 uses experts 0..7
    # Remaining m-1 tokens choose among 0..7 and new experts up to u_distinct-1
    ids = torch.zeros((m_tokens, TOPK), device=device, dtype=torch.int64)
    ids[0] = torch.arange(TOPK, device=device)

    # Distribute the u_distinct - 8 new experts across tokens 1..m-1
    new_pool = list(range(TOPK, u_distinct))
    pool_idx = 0

    for t in range(1, m_tokens):
        # Each token keeps ~2-3 experts from previous and picks new ones
        n_new = min(len(new_pool) - pool_idx, 5)
        shared = [i % TOPK for i in range(t, t + (TOPK - n_new))]
        fresh = new_pool[pool_idx : pool_idx + n_new]
        pool_idx += n_new
        while len(shared) + len(fresh) < TOPK:
            shared.append(0)
        tok_ids = shared + fresh
        ids[t] = torch.tensor(tok_ids, device=device, dtype=torch.int64)

    return ids


def benchmark_roofline(warm_l2: bool = False) -> dict:
    device = torch.device("cuda")
    n_replicas = 1 if warm_l2 else N_REPLICAS

    # Measured empirical union sizes from Phase 0 for M in [1, 2, 3, 4, 6, 8, 12, 16]
    empirical_unions = {
        1: 8,
        2: 13,
        3: 18,
        4: 21,
        6: 27,
        8: 32,
        12: 40,
        16: 45,
    }

    shapes = [
        ("swiglu", 1024, 2048),
        ("down", 2048, 512),
    ]

    results = {
        "mode": "warm_l2" if warm_l2 else "cold_l2",
        "swiglu": {},
        "down": {},
        "moe_layer": {},
    }

    for op_name, n_rows, k in shapes:
        nwords = k // 16
        packs_list = [_pack((E, n_rows, nwords)) for _ in range(n_replicas)]
        alpha_list = [
            torch.rand(E, n_rows, device=device, dtype=torch.float32) * 0.02
            for _ in range(n_replicas)
        ]

        op_data = {}

        for m_tokens, u_distinct in sorted(empirical_unions.items()):
            # -----------------------------------------------------------------
            # 1. Standard Speculative Decode: m_tokens * 8 slots, U distinct experts
            # -----------------------------------------------------------------
            std_ids = construct_standard_sd_ids(m_tokens, u_distinct, device)
            std_slots = m_tokens * TOPK

            if op_name == "swiglu":
                x_std = torch.randn(m_tokens, k, device=device, dtype=torch.bfloat16)
                y_std = torch.empty(std_slots, n_rows // 2, device=device, dtype=x_std.dtype)
                bn, bkw, warps, stages = _expert_launch_meta(nwords)

                fns_std = []
                for rep in range(n_replicas):
                    p_rep = packs_list[rep]
                    a_rep = alpha_list[rep]

                    def make_std(p=p_rep, a=a_rep):
                        _ternary_expert_swiglu_kernel[
                            (triton.cdiv(n_rows // 2, bn), std_slots)
                        ](
                            x_std, p, a, std_ids.reshape(std_slots), y_std, n_rows // 2,
                            nwords, x_std.stride(0), x_std.stride(1), p.stride(0),
                            p.stride(1), p.stride(2), a.stride(0),
                            a.stride(1), y_std.stride(0), y_std.stride(1),
                            TOPK=TOPK, BLOCK_N=bn, BLOCK_K_WORDS=bkw, STREAM_W=True,
                            num_warps=warps, num_stages=stages,
                        )
                    fns_std.append(make_std)

            else:  # down
                x_std = torch.randn(m_tokens, TOPK, k, device=device, dtype=torch.bfloat16)
                slot_x = x_std.reshape(std_slots, k)
                y_std = torch.empty(std_slots, n_rows, device=device, dtype=x_std.dtype)
                bn, bkw, warps, stages = _expert_launch_meta(nwords)

                fns_std = []
                for rep in range(n_replicas):
                    p_rep = packs_list[rep]
                    a_rep = alpha_list[rep]

                    def make_std(p=p_rep, a=a_rep):
                        _ternary_expert_gemv_kernel[(triton.cdiv(n_rows, bn), std_slots)](
                            slot_x, p, a, std_ids.reshape(std_slots), y_std, n_rows,
                            nwords, slot_x.stride(0), slot_x.stride(1), p.stride(0),
                            p.stride(1), p.stride(2), a.stride(0),
                            a.stride(1), y_std.stride(0), y_std.stride(1),
                            BLOCK_N=bn, BLOCK_K_WORDS=bkw, STREAM_W=True,
                            num_warps=warps, num_stages=stages,
                        )
                    fns_std.append(make_std)

            t_standard = _timed_loop(fns_std)

            # -----------------------------------------------------------------
            # 2. ERS Grouped Kernel: m_tokens share the exact 8 base experts (r_e = m)
            # -----------------------------------------------------------------
            ers_active = torch.arange(8, device=device, dtype=torch.int64)
            ers_ids = ers_active.unsqueeze(0).expand(m_tokens, -1).contiguous()
            ers_groups = expert_slot_groups(ers_ids, E)

            if op_name == "swiglu":
                x_ers = torch.randn(m_tokens, k, device=device, dtype=torch.bfloat16)
                fns_ers = []
                for rep in range(n_replicas):
                    p_rep = packs_list[rep]
                    a_rep = alpha_list[rep]

                    def make_ers(p=p_rep, a=a_rep):
                        ternary_expert_swiglu(x_ers, p, a, ers_ids, slot_groups=ers_groups, grouped=True)
                    fns_ers.append(make_ers)

            else:  # down
                x_ers = torch.randn(m_tokens, 8, k, device=device, dtype=torch.bfloat16)
                fns_ers = []
                for rep in range(n_replicas):
                    p_rep = packs_list[rep]
                    a_rep = alpha_list[rep]

                    def make_ers(p=p_rep, a=a_rep):
                        ternary_expert_gemv(x_ers, p, a, ers_ids, slot_groups=ers_groups, grouped=True)
                    fns_ers.append(make_ers)

            t_ers = _timed_loop(fns_ers)

            # Reference baseline: for m=1, t_ar = t_standard
            if m_tokens == 1:
                t_ers = t_standard

            op_data[m_tokens] = {
                "m_tokens": m_tokens,
                "u_distinct_experts": u_distinct,
                "t_standard_sd_us": t_standard,
                "t_ers_grouped_us": t_ers,
                "ers_speedup_vs_standard": t_standard / t_ers if t_ers > 0 else 1.0,
            }

        results[op_name] = op_data

    # Combined MoE Layer Metrics
    for m_tokens in sorted(empirical_unions.keys()):
        sw = results["swiglu"][m_tokens]
        dw = results["down"][m_tokens]
        t_moe_standard = sw["t_standard_sd_us"] + dw["t_standard_sd_us"]
        t_moe_ers = sw["t_ers_grouped_us"] + dw["t_ers_grouped_us"]

        results["moe_layer"][m_tokens] = {
            "m_tokens": m_tokens,
            "u_distinct": sw["u_distinct_experts"],
            "t_standard_sd_us": t_moe_standard,
            "t_ers_us": t_moe_ers,
            "ers_speedup": t_moe_standard / t_moe_ers if t_moe_ers > 0 else 1.0,
            "incremental_cost_us": 0.0,
            "relative_to_ar": 1.0,
        }

    base_ar_us = results["moe_layer"][1]["t_ers_us"]
    for m_tokens in sorted(empirical_unions.keys()):
        d = results["moe_layer"][m_tokens]
        d["relative_to_ar"] = d["t_ers_us"] / base_ar_us
        d["incremental_cost_us"] = d["t_ers_us"] - base_ar_us

    return results


def main():
    parser = argparse.ArgumentParser(description="Microbenchmark expert GEMV roofline on GB10")
    parser.add_argument("--output", default="runs/expert_roofline_benchmark.json", help="Output JSON path")
    parser.add_argument("--warm", action="store_true", help="Benchmark warm L2 instead of cold L2")
    args = parser.parse_args()

    print("=" * 88)
    print("PHASE 1: EXPERT GEMV ROOFLINE & ERS VS STANDARD SPECULATIVE DECODING ON GB10")
    print(f"Memory Hierarchy: {'WARM L2 (Weight in cache)' if args.warm else 'COLD L2 / HBM STREAMING (536 MB cycling)'}")
    print("=" * 88)

    res = benchmark_roofline(warm_l2=args.warm)

    print("\n[A] SWIGLU KERNEL (K=2048, N=512, 512 KB/expert):")
    print(f"{'Draft m':<8} | {'Standard SD (u exp)':<22} | {'ERS Grouped (8 exp)':<20} | {'ERS Speedup':<12}")
    print("-" * 72)
    for m, d in sorted(res["swiglu"].items()):
        print(f"m={m:<6} | {d['t_standard_sd_us']:<6.1f} us ({d['u_distinct_experts']} exp, {m*8} slots) | {d['t_ers_grouped_us']:<6.1f} us (8 exp)       | {d['ers_speedup_vs_standard']:<10.2f}x")

    print("\n[B] DOWN PROJECTION KERNEL (K=512, N=2048, 256 KB/expert):")
    print(f"{'Draft m':<8} | {'Standard SD (u exp)':<22} | {'ERS Grouped (8 exp)':<20} | {'ERS Speedup':<12}")
    print("-" * 72)
    for m, d in sorted(res["down"].items()):
        print(f"m={m:<6} | {d['t_standard_sd_us']:<6.1f} us ({d['u_distinct_experts']} exp, {m*8} slots) | {d['t_ers_grouped_us']:<6.1f} us (8 exp)       | {d['ers_speedup_vs_standard']:<10.2f}x")

    print("\n[C] COMBINED MOE LAYER LATENCY (SwiGLU + Down):")
    print(f"{'Draft m':<8} | {'Standard SD':<14} | {'ERS Grouped':<14} | {'Penalty vs AR':<16} | {'ERS Speedup':<12}")
    print("-" * 72)
    base_us = res["moe_layer"][1]["t_ers_us"]
    for m, d in sorted(res["moe_layer"].items()):
        penalty_pct = (d["relative_to_ar"] - 1.0) * 100
        print(f"m={m:<6} | {d['t_standard_sd_us']:<8.1f} us    | {d['t_ers_us']:<8.1f} us    | {penalty_pct:+5.1f}% (+{d['incremental_cost_us']:<4.1f} us) | {d['ers_speedup']:<10.2f}x")

    print("\n" + "=" * 88)
    print("KEY ROOFLINE TAKEAWAYS:")
    for m in [2, 4, 8, 16]:
        dm = res["moe_layer"][m]
        print(f"  • Draft m={m:<2}: ERS saves {dm['t_standard_sd_us'] - dm['t_ers_us']:<5.1f} us/layer ({dm['ers_speedup']:.2f}x speedup vs Standard SD: {dm['t_ers_us']:.1f} us vs {dm['t_standard_sd_us']:.1f} us)")
    print("=" * 88)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved detailed roofline benchmark to {out_path}")


if __name__ == "__main__":
    main()
