"""L2-cold tile sweep for grouped expert GEMM, plus per-slot vs grouped.

    uv run --extra cuda python bench/tune_batched_expert.py
    uv run --extra cuda python bench/tune_batched_expert.py compare
    uv run --extra cuda python bench/tune_batched_expert.py tune

Decode at B=32 reads ~160 distinct experts, so a single 256-expert stack is
already cold. Replicas still cycle so a later layer does not sit in L2 from
the previous call. Times include grouping (argsort + histogram).
"""

from __future__ import annotations

import itertools
import sys

import torch
import triton

from maple_run.kernels.ternary_expert import (
    _expert_launch_meta,
    _grouped_launch_meta,
    _ternary_expert_gemv_kernel,
    _ternary_expert_gemv_grouped_kernel,
    _ternary_expert_swiglu_kernel,
    _ternary_expert_swiglu_grouped_kernel,
    expert_slot_groups,
    ternary_expert_gemv,
    ternary_expert_swiglu,
)

E, TOPK = 256, 8
COLD_BYTES = 512 * 1024 * 1024


def _pack(shape):
    return (
        torch.randint(0, 2**31, shape, device="cuda", dtype=torch.int64)
        .to(torch.uint32)
        .contiguous()
    )


def _replicas(shape, nbytes):
    n = max(2, COLD_BYTES // nbytes)
    return [_pack(shape) for _ in range(n)]


def _ids(n_tok, seed=0, hot: int | None = None):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    ids = torch.randint(0, E, (n_tok, TOPK), device="cuda", generator=g)
    if hot is None:
        return ids
    # Half the slots pile onto ``hot`` experts so grouping has something to do.
    g2 = torch.Generator(device="cuda")
    g2.manual_seed(seed + 1)
    take = torch.rand((n_tok, TOPK), device="cuda", generator=g2) < 0.5
    hot_ids = torch.randint(0, hot, (n_tok, TOPK), device="cuda", generator=g)
    return torch.where(take, hot_ids, ids)


def _cuda_ms(fn, warmup=4, iters=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _graph_ms(build, replay, warmup=3, iters=20) -> float:
    for _ in range(warmup):
        build()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build()
    for _ in range(2):
        replay(g)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        replay(g)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def compare() -> None:
    """Per-slot baseline vs grouped, CUDA-graph captured (matches decode)."""
    shapes = [
        ("swiglu", 1024, 2048),
        ("down", 2048, 512),
    ]
    print(f"{'op':<8} {'B':>4} {'route':<8} {'per-slot us':>12} {'grouped us':>12} {'speedup':>8}")
    for name, n_rows, k in shapes:
        nwords = k // 16
        packs = _pack((E, n_rows, nwords))
        alpha = torch.rand(E, n_rows, device="cuda") * 0.02
        for hot, route in ((None, "uniform"), (16, "hot16")):
            for n_tok in (8, 16, 32, 128, 256):
                ids = _ids(n_tok, seed=n_tok, hot=hot)
                slots = n_tok * TOPK
                if name == "swiglu":
                    x = torch.randn(n_tok, k, device="cuda", dtype=torch.bfloat16)
                    y_slot = torch.empty(slots, n_rows // 2, device="cuda", dtype=x.dtype)

                    def per_slot():
                        bn, bkw, warps, stages = _expert_launch_meta(nwords)
                        _ternary_expert_swiglu_kernel[
                            (triton.cdiv(n_rows // 2, bn), slots)
                        ](
                            x, packs, alpha, ids.reshape(slots), y_slot, n_rows // 2,
                            nwords, x.stride(0), x.stride(1), packs.stride(0),
                            packs.stride(1), packs.stride(2), alpha.stride(0),
                            alpha.stride(1), y_slot.stride(0), y_slot.stride(1),
                            TOPK=TOPK, BLOCK_N=bn, BLOCK_K_WORDS=bkw, STREAM_W=n_tok == 1,
                            num_warps=warps, num_stages=stages,
                        )

                    def grouped():
                        ternary_expert_swiglu(x, packs, alpha, ids, grouped=True)
                else:
                    x = torch.randn(n_tok, TOPK, k, device="cuda", dtype=torch.bfloat16)
                    y_slot = torch.empty(slots, n_rows, device="cuda", dtype=x.dtype)
                    slot_x = x.reshape(slots, k)

                    def per_slot():
                        bn, bkw, warps, stages = _expert_launch_meta(nwords)
                        _ternary_expert_gemv_kernel[(triton.cdiv(n_rows, bn), slots)](
                            slot_x, packs, alpha, ids.reshape(slots), y_slot, n_rows,
                            nwords, slot_x.stride(0), slot_x.stride(1), packs.stride(0),
                            packs.stride(1), packs.stride(2), alpha.stride(0),
                            alpha.stride(1), y_slot.stride(0), y_slot.stride(1),
                            BLOCK_N=bn, BLOCK_K_WORDS=bkw, STREAM_W=n_tok == 1,
                            num_warps=warps, num_stages=stages,
                        )

                    def grouped():
                        ternary_expert_gemv(x, packs, alpha, ids, grouped=True)

                per_slot()
                grouped()
                torch.cuda.synchronize()

                gs = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gs):
                    per_slot()
                gg = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gg):
                    grouped()

                t_slot = _cuda_ms(gs.replay, warmup=5, iters=30)
                t_grp = _cuda_ms(gg.replay, warmup=5, iters=30)
                print(
                    f"{name:<8} {n_tok:>4} {route:<8} {t_slot*1000:>12.1f} {t_grp*1000:>12.1f} "
                    f"{t_slot / t_grp:>7.2f}x"
                )


def tune() -> None:
    """Sweep grouped tiles at B=16 and B=32, swiglu and down, cold-ish stacks."""
    jobs = [
        ("swiglu", 512, 2048, True),   # N=intermediate, K=hidden
        ("down", 2048, 512, False),
    ]
    for name, n_out, k, is_swiglu in jobs:
        nwords = k // 16
        packed_rows = n_out * 2 if is_swiglu else n_out
        nbytes = E * packed_rows * nwords * 4
        packs = _replicas((E, packed_rows, nwords), nbytes)
        alpha = torch.rand(E, packed_rows, device="cuda") * 0.02
        wbytes = nbytes
        print(f"\n=== {name} N={n_out} K={k}  {wbytes/1e6:.0f} MB x {len(packs)} ===")
        for n_tok in (8, 16, 32):
            x = torch.randn(n_tok, k, device="cuda", dtype=torch.bfloat16)
            ids = _ids(n_tok, seed=n_tok)
            order, offsets, unique_ids = expert_slot_groups(ids, E)
            use_compact = unique_ids is not None
            n_launch = int(unique_ids.shape[0]) if use_compact else E
            uniq = unique_ids if use_compact else order
            if is_swiglu:
                y = torch.empty(n_tok * TOPK, n_out, device="cuda", dtype=x.dtype)
            else:
                x = torch.randn(n_tok, TOPK, k, device="cuda", dtype=torch.bfloat16)
                y = torch.empty(n_tok * TOPK, n_out, device="cuda", dtype=x.dtype)
            cfgs = list(
                itertools.product(
                    (16, 32, 64),
                    (2, 4, 8, 16),
                    (8, 16, 32),
                    (2, 4),
                    (2, 3),
                )
            )
            best = []
            for bn, bkw, bb, warps, stages in cfgs:
                if bn * bkw * 16 > 16 * 256:
                    continue
                if not is_swiglu and bb < 16:
                    continue

                def call(
                    i,
                    bn=bn,
                    bkw=bkw,
                    bb=bb,
                    warps=warps,
                    stages=stages,
                ):
                    grid = (triton.cdiv(n_out, bn), n_launch)
                    if is_swiglu:
                        _ternary_expert_swiglu_grouped_kernel[grid](
                            x, packs[i], alpha, order, offsets, uniq, y,
                            n_out, nwords,
                            x.stride(0), x.stride(1), packs[i].stride(0),
                            packs[i].stride(1), packs[i].stride(2),
                            alpha.stride(0), alpha.stride(1), y.stride(0), y.stride(1),
                            USE_COMPACT=use_compact, TOPK=TOPK, STREAM_W=True,
                            BLOCK_N=bn, BLOCK_K_WORDS=bkw, BLOCK_B=bb,
                            num_warps=warps, num_stages=stages,
                        )
                    else:
                        _ternary_expert_gemv_grouped_kernel[grid](
                            x.reshape(-1, k), packs[i], alpha, order, offsets, uniq, y,
                            n_out, nwords,
                            x.reshape(-1, k).stride(0), x.reshape(-1, k).stride(1),
                            packs[i].stride(0), packs[i].stride(1), packs[i].stride(2),
                            alpha.stride(0), alpha.stride(1), y.stride(0), y.stride(1),
                            USE_COMPACT=use_compact, PER_SLOT_X=True, TOPK=1, STREAM_W=True,
                            BLOCK_N=bn, BLOCK_K_WORDS=bkw, BLOCK_B=bb,
                            num_warps=warps, num_stages=stages,
                        )

                try:
                    t = _cuda_ms(lambda: call(0), warmup=1, iters=5)
                except Exception:
                    continue
                best.append((t, bn, bkw, bb, warps, stages))
            best.sort()
            default = _grouped_launch_meta(nwords, n_tok)
            print(f"  B={n_tok}  default {default}")
            for t, bn, bkw, bb, warps, stages in best[:6]:
                print(
                    f"    {t*1000:7.1f} us  BLOCK_N={bn:<3} BLOCK_K_WORDS={bkw:<3} "
                    f"BLOCK_B={bb:<3} warps={warps} stages={stages}"
                )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if which in ("compare", "all"):
        compare()
    if which in ("tune", "all"):
        tune()
