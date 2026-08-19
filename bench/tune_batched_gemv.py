"""Tune the batch>1 packed GEMV kernels against cold weights.

Decode reads every weight exactly once per token, so a microbenchmark that
hammers one layer-sized tensor measures L2, not the machine the kernel actually
runs on. This cycles round-robin through enough layer-sized replicas to blow
past L2 before a tensor is reused, which is the same reason the batch-1 tuning
comments in these kernels distrust isolated numbers.

    uv run --extra cuda python bench/tune_batched_gemv.py [ternary|rtn4|all]
"""

from __future__ import annotations

import itertools
import sys

import torch
import triton

COLD_BYTES = 768 * 1024 * 1024


def _replicas(make, nbytes: int) -> list:
    reps = max(2, COLD_BYTES // nbytes)
    return [make() for _ in range(reps)]


def bench_cold(call, reps: int, iters: int = 3) -> float:
    """ms per call, cycling through ``reps`` distinct weight tensors."""
    for i in range(min(reps, 4)):
        call(i)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for it in range(iters):
        for i in range(reps):
            call(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * reps)


def baseline_ternary(packs, alpha, N, nwords, wbytes) -> float:
    """Cold time for the tuned batch-1 kernel on the same replicas."""
    from maple_run.kernels.ternary_gemv import _gemv_launch_meta, _ternary_gemv_kernel

    bn, bkw, _bb, warps, stages = _gemv_launch_meta(nwords, 1)
    x = torch.randn(1, nwords * 16, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(1, N, device="cuda", dtype=torch.bfloat16)

    def call(i):
        _ternary_gemv_kernel[(triton.cdiv(N, bn), 1)](
            x, packs[i], alpha, y, x, N, nwords,
            x.stride(0), x.stride(1), packs[i].stride(0), packs[i].stride(1),
            alpha.stride(0), y.stride(0), y.stride(1), 1e-6,
            HAS_RMS=False, STREAM_W=True, BLOCK_N=bn, BLOCK_K_WORDS=bkw,
            num_warps=warps, num_stages=stages,
        )

    t = bench_cold(call, len(packs))
    print(f"  B=1   batch-1 kernel: {t*1000:7.1f} us  {wbytes/t/1e6:6.0f} GB/s  "
          f"(BLOCK_N={bn} BLOCK_K_WORDS={bkw} warps={warps} stages={stages})")
    return t


def tune_ternary(batches=(2, 8, 32)) -> None:
    from maple_run.kernels.ternary_gemv import _ternary_gemm_kernel

    N, K = 3072, 2048          # QKV; O (2048) tunes identically, same nwords
    nwords = K // 16
    packs = _replicas(
        lambda: torch.randint(0, 2**31, (N, nwords), device="cuda", dtype=torch.int64)
        .to(torch.uint32)
        .contiguous(),
        N * nwords * 4,
    )
    alpha = torch.rand(N, device="cuda", dtype=torch.float32) * 0.02
    wbytes = N * nwords * 4
    print(f"\n=== ternary QKV N={N} K={K}  {wbytes/1e6:.1f} MB x {len(packs)} replicas ===")
    baseline_ternary(packs, alpha, N, nwords, wbytes)

    grid_cfgs = [
        (bn, bkw, w, st)
        for bn, bkw, w, st in itertools.product(
            (16, 32, 64, 128), (2, 4, 8, 16), (1, 2, 4, 8), (2, 3)
        )
        if bn * bkw * 16 <= 128 * 128
    ]
    # QKV -- the big ternary consumer -- always folds the input RMSNorm in, and
    # that path carries the residual dot, so it is what gets tuned. Tuning the
    # bare GEMV instead picks configs that spill once the second dot appears
    # (BLOCK_N=64/warps=8 measured 17 us bare and 455 us with the norm).
    rms = torch.rand(K, device="cuda", dtype=torch.float32) + 0.5
    for B in batches:
        x = torch.randn(B, K, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(B, N, device="cuda", dtype=torch.bfloat16)
        block_b = max(16, triton.next_power_of_2(B))
        best = []
        for bn, bkw, warps, stages in grid_cfgs:
            def call(i, bn=bn, bkw=bkw, warps=warps, stages=stages):
                _ternary_gemm_kernel[(triton.cdiv(N, bn), triton.cdiv(B, block_b))](
                    x, packs[i], alpha, y, rms, N, nwords, B,
                    x.stride(0), x.stride(1), packs[i].stride(0), packs[i].stride(1),
                    alpha.stride(0), y.stride(0), y.stride(1), 1e-6,
                    HAS_RMS=True, STREAM_W=True, IEEE=False, SPLIT_X=True,
                    BLOCK_N=bn, BLOCK_K_WORDS=bkw, BLOCK_B=block_b,
                    num_warps=warps, num_stages=stages,
                )
            try:
                t = bench_cold(call, len(packs))
            except Exception:
                continue
            best.append((t, bn, bkw, warps, stages))
        best.sort()
        print(f"  B={B:<3} BLOCK_B={block_b}")
        for t, bn, bkw, warps, stages in best[:5]:
            print(f"    {t*1000:7.1f} us  {wbytes/t/1e6:6.0f} GB/s  "
                  f"BLOCK_N={bn:<4} BLOCK_K_WORDS={bkw:<3} warps={warps} stages={stages}")


def baseline_rtn4(packs, scales, biases, N, nwords, G, wbytes) -> float:
    """Cold time for the tuned batch-1 kernel on the same replicas."""
    from maple_run.kernels.rtn4 import _rtn4_gemv_kernel, _rtn4_launch_meta

    bn, bkw, _bb, warps, stages = _rtn4_launch_meta(nwords, 1)
    x = torch.randn(1, nwords * 8, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(1, N, device="cuda", dtype=torch.bfloat16)

    def call(i):
        _rtn4_gemv_kernel[(triton.cdiv(N, bn), 1)](
            x, packs[i], scales, biases, y, N, nwords,
            x.stride(0), x.stride(1), packs[i].stride(0), packs[i].stride(1),
            scales.stride(0), scales.stride(1), biases.stride(0), biases.stride(1),
            y.stride(0), y.stride(1),
            GROUP_SIZE=G, BLOCK_N=bn, BLOCK_K_WORDS=bkw, STREAM_W=True,
            num_warps=warps, num_stages=stages,
        )

    t = bench_cold(call, len(packs), iters=2)
    print(f"  B=1   batch-1 kernel: {t*1000:7.1f} us  {wbytes/t/1e6:6.0f} GB/s  "
          f"(BLOCK_N={bn} BLOCK_K_WORDS={bkw} warps={warps} stages={stages})")
    return t


def tune_rtn4(batches=(2, 8, 32)) -> None:
    from maple_run.kernels.rtn4 import _rtn4_gemm_kernel

    N, K, G = 151936, 2048, 64          # lm_head
    nwords = K // 8
    packs = _replicas(
        lambda: torch.randint(0, 2**31, (N, nwords), device="cuda", dtype=torch.int64)
        .to(torch.uint32)
        .contiguous(),
        N * nwords * 4,
    )
    scales = torch.rand(N, K // G, device="cuda", dtype=torch.bfloat16) * 0.01
    biases = torch.rand(N, K // G, device="cuda", dtype=torch.bfloat16) * 0.01
    wbytes = N * nwords * 4 + scales.numel() * 2 * 2
    print(f"\n=== rtn4 lm_head N={N} K={K}  {wbytes/1e6:.0f} MB x {len(packs)} replicas ===")
    baseline_rtn4(packs, scales, biases, N, nwords, G, wbytes)

    words_per_group = G // 8
    grid_cfgs = [
        (bn, bkw, w, st)
        for bn, bkw, w, st in itertools.product(
            (16, 32, 64, 128), (8, 16, 32, 64), (1, 2, 4, 8), (2, 3, 4)
        )
        if bkw % words_per_group == 0 and bn * bkw * 8 <= 128 * 256
    ]
    for B in batches:
        x = torch.randn(B, K, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(B, N, device="cuda", dtype=torch.bfloat16)
        block_b = max(16, triton.next_power_of_2(B))
        best = []
        for bn, bkw, warps, stages in grid_cfgs:
            def call(i, bn=bn, bkw=bkw, warps=warps, stages=stages):
                _rtn4_gemm_kernel[(triton.cdiv(N, bn), triton.cdiv(B, block_b))](
                    x, packs[i], scales, biases, y, N, nwords, B,
                    x.stride(0), x.stride(1), packs[i].stride(0), packs[i].stride(1),
                    scales.stride(0), scales.stride(1), biases.stride(0), biases.stride(1),
                    y.stride(0), y.stride(1),
                    GROUP_SIZE=G, IEEE=False, STREAM_W=True,
                    BLOCK_N=bn, BLOCK_K_WORDS=bkw, BLOCK_B=block_b,
                    num_warps=warps, num_stages=stages,
                )
            try:
                t = bench_cold(call, len(packs), iters=2)
            except Exception:
                continue
            best.append((t, bn, bkw, warps, stages))
        best.sort()
        print(f"  B={B:<3} BLOCK_B={block_b}")
        for t, bn, bkw, warps, stages in best[:5]:
            print(f"    {t*1000:7.1f} us  {wbytes/t/1e6:6.0f} GB/s  "
                  f"BLOCK_N={bn:<4} BLOCK_K_WORDS={bkw:<3} warps={warps} stages={stages}")


def report(batches=(1, 2, 4, 8, 16, 32)) -> None:
    """Final ms / GB/s vs batch through the public dispatch, on cold weights."""
    from maple_run.kernels.rtn4 import rtn4_gemv
    from maple_run.kernels.ternary_gemv import ternary_gemv

    def sweep(title, packs, wbytes, call):
        print(f"\n{title}   ({wbytes/1e6:.1f} MB x {len(packs)} cold replicas)")
        print(f"{'B':>4} {'us/call':>9} {'GB/s':>7} {'us/token':>9} {'vs B=1':>7}")
        base = None
        for B in batches:
            t = bench_cold(lambda i, B=B: call(i, B), len(packs), iters=2)
            base = base or t
            print(f"{B:>4} {t*1000:>9.1f} {wbytes/t/1e6:>7.0f} {t*1000/B:>9.2f} "
                  f"{base*B/t:>6.1f}x")

    N, K = 3072, 2048
    nwords = K // 16
    packs = _replicas(
        lambda: torch.randint(0, 2**31, (N, nwords), device="cuda", dtype=torch.int64)
        .to(torch.uint32).contiguous(), N * nwords * 4)
    alpha = torch.rand(N, device="cuda", dtype=torch.float32) * 0.02
    xs = {B: torch.randn(B, K, device="cuda", dtype=torch.bfloat16) for B in batches}
    rms = torch.rand(K, device="cuda", dtype=torch.float32) + 0.5
    sweep(f"ternary QKV N={N} K={K}", packs, N * nwords * 4,
          lambda i, B: ternary_gemv(xs[B], packs[i], alpha))
    sweep(f"ternary QKV N={N} K={K} (fused RMSNorm)", packs, N * nwords * 4,
          lambda i, B: ternary_gemv(xs[B], packs[i], alpha, rms_weight=rms))

    N, K, G = 151936, 2048, 64
    nwords = K // 8
    packs = _replicas(
        lambda: torch.randint(0, 2**31, (N, nwords), device="cuda", dtype=torch.int64)
        .to(torch.uint32).contiguous(), N * nwords * 4)
    scales = torch.rand(N, K // G, device="cuda", dtype=torch.bfloat16) * 0.01
    biases = torch.rand(N, K // G, device="cuda", dtype=torch.bfloat16) * 0.01
    xs = {B: torch.randn(B, K, device="cuda", dtype=torch.bfloat16) for B in batches}
    sweep(f"rtn4 lm_head N={N} K={K}", packs, N * nwords * 4 + scales.numel() * 4,
          lambda i, B: rtn4_gemv(xs[B], packs[i], scales, biases, group_size=G))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("ternary", "all"):
        tune_ternary()
    if which in ("rtn4", "all"):
        tune_rtn4()
    if which == "report":
        report()
