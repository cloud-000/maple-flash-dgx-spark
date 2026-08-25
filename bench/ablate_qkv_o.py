"""Split QKV vs O decode ablation. Isolated cold GEMV is the 175 GB/s reference.

The production kernel is one ``ternary_gemv``; QKV always passes ``rms_weight``
and O never does. No-op one or the other *before* CUDA-graph capture, re-time
full sampled decode, subtract. Do not use a 1-4 MB hot-L2 microbench.

    uv run --extra cuda --extra tokenizer python bench/ablate_qkv_o.py

Default generate path is untouched (ablation flag stays None).
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from maple_run.generate import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    encode_user_prompt,
    load_packed,
    run_generate,
    warmup_model,
)
from maple_run.kernels.ternary_gemv import (
    GEMV_RMS_DUMMY,
    GEMV_RMS_STRIP,
    GEMV_SKIP_BOTH,
    GEMV_SKIP_O,
    GEMV_SKIP_QKV,
    configure_gemv_ablation,
    configure_gemv_decode_launch,
    ternary_gemv,
)

REPO = Path(__file__).resolve().parents[1]
PACKED_CKPT = REPO / "checkpoints" / "maple-2bit"
PROMPT = "Write a haiku on groves"
PEAK_GBS = 250.0
COLD_BYTES = 75 * 1024 * 1024
N_LAYERS = 24
HIDDEN = 2048
QKV_N = 3072  # 16 q + 4 k + 4 v, head_dim 128
O_N = 2048
K = 2048
CODES_PER_WORD = 16


def _packed_bytes(n_out: int, k_in: int = K) -> int:
    return n_out * (k_in // CODES_PER_WORD) * 4


QKV_BYTES = N_LAYERS * _packed_bytes(QKV_N)
O_BYTES = N_LAYERS * _packed_bytes(O_N)
BOTH_BYTES = QKV_BYTES + O_BYTES


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def _gbs(nbytes: int, us: float) -> float:
    if us <= 0:
        return 0.0
    return nbytes / (us * 1e-6) / 1e9


def _pct(gbs: float) -> float:
    return 100.0 * gbs / PEAK_GBS


def _rand_pack(n_out: int, nwords: int) -> torch.Tensor:
    return (
        torch.randint(0, 2**31, (n_out, nwords), device="cuda", dtype=torch.int64)
        .to(torch.uint32)
        .contiguous()
    )


def bench_isolated(n_out: int, *, has_rms: bool, min_bytes: int = COLD_BYTES) -> float:
    """us/call, CUDA-graph replay over enough replicas to pass ``min_bytes``.

    A Python loop around one launch measures Triton launch overhead; a single
    1-4 MB weight stays in L2. Graph the rotation, then time replay.
    """
    nwords = K // CODES_PER_WORD
    wbytes = _packed_bytes(n_out)
    n_rep = max(2, (min_bytes + wbytes - 1) // wbytes)
    packs = [_rand_pack(n_out, nwords) for _ in range(n_rep)]
    alpha = torch.rand(n_out, device="cuda", dtype=torch.float32) * 0.02
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    rms = torch.rand(K, device="cuda", dtype=torch.float32) + 0.5 if has_rms else None

    def launch(i: int) -> torch.Tensor:
        return ternary_gemv(x, packs[i], alpha, rms_weight=rms)

    for i in range(n_rep):
        launch(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(n_rep):
            launch(i)
    # Replay a few times; take the median of per-call us.
    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / n_rep)  # us
    return _median(times)


def _run_decode(model, tokenizer, input_ids, max_tokens: int, seed: int):
    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)

    result = run_generate(
        model,
        tokenizer,
        input_ids,
        max_tokens=max_tokens,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K,
        seed=seed,
        log=log,
        use_graph=True,
        early_stop=False,
    )
    graph_on = any("CUDA graph decode: replay matches" in m for m in lines)
    return result, graph_on, lines


CONFIGS = (
    ("baseline", None, None, 0),
    ("no-op QKV", GEMV_SKIP_QKV, None, QKV_BYTES),
    ("no-op O", GEMV_SKIP_O, None, O_BYTES),
    ("no-op both", GEMV_SKIP_BOTH, None, BOTH_BYTES),
    ("QKV strip RMS", None, GEMV_RMS_STRIP, 0),
    ("O dummy RMS", None, GEMV_RMS_DUMMY, 0),
)


def run_in_model(
    *,
    model_dir: Path,
    max_tokens: int,
    trials: int,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    print(f"Loading packed model from {model_dir}", flush=True)
    model, tokenizer = load_packed(str(model_dir), flash_head=False)
    warmup_model(model, flash_head=False)
    input_ids = encode_user_prompt(tokenizer, PROMPT, model.device)
    us: dict[str, list[float]] = {name: [] for name, *_ in CONFIGS}
    tok_s: dict[str, list[float]] = {name: [] for name, *_ in CONFIGS}

    # Round-robin: timing configs back-to-back favours whichever ran first.
    for trial in range(trials):
        print(f"\n=== trial {trial + 1}/{trials} ===", flush=True)
        for name, skip, rms, _nbytes in CONFIGS:
            configure_gemv_ablation(skip=skip, rms=rms)
            try:
                result, graph_on, lines = _run_decode(
                    model, tokenizer, input_ids, max_tokens, seed
                )
            finally:
                configure_gemv_ablation()
            if result.n_new != max_tokens:
                raise RuntimeError(
                    f"{name} trial {trial} decoded {result.n_new} tok, "
                    f"expected {max_tokens} (stop tokens must stay off)"
                )
            if not graph_on:
                raise RuntimeError(
                    f"{name} trial {trial} did not capture a CUDA graph:\n"
                    + "\n".join(lines)
                )
            us_tok = 1e6 * result.decode_s / result.n_new
            us[name].append(us_tok)
            tok_s[name].append(result.decode_tok_s)
            print(
                f"  {name:<16} {result.n_new} tok  {us_tok:7.1f} us/tok  "
                f"{result.decode_tok_s:6.1f} tok/s",
                flush=True,
            )
    return us, tok_s


def _print_isolated() -> dict[str, tuple[float, int]]:
    print(
        f"\nIsolated cold GEMV (rotate ≥{COLD_BYTES / 1e6:.0f} MB, CUDA-graph replay)\n"
        f"{'run':<16} {'us':>8} {'GB/s':>7} {'% of 250':>9}  bytes",
        flush=True,
    )
    rows = {
        "QKV RMS": (QKV_N, True, _packed_bytes(QKV_N)),
        "QKV bare": (QKV_N, False, _packed_bytes(QKV_N)),
        "O bare": (O_N, False, _packed_bytes(O_N)),
        "O dummy RMS": (O_N, True, _packed_bytes(O_N)),
    }
    out = {}
    for name, (n_out, has_rms, nbytes) in rows.items():
        us = bench_isolated(n_out, has_rms=has_rms)
        gbs = _gbs(nbytes, us)
        out[name] = (us, nbytes)
        print(
            f"{name:<16} {us:8.2f} {gbs:7.1f} {_pct(gbs):8.1f}%  "
            f"{nbytes / 1e6:.2f} MB",
            flush=True,
        )
    return out


def _print_in_model(us: dict[str, list[float]], tok_s: dict[str, list[float]]) -> None:
    print(
        f"\nIn-model sampled decode (T={DEFAULT_TEMPERATURE} top_p={DEFAULT_TOP_P} "
        f"top_k={DEFAULT_TOP_K}, graphs on, stop off, {PROMPT!r})\n"
        f"{'run':<16} {'us/tok':>8} {'tok/s':>7}  trials",
        flush=True,
    )
    med_us = {}
    med_tok = {}
    for name, _skip, _rms, _nbytes in CONFIGS:
        med_us[name] = _median(us[name])
        med_tok[name] = _median(tok_s[name])
        spread = max(us[name]) - min(us[name])
        print(
            f"{name:<16} {med_us[name]:8.1f} {med_tok[name]:7.1f}  "
            f"{[round(x, 1) for x in us[name]]}  spread {spread:.1f} us",
            flush=True,
        )

    t_full = med_us["baseline"]
    print(
        f"\nDerived  us/token = t_full − t_skip   (peak {PEAK_GBS:.0f} GB/s)\n"
        f"{'kernel':<16} {'us/tok':>8} {'GB/s':>7} {'% of 250':>9}  bytes",
        flush=True,
    )
    for name, nbytes in (
        ("QKV", QKV_BYTES),
        ("O", O_BYTES),
        ("QKV+O", BOTH_BYTES),
    ):
        skip_name = {"QKV": "no-op QKV", "O": "no-op O", "QKV+O": "no-op both"}[name]
        dt = t_full - med_us[skip_name]
        gbs = _gbs(nbytes, dt)
        print(
            f"{name:<16} {dt:8.1f} {gbs:7.1f} {_pct(gbs):8.1f}%  "
            f"{nbytes / 1e6:.1f} MB",
            flush=True,
        )

    qkv_rms_tax = t_full - med_us["QKV strip RMS"]
    o_rms_tax = med_us["O dummy RMS"] - t_full
    print(
        f"\nRMS tax in situ\n"
        f"  QKV fused RMS (baseline − strip): {qkv_rms_tax:+.1f} us/tok\n"
        f"  O dummy RMS   (dummy − baseline): {o_rms_tax:+.1f} us/tok",
        flush=True,
    )
    print(
        f"\nBaseline sampled tok/s: {med_tok['baseline']:.1f}  "
        f"(handoff ~368; do not regress)",
        flush=True,
    )


# batch=1: (BLOCK_N, BLOCK_K_WORDS, BLOCK_B, warps, stages). RMS stays on.
LAUNCH_SWEEP = (
    ("BN2 BKW64 w1 st3 (default)", (2, 64, 1, 1, 3)),
    ("BN2 BKW32 w1 st3", (2, 32, 1, 1, 3)),
    ("BN2 BKW64 w2 st3", (2, 64, 1, 2, 3)),
    ("BN2 BKW64 w4 st3", (2, 64, 1, 4, 3)),
    ("BN2 BKW32 w2 st3", (2, 32, 1, 2, 3)),
    ("BN2 BKW32 w4 st3", (2, 32, 1, 4, 3)),
)


def run_launch_sweep(
    *,
    model_dir: Path,
    max_tokens: int,
    trials: int,
    seed: int,
) -> None:
    print(f"Loading packed model from {model_dir}", flush=True)
    model, tokenizer = load_packed(str(model_dir), flash_head=False)
    warmup_model(model, flash_head=False)
    input_ids = encode_user_prompt(tokenizer, PROMPT, model.device)
    us: dict[str, list[float]] = {name: [] for name, _ in LAUNCH_SWEEP}
    tok_s: dict[str, list[float]] = {name: [] for name, _ in LAUNCH_SWEEP}

    print(
        "\nLaunch-config sweep in full sampled decode (RMS on, graphs on, stop off). "
        "Round-robin; median. A config that wins isolated and loses in the model "
        "is a regression.",
        flush=True,
    )
    for trial in range(trials):
        print(f"\n=== trial {trial + 1}/{trials} ===", flush=True)
        for name, meta in LAUNCH_SWEEP:
            configure_gemv_ablation()
            configure_gemv_decode_launch(meta)
            try:
                result, graph_on, lines = _run_decode(
                    model, tokenizer, input_ids, max_tokens, seed
                )
            finally:
                configure_gemv_decode_launch(None)
            if result.n_new != max_tokens:
                raise RuntimeError(
                    f"{name} trial {trial} decoded {result.n_new} tok, expected {max_tokens}"
                )
            if not graph_on:
                raise RuntimeError(
                    f"{name} trial {trial} did not capture a CUDA graph:\n"
                    + "\n".join(lines)
                )
            us_tok = 1e6 * result.decode_s / result.n_new
            us[name].append(us_tok)
            tok_s[name].append(result.decode_tok_s)
            print(
                f"  {name:<28} {us_tok:7.1f} us/tok  {result.decode_tok_s:6.1f} tok/s",
                flush=True,
            )

    print(
        f"\nLaunch sweep median (T={DEFAULT_TEMPERATURE} top_p={DEFAULT_TOP_P} "
        f"top_k={DEFAULT_TOP_K})\n"
        f"{'config':<28} {'us/tok':>8} {'tok/s':>7}  vs default",
        flush=True,
    )
    base = _median(us[LAUNCH_SWEEP[0][0]])
    for name, _meta in LAUNCH_SWEEP:
        mu = _median(us[name])
        mt = _median(tok_s[name])
        spread = max(us[name]) - min(us[name])
        print(
            f"{name:<28} {mu:8.1f} {mt:7.1f}  {mu - base:+7.1f} us  "
            f"{[round(x, 1) for x in us[name]]}  spread {spread:.1f} us",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(PACKED_CKPT))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--isolated-only",
        action="store_true",
        help="Skip the in-model table (cold QKV/O only)",
    )
    parser.add_argument(
        "--launch-sweep",
        action="store_true",
        help="Round-robin batch-1 launch configs in full sampled decode (RMS on)",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    configure_gemv_ablation()
    configure_gemv_decode_launch(None)
    model_dir = Path(args.model).expanduser()
    if args.launch_sweep:
        if not (model_dir / "config.json").exists():
            raise SystemExit(f"packed checkpoint missing: {model_dir}")
        run_launch_sweep(
            model_dir=model_dir,
            max_tokens=args.max_tokens,
            trials=args.trials,
            seed=args.seed,
        )
        return 0
    isolated = _print_isolated()
    del isolated
    if args.isolated_only:
        return 0
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"packed checkpoint missing: {model_dir}")
    us, tok_s = run_in_model(
        model_dir=model_dir,
        max_tokens=args.max_tokens,
        trials=args.trials,
        seed=args.seed,
    )
    _print_in_model(us, tok_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
