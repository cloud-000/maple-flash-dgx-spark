"""CLI entry point. Implementation lives in later modules; see docs/HANDOFF.md."""

from __future__ import annotations

import argparse
import sys

from maple_run import __version__

_EVAL_MOVED = (
    "maple-run eval moved to the bench project.\n"
    "Serve the packed checkpoint, then run the harness:\n"
    "  maple-run serve --model checkpoints/maple-2bit --port 8000\n"
    "  uv run --directory ../bench bench eval "
    "--base-url http://127.0.0.1:8000/v1 --model maple-2bit "
    "--output runs/maple-2bit"
)


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens (default 128). -1 fills remaining context",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (default 1.0). 0 is greedy argmax",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling cutoff after top-k (default 0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Keep the top-k softmax tokens before nucleus (default 20). 1 is greedy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="CUDA generator seed for sampling (ignored when greedy)",
    )
    parser.add_argument(
        "--flash-head",
        action="store_true",
        help="Approximate lm_head: score cluster centroids, then exact logits "
        "for the top 512 clusters (requires --flash-head-only on the checkpoint)",
    )


def _validate_sampling_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if args.top_p <= 0 or args.top_p > 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    if args.max_tokens < -1:
        parser.error("--max-tokens must be >= 0 or -1")


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "eval":
        print(_EVAL_MOVED, file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(
        prog="maple-run",
        description="Packed ternary CUDA runtime for DeepGrove Maple-Preview.",
    )
    parser.add_argument("--version", action="version", version=f"maple-run {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    convert = sub.add_parser(
        "convert",
        help="Pack HF bf16 Maple weights into 2-bit codes + row_alpha",
    )
    convert.add_argument(
        "source",
        help="Hugging Face repo id (uses the local hub cache if present) "
        "or a checkpoint directory",
    )
    convert.add_argument("-o", "--output", default=None, help="Output directory")
    convert.add_argument(
        "--threshold-scale",
        type=float,
        default=0.7,
        help="Ternarization threshold scale (DeepGrove default 0.7)",
    )
    convert.add_argument(
        "--flash-head",
        action="store_true",
        help="After conversion, cluster the lm_head and attach FlashHead data",
    )
    convert.add_argument(
        "--flash-head-only",
        action="store_true",
        help="Treat source as an already-packed directory; only attach FlashHead",
    )
    convert.add_argument(
        "--clusters",
        type=int,
        default=4748,
        help="FlashHead clusters (must divide vocab size; default 4748)",
    )
    convert.add_argument(
        "--probes",
        type=int,
        default=512,
        help="FlashHead probes per token (default 512)",
    )
    convert.add_argument(
        "--kmeans-iters",
        type=int,
        default=60,
        help="FlashHead k-means iterations (default 60)",
    )

    generate = sub.add_parser("generate", help="Decode from a packed checkpoint")
    generate.add_argument("--model", required=True, help="Packed checkpoint directory")
    generate.add_argument("--prompt", required=True)
    generate.add_argument(
        "--eggroll",
        default=None,
        help="Directory of saved EGGROLL adapters (disables FlashHead)",
    )
    _add_sampling_args(generate)

    sub.add_parser(
        "eval",
        help="Moved to the bench project (OpenAI-compatible quality evals)",
    )

    serve = sub.add_parser(
        "serve",
        help="OpenAI-compatible HTTP server for a packed checkpoint",
    )
    serve.add_argument("--model", required=True, help="Packed checkpoint directory")
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1; use 0.0.0.0 to listen on all interfaces)",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default 8000)",
    )
    serve.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Concurrent decode slots (default 8)",
    )
    serve.add_argument(
        "--max-len",
        type=int,
        default=8192,
        help="KV cache length per slot, prompt + new tokens (default 8192)",
    )
    _add_sampling_args(serve)

    eggroll = sub.add_parser(
        "eggroll",
        help="Post-train packed Maple with EGGROLL (rank-1 evolutionary strategy)",
    )
    eggroll.add_argument("--model", required=True, help="Packed checkpoint directory")
    eggroll.add_argument(
        "-o",
        "--output",
        required=True,
        help="Directory for EGGROLL adapters, history.jsonl, and episodes.jsonl",
    )
    eggroll.add_argument(
        "--resume",
        default=None,
        help="Existing adapter directory to continue from",
    )
    eggroll.add_argument("--steps", type=int, default=50, help="ES steps (default 50)")
    eggroll.add_argument(
        "--population",
        type=int,
        default=16,
        help="Population size, even, antithetic pairs (default 16)",
    )
    eggroll.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Perturbation rank (default 1; paper's fast path)",
    )
    eggroll.add_argument(
        "--r-max",
        type=int,
        default=32,
        help="SVD cap on fused residual rank (default 32)",
    )
    eggroll.add_argument(
        "--sigma",
        type=float,
        default=0.001,
        help="Perturbation std (default 0.001)",
    )
    eggroll.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="ES step size alpha (default 0.001)",
    )
    eggroll.add_argument("--seed", type=int, default=0)
    eggroll.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max new tokens per rollout (default 256)",
    )
    eggroll.add_argument(
        "--max-tool-rounds",
        type=int,
        default=6,
        help="Max tool-call rounds per agent task (default 6)",
    )
    eggroll.add_argument(
        "--prompts-per-step",
        type=int,
        default=4,
        help="Tasks mixed into each ES step (default 4)",
    )
    eggroll.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help=(
            "Concurrent decode slots for same-member single-turn rollouts "
            "(default 8). Tool-call episodes stay serial. Raise "
            "--prompts-per-step to fill the batch"
        ),
    )
    eggroll.add_argument(
        "--modules",
        default="qkv,o_proj,down,lm_head",
        help="Comma-separated adapters (router is frozen by default)",
    )
    eggroll.add_argument(
        "--eval-only",
        action="store_true",
        help="Score the suite without taking an ES step",
    )
    eggroll.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Write adapters every N steps (default 10)",
    )
    eggroll.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Rollout temperature (default 0 greedy, as in eggroll-vllm)",
    )
    eggroll.add_argument("--top-p", type=float, default=0.95)
    eggroll.add_argument("--top-k", type=int, default=20)
    eggroll.add_argument(
        "--env",
        default=None,
        help=(
            "Comma-separated env plugins from maple_run.eggroll_envs or --env-dir "
            "(e.g. RefusalEnv,DoomEnv,ProceduralSearch,NemotronIPI). "
            "Default is the built-in 21-prompt suite. Base types (SearchEnv, "
            "CodingEnv) must be subclassed."
        ),
    )
    eggroll.add_argument(
        "--env-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of extra env plugins (also auto-loads ./eggroll_envs if present)",
    )
    eggroll.add_argument(
        "--trace-text",
        action="store_true",
        help="Store full prompt/text/reasoning in episodes.jsonl (default: truncate)",
    )
    eggroll.add_argument(
        "--trace-chars",
        type=int,
        default=2000,
        help="Max chars of prompt/text/reasoning per episode (default 2000; 0 omits them)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "convert":
        if args.flash_head_only and args.flash_head:
            parser.error("use --flash-head-only or --flash-head, not both")
        if args.flash_head_only:
            from maple_run.flash_head import generate_flash_head

            generate_flash_head(
                args.source,
                n_clusters=args.clusters,
                n_iter=args.kmeans_iters,
                n_probes=args.probes,
            )
            return 0
        if args.output is None:
            parser.error("--output is required")
        from maple_run.convert import convert_checkpoint

        convert_checkpoint(
            args.source, args.output, threshold_scale=args.threshold_scale
        )
        if args.flash_head:
            from maple_run.flash_head import generate_flash_head

            generate_flash_head(
                args.output,
                n_clusters=args.clusters,
                n_iter=args.kmeans_iters,
                n_probes=args.probes,
            )
        return 0
    if args.cmd == "generate":
        _validate_sampling_args(parser, args)
        from maple_run.generate import generate

        generate(
            args.model,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            flash_head=args.flash_head,
            eggroll=args.eggroll,
        )
        return 0
    if args.cmd == "serve":
        _validate_sampling_args(parser, args)
        if args.port < 0 or args.port > 65535:
            parser.error("--port must be in 0..65535")
        if args.max_batch < 1:
            parser.error("--max-batch must be >= 1")
        if args.max_len < 1:
            parser.error("--max-len must be >= 1")
        from maple_run.server import serve

        serve(
            args.model,
            host=args.host,
            port=args.port,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed,
            flash_head=args.flash_head,
            max_batch=args.max_batch,
            max_len=args.max_len,
        )
        return 0
    if args.cmd == "eggroll":
        if args.population < 2 or args.population % 2:
            parser.error("--population must be even and >= 2")
        if args.rank < 1:
            parser.error("--rank must be >= 1")
        if args.r_max < args.rank:
            parser.error("--r-max must be >= --rank")
        if args.sigma <= 0:
            parser.error("--sigma must be > 0")
        if args.lr <= 0:
            parser.error("--lr must be > 0")
        if args.steps < 0:
            parser.error("--steps must be >= 0")
        if args.max_tool_rounds < 1:
            parser.error("--max-tool-rounds must be >= 1")
        if args.prompts_per_step < 1:
            parser.error("--prompts-per-step must be >= 1")
        if args.max_batch < 1:
            parser.error("--max-batch must be >= 1")
        if args.temperature < 0:
            parser.error("--temperature must be >= 0")
        if args.top_p <= 0 or args.top_p > 1:
            parser.error("--top-p must be in (0, 1]")
        if args.top_k < 0:
            parser.error("--top-k must be >= 0")
        if args.max_tokens < 1:
            parser.error("--max-tokens must be >= 1")
        if args.trace_chars < 0:
            parser.error("--trace-chars must be >= 0")
        env_dirs = tuple(args.env_dir or ())
        if args.env:
            from maple_run.eggroll.envs import get_envs, load_plugins

            names = [n.strip() for n in args.env.split(",") if n.strip()]
            if not names:
                parser.error("--env must list at least one plugin name")
            try:
                load_plugins(list(env_dirs) if env_dirs else None)
                get_envs(names)
            except (KeyError, TypeError, FileNotFoundError) as exc:
                parser.error(str(exc))
        elif env_dirs:
            from maple_run.eggroll.envs import load_plugins

            try:
                load_plugins(list(env_dirs))
            except FileNotFoundError as exc:
                parser.error(str(exc))
        from maple_run.eggroll.train import TrainConfig, train

        modules = tuple(m.strip() for m in args.modules.split(",") if m.strip())
        train(
            TrainConfig(
                model_dir=args.model,
                output_dir=args.output,
                resume=args.resume,
                steps=args.steps,
                population=args.population,
                rank=args.rank,
                r_max=args.r_max,
                sigma=args.sigma,
                lr=args.lr,
                seed=args.seed,
                max_tokens=args.max_tokens,
                max_tool_rounds=args.max_tool_rounds,
                prompts_per_step=args.prompts_per_step,
                max_batch=args.max_batch,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                modules=modules,
                eval_only=args.eval_only,
                save_every=args.save_every,
                env_names=args.env,
                env_dirs=env_dirs,
                trace_text=args.trace_text,
                trace_chars=args.trace_chars,
            )
        )
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
