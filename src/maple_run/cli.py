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


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str] | None) -> set[str]:
    """Dest names the user actually passed, including subparser flags."""
    tokens = list(argv if argv is not None else sys.argv[1:])
    opt_to_dest: dict[str, str] = {}

    def _walk(p: argparse.ArgumentParser) -> None:
        for action in p._actions:
            for opt in getattr(action, "option_strings", []) or []:
                opt_to_dest[opt] = action.dest
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                for sub in choices.values():
                    _walk(sub)

    _walk(parser)
    seen: set[str] = set()
    for tok in tokens:
        key = tok.split("=", 1)[0]
        if key in opt_to_dest:
            seen.add(opt_to_dest[key])
    return seen


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
    generate.add_argument(
        "--nextlat",
        action="store_true",
        help="Self-speculative decode with a trained NextLat p_ψ "
        "(nextlat.safetensors next to the packed checkpoint)",
    )
    generate.add_argument(
        "--draft-len",
        type=int,
        default=5,
        help="NextLat draft tokens per verify round (default 5)",
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
            "Overrides the environments from --recipe. Base types (SearchEnv, "
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
        "--recipe",
        default="default",
        help=(
            "Training recipe (default, agent, ipi, or a plugin from --recipe-dir). "
            "Supplies environments and hyperparameter defaults; CLI flags override."
        ),
    )
    eggroll.add_argument(
        "--recipe-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of extra recipe plugins (also auto-loads ./eggroll_recipes if present)",
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

    nextlat = sub.add_parser(
        "nextlat",
        help="Train NextLat latent dynamics p_ψ on frozen packed Maple",
    )
    nextlat.add_argument("--model", required=True, help="Packed checkpoint directory")
    nextlat.add_argument(
        "--data",
        required=True,
        help="Text file (one sequence per line) or JSONL with input_ids "
        "or text/content/prompt",
    )
    nextlat.add_argument(
        "-o",
        "--output",
        default=None,
        help="Directory for nextlat.safetensors (default: --model)",
    )
    nextlat.add_argument("--steps", type=int, default=200, help="AdamW steps (default 200)")
    nextlat.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Truncate each sequence to this many tokens (default 256)",
    )
    nextlat.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    nextlat.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Multi-step unroll d (default 1; paper FineWeb-Edu used 1 or 2)",
    )
    nextlat.add_argument(
        "--lambda-next-h",
        type=float,
        default=1.0,
        help="Weight on Smooth-L1 next-hidden loss (default 1.0)",
    )
    nextlat.add_argument(
        "--lambda-kl",
        type=float,
        default=1.0,
        help="Weight on KL through the frozen lm_head (default 1.0)",
    )
    nextlat.add_argument(
        "--mlp-hidden",
        type=int,
        default=None,
        help="p_ψ MLP width (default: FineWeb-Edu ratio, 6528 at hidden=2048)",
    )
    nextlat.add_argument("--seed", type=int, default=0)
    nextlat.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap the number of training sequences",
    )

    synth = sub.add_parser(
        "nextlat-synth",
        help="Generate Maple completions for NextLat on-policy jsonl",
    )
    synth.add_argument("--model", required=True, help="Packed checkpoint directory")
    synth.add_argument(
        "--prompts",
        required=True,
        help="Prompt jsonl, a text file, or a Hugging Face dataset id "
        "(e.g. HuggingFaceH4/ultrachat_200k)",
    )
    synth.add_argument(
        "-o",
        "--output",
        default="nextlat/data/onpolicy.jsonl",
        help="Output jsonl (default nextlat/data/onpolicy.jsonl)",
    )
    synth.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N prompts (default: all)",
    )
    synth.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max new tokens per prompt (default 256, to fill --seq-len 256)",
    )
    synth.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Match generate default (1.0). 0 is greedy",
    )
    synth.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Match generate default (0.95)",
    )
    synth.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Match generate default (20)",
    )
    synth.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base CUDA seed; prompt i uses seed+i (batch uses seed of first row)",
    )
    synth.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Decode this many prompts together (default 8; 1 is sequential)",
    )
    synth.add_argument(
        "--split",
        default=None,
        help="HF dataset split (default: train_sft, then train)",
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
        if args.draft_len < 1:
            parser.error("--draft-len must be >= 1")
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
            nextlat=args.nextlat,
            draft_len=args.draft_len,
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
        from maple_run.eggroll.recipes import get_recipe, load_recipe_plugins

        recipe_dirs = tuple(args.recipe_dir or ())
        try:
            load_recipe_plugins(list(recipe_dirs) if recipe_dirs else None)
            recipe = get_recipe(args.recipe)
        except (KeyError, FileNotFoundError, ImportError) as exc:
            parser.error(str(exc))
        explicit = _explicit_cli_dests(parser, argv)
        resolved, overrides = recipe.resolve(args, explicit)
        if resolved["population"] < 2 or resolved["population"] % 2:
            parser.error("--population must be even and >= 2")
        if resolved["rank"] < 1:
            parser.error("--rank must be >= 1")
        if resolved["r_max"] < resolved["rank"]:
            parser.error("--r-max must be >= --rank")
        if resolved["sigma"] <= 0:
            parser.error("--sigma must be > 0")
        if resolved["lr"] <= 0:
            parser.error("--lr must be > 0")
        if resolved["steps"] < 0:
            parser.error("--steps must be >= 0")
        if resolved["max_tool_rounds"] < 1:
            parser.error("--max-tool-rounds must be >= 1")
        if resolved["prompts_per_step"] < 1:
            parser.error("--prompts-per-step must be >= 1")
        if resolved["max_batch"] < 1:
            parser.error("--max-batch must be >= 1")
        if resolved["temperature"] < 0:
            parser.error("--temperature must be >= 0")
        if resolved["top_p"] <= 0 or resolved["top_p"] > 1:
            parser.error("--top-p must be in (0, 1]")
        if resolved["top_k"] < 0:
            parser.error("--top-k must be >= 0")
        if resolved["max_tokens"] < 1:
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

        train(
            TrainConfig(
                model_dir=args.model,
                output_dir=args.output,
                resume=args.resume,
                eval_only=args.eval_only,
                env_names=args.env,
                env_dirs=env_dirs,
                recipe_name=recipe.name,
                recipe_dirs=recipe_dirs,
                recipe_defaults=recipe.defaults(),
                cli_overrides=overrides,
                trace_text=args.trace_text,
                trace_chars=args.trace_chars,
                **resolved,
            )
        )
        return 0
    if args.cmd == "nextlat":
        if args.steps < 1:
            parser.error("--steps must be >= 1")
        if args.seq_len < 3:
            parser.error("--seq-len must be >= 3")
        if args.lr <= 0:
            parser.error("--lr must be > 0")
        if args.horizon < 1:
            parser.error("--horizon must be >= 1")
        if args.mlp_hidden is not None and args.mlp_hidden < 1:
            parser.error("--mlp-hidden must be >= 1")
        if args.max_sequences is not None and args.max_sequences < 1:
            parser.error("--max-sequences must be >= 1")
        from maple_run.nextlat_train import train_nextlat

        train_nextlat(
            args.model,
            args.data,
            output=args.output,
            steps=args.steps,
            seq_len=args.seq_len,
            lr=args.lr,
            horizon=args.horizon,
            lambda_next_h=args.lambda_next_h,
            lambda_kl=args.lambda_kl,
            mlp_hidden=args.mlp_hidden,
            seed=args.seed,
            max_sequences=args.max_sequences,
        )
        return 0
    if args.cmd == "nextlat-synth":
        _validate_sampling_args(parser, args)
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be >= 1")
        if args.batch_size < 1:
            parser.error("--batch-size must be >= 1")
        from maple_run.nextlat_synth import synthesize_onpolicy

        synthesize_onpolicy(
            args.model,
            args.prompts,
            output=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            limit=args.limit,
            batch_size=args.batch_size,
            split=args.split,
        )
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
