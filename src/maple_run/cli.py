"""CLI entry point. Implementation lives in later modules; see docs/HANDOFF.md."""

from __future__ import annotations

import argparse
import sys

from maple_run import __version__


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-tokens", type=int, default=128)
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
    if args.max_tokens < 0:
        parser.error("--max-tokens must be >= 0")


def main(argv: list[str] | None = None) -> int:
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
    _add_sampling_args(generate)

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
    _add_sampling_args(serve)

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
        )
        return 0
    if args.cmd == "serve":
        _validate_sampling_args(parser, args)
        if args.port < 0 or args.port > 65535:
            parser.error("--port must be in 0..65535")
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
        )
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
