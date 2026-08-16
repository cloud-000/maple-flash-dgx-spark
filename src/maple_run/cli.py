"""CLI entry point. Implementation lives in later modules; see docs/HANDOFF.md."""

from __future__ import annotations

import argparse
import sys

from maple_run import __version__


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
    convert.add_argument("-o", "--output", required=True, help="Output directory")
    convert.add_argument(
        "--threshold-scale",
        type=float,
        default=0.7,
        help="Ternarization threshold scale (DeepGrove default 0.7)",
    )

    generate = sub.add_parser("generate", help="Decode from a packed checkpoint")
    generate.add_argument("--model", required=True, help="Packed checkpoint directory")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", type=int, default=128)

    args = parser.parse_args(argv)
    if args.cmd == "convert":
        from maple_run.convert import convert_checkpoint

        convert_checkpoint(
            args.source, args.output, threshold_scale=args.threshold_scale
        )
        return 0
    if args.cmd == "generate":
        from maple_run.generate import generate

        generate(args.model, args.prompt, max_tokens=args.max_tokens)
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
