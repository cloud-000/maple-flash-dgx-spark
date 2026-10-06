"""CLI entry point for maple-run."""

from __future__ import annotations

import argparse
import sys

from maple_run import __version__


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


def _add_dense_head_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dense-head",
        action="store_true",
        help="Evaluation mode: use the original BF16 lm_head with the packed ternary body",
    )
    parser.add_argument(
        "--original",
        default="deepgrove/maple-preview",
        help="Original checkpoint supplying the dense BF16 head and canonical tokenizer",
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
    parser = argparse.ArgumentParser(
        prog="maple-run",
        description="Packed ternary CUDA runtime for DeepGrove Maple-Preview.",
    )
    parser.add_argument("--version", action="version", version=f"maple-run {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. convert
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

    # 2. generate
    generate = sub.add_parser("generate", help="Decode from a packed checkpoint")
    generate.add_argument("--model", required=True, help="Packed checkpoint directory")
    generate.add_argument("--prompt", required=True)
    _add_sampling_args(generate)
    _add_dense_head_args(generate)

    # 3. fidelity
    fidelity = sub.add_parser(
        "fidelity",
        help="Teacher-forced original bf16 vs packed exact-head fidelity report",
    )
    fidelity.add_argument("--model", required=True, help="Packed checkpoint directory")
    fidelity.add_argument(
        "--original",
        default="deepgrove/maple-preview",
        help="Original bf16 checkpoint directory or locally cached Hugging Face repo (default: deepgrove/maple-preview)",
    )
    fidelity.add_argument("--prompts", required=True, help="Text/JSONL prompt corpus")
    fidelity.add_argument("--output", required=True, help="JSON report path")
    fidelity.add_argument("--limit", type=int, default=None, help="Maximum prompts")
    fidelity.add_argument(
        "--split", default=None, help="Dataset split when --prompts is a dataset id"
    )
    fidelity.add_argument(
        "--max-seq-len", type=int, default=512, help="Maximum tokens per prompt (default 512)"
    )
    fidelity.add_argument("--max-kl", type=float, default=None, help="Fail if mean forward KL exceeds this")
    fidelity.add_argument(
        "--min-top1-agreement", type=float, default=None,
        help="Fail if aggregate top-1 agreement is lower",
    )

    # 4. serve
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
    _add_dense_head_args(serve)

    # 5. evaluate
    evaluate = sub.add_parser(
        "evaluate",
        help="System One (Jev) fast discriminative evaluation on packed checkpoint",
    )
    evaluate.add_argument("--model", required=True, help="Packed checkpoint directory")
    evaluate.add_argument(
        "--request",
        default=None,
        help="Path to JSON file containing System One request body",
    )
    evaluate.add_argument(
        "--state",
        default=None,
        help="Inline state context string (when not using --request)",
    )
    evaluate.add_argument(
        "--question",
        default=None,
        help="Inline JSON string of questions or a single question definition",
    )
    evaluate.add_argument(
        "--output",
        default=None,
        help="Optional path to write output JSON response",
    )

    args = parser.parse_args(argv)

    if args.cmd == "convert":
        if args.threshold_scale <= 0:
            parser.error("--threshold-scale must be > 0")
        if args.clusters <= 0:
            parser.error("--clusters must be > 0")
        if args.probes <= 0:
            parser.error("--probes must be > 0")
        if args.kmeans_iters <= 0:
            parser.error("--kmeans-iters must be > 0")
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
        if args.dense_head and args.flash_head:
            parser.error("--dense-head and --flash-head are mutually exclusive")
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
            dense_head_source=args.original if args.dense_head else None,
        )
        return 0

    if args.cmd == "fidelity":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be >= 1")
        if args.max_seq_len < 2:
            parser.error("--max-seq-len must be >= 2")
        if args.max_kl is not None and args.max_kl < 0:
            parser.error("--max-kl must be >= 0")
        if args.min_top1_agreement is not None and not 0 <= args.min_top1_agreement <= 1:
            parser.error("--min-top1-agreement must be in [0, 1]")
        from maple_run.fidelity import run_fidelity

        ok = run_fidelity(
            args.model,
            args.prompts,
            output=args.output,
            original_dir=args.original,
            limit=args.limit,
            split=args.split,
            max_seq_len=args.max_seq_len,
            max_kl=args.max_kl,
            min_top1_agreement=args.min_top1_agreement,
        )
        return 0 if ok else 1

    if args.cmd == "serve":
        _validate_sampling_args(parser, args)
        if args.dense_head and args.flash_head:
            parser.error("--dense-head and --flash-head are mutually exclusive")
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
            dense_head_source=args.original if args.dense_head else None,
            max_batch=args.max_batch,
            max_len=args.max_len,
        )
        return 0

    if args.cmd == "evaluate":
        import json
        from pathlib import Path
        from maple_run.generate import load_packed
        from maple_run.system_one import (
            SystemOneEvaluator,
            parse_system_one_request,
        )

        if args.request:
            req_path = Path(args.request)
            if not req_path.exists():
                parser.error(f"request file not found: {args.request}")
            body = json.loads(req_path.read_text(encoding="utf-8"))
        elif args.state and args.question:
            try:
                q_data = json.loads(args.question)
            except Exception as e:
                parser.error(f"invalid JSON in --question: {e}")
            if "type" in q_data:
                questions = {"eval": q_data}
            else:
                questions = q_data
            body = {
                "model": Path(args.model).name,
                "state": args.state,
                "questions": questions,
            }
        else:
            parser.error("must provide either --request <file.json> or both --state and --question")

        model, tokenizer = load_packed(args.model)
        evaluator = SystemOneEvaluator(model, tokenizer)
        req = parse_system_one_request(body, default_model=Path(args.model).name)
        resp = evaluator.evaluate_request(req)
        result_dict = resp.to_dict()
        out_json = json.dumps(result_dict, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(out_json, encoding="utf-8")
        print(out_json)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
