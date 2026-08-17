# maple-run

Packed ternary CUDA runtime for [deepgrove/maple-preview](https://huggingface.co/deepgrove/maple-preview) on NVIDIA DGX Spark.

This is **not** a vLLM wrapper. Maple's Hugging Face dump is native-ternary values stored as bf16 (~40 GB). The Mac 220 tok/s path uses packed 2-bit weights (~5.3 GB) plus custom kernels. This repo is that path on CUDA.

Phase 1 convert works (`uv run maple-run convert deepgrove/maple-preview -o checkpoints/maple-2bit`). Phase 2 packed GEMV, phase 3 packed decode, phase 4 FlashHead, and phase 5 HTTP serve are implemented. Plan: [`docs/HANDOFF.md`](docs/HANDOFF.md). Agent rules: [`AGENTS.md`](AGENTS.md). Local sources: [`docs/SOURCES.md`](docs/SOURCES.md).

```bash
uv sync --extra cuda --extra tokenizer --extra eval
uv run maple-run generate --model checkpoints/maple-2bit \
  --prompt "Write a haiku about a grove." --max-tokens 128
uv run maple-run generate --model checkpoints/maple-2bit --flash-head \
  --prompt "Write a haiku about a grove." --max-tokens 128
uv run maple-run serve --model checkpoints/maple-2bit --host 127.0.0.1 --port 8000
uv run maple-run eval --model checkpoints/maple-2bit --output evals/maple-2bit
uv run maple-run eval --model checkpoints/maple-2bit --bench aime2026 --limit 1 --n-samples 1
```

`eval` reproduces DeepGrove's quality table (LCBv6, AIME 2026, HMMT Feb 2026, GPQA-D) on the **dense** 4-bit `lm_head` with packed kernels. Default protocol is T=1.0 / top_p=0.95 / top_k=20, 4 samples, 64k max tokens. Results resume from `--output`.

```bash
uv sync
uv run maple-run --help
```
