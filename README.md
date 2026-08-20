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
```

Quality evals (LCBv6, AIME 2026, HMMT Feb 2026, GPQA-D) live in the sibling [`bench`](../bench) project and talk to `maple-run serve` over OpenAI `/v1/chat/completions`. Dense-head protocol is T=1.0 / top_p=0.95 / top_k=20, 4 samples, 64k max tokens.

```bash
uv run --directory ../bench bench eval \
  --base-url http://127.0.0.1:8000/v1 --model maple-2bit \
  --output runs/maple-2bit
```

```bash
uv sync
uv run maple-run --help
```
