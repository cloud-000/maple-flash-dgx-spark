# maple-run

Packed ternary CUDA runtime for [deepgrove/maple-preview](https://huggingface.co/deepgrove/maple-preview) on NVIDIA DGX Spark.

This is **not** a vLLM wrapper. Maple's Hugging Face dump is native-ternary values stored as bf16 (~40 GB). The Mac 220 tok/s path uses packed 2-bit weights (~5.3 GB) plus custom kernels. This repo is that path on CUDA.

Conversion, packed GEMV, packed decode, FlashHead, and HTTP serving are implemented. Correctness and quantization reference: [`docs/FIDELITY.md`](docs/FIDELITY.md). Chat template and reasoning retention: [`docs/CHAT_TEMPLATE.md`](docs/CHAT_TEMPLATE.md). Historical implementation record: [`docs/HANDOFF.md`](docs/HANDOFF.md). Agent rules: [`AGENTS.md`](AGENTS.md). Local sources: [`docs/SOURCES.md`](docs/SOURCES.md).

## Getting Started

### 1. Install dependencies

```bash
uv sync --extra cuda --extra tokenizer --extra eval
```

### 2. Download and convert weights

`maple-run` does not execute the raw 40 GB bf16 Hugging Face weights directly. Instead, the `convert` command downloads and packs them into the 2-bit ternary format (~5.4 GB). Shards are streamed from Hugging Face one at a time and converted in-flight, so you do not need 40 GB of free memory or disk:

```bash
# Convert from Hugging Face and generate FlashHead clusters
uv run maple-run convert deepgrove/maple-preview -o checkpoints/maple-2bit --flash-head
```

*(If you already have a packed checkpoint and only want to add FlashHead clusters later, run `uv run maple-run convert checkpoints/maple-2bit --flash-head-only`.)*

### 3. Run inference or serve

```bash
# Exact-head generation
uv run maple-run generate --model checkpoints/maple-2bit \
  --prompt "Write a haiku about a grove." --max-tokens 128

# Fast FlashHead generation (~480 tok/s)
uv run maple-run generate --model checkpoints/maple-2bit --flash-head \
  --prompt "Write a haiku about a grove." --max-tokens 128

# OpenAI-compatible API server (/v1/chat/completions)
uv run maple-run serve --model checkpoints/maple-2bit --host 127.0.0.1 --port 8000
```

## Quality Evaluations

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
