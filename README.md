# maple-run

Packed ternary CUDA runtime for [deepgrove/maple-preview](https://huggingface.co/deepgrove/maple-preview) on NVIDIA DGX Spark.

This is **not** a vLLM wrapper. Maple's Hugging Face dump is native-ternary values stored as bf16 (~40 GB). The Mac 220 tok/s path uses packed 2-bit weights (~5.3 GB) plus custom kernels. This repo is that path on CUDA.

Phase 1 convert works (`uv run maple-run convert deepgrove/maple-preview -o checkpoints/maple-2bit`). Phase 2 packed GEMV is implemented (`maple_run.kernels.ternary_gemv`). Plan: [`docs/HANDOFF.md`](docs/HANDOFF.md). Agent rules: [`AGENTS.md`](AGENTS.md). Local sources: [`docs/SOURCES.md`](docs/SOURCES.md).

```bash
uv sync
uv run maple-run --help
```
