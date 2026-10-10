# Agent Instructions

These instructions apply to the entire repository.

## What this is

`maple-run` is a uv Python project that packs DeepGrove Maple-Preview's
Hugging Face bf16 dump into 2-bit ternary codes and runs packed CUDA/Triton
kernels on DGX Spark (GB10, sm_121). It is a custom runtime, not a vLLM recipe.

## Documentation map

- `AGENTS.md` is the current operating guide and repository status summary.
- `docs/CHAT_TEMPLATE.md` defines the ChatML chat template syntax, role
  normalization, reasoning (`<think>`) retention scopes (conversation vs.
  stateless vs. turn-scoped), and verification suite.
- `docs/FIDELITY.md` defines the A/B/C model representations, quantization
  boundaries, fidelity methodology, known reference-loader pitfall, and
  September 2026 baseline measurements. Read it before changing conversion,
  packed model math, attention, routing, or the `fidelity` command.
- `docs/SOURCES.md` lists local and upstream evidence. Read it before searching
  the web for Maple, MLX conversion, or hardware numbers.
- `docs/HANDOFF.md` is a historical implementation handoff and engineering
  record. Its phase plan is complete; use it for design rationale, benchmark
  history, and host-specific measurement notes, not as the current task queue.
- `docs/sources/mlx_lm_ternary.py` is the vendored authoritative conversion
  algorithm.

## Current implementation status

The original implementation phases are complete. Preserve the default exact
head generate path and treat optional experimental paths separately.

1. **Pack** (`maple_run/pack.py`, `maple_run/convert.py`): **done.** CPU NumPy
   port of DeepGrove `ternarize` / `_pack_2bit` plus streaming convert. Packed
   checkpoint lives at `checkpoints/maple-2bit` (gitignored).
2. **Kernels** (`maple_run/kernels/ternary_gemv.py`): **done.** Packed GEMV that
   never unpacks to a dense bf16 matrix.
3. **Model** (`maple_run/model.py`): **done, then fused and tuned twice.**
   Packed Maple forward, fused RMS/QKV/SwiGLU, decode attn, fused router, fused
   sampler, greedy+sampled generate. Speed bench / CLI default is sampled
   (`T=1.0 top_p=0.95 top_k=20`), ~368 tok/s (256-tok bench) / ~364 tok/s
   (700-tok haiku); greedy France (`--temperature 0`) is correctness only.
   That is past the M4-scaled target computed against the bandwidth this host
   actually delivers (~250 GB/s, not the 273 GB/s spec) and ~5% short of the
   386 tok/s computed from the spec sheet. Default generate path — do not
   regress it.
4. **FlashHead** (`maple_run/flash_head.py`): **done.** `--flash-head` scores
   4748 centroids then exact logits for 512 clusters (~487 tok/s). Clusters
   are attached on `checkpoints/maple-2bit`. Prefill stays exact.
5. **HTTP** (`maple_run/server.py`): **done.** `maple-run serve` is an
   OpenAI-compatible `/v1/chat/completions` endpoint. Same sampling flags as
   `generate` are CLI defaults; the request body may override them.
6. **NextLat** (`maple_run/nextlat.py`): **implemented but opt-in and
   experimental.** Train `p_ψ` on frozen
   Maple (`maple-run nextlat`), optional on-policy dumps via
   `maple-run nextlat-synth` into `nextlat/data/`, then `--nextlat`
   speculative decode. Do not regress the default exact-head generate path.

Quality evaluations live in the sibling `bench` project rather than this
repository.

## Fidelity reference hierarchy

- **A:** original dense bf16 Transformers. Use it to measure quantization
  impact, after validating its tokenizer, attention layout, and RoPE state.
- **B:** dense dequantized packed weights. This is the primary mathematical
  reference for packed runtime correctness.
- **C:** production packed `maple-run`. B-versus-C isolates kernel/fusion
  behavior; A-versus-C also includes 4-bit embedding and LM-head quantization.

Do not assume a dense Transformers forward is ground truth merely because it
completes. See `docs/FIDELITY.md` for the reference RoPE incident and baseline.

## Environment

- Work from the repository root with `uv`.
- Python 3.12 (`requires-python = ">=3.12"`).
- CUDA extra is `torch==2.13.0` / `triton==3.7.1`, confirmed on this GB10
  (aarch64, CUDA 13.0, `2.13.0+cu130`). Do not swap in a CPU wheel.
- Do not install Dao-AILab `flash_attn` into `~/Code/spark-vllm-docker` or its
  container image. That image is a separate project; Maple is not a supported
  vLLM architecture (`MapleForCausalLM`).
- Do not launch vLLM, Docker builds, or `run-recipe.sh` from this repo.

## Boundaries

- Preserve unrelated user changes.
- Do not commit Hugging Face weight shards or packed checkpoints.
- Do not expose tokens, `.env`, or credentials.
- Prefer sources already in `docs/sources/` over new web searches.
- Preserve intentional `NotImplementedError` stubs unless the requested work
  explicitly covers them; do not delete module layout as cleanup.
