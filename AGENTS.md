# Agent Instructions

These instructions apply to the entire repository.

## What this is

`maple-run` is a uv Python project that will pack DeepGrove Maple-Preview's
Hugging Face bf16 dump into 2-bit ternary codes and run packed CUDA/Triton
kernels on DGX Spark (GB10, sm_121). It is a custom runtime, not a vLLM recipe.

Read `docs/HANDOFF.md` before writing kernels, a converter, or a generate loop.
Read `docs/SOURCES.md` before searching the web for Maple, MLX conversion, or
hardware numbers. Vendored algorithm source is `docs/sources/mlx_lm_ternary.py`.

## Choose the work from the handoff phases

1. **Pack** (`maple_run/pack.py`, `maple_run/convert.py`): **done.** CPU NumPy
   port of DeepGrove `ternarize` / `_pack_2bit` plus streaming convert. Packed
   checkpoint lives at `checkpoints/maple-2bit` (gitignored).
2. **Kernels** (`maple_run/kernels/ternary_gemv.py`): **done.** Packed GEMV that
   never unpacks to a dense bf16 matrix.
3. **Model** (`maple_run/model.py`): **done, then fused further.** Packed Maple
   forward, fused RMS/QKV/SwiGLU, decode attn, greedy generate. Exact-head
   ~144–171 tok/s; Spark should scale toward ~386 tok/s (`169 × 273/120`)
   before FlashHead. **Start here** (more kernel/launch efficiency).
4. **FlashHead** (`maple_run/generate.py` / head clustering): not until asked.

Stop at the phase the user asked for. Do not skip packing tests to jump to a
server.

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
- `NotImplementedError` stubs are intentional; replace them in the matching
  phase instead of deleting the module layout.
