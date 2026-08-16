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
2. **Kernels** (`maple_run/kernels/ternary_gemv.py`): packed GEMV that never
   unpacks to a dense bf16 matrix. **Start here.**
3. **Model** (`maple_run/model.py`): Maple forward with packed linears, fused
   expert dispatch, then decode. FlashHead is later.

Stop at the phase the user asked for. Do not skip packing tests to jump to a
server.

## Environment

- Work from the repository root with `uv`.
- Python 3.12 (`requires-python = ">=3.12"`).
- Do not `uv add torch` until you have confirmed this host's aarch64 CUDA
  PyTorch. Spark is not a generic x86 wheel.
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
