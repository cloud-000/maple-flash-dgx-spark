# Handoff: implement maple-run

This document is for the **next session**. The previous session diagnosed why
`vllm serve deepgrove/maple-preview` failed on Spark, why the Hugging Face CUDA
Transformers path is slow, and what a packed-kernel runtime has to do.

**Phase 1 (CPU pack + convert) is done.** Next session starts at **phase 2**
(packed GEMV kernel). Do not re-convert unless the packed checkpoint is
missing. Do not start with a web search for Maple, MLX, or Spark bandwidth;
read `docs/SOURCES.md` and `AGENTS.md` first.

## Status (2026-08-16)

| Phase | State |
|---|---|
| 1 Pack / convert | **Done.** NumPy `ternarize` / `pack_2bit` / 4-bit RTN; streaming converter; tests in `tests/test_pack.py` and `tests/test_convert.py`. |
| 2 Packed GEMV kernel | Not started (`maple_run/kernels/ternary_gemv.py` is still a stub). |
| 3 Model decode | Not started |
| 4 FlashHead | Not started |
| 5 HTTP | Not started |

Packed checkpoint on this host (gitignored, do not commit):

```text
~/Code/maple-run/checkpoints/maple-2bit
```

Produced with:

```bash
uv run maple-run convert deepgrove/maple-preview -o checkpoints/maple-2bit
```

That command resolved the local Hugging Face hub snapshot (9× ~4.7 GB bf16
shards, never loaded through Transformers) and wrote **5.41 GB** in three
packed shards (`model-00001-of-00003.safetensors` …). Experts are stacked as
`model.layers.{L}.mlp.switch_mlp.{proj}` with `row_alpha`; layout is recorded
in `checkpoints/maple-2bit/config.json` under `maple_run`.

Source dump (also already cached):

```text
~/.cache/huggingface/hub/models--deepgrove--maple-preview/snapshots/ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07/
```

Do not load that dump with `transformers.AutoModel`. Maple's `modeling_maple.py`
imports `fa3.py` / `flash_attn`, and `from_pretrained` materializes the 40 GB
file as dense `nn.Linear` weights. Convert streams safetensors with NumPy.

## Goal

Run Maple-Preview on this DGX Spark at Mac-like decode speed: **keep weights
packed at runtime**. High-level job:

1. Pack the HF bf16 dump into 2-bit codes + per-row `α`.  ← done
2. Write kernels that read that layout and never unpack to bf16.

Python + uv is the shell. Speed is the kernels.

## What already failed (do not retry)

Command:

```text
./launch-cluster.sh --solo exec vllm serve deepgrove/maple-preview \
  --port 8000 --host 0.0.0.0 --gpu-memory-utilization 0.8 --trust-remote-code
```

in `~/Code/spark-vllm-docker`. Two errors:

1. Hugging Face `check_imports` on `modeling_maple.py` / `fa3.py` wants Dao
   `flash_attn`. Warning only.
2. Fatal: `Model architectures ['MapleForCausalLM'] are not supported for now.`

Installing `flash_attn` into the vLLM image does not add a Maple model class,
and Dao-AILab `flash_attn` is a compiled CUDA extension that is a poor fit for
GB10/SM121. Leave `spark-vllm-docker` alone.

There was a `recipes/maple-preview-flash.yaml` that called the same `vllm serve`
and attached `mods/step-3.7-flash` (a **Step-3.7** patch, not Maple). That
recipe is not a working path.

vLLM `--model-impl transformers` is also a poor bet: MapleAttention calls
`flash_attention_forward` from `fa3.py` instead of Transformers
`ALL_ATTENTION_FUNCTIONS`, and `moe_infer` is an unfused Python loop with a
GPU→CPU sync (`tokens_per_expert.cpu().numpy()`).

## Native ternary vs the 40 GB file

Maple was **trained quantization-aware** in ternary. The public CUDA dump still
stores those values as **bf16 tensors** so `nn.Linear` works. Same information
as 2-bit codes + row scale; ~8× the bytes.

DeepGrove's converter (vendored at `docs/sources/mlx_lm_ternary.py`):

> Maple is trained quantization-aware, so its bf16 weights are recovered to
> ternary values {-alpha_row, 0, +alpha_row} by thresholding (this is not
> round-to-nearest quantization; `mlx_lm.convert` cannot produce it).

`config.json` has `"dtype": "bfloat16"` and `"quantize": true`. That is compute
container vs “these linears are ternary-valued.”

| Artifact | Size | What |
|---|---|---|
| `deepgrove/maple-preview` | ~40 GB (`total_size` 40428060672 in the HF index) | Unpacked QAT weights, Transformers-shaped |
| MLX packed (`maple-2bit-mlx` / convert output) | ~5.31 GB | 2-bit codes + `row_alpha`; 4-bit head/embed |
| Information per ternary weight | `log2(3) ≈ 1.58` bits | Storage is 2 bits/code |

Not every tensor is ternary (from the converter):

- **Ternary:** 2-D `.weight` except the router (attention Q/K/V/O, expert
  gate/up/down, etc.)
- **Stay float32:** `.mlp.gate.weight` (router)
- **4-bit RTN, group 64:** `lm_head.weight`, `model.word_embeddings.weight`
- **Float 1-D:** norms

Mac 218 tok/s (M4, `--flash-head`) is packed kernels + FlashHead, not the
Transformers file. User measured ~220 tok/s on an M5 Air with that MLX stack.

## Why Spark Transformers would still lose

Decode is memory-bound: tok/s ≈ bandwidth / bytes per token.

| Machine | Bandwidth | Source |
|---|---|---|
| DGX Spark GB10 | **273 GB/s** LPDDR5x, 128 GB | NVIDIA DGX Spark spec |
| M4 Mini (base) | ~120 GB/s | Apple M4 |
| M5 Air (base M5) | **153.6 GB/s** | Apple M5, LPDDR5X 9600 |

Spark has **more** bandwidth than the Air. The HF CUDA path still loses because
it is a different program:

- Unpacked bf16 traffic for active params + full `lm_head` ≈ **2.3 GB/token**
  → Spark ceiling ~`273/2.3 ≈ 120` tok/s even if fused.
- `moe_infer` syncs to CPU and loops 256 experts in Python every layer.
- No FlashHead (M4: 169 tok/s exact head → 218 with FlashHead).

A kernel that unpacks to bf16 then matmuls has failed. Packed + fused on Spark
should be able to beat the Air (`273/154 × 220 ≈ 390` tok/s as a back-of-envelope
if efficiency matched).

GB10 tensor cores want FP4/FP8/bf16, not `{-α,0,+α}`. Expect bit-packed
integer/add kernels (BitNet-style), not a vLLM FP8 recipe.

## Packing algorithm (copy this, do not reinvent)

From `docs/sources/mlx_lm_ternary.py`:

```text
DEFAULT_THRESHOLD_SCALE = 0.7
GROUP_SIZE = 128          # ternary
HEAD_GROUP_SIZE = 64      # lm_head / embeddings 4-bit
HEAD_BITS = 4
```

`ternarize(weight)` on `[..., N, K]` with `K % 128 == 0`:

1. Cast `w` to float32 (bf16 reductions flip ~0.15% of codes and perturb α).
2. `threshold = 0.7 * mean(|w|, axis=-1)`
3. `mask = |w| > threshold`
4. `alpha = mean(|w| of survivors)` per row; cast α back to weight dtype
5. `ternary = sign(w) * mask` in `{−1, 0, +1}`
6. `packed = pack_2bit(ternary + 1)` → codes `{0, 1, 2}`
7. Default storage: `weight` packed uint32 + `row_alpha` (one α per output row).
   `--group-scales` repeats α across `K/128` groups (~+0.6 GB); only for generic
   MLX quantized readers.

`_pack_2bit`: 16 codes per uint32, LSB first, matches `mx.quantize(..., bits=2,
mode="affine")`.

Expert weights in HF are per-expert
`model.layers.{L}.mlp.experts.{E}.{proj}.weight`. The MLX converter **stacks**
them into `model.layers.{L}.mlp.switch_mlp.{proj}` after all 256 experts of a
proj arrive. Either preserve HF names or document the stacked layout in the
packed config.

FlashHead (phase 4, not 1): 4748 equal-size clusters over `lm_head` rows
(vocab 151936 ÷ 4748 = 32), probe top 512 clusters, always score force tokens
(EOS / `<|im_end|>`). See `generate_flash_head()` in the same file.

## Maple architecture (from config + modeling)

Vendored config: `docs/sources/maple-preview-config.json`.

- 24 layers, hidden 2048, head_dim 128, 16 query heads, 4 KV heads
- MoE: 256 experts, top-8, `moe_intermediate_size` 512, **no shared experts**
- Attention: 3:1 SWA-512 : global (`layer_types` repeats 3 sliding + 1 full)
- Partial RoPE 0.5; `nope_on_global_attention: true`; QK RMSNorm
- Vocab 151936, context 131072, SiLU, RMSNorm 1e-6
- MLP: clamp gate max 7, up to ±7, then down-proj (see `MapleMLP.forward`)
- Router: softmax → topk → renormalize; **not** grouped DeepSeek routing
- `tie_word_embeddings: false`

HF modeling (this host, if the snapshot is cached):

```text
~/.cache/huggingface/hub/models--deepgrove--maple-preview/snapshots/ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07/modeling_maple.py
```

Do not keep `tokens_per_expert.cpu().numpy()` or a Python loop over 256 experts.

## Suggested implementation order

### Phase 1 — pack (CPU) — **done**

Implemented in `maple_run/pack.py` and `maple_run/convert.py` (CPU NumPy, no
MLX, no CUDA). Unit tests cover LSB-first 16 codes/uint32, unpacked codes vs
`sign/mask`, 4-bit RTN, expert stacking across shards, and BF16 promotion.

`uv run maple-run convert deepgrove/maple-preview -o checkpoints/maple-2bit`
completed on this host: **5.41 GB** packed output. Repo ids use the Hugging
Face hub cache when the snapshot is already on disk (`snapshot_download(...,
local_files_only=True)`); shards are read in place.

### Phase 2 — one kernel

`maple_run/kernels/ternary_gemv.py`: decode GEMV `y = x @ W` with `x` shape
`[1, K]` or `[B, K]`, `W` packed, scale by `row_alpha`. Correctness vs
dequantized `F.linear` on a small layer. **Fail the review if the kernel
materializes a dense bf16 `W`.**

Triton first. Raw `.cu` only if Triton is unusable on sm_121.

### Phase 3 — model decode

Wire `PackedTernaryLinear` into a Maple forward. Fused expert dispatch (grouped
GEMM / packed expert kernel), not 256 Python launches. Tokenizer from the HF
repo (`transformers` optional extra). Attention on activations: torch SDPA or
FlashInfer; do not add Dao `flash_attn` to the vLLM image.

Measure tok/s and bytes/token vs the ~2.3 GB/token unpacked estimate.

### Phase 4 — FlashHead (optional)

Port `generate_flash_head`. This is the 169→218 tok/s jump on M4, not the reason
the model is ternary.

### Phase 5 — HTTP (optional)

Only after packed decode works. Do not start here.

## Scaffold map

| Path | Role |
|---|---|
| `src/maple_run/cli.py` | `convert` / `generate` subcommands |
| `src/maple_run/pack.py` | CPU ternarize + pack_2bit + 4-bit RTN |
| `src/maple_run/convert.py` | Streaming HF → packed safetensors |
| `src/maple_run/kernels/ternary_gemv.py` | Packed GEMV (stub) |
| `src/maple_run/linear.py` | Packed linear module (stub) |
| `src/maple_run/model.py` | MapleForCausalLM (stub) |
| `src/maple_run/generate.py` | Generate helper (stub) |
| `docs/sources/mlx_lm_ternary.py` | **Authoritative packer** (DeepGrove, MIT) |
| `docs/sources/mlx_lm_deepgrove_README.md` | MLX runtime README |
| `docs/sources/maple-preview-config.json` | HF `config.json` copy |

## Dependencies

`pyproject.toml` has CPU convert deps (`safetensors`, `huggingface-hub`,
`numpy`). CUDA extras (`torch`, `triton`) are optional and **not pinned**:
confirm Spark's existing PyTorch before `uv add`. Tokenizer extra is
`transformers>=4.57.0` (model card `transformers_version` 4.57.1).

## Out of scope unless the user asks

- Editing `~/Code/spark-vllm-docker`
- `pip install flash_attn` anywhere
- Serving via vLLM
- Apple MLX runtime on this Linux box
- Committing weight shards
