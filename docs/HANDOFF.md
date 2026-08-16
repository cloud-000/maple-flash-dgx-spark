# Handoff: implement maple-run

This document is for the **next session**. The previous session diagnosed why
`vllm serve deepgrove/maple-preview` failed on Spark, why the Hugging Face CUDA
Transformers path is slow, and what a packed-kernel runtime has to do.

**Phase 1–3 are done, including a body-fusion pass.** Next session is **more
exact-head decode speed**, not FlashHead yet. Do not re-convert unless the
packed checkpoint is missing. Do not start with a web search for Maple, MLX, or
Spark bandwidth; read `docs/SOURCES.md` and `AGENTS.md` first.

## Status (2026-08-16)

| Phase | State |
|---|---|
| 1 Pack / convert | **Done.** NumPy `ternarize` / `pack_2bit` / 4-bit RTN; streaming converter; tests in `tests/test_pack.py` and `tests/test_convert.py`. |
| 2 Packed GEMV kernel | **Done.** Triton packed GEMV in `maple_run/kernels/ternary_gemv.py`; never unpacks to dense bf16. Tests in `tests/test_ternary_gemv.py`. |
| 3 Model decode | **Done, then fused further.** Packed forward + fused RMS/QKV/SwiGLU/decode attn + pinned decode GEMV tiles. Exact-head sampled ~254 tok/s on the 256-tok bench, ~234 tok/s on 700-tok haiku. CUDA graphs used when replay matches eager sampled ids. Still short of bandwidth scaling. |
| 4 FlashHead | **Do not start** until the user asks. Exact-head body/head kernels first. |
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

### Phase 2 — one kernel — **done**

`maple_run/kernels/ternary_gemv.py`: decode GEMV `y = x @ W` with `x` shape
`[1, K]` or `[B, K]`, `W` packed uint32, scale by `row_alpha`. Triton on this
host (GB10 sm_121, torch 2.13.0+cu130). Correctness vs dequantized `F.linear`
in fp32; the kernel does not materialize a dense bf16 `W`.
`PackedTernaryLinear.forward` calls it. 3-D stacked experts use
`ternary_expert_gemv` (phase 3).

### Phase 3 — model decode — **done**

`maple_run/model.py` loads `checkpoints/maple-2bit` onto CUDA, keeps ternary
weights as uint32 + `row_alpha`, and runs:

- `PackedTernaryLinear` for fused QKV / O
- `ternary_expert_gemv` over the selected top-8 experts (one launch, 3-D stacked
  `switch_mlp`; no 256-expert Python loop, no `tokens_per_expert.cpu()`)
- Decode: fused input-RMS into QKV, fused add+RMSNorm, fused expert SwiGLU,
  preallocated KV, Triton GQA for `q_len=1`; prefill still torch SDPA
- 4-bit RTN embedding gather + `rtn4_gemv` exact `lm_head` (FlashHead is phase 4)
- Chat-template generate. CLI defaults are sampled: `--temperature 1.0 --top-p 0.95 --top-k 20`. Greedy is `--temperature 0`.

Speed command (do not use greedy France for tok/s):

```bash
uv run maple-run generate --model checkpoints/maple-2bit \
  --prompt "Write a haiku on groves" --max-tokens 700
```

Or `uv run pytest tests/test_bench.py --bench -s`. Greedy `--temperature 0` / France
`--max-tokens 128` is only a correctness check (should EOS at Paris). Argmax skips
softmax/top-k/nucleus/multinomial and often stops at ~38 tok, which inflates tok/s.

Measured on this GB10 host (2026-08-16), exact 4-bit head, no FlashHead:

| | |
|---|---|
| Packed weight traffic | **462 MB/token** |
| Unpacked bf16 traffic (same active tensors) | **2384 MB/token** (~2.3 GB handoff estimate) |
| Bandwidth ceiling | `273 / 0.462 ≈ 590` tok/s if fully fused |
| M4-scaled target (no FlashHead) | `169 × 273 / 120 ≈ 386` tok/s |
| Decode **sampled** T=1.0 top-p=0.95 top-k=20 (haiku, 256 tok bench) | **~254 tok/s** |
| Decode **sampled** T=1.0 top-p=0.95 top-k=20 (haiku, 700 tok) | **~234 tok/s** |
| Decode greedy France 38 tok EOS (not a speed bench) | ~278 tok/s; still prints Paris |
| Isolated `rtn4_gemv` lm_head | ~0.97 ms / **~201 GB/s** (was ~1.36 ms / 143 GB/s) |
| Decode QKV GEMV | ~153 GB/s |
| Body SwiGLU | ~200 GB/s of 273 |

CUDA-graph capture of the full decode forward is enabled when an untimed
8-token replay matches eager sampled token ids at the same RNG state (KV
cache restored after warmup; default CUDA RNG restored when `--seed` is
unset). Capture sits between prefill and the decode timer so it does not
inflate tok/s. Previous capture without restore wrote warmup tokens into KV
and looked like wrong tokens / early EOS.

Effective traffic at 254 tok/s is `254 × 0.462 ≈ 117` GB/s — 43% of Spark
peak vs ~65% of M4 peak. Body GEMV tiles and graphs closed much of the launch
gap; the exact 4-bit head is still ~0.97 ms/token.

### Open issues (next session: more exact-head speed, not FlashHead)

1. **Bandwidth not scaling to 386 tok/s.** Sampled decode is ~254 tok/s
   (256-tok bench) / ~234 tok/s (700-tok haiku). Do not unpack ternary codes.
   Raise kernel % of 273 GB/s: exact `rtn4_gemv` is ~201 GB/s (fat K-tile with
   per-group scale; a single-scale fat tile hits ~270 GB/s but is wrong).
   QKV ~153 GB/s; SwiGLU ~200 GB/s. `ternary_expert_down_sum` is currently
   slower than split down-GEMV + `moe_reduce`. **Measure tok/s with sampled
   decode** (`--temperature 1.0 --top-p 0.95 --top-k 20`), not greedy France.
2. **Greedy think-trace loops.** Sampling is in `generate.py`. Greedy haiku
   `--max-tokens 700` still hits the cap recounting syllables. France T=0
   remains the greedy correctness check (must print Paris).
3. CUDA graphs are on when 8-token replay matches eager sampled ids. Do not
   skip the match check. Capture warmup must restore KV `seqlen` and the
   default CUDA RNG (CLI `--seed` unset uses that generator).

### Phase 4 — FlashHead (optional)

Port `generate_flash_head`. This is the 169→218 tok/s jump on M4, not the reason
the model is ternary. **Do not start until asked.**

### Phase 5 — HTTP (optional)

Only after packed decode works. Do not start here.

## Scaffold map

| Path | Role |
|---|---|
| `src/maple_run/cli.py` | `convert` / `generate` subcommands |
| `src/maple_run/pack.py` | CPU ternarize + pack_2bit + 4-bit RTN |
| `src/maple_run/convert.py` | Streaming HF → packed safetensors |
| `src/maple_run/kernels/ternary_gemv.py` | Packed GEMV (Triton); decode tiles pinned |
| `src/maple_run/kernels/ternary_expert.py` | Fused indexed expert GEMV + SwiGLU |
| `src/maple_run/kernels/fused_norm.py` | RMSNorm, add+RMSNorm, MoE reduce |
| `src/maple_run/kernels/decode_attn.py` | q_len=1 QK-RMS/RoPE + GQA (loops seqlen/SWA) |
| `src/maple_run/kernels/rtn4.py` | 4-bit RTN embedding + lm_head GEMV |
| `src/maple_run/linear.py` | PackedTernaryLinear / Experts / RTN4 |
| `src/maple_run/model.py` | MapleForCausalLM packed forward |
| `src/maple_run/generate.py` | Tokenizer + greedy/sampled decode + tok/s |
| `tests/test_bench.py` | Sampled decode speed (`pytest --bench`) |
| `docs/sources/mlx_lm_ternary.py` | **Authoritative packer** (DeepGrove, MIT) |
| `docs/sources/mlx_lm_deepgrove_README.md` | MLX runtime README |
| `docs/sources/maple-preview-config.json` | HF `config.json` copy |

## Dependencies

`pyproject.toml` has CPU convert deps (`safetensors`, `huggingface-hub`,
`numpy`). CUDA extra is pinned to this host's stack: `torch==2.13.0` /
`triton==3.7.1` (PyPI manylinux aarch64 CUDA 13.0, same as `2.13.0+cu130`).
Tokenizer extra is `transformers>=4.57.0` (model card `transformers_version`
4.57.1).

## Out of scope unless the user asks

- Editing `~/Code/spark-vllm-docker`
- `pip install flash_attn` anywhere
- Serving via vLLM
- Apple MLX runtime on this Linux box
- Committing weight shards
