# Handoff: implement maple-run

This document is for the **next session**. The previous session diagnosed why
`vllm serve deepgrove/maple-preview` failed on Spark, why the Hugging Face CUDA
Transformers path is slow, and what a packed-kernel runtime has to do.

**Phase 1–5 are done.** Exact-head sampled decode is ~368 tok/s. FlashHead
(`--flash-head`) is ~487 tok/s on the 256-tok bench / ~489 tok/s on 700-tok
haiku — the M4-class 169→218 jump, on this host. Default generate is still
exact-head; do not regress it. `maple-run serve` is an OpenAI-compatible HTTP
endpoint. Do not re-convert unless the packed checkpoint is missing; FlashHead
clusters are already attached (`--flash-head-only`).
Do not start with a web search for Maple, MLX, or Spark bandwidth; read
`docs/SOURCES.md` and `AGENTS.md` first — and read "How to measure on this host"
below before trusting any kernel benchmark.

## Status (2026-08-17)

| Phase | State |
|---|---|
| 1 Pack / convert | **Done.** NumPy `ternarize` / `pack_2bit` / 4-bit RTN; streaming converter; tests in `tests/test_pack.py` and `tests/test_convert.py`. |
| 2 Packed GEMV kernel | **Done.** Triton packed GEMV in `maple_run/kernels/ternary_gemv.py`; never unpacks to dense bf16. Tests in `tests/test_ternary_gemv.py`. |
| 3 Model decode | **Done, fused, and tuned twice.** Packed forward + fused RMS/QKV/SwiGLU/decode attn + fused router + fused sampler. Exact-head sampled **~368 tok/s** on the 256-tok bench, **~364 tok/s** on 700-tok haiku. CUDA graphs used when replay matches eager sampled ids. **Default generate path; do not regress.** |
| 4 FlashHead | **Done.** `maple_run/flash_head.py` + indexed 4-bit GEMV. 4748 clusters / 512 probes attached on this host. Sampled **~487 tok/s** (256-tok bench) / **~489 tok/s** (700-tok haiku). Prefill stays on the exact head. CLI: `--flash-head`. Tests in `tests/test_flash_head.py`. |
| 5 HTTP | **Done.** `maple-run serve` OpenAI `/v1/chat/completions` + `/v1/completions`. |

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
- `router_topk`: router GEMV + softmax + top-8 + renormalize in two launches
  (`kernels/router.py`). The torch chain it replaced was eight launches per
  layer over a 256-wide vector and cost 633 us/token — more than the router
  weight traffic itself
- 4-bit RTN embedding gather + `rtn4_gemv` exact `lm_head`; optional FlashHead
  (`--flash-head`) scores 4748 centroids then exact logits for 512 clusters
- `fused_sample`: top-k + temperature + nucleus + CDF inverse in two launches
  (`kernels/sampler.py`), replacing ~10 eager launches outside the graph
- Chat-template generate. CLI defaults are sampled: `--temperature 1.0 --top-p 0.95 --top-k 20`. Greedy is `--temperature 0`.

Packed codes stay in the checkpoint's `[N, nwords]` order. An earlier transpose
to `[nwords, N]` put consecutive output rows together but scattered each CTA's
K tile across `BLOCK_K_WORDS` addresses `N*4` bytes apart; read cold the
untransposed layout is faster everywhere (QKV 141 vs 125 GB/s, O 191 vs 165,
lm_head 215 vs 194). `packed_kn=True` still works in the kernels; nothing uses
it. That layout choice also flipped the right dtype for the RTN scales: with a
row's 32 groups contiguous, bf16 scales/biases win (820 vs 903 us on the head),
where under the transposed layout they had lost.

Speed command (do not use greedy France for tok/s):

```bash
uv run maple-run generate --model checkpoints/maple-2bit \
  --prompt "Write a haiku on groves" --max-tokens 700
```

FlashHead (clusters already attached on this host):

```bash
uv run maple-run convert checkpoints/maple-2bit --flash-head-only   # already done
uv run maple-run generate --model checkpoints/maple-2bit --flash-head \
  --prompt "Write a haiku on groves" --max-tokens 700
```

Or `uv run pytest tests/test_bench.py --bench -s`. Greedy `--temperature 0` / France
`--max-tokens 128` is only a correctness check (should EOS at Paris). Argmax skips
softmax/top-k/nucleus/multinomial and often stops at ~38 tok, which inflates tok/s.

Measured on this GB10 host (2026-08-17), exact 4-bit head, no FlashHead:

| | |
|---|---|
| Packed weight traffic | **417 MB/token** (was 462; router + RTN scales are bf16 now) |
| Unpacked bf16 traffic (same active tensors) | **2359 MB/token** (~2.3 GB handoff estimate) |
| **Achievable** read bandwidth, measured | **~250 GB/s** — see "273 GB/s is not reachable" below |
| Bandwidth ceiling at that rate | `250 / 0.417 ≈ 600` tok/s if fully fused |
| M4-scaled target, spec bandwidth | `169 × 273 / 120 ≈ 386` tok/s |
| M4-scaled target, **achievable** bandwidth | `169 × 250 / 120 ≈ 352` tok/s |
| Decode **sampled** T=1.0 top-p=0.95 top-k=20 (haiku, 256 tok bench) | **~368 tok/s** (was 254) |
| Decode **sampled** T=1.0 top-p=0.95 top-k=20 (haiku, 700 tok) | **~364 tok/s** (was 234) |
| Decode greedy France 42 tok EOS (not a speed bench) | ~390 tok/s; still prints Paris |

FlashHead (`--flash-head`), same host, clusters already attached:

| | |
|---|---|
| Packed weight traffic | **267 MB/token** (centroids + 512×32 probed rows, not the 152 k head) |
| Decode **sampled** (haiku, 256 tok bench) | **~487 tok/s** |
| Decode **sampled** (haiku, 700 tok) | **~489 tok/s** |
| Decode greedy France (not a speed bench) | still prints Paris; force tokens EOS / `</think>` / `<|im_end|>` |

Default generate is exact-head. Do not regress the ~368 tok/s path. FlashHead
is opt-in and approximate: greedy is exact when the true argmax is in a probed
cluster. Prefill always uses the exact `lm_head`.

Effective traffic at 368 tok/s is `368 × 0.417 ≈ 153` GB/s — **61% of the 250 GB/s
this host actually delivers**, against M4's ~65% of its own peak. Exact-head
efficiency is now roughly M4-class. FlashHead is the 169→218 jump (29% on M4,
~32% here).

### 273 GB/s is not reachable — calibrate against 250

The 273 GB/s in the DGX Spark spec is not what a kernel can get. A trivial
Triton streaming reduction over 175 MB tops out at **250 GB/s**, and torch's own
`sum`/`max` over 200-800 MB get 236-241 GB/s. Every "% of peak" below is against
250, not 273. This also moves the M4-scaled target: scaled by bandwidth the
machine actually has, it is ~352 tok/s, which decode is already past. 386 tok/s
is still a fair stretch goal (it needs 161 GB/s effective, 64% of achievable)
but it is not what "Spark at M4 efficiency" works out to.

### Where the 2.7 ms/token goes

By ablation (replace one kernel with a memoized no-op and re-time full decode --
the profiler's per-kernel self time over-attributes by ~35% here):

| Kernel | us/token | of which | % of achievable peak |
|---|---|---|---|
| `rtn4_gemv` lm_head | ~720-825 | 175 MB | ~85% |
| `ternary_gemv` QKV + O | ~350-480 | 63 MB | ~55% |
| `ternary_expert_swiglu` | ~365-460 | 101 MB | ~85% |
| `ternary_expert_gemv` down | ~290 | 50 MB | ~70% |
| `router_topk` (both launches) | ~120-200 | 25 MB | — |
| fused sampler (outside the graph) | ~37 | — | — |
| `add_rms_norm` / `moe_reduce_add` | ~11 / ~19 | — | — |

Ranges are run-to-run spread, not uncertainty in the method; see the
measurement notes below.

CUDA-graph capture of the full decode forward is enabled when an untimed
8-token replay matches eager sampled token ids at the same RNG state (KV
cache restored after warmup; default CUDA RNG restored when `--seed` is
unset). Capture sits between prefill and the decode timer so it does not
inflate tok/s. Previous capture without restore wrote warmup tokens into KV
and looked like wrong tokens / early EOS.

### How to measure on this host (read before tuning anything)

This box misled three separate rounds of tuning. All four traps are real:

1. **A microbench in a Python loop measures Triton launch overhead, not the
   kernel.** ~10 us/launch of CPU time swamps anything under ~20 us. Capture N
   calls in a CUDA graph and time the replay.
2. **A single-buffer microbench measures L2.** A 1-4 MB weight stays resident
   and reports 2-3x the bandwidth the same kernel gets in the model, where every
   launch reads a different layer. Rotate over enough distinct buffers to pass
   ~75 MB; past that the number stops moving.
3. **Back-to-back launches in a sweep overlap; the model's do not** — but this
   turned out to cost only ~0.5 us/launch, so it is *not* where the gap is. The
   gap was L2 (trap 2) plus program count.
4. **Run-to-run spread is ~5%, and a long sweep drifts.** Timing configs one
   after another reliably favours whichever ran first. Round-robin the
   candidates and take each one's median. Two "wins" of ~100 us/token
   evaporated under this treatment, and one was a regression.

The honest tools, in order of trust: full sampled decode timed several times
(`tests/test_bench.py --bench`), ablation against full decode, round-robin A/B
of launch configs, L2-cold microbench, and last a plain microbench.

### Open issues (next session: more body-kernel speed)

1. **QKV/O is the weakest kernel**, ~55% of achievable against ~85% for the
   lm_head and the expert SwiGLU. It is the only body GEMV with a fused input
   RMSNorm. That norm no longer costs a pre-pass (rstd is a scalar and factors
   out of the dot product, so sumsq accumulates inside the main loop), but the
   O projection with no norm at all still runs materially better, so there may
   be more here. Do not unpack ternary codes to get it.
2. **FlashHead is done** (phase 4). Exact-head `lm_head` remains ~30% of the
   token when `--flash-head` is off. Indexed probe GEMV must keep the same
   BLOCK_N=16 tiles as `rtn4_gemv`; a one-CTA-per-cluster launch was a wash.
3. **The fused sampler is the only thing left outside the CUDA graph** (~37
   us/token, three eager launches including the `torch.rand` for the uniform).
   A self-feeding graph that samples and writes its own next `token_ids` would
   absorb it; PyTorch does support captured RNG, but the graph-vs-eager id
   check would need rework.
4. `ternary_expert_down_sum` (down GEMV + top-k reduce fused) is still unused:
   `moe_reduce_add` only costs ~19 us/token in situ, so the fusion is worth
   less than it looks.
5. **Greedy think-trace loops.** Greedy haiku `--max-tokens 700` still hits the
   cap recounting syllables. France T=0 remains the greedy correctness check
   (must print Paris, with or without `--flash-head`). **Measure tok/s with
   sampled decode**, never greedy.
6. CUDA graphs are on when 8-token replay matches eager sampled ids. Do not
   skip the match check. Capture warmup must restore KV `seqlen` and the
   default CUDA RNG (CLI `--seed` unset uses that generator). FlashHead is
   inside the captured forward (seq_len==1 only).

### Phase 4 — FlashHead — **done**

Port of `generate_flash_head` / `FlashHead`. 4748 equal-size clusters, probe
top 512, always score force tokens. Attach with
`maple-run convert checkpoints/maple-2bit --flash-head-only` (already run).
Enable at decode with `--flash-head`. Default generate stays exact-head.

### Phase 5 — HTTP — **done**

OpenAI-compatible stdlib server. Same sampling flags as `generate` are CLI
defaults; the request body may override them (`temperature`, `top_p`, `top_k`,
`max_tokens` / `max_completion_tokens`, `seed`, `stream`).

```bash
uv run maple-run serve --model checkpoints/maple-2bit --host 127.0.0.1 --port 8000
uv run maple-run serve --model checkpoints/maple-2bit --flash-head \
  --temperature 1.0 --top-p 0.95 --top-k 20 --max-tokens 128
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`.
One CUDA generate at a time. Prefill stays on the exact head. Do not regress
the CLI generate path.

## Scaffold map

| Path | Role |
|---|---|
| `src/maple_run/cli.py` | `convert` / `generate` / `serve` subcommands |
| `src/maple_run/server.py` | OpenAI-compatible HTTP (`/v1/chat/completions`) |
| `src/maple_run/pack.py` | CPU ternarize + pack_2bit + 4-bit RTN |
| `src/maple_run/convert.py` | Streaming HF → packed safetensors |
| `src/maple_run/kernels/ternary_gemv.py` | Packed GEMV (Triton); decode tiles pinned |
| `src/maple_run/kernels/ternary_expert.py` | Fused indexed expert GEMV + SwiGLU |
| `src/maple_run/kernels/fused_norm.py` | RMSNorm, add+RMSNorm, MoE reduce |
| `src/maple_run/kernels/router.py` | Fused router GEMV + softmax + top-k + renorm |
| `src/maple_run/kernels/sampler.py` | Fused top-k / temperature / nucleus sampling |
| `src/maple_run/kernels/decode_attn.py` | q_len=1 QK-RMS/RoPE + GQA (loops seqlen/SWA) |
| `src/maple_run/kernels/rtn4.py` | 4-bit RTN embedding + lm_head GEMV + indexed FlashHead GEMV |
| `src/maple_run/flash_head.py` | Cluster attach (`generate_flash_head`) + decode FlashHead |
| `src/maple_run/linear.py` | PackedTernaryLinear / Experts / RTN4 |
| `src/maple_run/model.py` | MapleForCausalLM packed forward |
| `src/maple_run/generate.py` | Tokenizer + greedy/sampled decode + tok/s |
| `tests/test_bench.py` | Sampled decode speed (`pytest --bench`); FlashHead when attached |
| `tests/test_flash_head.py` | Clustering, exact-when-all-probes, prefill stays exact |
| `tests/test_server.py` | OpenAI request parsing, CLI `serve` flags, fake-engine HTTP |
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
