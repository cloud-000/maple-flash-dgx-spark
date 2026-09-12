# Quantization and fidelity

This document defines the model representations used when validating
`maple-run`, records the September 2026 fidelity investigation, and explains
which comparison answers which question. The labels A, B, and C are local
diagnostic terminology, not names published by DeepGrove.

## Model representations

### A. Original dense bf16 Transformers

Path A loads the original `deepgrove/maple-preview` checkpoint through its
custom Hugging Face Transformers implementation.

- Transformer linear weights are stored as bf16 tensors and executed with
  `nn.Linear`.
- Those transformer tensors are already exact row-scaled ternary values. For
  the inspected Q/K/V/O and expert tensors, applying the published ternarizer
  reconstructs them with zero error.
- Embeddings and the language-model head remain in their original bf16 form.

Path A is useful for measuring the effect of packing and 4-bit boundary
quantization. It is not the closest implementation oracle for packed kernels,
because its embedding and output-head weights differ from the packed model.

### B. Dense dequantized packed reference

Path B expresses the packed checkpoint as ordinary dense tensors and executes
the model with straightforward PyTorch operations:

- ternary transformer codes become `{-alpha, 0, +alpha}` dense weights;
- 4-bit embedding and LM-head codes are dequantized with their group scales
  and biases;
- routing remains floating-point;
- standard dense operations replace custom packed kernels and fusions.

This is the primary correctness reference for path C. B and C consume the same
quantized model parameters, so their comparison isolates packed-kernel,
fusion, accumulation, and routing behavior.

Path B is a diagnostic construction, not a separately shipped runtime or a
claim that the original bf16 model is invalid.

### C. Packed `maple-run`

Path C is the production runtime in this repository. It keeps ternary and
4-bit weights packed and executes custom CUDA/Triton kernels, including fused
normalization, attention, MoE, routing, and sampling paths.

## Quantization boundaries

| Component | A: original | B and C: packed-model representation |
| --- | --- | --- |
| Attention Q/K/V/O | bf16 tensors containing exact ternary values | 2-bit ternary codes plus one `row_alpha` per output row |
| Expert gate/up/down | bf16 tensors containing exact ternary values | 2-bit ternary codes plus `row_alpha` |
| MoE router | floating-point | floating-point; not ternarized |
| Token embedding | bf16 | 4-bit affine RTN, group size 64 |
| LM head | bf16 | 4-bit affine RTN, group size 64 |
| Norm weights | floating-point | floating-point |

The authoritative conversion algorithm is vendored in
`docs/sources/mlx_lm_ternary.py`. Ternarization uses a threshold of
`0.7 * mean(abs(weight))`, performs reductions in float32, and stores 16
2-bit codes per uint32. The embedding and LM head use 4-bit affine
round-to-nearest quantization with group size 64.

## What each comparison means

| Comparison | Main question |
| --- | --- |
| A versus B | What changes when the original bf16 embedding/head are replaced by the packed model's 4-bit representation? |
| B versus C | Do the optimized packed runtime and kernels reproduce the same quantized model as a simple dense implementation? |
| A versus C | What is the total observed difference between original bf16 Transformers and production `maple-run`? |

### Dense-head capability evaluation

DeepGrove model card states that its reported capability comparison uses a
dense output head. For a like-for-like quality evaluation without loading the
complete dense Transformers model, `maple-run` provides an opt-in hybrid:

- packed ternary transformer projections and production kernels;
- packed 4-bit token embedding;
- original checkpoint dense bf16 `lm_head.weight`;
- original checkpoint tokenizer as the canonical token-ID source.

Run it with
`maple-run serve --dense-head --max-len 65536 --max-tokens 64000`.
The normal exact-head path remains the full-vocabulary 4-bit packed head and
is unchanged. Dense-head mode is for quality attribution, not speed
benchmarking, and cannot be combined with FlashHead. The server context must
cover the benchmark generation cap: a 32,768-token server silently turns a
64,000-token evaluation request into a length-truncated response.

Top-1 agreement need not be 100% even when two paths implement the same
quantized weights. Dense and packed kernels accumulate floating-point sums in
different orders and round at different points. Attention softmax can amplify
small Q/K changes, MoE top-k routing is discontinuous near expert boundaries,
and nearly tied output logits can swap the winning token while the overall
distributions remain close. Forward KL is therefore more informative than
top-1 agreement alone.

## September 2026 reference-loader incident

The first `maple-run fidelity` smoke report appeared to show a severe packed
runtime regression:

| Metric | Initial report | After reference fix |
| --- | ---: | ---: |
| Compared tokens | 63 | 63 |
| Mean forward KL, A versus C | 4.04007 | 0.0623718 |
| p95 forward KL | 9.59830 | 0.174415 |
| Top-1 agreement | 20.63% | 85.71% |

The packed runtime was not the cause. The bf16 reference model's
non-persistent RoPE `inv_freq` buffer was corrupted or left uninitialized by
the installed Transformers low-memory/meta-device loading path. The forward
still completed, but sliding-attention layers used invalid rotary frequencies.
The first meaningful divergence was layer 0 attention; global layers that skip
RoPE matched closely.

`load_original_reference` now deterministically rebuilds Maple's RoPE buffer
after `from_pretrained`. The SDPA adapter also validates the live Q/K/V layout.
Maple passes head-major tensors:

- Q: `[batch, 16, sequence, 128]`
- K/V: `[batch, 4, sequence, 128]`

At sequence length 64 these are Q `[1, 16, 64, 128]` and K/V
`[1, 4, 64, 128]`. A sequence-major assumption can remain shape-compatible in
some code, so a completed forward is not sufficient evidence of correctness.

## Three-path diagnostic results

The corrected 64-token smoke prompt produced:

| Comparison | Mean forward KL | Top-1 agreement |
| --- | ---: | ---: |
| A versus B | 0.06035 | 85.71% |
| B versus C | 0.01344 | 93.65% |
| A versus C | 0.06237 | 85.71% |

Additional isolation on the same prompt:

- applying the dense bf16 and packed 4-bit heads to A's hidden state gave
  mean KL 0.02872;
- the embedding/body effect measured with the dense head gave mean KL 0.03602;
- B versus C final hidden states had relative RMS error 0.10042 and cosine
  similarity 0.99496;
- same-input packed checks had about 0.35% QKV relative RMS error, 0.51%
  attention-core error, and 0.35% output-projection error;
- same-input router top-8 overlap was 100%, and same-input MLP error was about
  0.1-0.4% in the inspected layers.

Interpretation: the remaining A-versus-C difference is primarily the intended
4-bit embedding/LM-head boundary and its downstream amplification. B versus C
is much closer and did not reveal a material packed model-forward or kernel
mismatch. Small accumulated numerical differences can still alter later MoE
routes.

These are smoke-test measurements from one prompt, not quality-evaluation
scores or universal acceptance thresholds. Quality evaluations live in the
sibling `bench` project.

## Reproducing the smoke check

Use the original checkpoint tokenizer as the canonical token-ID source for
both paths:

```bash
HF_MODULES_CACHE="$PWD/tmp/hf-modules" \
uv run maple-run fidelity \
  --model checkpoints/maple-2bit \
  --prompts tmp/prompts.jsonl \
  --limit 1 \
  --max-seq-len 64 \
  --output runs/fidelity-smoke.json
```

The current `maple-run fidelity` command compares A directly with C. It does
not construct B or emit the three-path decomposition on every run; B was built
as a focused diagnostic reference during the investigation. If B becomes a
maintained command mode, keep its results separate from the default A-versus-C
report and label the representation used by every metric.

When diagnosing a regression, check in this order:

1. Confirm identical token IDs and the head-major Q/K/V layout.
2. Validate reference RoPE values before treating Transformers as ground truth.
3. Compare embedding outputs.
4. Compare attention output and residual after each layer.
5. Compare router logits and top-8 expert overlap.
6. Compare MLP output and the final normalized hidden state.
7. Apply both output heads to the same hidden state.
8. Compare packed kernels against dequantized packed weights before comparing
   them against a differently quantized model representation.
