# Sources

Use these instead of searching. Files under `docs/sources/` are snapshots from
the previous session (2026-08-16). Prefer them over live GitHub if they
disagree only in comments; if you must refresh, the URLs below are the
origins.

## Vendored in this repo

| File | What | Origin |
|---|---|---|
| `docs/sources/mlx_lm_ternary.py` | **Authoritative** bf16→2-bit converter (`ternarize`, `_pack_2bit`, FlashHead) | https://raw.githubusercontent.com/deepgrove-ai/mlx-lm-deepgrove/main/mlx_lm/ternary.py |
| `docs/sources/mlx_lm_deepgrove_README.md` | MLX runtime, convert CLI, M4/M5 tok/s table | https://raw.githubusercontent.com/deepgrove-ai/mlx-lm-deepgrove/main/README.md |
| `docs/sources/maple-preview-config.json` | HF `config.json` for `MapleForCausalLM` | https://huggingface.co/deepgrove/maple-preview/blob/main/config.json |

`mlx_lm_ternary.py` is Copyright 2026 DeepGrove AI, MIT (same as the model).
Keep the copyright header if you port code.

## Hugging Face / GitHub (do not need a search)

- Model card (CUDA Transformers + FA/Triton note; 5.31 GB / 218 tok/s claims):
  https://huggingface.co/deepgrove/maple-preview
- HF modeling (attention, unfused MoE, `MapleMLP` clamps):
  https://huggingface.co/deepgrove/maple-preview/blob/main/modeling_maple.py
- HF FlashAttention wrapper imported by modeling:
  https://huggingface.co/deepgrove/maple-preview/blob/main/fa3.py
- Packed MLX checkpoint (if you consume instead of converting):
  https://huggingface.co/deepgrove/maple-2bit-mlx
  (also referred to as `deepgrove/maple-preview-2bit-mlx` in some blobs)
- MLX fork:
  https://github.com/deepgrove-ai/mlx-lm-deepgrove
- Community GGUF / llama.cpp (not this project; mainline llama.cpp cannot load Maple):
  https://huggingface.co/stamsam/maple-preview-gguf
  fork branch `prism`: https://github.com/stamsam/llama.cpp

On this host the HF snapshot, if downloaded, is typically:

```text
~/.cache/huggingface/hub/models--deepgrove--maple-preview/snapshots/ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07/
```

Files there that matter: `modeling_maple.py`, `fa3.py`, `configuration_maple.py`,
`config.json`, `model.safetensors.index.json` (`metadata.total_size` =
40428060672). Weight shards may or may not be present.

## Hardware numbers used in the handoff

- DGX Spark: 128 GB LPDDR5x, **273 GB/s**, GB10 Grace Blackwell
  https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- Base M5 unified bandwidth **153.6 GB/s** (LPDDR5X 9600)
  https://en.wikipedia.org/wiki/Apple_M5
- M4 Mac mini ~120 GB/s (base M4). DeepGrove M4 table: 169 tok/s exact head,
  **218 tok/s `--flash-head`** (in the vendored MLX README).

## Why vLLM rejected Maple

vLLM native registry has no `MapleForCausalLM`. Transformers fallback
(`--model-impl transformers`) expects Hub models to use
`ALL_ATTENTION_FUNCTIONS` and `_supports_attention_backend`; Maple hardcodes
`flash_attention_forward` in `fa3.py`. Docs:

- https://docs.vllm.ai/en/latest/models/supported_models/
- https://huggingface.co/docs/transformers/en/community_integrations/vllm
- https://huggingface.co/blog/native-speed-vllm-transformers-backend

(`spark-vllm-docker` is a separate repo. Do not modify it for this work.)

## Algorithm constants (from vendored ternary.py)

```text
DEFAULT_THRESHOLD_SCALE = 0.7
GROUP_SIZE = 128
HEAD_GROUP_SIZE = 64
HEAD_BITS = 4
TERNARY_EXCLUDE = (".mlp.gate.weight",)
RTN_KEYS = ("lm_head.weight", "model.word_embeddings.weight")
FlashHead: n_clusters=4748, n_probes=512, vocab=151936
```

Packing: 16×2-bit codes per uint32, LSB first, codes = `{−1,0,+1} + 1`.
Ternarize arithmetic in **float32**; only final `alpha` returns to weight dtype.

## Quality benches (DeepGrove Maple-Preview table)

DeepGrove reported dense-head scores on LCBv6 / AIME 2026 / HMMT 2026 / GPQA-D
(78.7% mean). They did not publish a harness. The sibling `bench` project uses:

- MathArena AIME 2026: https://huggingface.co/datasets/MathArena/aime_2026
  config https://github.com/eth-sri/matharena/blob/main/configs/competitions/aime/aime_2026.yaml
- MathArena HMMT Feb 2026: https://huggingface.co/datasets/MathArena/hmmt_feb_2026
- GPQA Diamond CSV (OpenAI simple-evals copy):
  https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv
  prompt: https://github.com/openai/simple-evals/blob/main/gpqa_eval.py
- LiveCodeBench v6 (175 problems): https://huggingface.co/datasets/livecodebench/code_generation_lite
  prompts/tests: https://github.com/LiveCodeBench/LiveCodeBench
- Launch page (JS app; table is also on the HF model card image):
  https://deepgrove.ai/maple-preview
