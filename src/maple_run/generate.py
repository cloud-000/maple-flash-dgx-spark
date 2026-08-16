"""Text generation CLI helper."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import torch


def generate(model_dir: str, prompt: str, max_tokens: int = 128) -> str:
    """Load a packed checkpoint, greedy-decode, print text and tok/s."""
    from transformers import AutoTokenizer

    from maple_run.model import MapleForCausalLM

    model_dir = str(Path(model_dir).expanduser())
    print(f"Loading packed model from {model_dir}", flush=True)
    model = MapleForCausalLM.from_packed(model_dir)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using a model of type")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    else:
        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
    input_ids = input_ids.to(model.device).long()
    prompt_len = int(input_ids.shape[-1])

    traffic = model.decode_traffic()
    packed_mb = traffic["packed_weight_bytes"] / 1e6
    unpacked_mb = traffic["unpacked_bf16_bytes"] / 1e6
    print(
        f"Decode weight traffic: {packed_mb:.0f} MB/token packed vs "
        f"{unpacked_mb:.0f} MB/token unpacked bf16 "
        f"(handoff ~{traffic['unpacked_handoff_bytes'] / 1e9:.1f} GB/token)",
        flush=True,
    )

    cache = model.make_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model.forward(input_ids, cache=cache, logits_to_keep=1)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        eos = model.eos_token_id
        new_tokens: list[int] = []
        for _ in range(max_tokens):
            next_id = int(logits[0, -1].argmax().item())
            new_tokens.append(next_id)
            if eos is not None and next_id == int(eos):
                break
            nxt = torch.tensor([[next_id]], device=model.device, dtype=torch.long)
            logits = model.forward(nxt, cache=cache, logits_to_keep=1)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    prefill_s = t_prefill - t0
    decode_s = t_end - t_prefill
    n_new = len(new_tokens)
    if prompt_len and prefill_s > 0:
        print(
            f"Prefill: {prompt_len} tok in {prefill_s:.2f}s "
            f"({prompt_len / prefill_s:.1f} tok/s)",
            flush=True,
        )
    if n_new and decode_s > 0:
        print(
            f"Decode:  {n_new} tok in {decode_s:.2f}s "
            f"({n_new / decode_s:.1f} tok/s)",
            flush=True,
        )

    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(text, flush=True)
    return text
