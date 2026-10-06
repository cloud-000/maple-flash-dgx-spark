"""Packed ternary decode: single-sequence generation and CLI entrypoint."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from maple_run.batch import (
    BatchedResult,
    _first_batched_id,
    batched_generate,
)
from maple_run.chat import (
    CompletionDecoder,
    _as_token_id_set,
    _chat_message,
    _message_role,
    _stub_tool,
    _to_input_ids,
    _tool_function_name,
    decode_holding_incomplete,
    describe_chat_prompt,
    encode_messages,
    encode_prompt,
    encode_user_prompt,
    infer_tools_from_messages,
    merge_tools,
    message_text,
    normalize_chat_messages,
    parse_tool_call_json,
    prompt_in_think,
    resolve_chat_tools,
    skip_decode_ids,
    split_content_and_tools,
    stop_token_ids,
    think_token_ids,
)
from maple_run.graphs import (
    _DECODE_BUCKETS,
    _GRAPH_CHECK_TOKENS,
    _GRAPH_WARMUP,
    BucketedDecodeGraphs,
    _DecodeGraph,
    _capture_decode_graph,
    _decode_n,
    _next_token,
    _try_batched_decode_graphs,
    _try_decode_graph,
    decode_buckets,
)
from maple_run.kernels.sampler import sampler_workspace
from maple_run.sample import (
    DEFAULT_MAX_CONTEXT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    _cuda_rng,
    _nucleus,
    _rng_restore,
    _rng_snapshot,
    can_fuse_sampler,
    is_greedy,
    resolve_max_tokens,
    sample_next,
)

__all__ = [
    "DEFAULT_MAX_CONTEXT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_P",
    "GenerateResult",
    "BatchedResult",
    "CompletionDecoder",
    "BucketedDecodeGraphs",
    "batched_generate",
    "can_fuse_sampler",
    "decode_buckets",
    "decode_holding_incomplete",
    "describe_chat_prompt",
    "encode_messages",
    "encode_prompt",
    "encode_user_prompt",
    "generate",
    "infer_tools_from_messages",
    "is_greedy",
    "load_packed",
    "log_model_traffic",
    "merge_tools",
    "message_text",
    "normalize_chat_messages",
    "parse_tool_call_json",
    "prompt_in_think",
    "resolve_chat_tools",
    "resolve_max_tokens",
    "run_generate",
    "sample_next",
    "skip_decode_ids",
    "split_content_and_tools",
    "stop_token_ids",
    "think_token_ids",
    "warmup_model",
]


def _log_print(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class GenerateResult:
    text: str
    prompt_len: int
    n_new: int
    prefill_s: float
    decode_s: float
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def decode_tok_s(self) -> float:
        return self.n_new / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def prefill_tok_s(self) -> float:
        return self.prompt_len / self.prefill_s if self.prefill_s > 0 else 0.0


def load_packed(
    model_dir: str,
    flash_head: bool = False,
    dense_head_source: str | None = None,
):
    """Load a packed Maple checkpoint and its tokenizer."""
    from transformers import AutoTokenizer

    from maple_run.model import MapleForCausalLM

    model_dir = str(Path(model_dir).expanduser())
    model = MapleForCausalLM.from_packed(
        model_dir,
        use_flash_head=flash_head,
        dense_head_source=dense_head_source,
    )
    tokenizer_source = dense_head_source or model_dir
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using a model of type")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    return model, tokenizer


def warmup_model(model, flash_head: bool = False) -> None:
    """JIT the decode (and exact-head prefill) kernels."""
    with torch.inference_mode():
        model.forward(
            torch.zeros(1, 1, dtype=torch.long, device=model.device),
            cache=model.make_cache(max_len=8),
            logits_to_keep=1,
        )
        if flash_head:
            model.forward(
                torch.zeros(1, 2, dtype=torch.long, device=model.device),
                cache=model.make_cache(max_len=8),
                logits_to_keep=1,
            )
        torch.cuda.synchronize()


def log_model_traffic(model, flash_head: bool = False, log: Callable[[str], None] = _log_print) -> None:
    traffic = model.decode_traffic()
    packed_mb = traffic["packed_weight_bytes"] / 1e6
    unpacked_mb = traffic["unpacked_bf16_bytes"] / 1e6
    log(
        f"Decode weight traffic: {packed_mb:.0f} MB/token packed vs "
        f"{unpacked_mb:.0f} MB/token unpacked bf16 "
        f"(handoff ~{traffic['unpacked_handoff_bytes'] / 1e9:.1f} GB/token)"
    )
    if flash_head:
        meta = model.config.get("flash_head") or {}
        log(
            f"FlashHead: {meta.get('n_clusters')} clusters, "
            f"{meta.get('n_probes')} probes, force={meta.get('force_tokens')}"
        )


def run_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    max_tokens: int = 128,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = None,
    log: Callable[[str], None] | None = _log_print,
    on_token: Callable[[int], None] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    on_tool_calls: Callable[[list[dict]], None] | None = None,
    use_graph: bool | None = None,
    stop_ids: set[int] | None = None,
    early_stop: bool = True,
) -> GenerateResult:
    """Decode from ``input_ids`` on an already-loaded packed model."""
    def emit(msg: str) -> None:
        if log is not None:
            log(msg)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(model.device).long()
    prompt_len = int(input_ids.shape[-1])
    max_context = int(
        (getattr(model, "config", None) or {}).get(
            "max_position_embeddings", DEFAULT_MAX_CONTEXT
        )
    )
    max_tokens = resolve_max_tokens(max_tokens, prompt_len, max_context)
    greedy = is_greedy(temperature, top_k)
    generator = None
    if seed is not None and not greedy:
        generator = torch.Generator(device=model.device)
        generator.manual_seed(int(seed))
    sampler_ws = None
    if can_fuse_sampler(temperature, top_p, top_k):
        sampler_ws = sampler_workspace(model.vocab_size, top_k, model.device)
    if not greedy:
        emit(
            f"Sampling: temperature={temperature} top_p={top_p} "
            f"top_k={top_k} seed={seed} "
            f"({'fused' if sampler_ws is not None else 'torch'} sampler)"
        )

    if stop_ids is None:
        stop_ids = stop_token_ids(model, tokenizer)
    decoder = CompletionDecoder(
        tokenizer,
        in_think=prompt_in_think(tokenizer, input_ids),
        skip_ids=skip_decode_ids(tokenizer, extra=stop_ids),
    )
    cache = model.make_cache(max_len=prompt_len + max(max_tokens, 1))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_prefill = t0
    t_decode = t0
    t_end = t0
    new_tokens: list[int] = []
    hit_eos = False
    with torch.inference_mode():
        logits = model.forward(input_ids, cache=cache, logits_to_keep=1)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        graph = None
        allow_graph = use_graph if use_graph is not None else True
        if max_tokens > 1 and allow_graph:
            graph = _try_decode_graph(
                model,
                cache,
                logits,
                greedy=greedy,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
                sampler_ws=sampler_ws,
                log=emit,
            )
            torch.cuda.synchronize()
        t_decode = time.perf_counter()
        pinned = torch.empty((), dtype=torch.long, pin_memory=True)
        eos_ready = torch.cuda.Event()
        sample_in_graph = graph is not None and graph.next_id is not None
        next_t: torch.Tensor | None = None
        for i in range(max_tokens):
            if next_t is None:
                next_t = _next_token(
                    logits,
                    greedy=greedy,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=generator,
                    sampler_ws=sampler_ws,
                )
            pinned.copy_(next_t, non_blocking=True)
            eos_ready.record()
            if i + 1 < max_tokens:
                if graph is not None:
                    graph.token_ids.copy_(next_t.view(1, 1))
                    if sample_in_graph:
                        assert graph.uniform is not None
                        graph.uniform.copy_(
                            torch.rand((), device=model.device, generator=generator)
                        )
                    graph.graph.replay()
                    logits = graph.logits
                    if sample_in_graph:
                        next_t = graph.next_id
                    else:
                        next_t = None
                else:
                    logits = model.forward(
                        next_t.view(1, 1), cache=cache, logits_to_keep=1
                    )
                    next_t = None
            eos_ready.synchronize()
            next_id = int(pinned.item())
            new_tokens.append(next_id)
            if on_token is not None:
                on_token(next_id)
            r_delta, c_delta = decoder.push(next_id)
            if r_delta and on_reasoning is not None:
                on_reasoning(r_delta)
            if c_delta and on_text is not None:
                on_text(c_delta)
            new_calls = decoder.take_new_tool_calls()
            if new_calls and on_tool_calls is not None:
                on_tool_calls(new_calls)
            if early_stop:
                if stop_ids and next_id in stop_ids:
                    hit_eos = True
                    break
                if decoder.should_stop:
                    break
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    prefill_s = t_prefill - t0
    decode_s = t_end - t_decode
    n_new = len(new_tokens)
    if prompt_len and prefill_s > 0:
        emit(
            f"Prefill: {prompt_len} tok in {prefill_s:.2f}s "
            f"({prompt_len / prefill_s:.1f} tok/s)"
        )
    if n_new and decode_s > 0:
        emit(
            f"Decode:  {n_new} tok in {decode_s:.2f}s "
            f"({n_new / decode_s:.1f} tok/s)"
        )

    reasoning, text = decoder.finalize()
    if decoder.tool_calls:
        finish_reason = "tool_calls"
    elif hit_eos:
        finish_reason = "stop"
    else:
        finish_reason = "length"
    return GenerateResult(
        text=text,
        prompt_len=prompt_len,
        n_new=n_new,
        prefill_s=prefill_s,
        decode_s=decode_s,
        token_ids=new_tokens,
        finish_reason=finish_reason,
        reasoning=reasoning,
        tool_calls=decoder.tool_calls,
    )


def generate(
    model_dir: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = None,
    flash_head: bool = False,
    dense_head_source: str | None = None,
) -> GenerateResult:
    """Load a packed checkpoint, decode, print text and tok/s."""
    model_dir = str(Path(model_dir).expanduser())
    print(f"Loading packed model from {model_dir}", flush=True)
    model, tokenizer = load_packed(
        model_dir,
        flash_head=flash_head,
        dense_head_source=dense_head_source,
    )
    log_model_traffic(model, flash_head=flash_head)
    warmup_model(model, flash_head=flash_head)
    input_ids = encode_user_prompt(tokenizer, prompt, model.device)
    result = run_generate(
        model,
        tokenizer,
        input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    if result.reasoning:
        print(f"<think>\n{result.reasoning}\n</think>\n\n{result.text}", flush=True)
    else:
        print(result.text, flush=True)
    return result
