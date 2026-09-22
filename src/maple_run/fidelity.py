"""Teacher-forced fidelity checks between original bf16 and packed Maple."""

from __future__ import annotations

import contextlib
import importlib.machinery
import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DEFAULT_ORIGINAL = "deepgrove/maple-preview"


class FidelityError(RuntimeError):
    """A fidelity run cannot produce a meaningful comparison."""


def _legacy_default_rope_init(config, device=None, **_kwargs):
    """Transformers <5 ``default`` RoPE initializer retained by Maple's code."""
    import torch

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    base = float(getattr(config, "rope_theta", 10000.0))
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
            / dim
        )
    )
    return inv_freq, 1.0


@dataclass(frozen=True)
class TokenMetrics:
    kl: list[float]
    nll_delta: list[float]
    top1: list[bool]
    top5: list[bool]


def sdpa_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    **_kwargs,
):
    """Drop-in upstream FA wrapper for cache-free, causal reference prefills."""
    import torch
    import torch.nn.functional as F

    if dropout:
        raise FidelityError("reference model must be in eval mode (dropout=0)")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise FidelityError("reference attention expects rank-4 Q/K/V tensors")
    expected_heads = getattr(module, "num_heads", None)
    expected_kv_heads = getattr(module, "num_key_value_heads", None)
    if expected_heads is not None and query.shape[1] != expected_heads:
        raise FidelityError(
            "reference attention received Q in the wrong layout: expected "
            f"[batch, heads={expected_heads}, sequence, head_dim], got {tuple(query.shape)}"
        )
    if expected_kv_heads is not None and (
        key.shape[1] != expected_kv_heads or value.shape[1] != expected_kv_heads
    ):
        raise FidelityError(
            "reference attention received K/V in the wrong layout: expected "
            f"head axis 1 with {expected_kv_heads} heads, got "
            f"K={tuple(key.shape)} V={tuple(value.shape)}"
        )
    bsz, n_heads, q_len, _ = query.shape
    _, n_kv, k_len, _ = key.shape
    if value.shape[:3] != (bsz, n_kv, k_len):
        raise FidelityError(
            f"reference attention K/V shapes differ: K={tuple(key.shape)} V={tuple(value.shape)}"
        )
    if n_heads % n_kv:
        raise FidelityError(f"query heads {n_heads} are not divisible by KV heads {n_kv}")
    if n_heads != n_kv:
        repeat = n_heads // n_kv
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    q_pos = torch.arange(q_len, device=query.device) + (k_len - q_len)
    k_pos = torch.arange(k_len, device=query.device)
    allowed = k_pos.view(1, 1, 1, k_len) <= q_pos.view(1, 1, q_len, 1)
    if sliding_window is not None:
        allowed &= (
            q_pos.view(1, 1, q_len, 1) - k_pos.view(1, 1, 1, k_len)
            < int(sliding_window)
        )
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != (bsz, k_len):
            raise FidelityError("reference SDPA only supports a [batch, sequence] attention mask")
        allowed = allowed & attention_mask.to(dtype=torch.bool).view(bsz, 1, 1, k_len)
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=allowed, dropout_p=0.0, is_causal=False, scale=scaling
    )
    return out.transpose(1, 2).contiguous(), None


def _reset_reference_rope(model) -> None:
    """Rebuild Maple's non-persistent RoPE buffer after low-memory loading.

    Current Transformers constructs remote-code models on the meta device in
    ``from_pretrained``.  Maple's ``inv_freq`` is a non-persistent buffer, so
    it is not restored from the checkpoint and can be materialized without
    the constructor values.  The resulting model still runs, but rotary
    attention uses corrupt frequencies.  Recompute the deterministic default
    table from the config before moving the model to its execution device.
    """
    rotary = model.model.rotary_emb
    inv_freq, scaling = _legacy_default_rope_init(model.config, device="cpu")
    rotary.register_buffer("inv_freq", inv_freq, persistent=False)
    rotary.original_inv_freq = inv_freq
    rotary.attention_scaling = scaling


@contextlib.contextmanager
def _fake_flash_attention_imports() -> Iterator[None]:
    """Permit upstream ``fa3.py`` to import before its call is patched out."""
    saved: dict[str, object | None] = {}
    names = ("flash_attn", "flash_attn.bert_padding", "flash_attn_interface")
    for name in names:
        saved[name] = sys.modules.get(name)
    package = types.ModuleType("flash_attn")
    package.__path__ = []  # type: ignore[attr-defined]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None, is_package=True
    )

    def unavailable(*_args, **_kwargs):
        raise FidelityError("FlashAttention fallback was not patched into the reference model")

    package.flash_attn_func = unavailable
    package.flash_attn_varlen_func = unavailable
    padding = types.ModuleType("flash_attn.bert_padding")
    padding.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.bert_padding", loader=None
    )
    padding.index_first_axis = unavailable
    padding.pad_input = unavailable
    padding.unpad_input = unavailable
    interface = types.ModuleType("flash_attn_interface")
    interface.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn_interface", loader=None
    )
    interface.flash_attn_func = unavailable
    interface.flash_attn_varlen_func = unavailable
    sys.modules["flash_attn"] = package
    sys.modules["flash_attn.bert_padding"] = padding
    sys.modules["flash_attn_interface"] = interface
    try:
        yield
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def resolve_original_checkpoint(original: str | Path) -> Path:
    """Resolve a local path or an already-cached Hugging Face repo id."""
    path = Path(original).expanduser()
    if path.is_dir():
        return path
    from maple_run.convert import _try_local_hub_snapshot

    snapshot = _try_local_hub_snapshot(str(original))
    if snapshot is None:
        raise FidelityError(
            f"original checkpoint is not a directory and is not cached locally: {original}. "
            "Pass --original /path/to/snapshot or cache the repository first."
        )
    return snapshot


def load_original_reference(
    original_dir: str | Path = DEFAULT_ORIGINAL,
    device: str = "cuda",
    log=print,
):
    """Load local upstream Maple and replace its unavailable FA call with SDPA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    original = resolve_original_checkpoint(original_dir)
    if not (original / "modeling_maple.py").is_file():
        raise FidelityError(f"original checkpoint is missing modeling_maple.py: {original}")
    # Maple's remote code uses the old ``default`` spelling, removed from the
    # Transformers 5 dispatch table.  This is its pre-5 behavior exactly.
    ROPE_INIT_FUNCTIONS.setdefault("default", _legacy_default_rope_init)
    log(f"Loading original tokenizer/config from {original}")
    with _fake_flash_attention_imports():
        tokenizer = AutoTokenizer.from_pretrained(
            str(original), trust_remote_code=True, fix_mistral_regex=True
        )
        log("Constructing original bf16 model and loading weights")
        model = AutoModelForCausalLM.from_pretrained(
            str(original), trust_remote_code=True, dtype=torch.bfloat16
        )
    _reset_reference_rope(model)
    try:
        attention_module = sys.modules[model.model.layers[0].self_attn.__class__.__module__]
    except (AttributeError, IndexError, KeyError) as exc:
        raise FidelityError("loaded original model does not expose Maple attention layers") from exc
    attention_module.flash_attention_forward = sdpa_attention_forward
    log(f"Moving original model to {device}")
    model.eval().to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log("Original model ready")
    return model, tokenizer


def token_metrics(original_logits, packed_logits, targets) -> TokenMetrics:
    """Return one fidelity record per next-token slot, computed in float32."""
    import torch

    if original_logits.shape != packed_logits.shape:
        raise FidelityError(
            f"logit shapes differ: original={tuple(original_logits.shape)} packed={tuple(packed_logits.shape)}"
        )
    if original_logits.ndim != 2:
        raise FidelityError(f"expected [tokens, vocab] logits, got {tuple(original_logits.shape)}")
    if targets.shape != (original_logits.shape[0],):
        raise FidelityError("targets must contain one id for every logit row")
    p_log = torch.log_softmax(original_logits.float(), dim=-1)
    q_log = torch.log_softmax(packed_logits.float(), dim=-1)
    if not torch.isfinite(p_log).all() or not torch.isfinite(q_log).all():
        raise FidelityError("non-finite logits encountered")
    p = p_log.exp()
    kl = (p * (p_log - q_log)).sum(dim=-1)
    orig_nll = -p_log.gather(-1, targets[:, None]).squeeze(-1)
    packed_nll = -q_log.gather(-1, targets[:, None]).squeeze(-1)
    top1 = original_logits.argmax(dim=-1) == packed_logits.argmax(dim=-1)
    top5 = (packed_logits.topk(min(5, packed_logits.shape[-1]), dim=-1).indices == targets[:, None]).any(dim=-1)
    return TokenMetrics(
        kl=[float(x) for x in kl.cpu()],
        nll_delta=[float(x) for x in (packed_nll - orig_nll).cpu()],
        top1=[bool(x) for x in top1.cpu()],
        top5=[bool(x) for x in top5.cpu()],
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    kl = [x for rec in records for x in rec["kl_values"]]
    nll = [x for rec in records for x in rec["nll_delta_values"]]
    top1 = [x for rec in records for x in rec["top1_values"]]
    top5 = [x for rec in records for x in rec["top5_values"]]
    n = len(kl)
    if not n:
        raise FidelityError("no next-token slots available; prompts need at least two tokens")
    return {
        "tokens": n, "mean_kl": sum(kl) / n, "p50_kl": _percentile(kl, 0.50),
        "p95_kl": _percentile(kl, 0.95), "max_kl": max(kl),
        "mean_nll_delta": sum(nll) / n, "top1_agreement": sum(top1) / n,
        "top5_target_agreement": sum(top5) / n,
    }


def threshold_outcomes(summary: dict[str, Any], max_kl: float | None, min_top1: float | None) -> dict[str, Any]:
    outcomes: dict[str, Any] = {"passed": True}
    if max_kl is not None:
        outcomes["max_kl"] = {"limit": max_kl, "actual": summary["mean_kl"], "passed": summary["mean_kl"] <= max_kl}
        outcomes["passed"] &= outcomes["max_kl"]["passed"]
    if min_top1 is not None:
        outcomes["min_top1_agreement"] = {"limit": min_top1, "actual": summary["top1_agreement"], "passed": summary["top1_agreement"] >= min_top1}
        outcomes["passed"] &= outcomes["min_top1_agreement"]["passed"]
    return outcomes


def user_prompt_from_obj(obj: dict) -> str:
    """First user string: UltraChat ``messages``, then prompt/text/content."""
    messages = obj.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() == "user":
                text = str(msg.get("content") or "").strip()
                if text:
                    return text
    raw = obj.get("prompt", None)
    if raw is None:
        raw = obj.get("text") or obj.get("content")
    if raw is None:
        raise ValueError("row missing messages/prompt/text/content")
    if isinstance(raw, list):
        if not raw:
            raise ValueError("jsonl prompt list is empty")
        raw = raw[0]
    text = str(raw).strip()
    if not text:
        raise ValueError("jsonl prompt is empty")
    return text


def _iter_hf_prompts(repo: str, *, split: str | None, limit: int | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face --prompts needs the eval extra: "
            "uv sync --extra cuda --extra tokenizer --extra eval"
        ) from exc
    splits = [split] if split else ["train_sft", "train"]
    last_err: Exception | None = None
    ds = None
    for name in splits:
        if not name:
            continue
        try:
            ds = load_dataset(repo, split=name, streaming=True)
            break
        except Exception as exc:
            last_err = exc
            ds = None
    if ds is None:
        raise ValueError(
            f"could not load {repo!r} (tried splits {splits}): {last_err}"
        )
    n = 0
    for row in ds:
        yield user_prompt_from_obj(dict(row))
        n += 1
        if limit is not None and n >= int(limit):
            return


def iter_user_prompts(
    path: str | Path, *, limit: int | None = None, split: str | None = None
):
    raw = str(path)
    file_path = Path(raw)
    if not file_path.exists() and "/" in raw and not raw.endswith(".jsonl"):
        yield from _iter_hf_prompts(raw, split=split, limit=limit)
        return
    n = 0
    if file_path.suffix.lower() == ".jsonl":
        with file_path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                try:
                    yield user_prompt_from_obj(obj)
                except ValueError as exc:
                    raise ValueError(f"{file_path}:{line_no}: {exc}") from exc
                n += 1
                if limit is not None and n >= int(limit):
                    return
        return
    with file_path.open() as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            yield text
            n += 1
            if limit is not None and n >= int(limit):
                return


def run_fidelity(
    model_dir: str, prompts: str, *, output: str, original_dir: str = DEFAULT_ORIGINAL,
    limit: int | None = None, split: str | None = None, max_seq_len: int = 512,
    max_kl: float | None = None, min_top1_agreement: float | None = None, log=print,
) -> bool:
    """Evaluate a prompt corpus and write a JSON report. Returns gate status."""
    import torch
    from maple_run.generate import encode_user_prompt
    from maple_run.model import MapleForCausalLM

    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2")
    started = time.perf_counter()
    log("Loading packed model")
    # Fidelity needs exactly one canonical tokenizer. Transformers' Mistral
    # regex compatibility logic treats the packed ``model_type: maple`` config
    # differently from the original remote config even though their tokenizer
    # files are identical. Loading the packed serving tokenizer here can thus
    # produce different token IDs before either model runs.
    packed = MapleForCausalLM.from_packed(model_dir, use_flash_head=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log(f"Packed model ready ({time.perf_counter() - started:.1f}s)")
    original, original_tokenizer = load_original_reference(
        original_dir, str(packed.device), log=log
    )
    log(f"Both models ready ({time.perf_counter() - started:.1f}s)")
    records: list[dict[str, Any]] = []
    prompts_seen = 0
    for prompt in iter_user_prompts(prompts, limit=limit, split=split):
        # KL compares distributions conditioned on the same token IDs. Use the
        # corrected original tokenizer as that single source of truth.
        ids = encode_user_prompt(
            original_tokenizer, prompt, packed.device
        ).reshape(-1)[:max_seq_len]
        prompts_seen += 1
        if ids.numel() < 2:
            log(f"Prompt {prompts_seen}: skipped (fewer than two tokens)")
            continue
        prompt_started = time.perf_counter()
        log(f"Prompt {prompts_seen}: original forward ({ids.numel()} tokens)")
        with torch.inference_mode():
            ref = original(input_ids=ids[None], use_cache=False, logits_to_keep=0).logits[0, :-1]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log(f"Prompt {prompts_seen}: packed forward ({time.perf_counter() - prompt_started:.1f}s)")
            got = packed.forward(ids[None], cache=None, logits_to_keep=int(ids.numel()))[0, :-1]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        metrics = token_metrics(ref, got, ids[1:])
        log(f"Prompt {prompts_seen}: metrics ready ({time.perf_counter() - prompt_started:.1f}s)")
        records.append({
            "prompt_index": prompts_seen - 1, "input_tokens": int(ids.numel()),
            "compared_tokens": len(metrics.kl), "mean_kl": sum(metrics.kl) / len(metrics.kl),
            "max_kl": max(metrics.kl), "kl_values": metrics.kl,
            "nll_delta_values": metrics.nll_delta, "top1_values": metrics.top1,
            "top5_values": metrics.top5,
        })
    summary = summarize(records)
    gates = threshold_outcomes(summary, max_kl, min_top1_agreement)
    for rec in records:
        for key in ("kl_values", "nll_delta_values", "top1_values", "top5_values"):
            rec.pop(key)
    report = {
        "version": 1, "mode": "original_bf16_vs_packed_exact_head",
        "inputs": {"model": str(model_dir), "original": str(original_dir), "prompts": str(prompts), "limit": limit, "split": split, "max_seq_len": max_seq_len},
        "prompts_seen": prompts_seen, "prompts_compared": len(records),
        "summary": summary, "thresholds": gates, "prompts": records,
    }
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log("Writing fidelity report")
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    log(f"Fidelity: {summary['tokens']} tokens, mean KL {summary['mean_kl']:.6g}, top-1 {summary['top1_agreement']:.2%}")
    log(f"Report: {output_path}")
    return bool(gates["passed"])
