"""Generate Maple completions for NextLat on-policy training jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from maple_run.generate import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    batched_generate,
    can_fuse_sampler,
    encode_user_prompt,
    is_greedy,
    load_packed,
    run_generate,
    stop_token_ids,
    warmup_model,
)

DEFAULT_SYNTH_MAX_TOKENS = 256
DEFAULT_SYNTH_BATCH = 8
DEFAULT_SYNTH_OUTPUT = "nextlat/data/onpolicy.jsonl"


def user_prompt_from_obj(obj: dict) -> str:
    """First user string: UltraChat ``messages``, then prompt/text/content.

    MT-Bench ``prompt`` lists use turn 0. Do not pass a flattened full
    dialogue in ``text`` (Maple would see the assistant turns as the user).
    """
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


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _write_row(out, tokenizer, prompt: str, prompt_ids, new_ids, finish: str, log) -> None:
    prompt_list = (
        prompt_ids if isinstance(prompt_ids, list) else prompt_ids.reshape(-1).tolist()
    )
    new_list = [int(t) for t in new_ids]
    full_ids = prompt_list + new_list
    rec = {
        "text": tokenizer.decode(full_ids, skip_special_tokens=False),
        "input_ids": full_ids,
        "prompt": prompt,
        "prompt_len": len(prompt_list),
        "n_new": len(new_list),
        "finish_reason": finish,
    }
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()
    log(
        f"synth prompt_len={rec['prompt_len']} n_new={rec['n_new']} "
        f"finish={finish}"
    )


def synthesize_onpolicy(
    model_dir: str,
    prompts: str,
    *,
    output: str = DEFAULT_SYNTH_OUTPUT,
    max_tokens: int = DEFAULT_SYNTH_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    seed: int | None = None,
    limit: int | None = None,
    batch_size: int = DEFAULT_SYNTH_BATCH,
    split: str | None = None,
    log=print,
) -> Path:
    """Decode Maple replies (exact head, no NextLat) and write train jsonl.

    Each row stores ``input_ids`` = chat-templated prompt + sampled completion
    so ``maple-run nextlat`` can train on the real generate tokens.

    ``batch_size > 1`` prefills each prompt on its own (same as
    ``batched_generate``) and decodes the chunk together so packed weights
    are read once per step for the live rows.
    """
    if max_tokens < 1 and max_tokens != -1:
        raise ValueError("max_tokens must be >= 1 or -1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = _count_jsonl(out_path)

    model_dir = str(Path(model_dir).expanduser())
    log(f"Loading packed model from {model_dir}")
    model, tokenizer = load_packed(model_dir, flash_head=False)
    warmup_model(model, flash_head=False)
    wide = int(batch_size) > 1
    if wide and not is_greedy(temperature, top_k) and not can_fuse_sampler(
        temperature, top_p, top_k
    ):
        raise ValueError(
            "batched synth needs fused sampling (0 < top_k <= 64, 0 < top_p <= 1) "
            "or --batch-size 1"
        )
    log(
        "On-policy synth sampling: "
        f"temperature={temperature} top_p={top_p} top_k={top_k} "
        f"max_tokens={max_tokens} seed={seed} batch_size={batch_size} "
        "(exact head, no NextLat)"
    )

    written = 0
    chunk: list[tuple[int, str]] = []
    stop_ids = stop_token_ids(model, tokenizer) if wide else None

    def flush(rows: list[tuple[int, str]]) -> int:
        if not rows:
            return 0
        prompt_ids = [
            encode_user_prompt(tokenizer, prompt, model.device) for _, prompt in rows
        ]
        batch_seed = None if seed is None else int(seed) + rows[0][0]
        batched = batched_generate(
            model,
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=batch_seed,
            stop_ids=stop_ids,
            graphs=False,
            log=log,
        )
        n = 0
        for (_, prompt), ids, _plen, reason, pids in zip(
            rows,
            batched.token_ids,
            batched.prompt_lens,
            batched.finish_reasons,
            prompt_ids,
            strict=True,
        ):
            _write_row(out, tokenizer, prompt, pids, ids, reason, log)
            n += 1
        log(
            f"synth batch {len(rows)}: prefill {batched.prefill_s:.2f}s "
            f"decode {batched.n_new} tok in {batched.decode_s:.2f}s "
            f"({batched.decode_tok_s:.1f} tok/s aggregate)"
        )
        return n

    with out_path.open("a") as out:
        for i, prompt in enumerate(
            iter_user_prompts(prompts, limit=limit, split=split)
        ):
            if i < skip:
                continue
            if not wide:
                prompt_ids = encode_user_prompt(tokenizer, prompt, model.device)
                row_seed = None if seed is None else int(seed) + i
                result = run_generate(
                    model,
                    tokenizer,
                    prompt_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=row_seed,
                    log=None,
                )
                _write_row(
                    out,
                    tokenizer,
                    prompt,
                    prompt_ids,
                    result.token_ids,
                    result.finish_reason,
                    log,
                )
                written += 1
                continue
            chunk.append((i, prompt))
            if len(chunk) >= int(batch_size):
                written += flush(chunk)
                chunk = []
        if chunk:
            written += flush(chunk)
    if skip:
        log(f"Resumed past {skip} existing rows in {out_path}")
    log(f"Wrote {written} new rows to {out_path}")
    return out_path
