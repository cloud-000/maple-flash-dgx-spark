"""Train NextLat ``p_ψ`` on frozen packed Maple hidden states."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from maple_run.nextlat import (
    DEFAULT_HORIZON,
    DEFAULT_LAMBDA_KL,
    DEFAULT_LAMBDA_NEXT_H,
    NEXTLAT_WEIGHTS,
    LatentDynamics,
    attach_nextlat_metadata,
    default_mlp_hidden,
    dense_rtn4_weight,
    nextlat_loss,
)


def _read_examples(path: str | Path) -> list[dict]:
    """Load train rows. JSONL may carry exact ``input_ids`` from on-policy synth."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        examples: list[dict] = []
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids = obj.get("input_ids")
                if ids:
                    examples.append({"input_ids": [int(x) for x in ids]})
                    continue
                text = obj.get("text") or obj.get("content") or obj.get("prompt")
                if not text:
                    raise ValueError(
                        f"{path}:{line_no} missing input_ids or text/content/prompt"
                    )
                if isinstance(text, list):
                    text = "\n".join(str(t) for t in text)
                examples.append({"text": str(text)})
        return examples
    return [{"text": ln} for ln in path.read_text().splitlines() if ln.strip()]


def _tokenize_rows(tokenizer, examples: list[dict], seq_len: int, device) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    for ex in examples:
        ids = ex.get("input_ids")
        if ids is None:
            text = ex["text"]
            encode = getattr(tokenizer, "encode", None)
            if encode is not None:
                ids = encode(text, add_special_tokens=False)
            else:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        ids = list(ids)[: int(seq_len)]
        if len(ids) < 3:
            continue
        rows.append(torch.tensor(ids, dtype=torch.long, device=device))
    if not rows:
        raise ValueError("No tokenized sequences of length >= 3 in --data")
    return rows


def train_nextlat(
    model_dir: str,
    data: str,
    *,
    output: str | None = None,
    steps: int = 200,
    seq_len: int = 256,
    lr: float = 1e-3,
    horizon: int = DEFAULT_HORIZON,
    lambda_next_h: float = DEFAULT_LAMBDA_NEXT_H,
    lambda_kl: float = DEFAULT_LAMBDA_KL,
    mlp_hidden: int | None = None,
    seed: int = 0,
    max_sequences: int | None = None,
    log=print,
) -> LatentDynamics:
    """Fit ``p_ψ`` with Maple frozen. Packed ternary weights are not updated."""
    from maple_run.generate import load_packed, warmup_model

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if seq_len < horizon + 2:
        raise ValueError("seq_len must be >= horizon + 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if lr <= 0:
        raise ValueError("lr must be > 0")

    model_dir = str(Path(model_dir).expanduser())
    out_dir = Path(output or model_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading packed model from {model_dir}")
    model, tokenizer = load_packed(model_dir, flash_head=False)
    warmup_model(model, flash_head=False)
    torch.manual_seed(int(seed))

    examples = _read_examples(data)
    if max_sequences is not None:
        examples = examples[: int(max_sequences)]
    rows = _tokenize_rows(tokenizer, examples, seq_len, model.device)
    log(f"NextLat train: {len(rows)} sequences, steps={steps} horizon={horizon}")

    width = mlp_hidden if mlp_hidden is not None else default_mlp_hidden(model.hidden_size)
    dynamics = LatentDynamics(model.hidden_size, width, bias=False).to(
        device=model.device, dtype=torch.float32
    )
    opt = torch.optim.AdamW(dynamics.parameters(), lr=float(lr), betas=(0.9, 0.95))
    lm_weight = None
    if lambda_kl != 0.0:
        lm_weight = dense_rtn4_weight(model.lm_head, dtype=torch.float32).detach()
        lm_weight.requires_grad_(False)

    t0 = time.perf_counter()
    last = {}
    for step in range(int(steps)):
        ids = rows[step % len(rows)].view(1, -1)
        with torch.inference_mode():
            _, hidden = model.forward(
                ids, cache=None, logits_to_keep=0, return_hidden=True
            )
            embeds = model.word_embeddings(ids)
        hidden = hidden.float().detach()
        embeds = embeds.float().detach()
        dynamics.train()
        opt.zero_grad(set_to_none=True)
        stats = nextlat_loss(
            dynamics,
            hidden,
            embeds,
            lm_weight=lm_weight,
            horizon=int(horizon),
            lambda_next_h=float(lambda_next_h),
            lambda_kl=float(lambda_kl),
        )
        stats["loss"].backward()
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
        opt.step()
        last = {k: float(v) for k, v in stats.items()}
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == steps:
            log(
                f"step {step + 1}/{steps} loss={last['loss']:.4f} "
                f"next_h={last['next_h']:.4f} kl={last['kl']:.4f}"
            )

    elapsed = time.perf_counter() - t0
    log(f"NextLat train done in {elapsed:.1f}s")
    meta = {
        **dynamics.config_dict(),
        "horizon": int(horizon),
        "lambda_next_h": float(lambda_next_h),
        "lambda_kl": float(lambda_kl),
        "steps": int(steps),
        "seq_len": int(seq_len),
        "lr": float(lr),
        "seed": int(seed),
        "paper": "arXiv:2511.05963",
    }
    weights_path = out_dir / NEXTLAT_WEIGHTS
    dynamics.eval()
    dynamics.save(weights_path, extra=meta)
    attach_nextlat_metadata(out_dir, {**meta, "weights": NEXTLAT_WEIGHTS})
    log(f"Wrote {weights_path}")
    return dynamics
