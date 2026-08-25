"""Next-Latent Prediction (NextLat) for packed Maple.

Port of Teoh et al. 2025, arXiv:2511.05963, and the official
``NextLatDynamicsModel`` / Algorithm 1. Maple's ternary body has no
autograd path, so this trains the latent dynamics MLP ``p_ψ`` on frozen
hidden states and uses it for variable-length self-speculative decode.

``p_ψ(h_t, x_{t+1}) = MLP(LN([embed(x_{t+1}); h_t])) + h_t``.

Default generate stays exact-head autoregressive. Opt in with
``maple-run generate --nextlat`` after ``maple-run nextlat``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from maple_run.kernels.rtn4 import unpack_4bit_torch
from maple_run.pack import HEAD_GROUP_SIZE

NEXTLAT_WEIGHTS = "nextlat.safetensors"
DEFAULT_HORIZON = 1
DEFAULT_LAMBDA_NEXT_H = 1.0
DEFAULT_LAMBDA_KL = 1.0
DEFAULT_DRAFT_LEN = 5
# FineWeb-Edu (d_model=2048) used MLP width 6528; Maple is the same width.
_FINEWEB_HIDDEN = 2048
_FINEWEB_MLP_HIDDEN = 6528


def default_mlp_hidden(hidden_size: int) -> int:
    """MLP width, rounded to 128, matching the FineWeb-Edu ratio."""
    input_dim = 2 * int(hidden_size)
    raw = (_FINEWEB_MLP_HIDDEN / (2 * _FINEWEB_HIDDEN)) * input_dim
    return max(128, 128 * int(round(raw / 128)))


def dense_rtn4_weight(linear, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Dequant packed 4-bit ``lm_head`` to a frozen dense ``[V, K]`` matrix."""
    codes = unpack_4bit_torch(linear.packed_weight).to(torch.float32)
    group = int(getattr(linear, "group_size", HEAD_GROUP_SIZE))
    n, k = codes.shape[0], codes.shape[-1]
    grouped = codes.reshape(n, k // group, group)
    recon = grouped * linear.scales.float().unsqueeze(-1) + linear.biases.float().unsqueeze(
        -1
    )
    out = recon.reshape(n, k)
    if dtype is not None:
        out = out.to(dtype)
    return out


class LatentDynamics(nn.Module):
    """Three-layer GELU MLP with a residual update (paper Appendix C)."""

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden: int | None = None,
        *,
        bias: bool = False,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        mlp_hidden = int(mlp_hidden) if mlp_hidden is not None else default_mlp_hidden(
            hidden_size
        )
        if mlp_hidden < 1:
            raise ValueError("mlp_hidden must be >= 1")
        input_dim = hidden_size * 2
        self.hidden_size = hidden_size
        self.mlp_hidden = mlp_hidden
        self.bias = bool(bias)
        self.norm_x = nn.LayerNorm(input_dim, elementwise_affine=True, bias=self.bias)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden, bias=self.bias),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden, bias=self.bias),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size, bias=self.bias),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, current_states: torch.Tensor, next_token_embeds: torch.Tensor
    ) -> torch.Tensor:
        if current_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden last dim {current_states.shape[-1]} != {self.hidden_size}"
            )
        if next_token_embeds.shape != current_states.shape:
            raise ValueError(
                f"embed shape {tuple(next_token_embeds.shape)} != "
                f"hidden {tuple(current_states.shape)}"
            )
        x = torch.cat([next_token_embeds, current_states], dim=-1)
        delta = self.mlp(self.norm_x(x))
        return delta + current_states

    def config_dict(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "mlp_hidden": self.mlp_hidden,
            "n_layers": 3,
            "bias": self.bias,
        }

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu() for k, v in self.state_dict().items()}, str(path))
        meta_path = path.with_suffix(".json")
        meta = self.config_dict()
        if extra:
            meta.update(extra)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def from_config(cls, meta: dict, *, device=None, dtype: torch.dtype | None = None):
        model = cls(
            int(meta["hidden_size"]),
            int(meta.get("mlp_hidden") or default_mlp_hidden(int(meta["hidden_size"]))),
            bias=bool(meta.get("bias", False)),
        )
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        return model

    @classmethod
    def load(cls, path: str | Path, *, device=None, dtype: torch.dtype | None = None):
        path = Path(path)
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"NextLat metadata missing: {meta_path}")
        meta = json.loads(meta_path.read_text())
        model = cls.from_config(meta, device=device, dtype=dtype)
        state = load_file(str(path), device=str(device) if device is not None else "cpu")
        model.load_state_dict(state)
        model.eval()
        return model


def attach_nextlat_metadata(model_dir: str | Path, meta: dict) -> None:
    """Write ``maple_run.nextlat`` on the packed checkpoint config."""
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return
    config = json.loads(cfg_path.read_text())
    maple = dict(config.get("maple_run") or {})
    maple["nextlat"] = dict(meta)
    config["maple_run"] = maple
    config["nextlat"] = dict(meta)
    cfg_path.write_text(json.dumps(config, indent=2) + "\n")


def nextlat_checkpoint_paths(model_dir: str | Path) -> tuple[Path, Path]:
    model_dir = Path(model_dir)
    weights = model_dir / NEXTLAT_WEIGHTS
    return weights, weights.with_suffix(".json")


def load_nextlat_for_model(model, model_dir: str | Path) -> LatentDynamics:
    """Load ``p_ψ`` next to a packed Maple, checking hidden size."""
    weights, meta_path = nextlat_checkpoint_paths(model_dir)
    if not weights.exists():
        raise FileNotFoundError(
            "Checkpoint has no NextLat dynamics. Train them with "
            "`maple-run nextlat --model <packed> --data <text>`."
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"NextLat metadata missing: {meta_path}")
    meta = json.loads(meta_path.read_text())
    hidden = int(getattr(model, "hidden_size", 0) or model.config["hidden_size"])
    if int(meta["hidden_size"]) != hidden:
        raise ValueError(
            f"NextLat hidden_size {meta['hidden_size']} != model {hidden}"
        )
    dynamics = LatentDynamics.load(weights, device=model.device, dtype=torch.float32)
    return dynamics


def nextlat_loss(
    dynamics: LatentDynamics,
    hidden: torch.Tensor,
    token_embeds: torch.Tensor,
    *,
    lm_weight: torch.Tensor | None = None,
    horizon: int = DEFAULT_HORIZON,
    lambda_next_h: float = DEFAULT_LAMBDA_NEXT_H,
    lambda_kl: float = DEFAULT_LAMBDA_KL,
    mse_mask: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Algorithm 1: Smooth-L1 next-hidden + optional KL through a frozen head.

    ``hidden`` / ``token_embeds`` are ``[B, T, D]`` and should be detached.
    Teacher logits use ``lm_weight`` (stop-grad) so KL does not train the head.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if hidden.ndim != 3 or token_embeds.shape != hidden.shape:
        raise ValueError("hidden and token_embeds must be [B, T, D] with equal shape")
    bsz, seq_len, dim = hidden.shape
    if seq_len < horizon + 1:
        raise ValueError(f"sequence length {seq_len} is shorter than horizon+1")

    current = hidden
    next_tokens = token_embeds
    next_states = hidden
    teacher = None
    if lm_weight is not None and lambda_kl != 0.0:
        teacher = F.linear(hidden, lm_weight)

    loss_h = hidden.new_zeros(())
    loss_kl = hidden.new_zeros(())
    for step in range(horizon):
        current = current[:, :-1]
        next_tokens = next_tokens[:, 1:]
        next_states = next_states[:, 1:]
        pred = dynamics(current, next_tokens)
        target = next_states.detach()
        elem = F.smooth_l1_loss(pred, target, reduction="none")
        if mse_mask is not None:
            w = mse_mask[:, step + 1 :].unsqueeze(-1).to(dtype=elem.dtype)
            denom = w.expand_as(elem).sum().clamp_min(1.0)
            loss_h = loss_h + (elem * w).sum() / denom
        else:
            loss_h = loss_h + elem.mean()
        if teacher is not None:
            teacher = teacher[:, 1:]
            student = F.linear(pred, lm_weight)
            log_q = F.log_softmax(teacher.detach(), dim=-1)
            log_p = F.log_softmax(student, dim=-1)
            kl = F.kl_div(log_p, log_q, log_target=True, reduction="none").sum(dim=-1)
            if token_mask is not None:
                m = token_mask[:, step:].to(dtype=kl.dtype)
                # token_mask is [B, T-1] aligned with positions 0..T-2 of the
                # original sequence (next-token slots). After `step` shifts of
                # the teacher, valid rows are token_mask[:, step:].
                if m.shape[-1] != kl.shape[-1]:
                    m = m[:, : kl.shape[-1]]
                loss_kl = loss_kl + (kl * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss_kl = loss_kl + kl.mean()
        current = pred

    loss_h = loss_h / horizon
    loss_kl = loss_kl / horizon
    total = lambda_next_h * loss_h + lambda_kl * loss_kl
    return {
        "loss": total,
        "next_h": loss_h.detach(),
        "kl": loss_kl.detach(),
    }


def _nucleus_from_sorted(sorted_probs: torch.Tensor, top_p: float) -> torch.Tensor:
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    remove = cumsum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    return sorted_probs.masked_fill(remove, 0)


def token_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    """Softmax distribution used for both draft q and target p."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    x = logits.float()
    vocab = x.shape[-1]
    if temperature <= 0.0 or top_k == 1:
        idx = x.argmax(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        return out.scatter_(-1, idx, 1.0)
    if top_k > 0 and top_k < vocab:
        kth = torch.topk(x, int(top_k), dim=-1).values[..., -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    probs = torch.softmax(x / max(float(temperature), 1e-6), dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        kept = _nucleus_from_sorted(sorted_probs, float(top_p))
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, kept)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def sample_from_probs(
    probs: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, num_samples=1, generator=generator)
    # Pass the shape tuple (do not unpack). A 1D vocab row is [V], so
    # shape[:-1] is (); *() would call reshape() with no args and crash.
    return idx.reshape(probs.shape[:-1])


def propose_drafts(
    dynamics: LatentDynamics,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    embed_fn,
    head_fn,
    gamma: int,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll ``p_ψ`` for ``gamma`` tokens. Returns ids ``[gamma]`` and q ``[gamma, V]``."""
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    h = hidden.float()
    while h.ndim > 2:
        h = h.squeeze(1)
    if h.ndim == 1:
        h = h.unsqueeze(0)
    ids: list[torch.Tensor] = []
    qs: list[torch.Tensor] = []
    cur_logits = logits
    for i in range(gamma):
        q = token_probs(
            cur_logits.reshape(-1),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if temperature <= 0.0 or top_k == 1:
            tok = cur_logits.reshape(-1).argmax().view(1)
        else:
            tok = sample_from_probs(q, generator=generator).view(1)
        ids.append(tok)
        qs.append(q.reshape(-1))
        if i + 1 < gamma:
            emb = embed_fn(tok.view(1, 1)).float()
            while emb.ndim > 2:
                emb = emb.squeeze(1)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            h = dynamics(h, emb)
            cur_logits = head_fn(h)
    return torch.cat(ids, dim=0), torch.stack(qs, dim=0)


def accept_drafts(
    draft_ids: torch.Tensor,
    q_probs: torch.Tensor,
    p_probs: torch.Tensor,
    p_bonus: torch.Tensor,
    *,
    greedy: bool,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Leviathan accept/reject. Returns ``(n_accepted, bonus_or_correction_id)``."""
    gamma = int(draft_ids.numel())
    if greedy:
        for i in range(gamma):
            target = int(p_probs[i].argmax())
            if int(draft_ids[i]) != target:
                return i, target
        return gamma, int(p_bonus.argmax())

    device = draft_ids.device
    for i in range(gamma):
        tok = draft_ids[i].view(1)
        q_tok = q_probs[i].gather(0, tok).clamp_min(1e-12)
        p_tok = p_probs[i].gather(0, tok)
        accept_p = torch.minimum(torch.ones((), device=device), (p_tok / q_tok).reshape(()))
        u = torch.rand((), device=device, generator=generator)
        if u < accept_p:
            continue
        correction = torch.clamp(p_probs[i] - q_probs[i], min=0.0)
        if float(correction.sum()) <= 0:
            nxt = sample_from_probs(p_probs[i], generator=generator)
        else:
            nxt = sample_from_probs(correction / correction.sum(), generator=generator)
        return i, int(nxt.reshape(()))
    bonus = sample_from_probs(p_bonus, generator=generator)
    return gamma, int(bonus.reshape(()))


def _as_token_row(ids) -> torch.Tensor:
    t = ids if torch.is_tensor(ids) else torch.tensor(ids)
    t = t.long().reshape(-1)
    return t.view(1, -1)


def run_nextlat_generate(
    model,
    dynamics: LatentDynamics,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None,
    stop_ids: set[int],
    draft_len: int,
    decoder,
    log,
    on_token=None,
    on_text=None,
    on_reasoning=None,
    on_tool_calls=None,
):
    """Speculative decode using ``p_ψ`` drafts and Maple as the verifier.

    CUDA graphs stay off: accepted length varies per round. Default
    ``run_generate`` is unchanged when this is not called.
    """
    import time

    from maple_run.generate import GenerateResult, is_greedy, resolve_max_tokens

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(model.device).long()
    prompt_len = int(input_ids.shape[-1])
    max_context = int(
        (getattr(model, "config", None) or {}).get("max_position_embeddings", 131072)
    )
    max_tokens = resolve_max_tokens(max_tokens, prompt_len, max_context)
    greedy = is_greedy(temperature, top_k)
    gamma = max(int(draft_len), 1)
    dynamics = dynamics.eval()

    def embed_fn(tok: torch.Tensor) -> torch.Tensor:
        return model.word_embeddings(tok.to(model.device)).to(dtype=torch.float32)

    def head_fn(h: torch.Tensor) -> torch.Tensor:
        x = h.to(dtype=model.dtype)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return model.lm_head(x)

    cache = model.make_cache(max_len=prompt_len + max(max_tokens, 1))
    new_tokens: list[int] = []
    hit_eos = False
    proposed = 0
    accepted_drafts = 0
    rounds = 0

    def emit_token(tok_id: int) -> None:
        nonlocal hit_eos
        new_tokens.append(tok_id)
        if on_token is not None:
            on_token(tok_id)
        r_delta, c_delta = decoder.push(tok_id)
        if r_delta and on_reasoning is not None:
            on_reasoning(r_delta)
        if c_delta and on_text is not None:
            on_text(c_delta)
        new_calls = decoder.take_new_tool_calls()
        if new_calls and on_tool_calls is not None:
            on_tool_calls(new_calls)
        if stop_ids and tok_id in stop_ids:
            hit_eos = True

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits, hidden = model.forward(
            input_ids, cache=cache, logits_to_keep=1, return_hidden=True
        )
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        while len(new_tokens) < max_tokens and not hit_eos and not decoder.should_stop:
            remaining = max_tokens - len(new_tokens)
            steps = min(gamma, remaining)
            draft_ids, q_probs = propose_drafts(
                dynamics,
                hidden,
                logits[0, -1],
                embed_fn,
                head_fn,
                steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
            proposed += steps
            rounds += 1
            if steps == 1:
                tok = int(draft_ids[0])
                emit_token(tok)
                if hit_eos or decoder.should_stop or len(new_tokens) >= max_tokens:
                    break
                logits, hidden = model.forward(
                    draft_ids.view(1, 1),
                    cache=cache,
                    logits_to_keep=1,
                    return_hidden=True,
                )
                continue

            snap = cache.snapshot()
            p_logits, p_hidden = model.forward(
                draft_ids.view(1, -1),
                cache=cache,
                logits_to_keep=steps,
                return_hidden=True,
            )
            p_for_draft = torch.cat([logits[:, -1:, :], p_logits[:, :-1, :]], dim=1)
            p_probs = torch.stack(
                [
                    token_probs(
                        p_for_draft[0, i],
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    ).reshape(-1)
                    for i in range(steps)
                ],
                dim=0,
            )
            p_bonus = token_probs(
                p_logits[0, -1],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ).reshape(-1)
            n_acc, extra = accept_drafts(
                draft_ids,
                q_probs,
                p_probs,
                p_bonus,
                greedy=greedy,
                generator=generator,
            )
            accepted_drafts += n_acc
            if n_acc == steps:
                stopped = False
                for tok in draft_ids.tolist():
                    emit_token(int(tok))
                    if hit_eos or decoder.should_stop or len(new_tokens) >= max_tokens:
                        stopped = True
                        break
                if stopped:
                    break
                emit_token(int(extra))
                if hit_eos or decoder.should_stop or len(new_tokens) >= max_tokens:
                    break
                extra_t = torch.tensor([[extra]], device=model.device, dtype=torch.long)
                logits, hidden = model.forward(
                    extra_t, cache=cache, logits_to_keep=1, return_hidden=True
                )
                continue

            cache.restore(snap)
            keep = draft_ids[:n_acc].tolist() + [int(extra)]
            stopped = False
            for tok in keep:
                emit_token(int(tok))
                if hit_eos or decoder.should_stop or len(new_tokens) >= max_tokens:
                    stopped = True
                    break
            if stopped:
                break
            keep_t = _as_token_row(keep).to(model.device)
            logits, hidden = model.forward(
                keep_t, cache=cache, logits_to_keep=1, return_hidden=True
            )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    if log is not None and rounds:
        log(
            f"NextLat: draft_len={gamma} rounds={rounds} "
            f"accepted {accepted_drafts}/{proposed} drafts "
            f"({(accepted_drafts / proposed) if proposed else 0:.2f})"
        )
    prefill_s = t_prefill - t0
    decode_s = t_end - t_prefill
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
        n_new=len(new_tokens),
        prefill_s=prefill_s,
        decode_s=decode_s,
        token_ids=new_tokens,
        finish_reason=finish_reason,
        reasoning=reasoning,
        tool_calls=decoder.tool_calls,
    )


def effective_latent_rank(hidden: torch.Tensor, eps: float = 1e-12) -> float:
    """Roy & Vetterli effective rank of a ``[N, D]`` (or batched) hidden matrix."""
    x = hidden.reshape(-1, hidden.shape[-1]).float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x)
    except RuntimeError:
        s = torch.linalg.svdvals(x.cpu()).to(x.device)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    ent = -(p * p.clamp_min(eps).log()).sum()
    return float(math.exp(float(ent)))
