"""Continuous batch decoding and slot management for multi-sequence inference."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from maple_run.chat import stop_token_ids
from maple_run.graphs import BucketedDecodeGraphs, _log_print, _try_batched_decode_graphs
from maple_run.kernels.sampler import (
    MAX_TOP_K,
    batched_sampler_workspace,
    fused_sample,
    fused_sample_batched,
)
from maple_run.sample import can_fuse_sampler, is_greedy


@dataclass
class BatchedResult:
    """Per-row token ids plus the timings for an aggregate tok/s."""

    token_ids: list[list[int]]
    prompt_lens: list[int]
    finish_reasons: list[str]
    prefill_s: float
    decode_s: float
    steps: int
    # Graph width replayed at each decode step (empty when decoding eager).
    widths: list[int] = field(default_factory=list)

    @property
    def n_new(self) -> int:
        return sum(len(t) for t in self.token_ids)

    @property
    def decode_tok_s(self) -> float:
        """Aggregate: every row's tokens over the shared wall clock."""
        return self.n_new / self.decode_s if self.decode_s > 0 else 0.0


def _first_batched_id(logits, greedy, temperature, top_p, top_k, gen):
    """Pick the token that follows a prompt, from its batch-1 prefill logits."""
    last = logits[0, -1]
    if greedy:
        return last.argmax()
    return fused_sample(
        last,
        torch.rand((), device=last.device, generator=gen),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


@torch.inference_mode()
def batched_generate(
    model,
    prompts,
    *,
    max_tokens: int = 128,
    max_new: list[int] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    stop_ids: set[int] | None = None,
    cache=None,
    graphs: bool = False,
    buckets: tuple[int, ...] | None = None,
    runner: BucketedDecodeGraphs | None = None,
    log: Callable[[str], None] = _log_print,
) -> BatchedResult:
    """Decode several sequences in one batch.

    Prefill stays batch-1 -- each prompt is run into its own cache slot via
    :meth:`KVCache.prefill_slot` -- and only decode goes wide, which is where
    the packed weights get read once for the whole batch instead of once per
    row.

    Greedy by default (``temperature <= 0`` or ``top_k == 1``); otherwise the
    batched fused sampler draws one id per row from a per-row uniform.

    With ``graphs``, decode replays a captured graph chosen by the *live* row
    count rather than the starting batch. Finished rows are swapped out of the
    low slots so the live ones stay contiguous, so a batch of 32 that is down
    to 5 live rows replays the width-8 graph instead of paying for 27 rows
    that already stopped. Replay is checked against eager first and falls back
    if it disagrees, as the single-stream path does. ``buckets`` overrides the
    captured widths; the default is :func:`decode_buckets`.

    ``max_new`` is an optional per-row token cap. The decode horizon is
    ``max(max_new)`` (or ``max_tokens`` when omitted); a row retires when it
    hits its own cap rather than waiting for the longest companion.
    """
    device = model.device
    rows = [
        torch.as_tensor(p, device=device, dtype=torch.long).reshape(-1)
        for p in prompts
    ]
    if not rows:
        raise ValueError("batched_generate needs at least one prompt.")
    bsz = len(rows)
    prompt_lens = [int(t.numel()) for t in rows]
    if max_new is None:
        caps = [int(max_tokens)] * bsz
    else:
        if len(max_new) != bsz:
            raise ValueError(f"max_new length {len(max_new)} != {bsz} prompts.")
        caps = [int(c) for c in max_new]
        if any(c < 1 for c in caps):
            raise ValueError("each max_new row must be >= 1")
    horizon = max(caps)
    if stop_ids is None:
        stop_ids = stop_token_ids(model)
    if cache is None:
        cache = model.make_cache(
            max_len=max(prompt_lens) + max(horizon, 1), batch=bsz
        )
    elif cache.batch < bsz:
        raise ValueError(f"cache batch {cache.batch} < {bsz} prompts.")
    elif cache.max_len < max(prompt_lens) + max(horizon, 1):
        # Retired rows keep stepping in their bucket, so every slot advances
        # for the whole run and the cache has to cover the longest one. The
        # decode K/V store indexes by seqlen without a bound check.
        raise ValueError(
            f"cache max_len {cache.max_len} < longest prompt "
            f"{max(prompt_lens)} + {horizon} tokens."
        )
    else:
        cache.reset_identity()

    greedy = is_greedy(temperature, top_k)
    if not greedy and not can_fuse_sampler(temperature, top_p, top_k):
        raise ValueError(
            f"batched decode needs 0 < top_k <= {MAX_TOP_K} and 0 < top_p <= 1."
        )
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pending = [0] * bsz
    for b, row in enumerate(rows):
        with cache.prefill_slot(b):
            logits = model.forward(row.view(1, -1), cache=cache, logits_to_keep=1)
        pending[b] = int(_first_batched_id(logits, greedy, temperature, top_p, top_k, gen))
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # Slots get permuted as rows finish, so the kernels have to go through the
    # slot -> cache-row map from here on.
    cache.remap = True
    sample_ws = None
    if not greedy:
        # Sized at the cache width, not the live batch: the widest captured
        # graph runs cache.batch rows, and a narrower replay slices this.
        sample_ws = batched_sampler_workspace(
            int(model.config["vocab_size"]), int(top_k), cache.batch, device
        )
    if graphs and runner is None:
        runner = _try_batched_decode_graphs(
            model,
            cache,
            pending,
            sample=None if greedy else (sample_ws, temperature, top_p, top_k),
            buckets=buckets,
            log=log,
        )

    def step(tokens: torch.Tensor) -> torch.Tensor:
        live = int(tokens.shape[0])
        u = None if greedy else torch.rand(live, device=device, generator=gen)
        if runner is not None:
            return runner.replay(tokens, u)
        out_logits = model.forward(tokens, cache=cache, logits_to_keep=1)[:, -1, :]
        if greedy:
            return out_logits.argmax(dim=-1)
        return fused_sample_batched(
            out_logits,
            u,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            workspace=sample_ws,
        )

    out: list[list[int]] = [[] for _ in range(bsz)]
    reasons = ["length"] * bsz
    owner = list(range(bsz))  # slot -> output row
    live = bsz
    steps = 0
    widths: list[int] = []
    t0 = time.perf_counter()
    for i in range(int(horizon)):
        finished = []
        for slot in range(live):
            b = owner[slot]
            out[b].append(pending[slot])
            if pending[slot] in stop_ids:
                reasons[b] = "stop"
                finished.append(slot)
            elif len(out[b]) >= caps[b]:
                reasons[b] = "length"
                finished.append(slot)
        if i + 1 >= int(horizon):
            break
        # Descending, so a slot pulled down from the top is never one we have
        # already retired.
        for slot in reversed(finished):
            last = live - 1
            if slot != last:
                cache.swap_slots(slot, last)
                owner[slot], owner[last] = owner[last], owner[slot]
                pending[slot], pending[last] = pending[last], pending[slot]
            live -= 1
        if live == 0:
            break
        tokens = torch.tensor(
            pending[:live], device=device, dtype=torch.long
        ).view(live, 1)
        nxt = step(tokens)
        pending[:live] = [int(x) for x in nxt.tolist()]
        steps += 1
        if runner is not None:
            width = runner.width_for(live)
            widths.append(width)
            cache.advance_host(1, width)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    return BatchedResult(
        token_ids=out,
        prompt_lens=prompt_lens,
        finish_reasons=reasons,
        prefill_s=prefill_s,
        decode_s=decode_s,
        steps=steps,
        widths=widths,
    )
