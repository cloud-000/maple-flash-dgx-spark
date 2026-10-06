"""CUDA graph capture and runners for single-sequence and bucketed batch decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from maple_run.chat import stop_token_ids
from maple_run.kernels.sampler import fused_sample, fused_sample_batched
from maple_run.sample import _rng_restore, _rng_snapshot, sample_next

_GRAPH_WARMUP = 3
_GRAPH_CHECK_TOKENS = 8
_DECODE_BUCKETS = (1, 2, 4, 8, 16, 32)


def _log_print(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class _DecodeGraph:
    graph: torch.cuda.CUDAGraph
    token_ids: torch.Tensor
    logits: torch.Tensor
    uniform: torch.Tensor | None = None
    next_id: torch.Tensor | None = None


def _next_token(
    logits: torch.Tensor,
    *,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None,
    sampler_ws: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    last = logits[0, -1]
    if greedy:
        return last.argmax()
    if sampler_ws is not None:
        u = torch.rand((), device=last.device, generator=generator)
        return fused_sample(
            last,
            u,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            workspace=sampler_ws,
        )
    return sample_next(
        last,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
    )


def _capture_decode_graph(
    model,
    cache,
    token_ids: torch.Tensor,
    *,
    sample: tuple | None = None,
) -> _DecodeGraph:
    """Capture ``forward(token_ids)`` on ``cache``. Caller restores KV afterwards.

    ``sample`` is ``(workspace, temperature, top_p, top_k)`` to also capture
    ``fused_sample`` so the two sampler launches sit inside the replay instead
    of paying eager launch latency between tokens. ``uniform`` is a graph input
    copied each replay so ``--seed`` stays on the same generator as eager.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    logits = None
    uniform = None
    next_id = None
    with torch.cuda.stream(stream):
        for _ in range(_GRAPH_WARMUP):
            logits = model.forward(token_ids, cache=cache, logits_to_keep=1)
            if sample is not None:
                ws, temperature, top_p, top_k = sample
                fused_sample(
                    logits[0, -1],
                    torch.rand((), device=token_ids.device),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    workspace=ws,
                )
        graph = torch.cuda.CUDAGraph()
        if sample is not None:
            ws, temperature, top_p, top_k = sample
            uniform = torch.empty((), device=token_ids.device)
            uniform.uniform_()
            with torch.cuda.graph(graph):
                logits = model.forward(token_ids, cache=cache, logits_to_keep=1)
                next_id = fused_sample(
                    logits[0, -1],
                    uniform,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    workspace=ws,
                )
        else:
            with torch.cuda.graph(graph):
                logits = model.forward(token_ids, cache=cache, logits_to_keep=1)
    torch.cuda.current_stream().wait_stream(stream)
    if logits is None:
        raise RuntimeError("CUDA graph capture did not produce logits.")
    return _DecodeGraph(
        graph=graph,
        token_ids=token_ids,
        logits=logits,
        uniform=uniform,
        next_id=next_id,
    )


def _decode_n(
    *,
    n: int,
    logits: torch.Tensor,
    replay,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None,
    stop_ids: set[int],
    sampler_ws: tuple[torch.Tensor, ...] | None = None,
) -> list[int]:
    ids: list[int] = []
    for i in range(n):
        next_t = _next_token(
            logits,
            greedy=greedy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
            sampler_ws=sampler_ws,
        )
        ids.append(int(next_t.item()))
        if stop_ids and ids[-1] in stop_ids:
            break
        if i + 1 < n:
            logits = replay(next_t)
    return ids


def _try_decode_graph(
    model,
    cache,
    prefill_logits: torch.Tensor,
    *,
    greedy: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None,
    sampler_ws: tuple[torch.Tensor, ...] | None = None,
    log: Callable[[str], None] = _log_print,
) -> _DecodeGraph | None:
    """Capture a decode graph; keep it only if replay matches eager token ids."""
    snap = cache.snapshot()
    token_ids = torch.zeros(1, 1, dtype=torch.long, device=model.device)
    try:
        captured = _capture_decode_graph(model, cache, token_ids)
    except Exception as exc:
        cache.restore(snap)
        log(f"CUDA graph capture failed ({exc}); using eager decode")
        return None
    cache.restore(snap)

    n_check = _GRAPH_CHECK_TOKENS
    rng_state = _rng_snapshot(model.device, generator)
    stop_ids = stop_token_ids(model)

    def eager_replay(next_t: torch.Tensor) -> torch.Tensor:
        return model.forward(next_t.view(1, 1), cache=cache, logits_to_keep=1)

    def graph_replay(next_t: torch.Tensor) -> torch.Tensor:
        captured.token_ids.copy_(next_t.view(1, 1))
        captured.graph.replay()
        return captured.logits

    eager_ids = _decode_n(
        n=n_check,
        logits=prefill_logits,
        replay=eager_replay,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
        stop_ids=stop_ids,
        sampler_ws=sampler_ws,
    )
    cache.restore(snap)
    _rng_restore(model.device, rng_state, generator)

    graph_ids = _decode_n(
        n=n_check,
        logits=prefill_logits,
        replay=graph_replay,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
        stop_ids=stop_ids,
        sampler_ws=sampler_ws,
    )
    cache.restore(snap)
    _rng_restore(model.device, rng_state, generator)

    if eager_ids != graph_ids:
        log(
            f"CUDA graph replay diverged (eager {eager_ids[:8]} vs graph "
            f"{graph_ids[:8]}); using eager decode"
        )
        return None
    log("CUDA graph decode: replay matches eager sampled ids")
    if sampler_ws is not None and not greedy:
        try:
            captured = _capture_decode_graph(
                model,
                cache,
                token_ids,
                sample=(sampler_ws, temperature, top_p, top_k),
            )
            cache.restore(snap)
            log("CUDA graph decode: fused sampler captured")
        except Exception as exc:
            cache.restore(snap)
            log(
                f"CUDA graph sampler capture failed ({exc}); "
                "using forward-only graph"
            )
    return captured


def decode_buckets(max_batch: int, buckets=_DECODE_BUCKETS) -> tuple[int, ...]:
    """Graph widths to capture for a cache of ``max_batch`` rows.

    Powers of two up to the cache width, plus the cache width itself. A batch
    that shrinks as rows finish then always has a graph within 2x of it, which
    is the point of bucketing: one graph pinned at the starting width would
    keep paying for rows that have already stopped.
    """
    widths = [w for w in buckets if w < max_batch]
    widths.append(int(max_batch))
    return tuple(sorted(set(widths)))


class BucketedDecodeGraphs:
    """One captured decode graph per batch width, replayed at the narrowest fit.

    Every width shares the same input buffers, so a step is: write the live
    rows' tokens (and uniforms), replay, read the ids back. Rows past the live
    count are padding -- they run, and they write into the cache slots above
    the live ones, which is why the caller has to keep live sequences in the
    low slots (``KVCache.swap_slots``).

    Capture needs ``cache.remap`` on: the graph bakes in a kernel that reads
    the slot -> cache-row map, so compaction can permute slots between replays
    without re-capturing.
    """

    def __init__(
        self,
        model,
        cache,
        *,
        widths: tuple[int, ...],
        sample: tuple | None = None,
        log: Callable[[str], None] = _log_print,
    ):
        if not cache.remap:
            raise ValueError("BucketedDecodeGraphs needs cache.remap enabled.")
        device = model.device
        self.max_width = max(widths)
        self.widths = tuple(sorted(widths))
        self.sample = sample
        self.token_ids = torch.zeros(self.max_width, 1, dtype=torch.long, device=device)
        self.uniform = torch.rand(self.max_width, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.next_id: dict[int, torch.Tensor] = {}
        self.logits: dict[int, torch.Tensor] = {}

        snap = cache.snapshot()
        stream = torch.cuda.Stream()
        try:
            for w in self.widths:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(_GRAPH_WARMUP):
                        self._step(model, cache, w)
                torch.cuda.current_stream().wait_stream(stream)
                cache.restore(snap)
                # Each width keeps its own pool. Sharing one across the set
                # measured no smaller (208 vs 175 MB for six widths at B=16)
                # and would tie replay safety to replay order, which here is
                # whatever order the live count happens to take.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    logits, nxt = self._step(model, cache, w)
                cache.restore(snap)
                self.graphs[w] = graph
                self.logits[w] = logits
                self.next_id[w] = nxt
        finally:
            cache.restore(snap)
        log(
            f"Bucketed decode graphs: widths {list(self.widths)}"
            + (" with fused sampler" if sample is not None else " greedy")
        )

    def _step(self, model, cache, w: int):
        logits = model.forward(self.token_ids[:w], cache=cache, logits_to_keep=1)
        last = logits[:, -1, :]
        if self.sample is None:
            nxt = last.argmax(dim=-1)
        else:
            ws, temperature, top_p, top_k = self.sample
            nxt = fused_sample_batched(
                last,
                self.uniform[:w],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                workspace=ws,
            )
        return logits, nxt

    def width_for(self, live: int) -> int:
        for w in self.widths:
            if w >= live:
                return w
        raise ValueError(f"live batch {live} exceeds widest graph {self.max_width}.")

    def replay(
        self, tokens: torch.Tensor, uniform: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Step ``tokens`` ``[live, 1]``; returns the next ids ``[live]``."""
        live = int(tokens.shape[0])
        w = self.width_for(live)
        self.token_ids[:live].copy_(tokens.reshape(live, 1))
        if live < w:
            # Padding rows decode against the free slots above the live ones.
            self.token_ids[live:w].zero_()
        if uniform is not None:
            self.uniform[:live].copy_(uniform.reshape(-1)[:live])
        self.graphs[w].replay()
        return self.next_id[w][:live]

    def replay_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Step ``tokens`` ``[live, 1]``; returns last-token logits ``[live, vocab]``.

        Ignores the in-graph sample (greedy argmax or fused draw). A scheduler
        that mixes per-request sampling reads these logits and samples outside.
        """
        live = int(tokens.shape[0])
        w = self.width_for(live)
        self.token_ids[:live].copy_(tokens.reshape(live, 1))
        if live < w:
            self.token_ids[live:w].zero_()
        self.graphs[w].replay()
        return self.logits[w][:live].reshape(live, -1)


def _try_batched_decode_graphs(
    model,
    cache,
    pending: list[int],
    *,
    sample: tuple | None,
    buckets: tuple[int, ...] | None = None,
    log: Callable[[str], None] = _log_print,
) -> BucketedDecodeGraphs | None:
    """Capture bucketed graphs and keep them only if replay matches eager."""
    live = len(pending)
    widths = decode_buckets(cache.batch) if buckets is None else tuple(sorted(buckets))
    try:
        runner = BucketedDecodeGraphs(
            model, cache, widths=widths, sample=sample, log=log
        )
    except Exception as exc:
        log(f"Bucketed graph capture failed ({exc}); using eager batched decode")
        return None

    # Same check the single-stream path makes: the same few steps, eager and
    # replayed, from the same cache state. The uniforms are fixed rather than
    # redrawn so a sampled comparison is about the graph, not RNG ordering.
    snap = cache.snapshot()
    n_check = min(_GRAPH_CHECK_TOKENS, 4)
    start_tok = torch.tensor(pending, device=model.device, dtype=torch.long).view(live, 1)
    fixed_u = [
        torch.rand(live, device=model.device) if sample is not None else None
        for _ in range(n_check)
    ]

    def eager_step(tokens, u):
        last = model.forward(tokens, cache=cache, logits_to_keep=1)[:, -1, :]
        if sample is None:
            return last.argmax(dim=-1)
        ws, temperature, top_p, top_k = sample
        return fused_sample_batched(
            last, u, temperature=temperature, top_p=top_p, top_k=top_k, workspace=ws
        )

    def run(step_fn):
        ids, tok = [], start_tok
        for u in fixed_u:
            tok = step_fn(tok, u).reshape(-1, 1)
            ids.append(tok.reshape(-1).tolist())
        return ids

    eager_ids = run(eager_step)
    cache.restore(snap)
    graph_ids = run(lambda tokens, u: runner.replay(tokens, u))
    cache.restore(snap)

    if eager_ids != graph_ids:
        log(
            f"Bucketed graph replay diverged (eager {eager_ids[0][:4]} vs graph "
            f"{graph_ids[0][:4]}); using eager batched decode"
        )
        return None
    log("Bucketed decode graphs: replay matches eager ids")
    return runner
