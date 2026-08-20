"""EGGROLL train loop on packed Maple."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from maple_run.eggroll.es import EggrollRuntime, centered_advantages
from maple_run.eggroll.perturb import DEFAULT_MODULES, DEFAULT_RANK, DEFAULT_R_MAX, DEFAULT_SIGMA
from maple_run.eggroll.rollout import (
    EpisodeResult,
    GenerateFn,
    needs_agent_loop,
    run_single_turn_batch,
    run_task,
)
from maple_run.eggroll.tasks import Task, default_suite, sample_batch
from maple_run.generate import DEFAULT_TOP_K, DEFAULT_TOP_P, load_packed, warmup_model

DEFAULT_TRACE_CHARS = 2000


@dataclass
class TrainConfig:
    model_dir: str
    output_dir: str
    resume: str | None = None
    steps: int = 50
    population: int = 16
    rank: int = DEFAULT_RANK
    r_max: int = DEFAULT_R_MAX
    sigma: float = DEFAULT_SIGMA
    lr: float = 0.001
    seed: int = 0
    max_tokens: int = 256
    max_tool_rounds: int = 6
    prompts_per_step: int = 4
    max_batch: int = 8
    temperature: float = 0.0
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    modules: tuple[str, ...] = DEFAULT_MODULES
    eval_only: bool = False
    save_every: int = 10
    env_names: str | None = None
    env_dirs: tuple[str, ...] = ()
    trace_text: bool = False
    trace_chars: int = DEFAULT_TRACE_CHARS
    log: Callable[[str], None] = print


@dataclass
class StepStats:
    step: int
    mean_fitness: float
    pair_advantages: list[float]
    by_kind: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    pair_fitness: list[list[float]] = field(default_factory=list)
    prefill_s: float = 0.0
    decode_s: float = 0.0
    n_new: int = 0
    n_episodes: int = 0
    residual_rank_max: int = 0

    def to_record(self) -> dict:
        return {
            "step": self.step,
            "mean_fitness": self.mean_fitness,
            "by_kind": self.by_kind,
            "pair_advantages": self.pair_advantages,
            "pair_fitness": self.pair_fitness,
            "elapsed_s": self.elapsed_s,
            "prefill_s": self.prefill_s,
            "decode_s": self.decode_s,
            "n_new": self.n_new,
            "n_episodes": self.n_episodes,
            "decode_tok_s": (self.n_new / self.decode_s) if self.decode_s > 0 else 0.0,
            "residual_rank_max": self.residual_rank_max,
        }


def _log(cfg: TrainConfig, msg: str) -> None:
    cfg.log(msg)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _clip(text: str, n: int) -> str:
    if n <= 0 or len(text) <= n:
        return text
    return text[:n] + "…"


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _config_public(cfg: TrainConfig, *, suite: list) -> dict:
    kinds = sorted({getattr(t, "kind", type(t).__name__) for t in suite})
    return {
        "model_dir": cfg.model_dir,
        "output_dir": cfg.output_dir,
        "resume": cfg.resume,
        "steps": cfg.steps,
        "population": cfg.population,
        "rank": cfg.rank,
        "r_max": cfg.r_max,
        "sigma": cfg.sigma,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "max_tokens": cfg.max_tokens,
        "max_tool_rounds": cfg.max_tool_rounds,
        "prompts_per_step": cfg.prompts_per_step,
        "max_batch": cfg.max_batch,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "modules": list(cfg.modules),
        "eval_only": cfg.eval_only,
        "save_every": cfg.save_every,
        "env_names": cfg.env_names,
        "env_dirs": list(cfg.env_dirs),
        "trace_text": cfg.trace_text,
        "trace_chars": cfg.trace_chars,
        "suite_size": len(suite),
        "suite_kinds": kinds,
    }


def _timing_totals(episodes: list[EpisodeResult]) -> tuple[float, float, int]:
    """Sum generate timings once per packed batch; serial episodes count fully."""
    seen: set[int] = set()
    prefill = decode = 0.0
    n_new = 0
    for ep in episodes:
        n_new += ep.n_new
        if ep.batch_id is not None:
            if ep.batch_id in seen:
                continue
            seen.add(ep.batch_id)
        prefill += ep.prefill_s
        decode += ep.decode_s
    return prefill, decode, n_new


def _episode_record(
    ep: EpisodeResult,
    *,
    cfg: TrainConfig,
    phase: str,
    step: int | None,
    pair: int | None,
    sign: float | None,
    seed: int | None,
    task_i: int,
) -> dict:
    include_text = cfg.trace_text or cfg.trace_chars > 0
    limit = None if cfg.trace_text else cfg.trace_chars
    rec: dict = {
        "phase": phase,
        "step": step,
        "pair": pair,
        "sign": sign,
        "member": None if pair is None or sign is None else 2 * pair + (0 if sign > 0 else 1),
        "task_i": task_i,
        "kind": ep.kind,
        "fitness": ep.fitness,
        "parts": ep.parts,
        "prompt_len": ep.prompt_len,
        "n_new": ep.n_new,
        "prefill_s": ep.prefill_s,
        "decode_s": ep.decode_s,
        "decode_tok_s": (ep.n_new / ep.decode_s) if ep.decode_s > 0 else 0.0,
        "finish_reason": ep.finish_reason,
        "tool_calls": ep.tool_calls,
        "rounds": ep.rounds,
        "timing_shared": ep.batch_id is not None,
        "seed": seed,
    }
    if ep.turns:
        rec["turns"] = ep.turns
    task = ep.task
    if task is not None:
        rec["weight"] = task.weight
        rec["should_refuse"] = task.should_refuse
        rec["gold"] = task.gold
        rec["inject_error"] = task.inject_error
        if task.env is not None:
            rec["env"] = type(task.env).__name__
        if include_text:
            rec["prompt"] = task.prompt if limit is None else _clip(task.prompt, limit)
    if include_text:
        rec["text"] = ep.text if limit is None else _clip(ep.text, limit)
        rec["reasoning"] = ep.reasoning if limit is None else _clip(ep.reasoning, limit)
    return rec


def _write_episodes(
    path: Path,
    episodes: list[EpisodeResult],
    *,
    cfg: TrainConfig,
    phase: str,
    step: int | None,
    pair: int | None,
    sign: float | None,
    seed: int | None,
) -> None:
    for task_i, ep in enumerate(episodes):
        _append_jsonl(
            path,
            _episode_record(
                ep,
                cfg=cfg,
                phase=phase,
                step=step,
                pair=pair,
                sign=sign,
                seed=seed,
                task_i=task_i,
            ),
        )


def evaluate_tasks(
    model,
    tokenizer,
    tasks: list,
    cfg: TrainConfig,
    *,
    generate_fn: GenerateFn | None = None,
    seed: int | None = None,
) -> tuple[float, dict[str, float], list[EpisodeResult]]:
    if generate_fn is not None or cfg.max_batch <= 1:
        episodes = [
            run_task(
                model,
                tokenizer,
                task,
                max_tokens=cfg.max_tokens,
                max_tool_rounds=cfg.max_tool_rounds,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                seed=seed,
                generate_fn=generate_fn,
            )
            for task in tasks
        ]
    else:
        single = [t for t in tasks if not needs_agent_loop(t)]
        agent = [t for t in tasks if needs_agent_loop(t)]
        episodes = []
        for i in range(0, len(single), cfg.max_batch):
            chunk = single[i : i + cfg.max_batch]
            episodes.extend(
                run_single_turn_batch(
                    model,
                    tokenizer,
                    chunk,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    seed=seed,
                )
            )
        for task in agent:
            episodes.append(
                run_task(
                    model,
                    tokenizer,
                    task,
                    max_tokens=cfg.max_tokens,
                    max_tool_rounds=cfg.max_tool_rounds,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    seed=seed,
                )
            )
    by_kind: dict[str, list[float]] = {}
    for ep in episodes:
        by_kind.setdefault(ep.kind, []).append(ep.fitness)
    means = {k: _mean(v) for k, v in by_kind.items()}
    return _mean([ep.fitness for ep in episodes]), means, episodes


def expand_suite(suite: list) -> list[Task]:
    """Frozen tasks pass through; env plugins expand via ``catalog()``."""
    out: list[Task] = []
    for item in suite:
        if isinstance(item, Task):
            out.append(item)
        else:
            out.extend(item.bind(inst) for inst in item.catalog())
    return out


def resolve_suite(cfg: TrainConfig, suite: list | None) -> list:
    if suite is not None:
        return suite
    from maple_run.eggroll.envs import get_envs, load_plugins

    load_plugins(list(cfg.env_dirs) if cfg.env_dirs else None)
    if cfg.env_names:
        names = [n.strip() for n in cfg.env_names.split(",") if n.strip()]
        return get_envs(names)
    return default_suite()


def evaluate_baseline(
    model,
    tokenizer,
    cfg: TrainConfig,
    *,
    suite: list | None = None,
    generate_fn: GenerateFn | None = None,
) -> dict[str, float]:
    suite = expand_suite(suite if suite is not None else default_suite())
    fit, by_kind, _ = evaluate_tasks(
        model, tokenizer, suite, cfg, generate_fn=generate_fn, seed=cfg.seed
    )
    out = {"mean": fit, **{f"kind/{k}": v for k, v in by_kind.items()}}
    return out


def _load_model(cfg: TrainConfig):
    _log(cfg, f"Loading packed model from {cfg.model_dir}")
    model, tokenizer = load_packed(cfg.model_dir, flash_head=False)
    warmup_model(model, flash_head=False)
    return model, tokenizer


def _attach_runtime(model, cfg: TrainConfig) -> EggrollRuntime:
    if cfg.resume:
        rt = EggrollRuntime.load(cfg.resume)
        rt.attach_and_restore(model)
        _log(cfg, f"Restored EGGROLL adapters from {cfg.resume}")
        return rt
    rt = EggrollRuntime(
        rank=cfg.rank,
        r_max=cfg.r_max,
        sigma=cfg.sigma,
        modules=cfg.modules,
        base_seed=cfg.seed,
    )
    rt.attach(model)
    return rt


def train(
    cfg: TrainConfig,
    *,
    model=None,
    tokenizer=None,
    suite: list | None = None,
    generate_fn: GenerateFn | None = None,
) -> list[StepStats]:
    if cfg.population < 2 or cfg.population % 2:
        raise ValueError("population must be even and >= 2 (antithetic pairs)")
    if cfg.max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    if cfg.trace_chars < 0:
        raise ValueError("trace_chars must be >= 0")
    suite = resolve_suite(cfg, suite)
    if model is None or tokenizer is None:
        model, tokenizer = _load_model(cfg)
    runtime = _attach_runtime(model, cfg)
    _log(
        cfg,
        f"decode batch {cfg.max_batch} "
        f"(single-turn packed GEMM; tool episodes serial)",
    )
    rng = random.Random(cfg.seed)
    n_pairs = cfg.population // 2
    history: list[StepStats] = []
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    episodes_path = output / "episodes.jsonl"
    if not cfg.resume:
        history_path.write_text("")
        episodes_path.write_text("")
    (output / "config.json").write_text(
        json.dumps(_config_public(cfg, suite=suite), indent=2) + "\n"
    )

    if cfg.eval_only:
        expanded = expand_suite(suite)
        fit, by_kind, episodes = evaluate_tasks(
            model, tokenizer, expanded, cfg, generate_fn=generate_fn, seed=cfg.seed
        )
        metrics = {"mean": fit, **{f"kind/{k}": v for k, v in by_kind.items()}}
        _log(cfg, f"baseline {json.dumps(metrics)}")
        (output / "baseline.json").write_text(json.dumps(metrics, indent=2) + "\n")
        _write_episodes(
            episodes_path,
            episodes,
            cfg=cfg,
            phase="eval",
            step=None,
            pair=None,
            sign=None,
            seed=cfg.seed,
        )
        return history

    for step in range(cfg.steps):
        t0 = time.perf_counter()
        batch = sample_batch(suite, cfg.prompts_per_step, rng)
        raw: list[float] = []
        kind_acc: dict[str, list[float]] = {}
        step_episodes: list[EpisodeResult] = []
        pair_fitness: list[list[float]] = []
        for pair in range(n_pairs):
            pair_fits = []
            for sign in (1.0, -1.0):
                runtime.set_member(step, pair, sign=sign)
                member_seed = cfg.seed + step * 1009 + pair
                fit, by_kind, episodes = evaluate_tasks(
                    model,
                    tokenizer,
                    batch,
                    cfg,
                    generate_fn=generate_fn,
                    seed=member_seed,
                )
                _write_episodes(
                    episodes_path,
                    episodes,
                    cfg=cfg,
                    phase="train",
                    step=step,
                    pair=pair,
                    sign=sign,
                    seed=member_seed,
                )
                pair_fits.append(fit)
                raw.append(fit)
                step_episodes.extend(episodes)
                for k, v in by_kind.items():
                    kind_acc.setdefault(k, []).append(v)
            pair_fitness.append(pair_fits)
        runtime.clear_member()
        shaped = centered_advantages(raw)
        pair_adv = [
            shaped[2 * i] - shaped[2 * i + 1] for i in range(n_pairs)
        ]
        runtime.fuse_antithetic(
            step, pair_adv, lr=cfg.lr, population=cfg.population
        )
        prefill_s, decode_s, n_new = _timing_totals(step_episodes)
        rank_max = max(
            (adapter.residual_rank for adapter in runtime.adapters.values()),
            default=0,
        )
        stats = StepStats(
            step=step,
            mean_fitness=_mean(raw),
            pair_advantages=pair_adv,
            by_kind={k: _mean(v) for k, v in kind_acc.items()},
            elapsed_s=time.perf_counter() - t0,
            pair_fitness=pair_fitness,
            prefill_s=prefill_s,
            decode_s=decode_s,
            n_new=n_new,
            n_episodes=len(step_episodes),
            residual_rank_max=rank_max,
        )
        history.append(stats)
        rec = stats.to_record()
        _append_jsonl(history_path, rec)
        tok = f" {rec['decode_tok_s']:.1f} tok/s" if decode_s > 0 else ""
        _log(
            cfg,
            f"step {step} fit={stats.mean_fitness:.3f} "
            + " ".join(f"{k}={v:.3f}" for k, v in stats.by_kind.items())
            + f" {stats.elapsed_s:.1f}s{tok}",
        )
        if cfg.save_every and (step + 1) % cfg.save_every == 0:
            runtime.save(output)

    runtime.save(output)
    _log(cfg, f"saved EGGROLL adapters to {output}")
    return history
