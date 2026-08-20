"""Builtin single-turn envs wrapping the original prompt catalogs."""

from __future__ import annotations

from typing import Any

from maple_run.eggroll.envs.base import Episode, SingleTurnEnv
from maple_run.eggroll.envs.registry import register
from maple_run.eggroll.rewards import reasoning_score, refusal_score
from maple_run.eggroll.tasks import doom_tasks, reasoning_tasks, refusal_tasks


@register
class RefusalEnv(SingleTurnEnv):
    kind = "refusal"

    def catalog(self) -> list[Any]:
        return [
            {"prompt": t.prompt, "should_refuse": t.should_refuse, "max_tokens": t.max_tokens}
            for t in refusal_tasks()
        ]

    def sample(self, rng) -> dict[str, Any]:
        return rng.choice(self.catalog())

    def prompt(self, inst: Any) -> str:
        return str(inst["prompt"])

    def max_new(self, inst: Any) -> int | None:
        return inst.get("max_tokens")

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        ref = refusal_score(episode.text, should_refuse=bool(inst.get("should_refuse")))
        return ref, {"refusal": ref}


@register
class DoomEnv(SingleTurnEnv):
    kind = "doom"

    def catalog(self) -> list[Any]:
        return [{"prompt": t.prompt, "max_tokens": t.max_tokens} for t in doom_tasks()]

    def sample(self, rng) -> dict[str, Any]:
        return rng.choice(self.catalog())

    def prompt(self, inst: Any) -> str:
        return str(inst["prompt"])

    def max_new(self, inst: Any) -> int | None:
        return inst.get("max_tokens")

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        return 0.0, {}

    def combine(self, verify: float, doom: float) -> float:
        return float(doom)


@register
class ReasonEnv(SingleTurnEnv):
    kind = "reason"

    def catalog(self) -> list[Any]:
        return [
            {"prompt": t.prompt, "gold": t.gold, "max_tokens": t.max_tokens}
            for t in reasoning_tasks()
        ]

    def sample(self, rng) -> dict[str, Any]:
        return rng.choice(self.catalog())

    def prompt(self, inst: Any) -> str:
        return str(inst["prompt"])

    def max_new(self, inst: Any) -> int | None:
        return inst.get("max_tokens")

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        gold = inst.get("gold")
        score = reasoning_score(episode.text, gold, reasoning=episode.reasoning)
        return score, {"reason": score}

    def combine(self, verify: float, doom: float) -> float:
        return 0.8 * float(verify) + 0.2 * float(doom)
