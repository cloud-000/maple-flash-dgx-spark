"""Register-machine environment (write / read / add / submit)."""

from __future__ import annotations

from typing import Any

from maple_run.eggroll.envs.base import Episode, ToolEnv
from maple_run.eggroll.envs.machine import RegisterMachine
from maple_run.eggroll.envs.registry import register
from maple_run.eggroll.rewards import tool_episode_score
from maple_run.eggroll.tasks import AGENT_TOOLS, tool_tasks


@register
class RegisterEnv(ToolEnv):
    """Default agentic env: named registers + verifier target."""

    kind = "tool"

    def __init__(self) -> None:
        super().__init__()
        self._machine: RegisterMachine | None = None

    def catalog(self) -> list[Any]:
        return [_inst_from_task(t) for t in tool_tasks()]

    def sample(self, rng) -> dict[str, Any]:
        return rng.choice(self.catalog())

    def prompt(self, inst: Any) -> str:
        return str(inst["prompt"])

    def tools(self, inst: Any) -> list:
        return list(inst.get("tools") or AGENT_TOOLS)

    def reset(self, inst: Any) -> None:
        self._machine = RegisterMachine(
            dict(inst.get("initial_state") or {}),
            dict(inst.get("target") or {}),
            inject_error=bool(inst.get("inject_error")),
        )

    def step(self, name: str, args: dict | None) -> tuple[str, bool]:
        assert self._machine is not None
        return self._machine.execute(name, args)

    def done(self) -> bool:
        return bool(self._machine and self._machine.submitted)

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        machine = self._machine
        verifier = bool(machine and machine.verifier_ok())
        submitted = bool(machine and machine.submitted) or episode.submitted
        fit = tool_episode_score(
            valid_calls=episode.valid_calls,
            invalid_calls=episode.invalid_calls,
            recovered=episode.recovered,
            verifier_ok=verifier,
            doom=episode.doom,
            submitted=submitted,
        )
        return fit, {
            "tools": fit,
            "verifier": 1.0 if verifier else 0.0,
            "recovered": 1.0 if episode.recovered else 0.0,
        }

    def combine(self, verify: float, doom: float) -> float:
        return float(verify)


def _inst_from_task(task) -> dict[str, Any]:
    return {
        "prompt": task.prompt,
        "tools": task.tools,
        "initial_state": dict(task.initial_state),
        "target": dict(task.target or {}),
        "inject_error": task.inject_error,
        "max_tokens": task.max_tokens,
    }
