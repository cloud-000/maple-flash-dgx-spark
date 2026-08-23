"""Register-machine environment (write / read / add / submit)."""

from __future__ import annotations

from typing import Any

from maple_run.eggroll.envs.base import Episode, ToolEnv
from maple_run.eggroll.envs.machine import RegisterMachine
from maple_run.eggroll.envs.registry import register
from maple_run.eggroll.rewards import tool_episode_score
from maple_run.eggroll.tools import REGISTER_TOOLS

_SYSTEM = (
    "You are a register machine. Use the write/read/add/submit tools. "
    "Do not call the same tool with the same arguments twice. "
    "When the target is reached, submit the register."
)

_CATALOG = (
    {
        "prompt": f"{_SYSTEM} Set register acc to 3, add 4, then submit acc. The verifier expects 7.",
        "initial_state": {},
        "target": {"acc": 7},
        "inject_error": False,
    },
    {
        "prompt": f"{_SYSTEM} write x=10, add -3 to x, read x, submit x. The verifier expects 7.",
        "initial_state": {},
        "target": {"x": 7},
        "inject_error": False,
    },
    {
        "prompt": (
            f"{_SYSTEM} write score=1, add 1 three times, submit score. "
            "The verifier expects 4."
        ),
        "initial_state": {},
        "target": {"score": 4},
        "inject_error": False,
    },
    {
        "prompt": (
            f"{_SYSTEM} write n=2. If a tool errors, fix the arguments and continue. "
            "Add 5, then submit n. The verifier expects 7."
        ),
        "initial_state": {},
        "target": {"n": 7},
        "inject_error": True,
    },
)


@register
class RegisterEnv(ToolEnv):
    """Default agentic env: named registers + verifier target."""

    kind = "tool"

    def catalog(self) -> list[Any]:
        return [
            {
                "prompt": row["prompt"],
                "initial_state": dict(row["initial_state"]),
                "target": dict(row["target"]),
                "inject_error": bool(row["inject_error"]),
            }
            for row in _CATALOG
        ]

    def sample(self, rng) -> dict[str, Any]:
        return rng.choice(self.catalog())

    def prompt(self, inst: Any) -> str:
        return str(inst["prompt"])

    def meta(self, inst: Any) -> dict[str, Any]:
        return {
            "inject_error": bool(inst.get("inject_error")),
            "target": dict(inst.get("target") or {}),
        }

    def tools(self, inst: Any) -> list:
        return list(REGISTER_TOOLS)

    def initial_state(self, inst: Any) -> RegisterMachine:
        return RegisterMachine(
            dict(inst.get("initial_state") or {}),
            dict(inst.get("target") or {}),
            inject_error=bool(inst.get("inject_error")),
        )

    def done(self, state: Any = None) -> bool:
        machine = state if state is not None else getattr(self, "_state", None)
        return bool(machine and machine.submitted)

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        machine = episode.state if episode.state is not None else getattr(self, "_state", None)
        verifier = bool(machine and machine.verifier_ok())
        submitted = bool(machine and machine.submitted)
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
