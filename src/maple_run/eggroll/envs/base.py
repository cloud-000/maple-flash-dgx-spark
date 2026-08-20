"""Plugin RL environments for EGGROLL.

Write a subclass (prompts, tools, step, verify). The harness owns generate,
doom scoring, population, and rank-1 fuse. JSON is data your ``sample()``
may load, not a required schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from maple_run.eggroll.tasks import Task


def function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """OpenAI-style tool schema, same shape Maple generate already parses."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required or []),
            },
        },
    }


@dataclass
class Episode:
    """Filled by the shared rollout loop, then passed to ``Env.verify``."""

    inst: Any
    messages: list[dict[str, Any]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    valid_calls: int = 0
    invalid_calls: int = 0
    recovered: bool = False
    submitted: bool = False
    n_new: int = 0
    doom: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(self.texts)


class Env(ABC):
    """One RL environment type. Subclass and ``@register``; do not touch ES."""

    kind: str = "tool"
    weight: float = 1.0
    abstract: bool = False

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def sample(self, rng) -> Any:
        """Draw one instance. Schema is yours."""

    def catalog(self) -> list[Any]:
        """Eval-all instances. Default: eight draws from ``Random(0)``."""
        import random

        rng = random.Random(0)
        return [self.sample(rng) for _ in range(8)]

    @abstractmethod
    def prompt(self, inst: Any) -> str:
        ...

    def tools(self, inst: Any) -> list | None:
        return None

    def max_new(self, inst: Any) -> int | None:
        return None

    def max_rounds(self, inst: Any) -> int | None:
        return None

    def reset(self, inst: Any) -> None:
        """Called once per episode before the first generate (tool envs)."""

    def step(self, name: str, args: dict | None) -> tuple[str, bool]:
        """Return ``(observation, valid)``. Unused for single-turn envs."""
        return f"ERROR: unknown tool {name}", False

    def done(self) -> bool:
        return False

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        """Task-specific score (not doom). Return ``(value, parts)``."""
        return 0.0, {}

    def combine(self, verify: float, doom: float) -> float:
        """Mix verifier with shared ``doom_score``. Override if verify already includes doom."""
        return 0.7 * float(verify) + 0.3 * float(doom)

    def messages(self, inst: Any) -> list[dict[str, Any]] | None:
        """Optional chat messages (system + user). Default: ``prompt()`` only."""
        return None

    def bind(self, inst: Any) -> Task:
        return Task(
            kind=self.kind,
            prompt=self.prompt(inst),
            weight=self.weight,
            tools=self.tools(inst),
            max_tokens=self.max_new(inst),
            env=self,
            inst=inst,
            messages=self.messages(inst),
        )


class SingleTurnEnv(Env):
    """Generate once, then ``verify`` the text. No tools."""

    kind = "doom"

    def tools(self, inst: Any) -> list | None:
        return None


class ToolEnv(Env):
    """Multi-turn tool loop. Implement ``reset`` / ``step`` / ``verify``."""

    kind = "tool"
    abstract = True
