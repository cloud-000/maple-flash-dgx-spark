"""Bundled training recipes (compositions of ``eggroll_envs``).

The registry and ``TrainingRecipe`` protocol live in ``maple_run.eggroll.recipes``.
Register extra recipes from ``./eggroll_recipes`` or ``--recipe-dir``.
"""

from __future__ import annotations

from typing import Any

from maple_run.eggroll.recipes import TrainingRecipe, register_recipe
from maple_run.eggroll_envs.builtin import DoomEnv, ReasonEnv, RefusalEnv
from maple_run.eggroll_envs.ipi import NemotronIPI
from maple_run.eggroll_envs.registers import RegisterEnv
from maple_run.eggroll_envs.search import ProceduralSearch


@register_recipe
class DefaultRecipe(TrainingRecipe):
    """Refusal + doom + reasoning + register machine (the original 21-prompt mix)."""

    name = "default"

    def environments(self) -> list:
        return [RefusalEnv(), DoomEnv(), ReasonEnv(), RegisterEnv()]


@register_recipe
class AgentRecipe(TrainingRecipe):
    """Multi-step tool use: registers + frozen search."""

    name = "agent"

    def defaults(self) -> dict[str, Any]:
        out = super().defaults()
        out["max_tokens"] = 512
        out["max_tool_rounds"] = 8
        return out

    def environments(self) -> list:
        return [RegisterEnv(), ProceduralSearch()]


@register_recipe
class IPIRecipe(TrainingRecipe):
    """Indirect prompt-injection (Nemotron IPI)."""

    name = "ipi"

    def defaults(self) -> dict[str, Any]:
        out = super().defaults()
        out["max_tokens"] = 512
        out["max_tool_rounds"] = 8
        return out

    def environments(self) -> list:
        return [NemotronIPI()]
