"""Bundled EGGROLL environments and training recipes.

Core harness lives in ``maple_run.eggroll`` / ``maple_run.eggroll.envs``.
Put project-local env plugins in ``./eggroll_envs`` (or ``--env-dir``) and
recipes in ``./eggroll_recipes`` (or ``--recipe-dir``).
"""

from maple_run.eggroll_envs.builtin import DoomEnv, ReasonEnv, RefusalEnv
from maple_run.eggroll_envs.ipi import NemotronIPI
from maple_run.eggroll_envs.recipes import AgentRecipe, DefaultRecipe, IPIRecipe
from maple_run.eggroll_envs.registers import RegisterEnv
from maple_run.eggroll_envs.search import ProceduralSearch

__all__ = [
    "AgentRecipe",
    "DefaultRecipe",
    "DoomEnv",
    "IPIRecipe",
    "NemotronIPI",
    "ProceduralSearch",
    "ReasonEnv",
    "RefusalEnv",
    "RegisterEnv",
]
