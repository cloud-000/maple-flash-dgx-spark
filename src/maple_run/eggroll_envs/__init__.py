"""Bundled EGGROLL environment extensions (datasets / concrete envs).

Core harness lives in ``maple_run.eggroll`` / ``maple_run.eggroll.envs``.
Put project-local plugins in ``./eggroll_envs`` (auto-loaded) or ``--env-dir``.
"""

from maple_run.eggroll_envs.builtin import DoomEnv, ReasonEnv, RefusalEnv
from maple_run.eggroll_envs.ipi import NemotronIPI
from maple_run.eggroll_envs.registers import RegisterEnv
from maple_run.eggroll_envs.search import ProceduralSearch

__all__ = [
    "DoomEnv",
    "NemotronIPI",
    "ProceduralSearch",
    "ReasonEnv",
    "RefusalEnv",
    "RegisterEnv",
]
