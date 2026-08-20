"""Plugin API for EGGROLL environments (harness, not datasets).

Write extensions in ``maple_run.eggroll_envs`` (bundled) or ``./eggroll_envs``
(project-local). Subclass ``SearchEnv`` / ``SingleTurnEnv`` / ``ToolEnv`` /
``CodingEnv`` and ``@register``.
"""

from maple_run.eggroll.envs.base import (
    Env,
    Episode,
    SingleTurnEnv,
    ToolEnv,
    function_tool,
)
from maple_run.eggroll.envs.coding import CodingEnv
from maple_run.eggroll.envs.machine import RegisterMachine
from maple_run.eggroll.envs.registry import (
    EggrollAPI,
    get_envs,
    instantiate,
    load_env_dir,
    load_env_file,
    load_plugins,
    register,
    registry,
)
from maple_run.eggroll.envs.search import SearchEnv, WEB_FETCH_TOOL, WEB_SEARCH_TOOL

__all__ = [
    "CodingEnv",
    "EggrollAPI",
    "Env",
    "Episode",
    "RegisterMachine",
    "SearchEnv",
    "SingleTurnEnv",
    "ToolEnv",
    "WEB_FETCH_TOOL",
    "WEB_SEARCH_TOOL",
    "function_tool",
    "get_envs",
    "instantiate",
    "load_env_dir",
    "load_env_file",
    "load_plugins",
    "register",
    "registry",
]
