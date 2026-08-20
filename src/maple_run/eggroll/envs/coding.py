"""Coding sandbox stub (core base type).

Subclass this in ``maple_run.eggroll_envs`` or ``./eggroll_envs``. The inner
ES loop must stay in-process; do not shell out from ``step()``.
"""

from __future__ import annotations

from typing import Any

from maple_run.eggroll.envs.base import Episode, ToolEnv, function_tool
from maple_run.eggroll.envs.registry import register

WRITE_FILE_TOOL = function_tool(
    "write_file",
    "Write a file in the stub workspace (not implemented).",
    {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    ["path", "content"],
)


@register
class CodingEnv(ToolEnv):
    """Base coding env. ``sample`` / ``files`` / ``tests`` are not implemented yet."""

    kind = "code"
    abstract = True

    def sample(self, rng) -> Any:
        raise NotImplementedError(
            "CodingEnv is a stub. Subclass it and implement sample(), files(), "
            "and tests(); do not run a subprocess from the ES inner loop."
        )

    def prompt(self, inst: Any) -> str:
        raise NotImplementedError("CodingEnv is a stub")

    def files(self, inst: Any) -> dict[str, str]:
        raise NotImplementedError("CodingEnv is a stub")

    def tests(self, inst: Any) -> list[Any]:
        raise NotImplementedError("CodingEnv is a stub")

    def tools(self, inst: Any) -> list:
        return [WRITE_FILE_TOOL]

    def step(self, name: str, args: dict | None) -> tuple[str, bool]:
        return "ERROR: CodingEnv is a stub", False

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        return 0.0, {"code": 0.0}
