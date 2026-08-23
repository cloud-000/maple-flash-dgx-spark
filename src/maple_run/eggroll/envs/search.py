"""Frozen / in-process web search base. No HTTP in the ES inner loop.

Concrete datasets subclass this in ``maple_run.eggroll_envs``.
Reusable ``WebSearchTool`` / ``WebFetchTool`` own the handlers; this env
owns SERP data and citation scoring.
"""

from __future__ import annotations

from typing import Any

from maple_run.eggroll.envs.base import Episode, ToolEnv
from maple_run.eggroll.envs.registry import register
from maple_run.eggroll.tools import WebFetchTool, WebSearchTool

WEB_SEARCH_TOOL = WebSearchTool().schema()
WEB_FETCH_TOOL = WebFetchTool().schema()


@register
class SearchEnv(ToolEnv):
    """Base search env. Override ``search`` / ``fetch`` / ``verify`` / ``sample``.

    Default tools: ``web_search`` (once) then optional ``web_fetch`` of a hit URL.
    """

    kind = "search"
    abstract = True

    def tools(self, inst: Any) -> list:
        return [WebSearchTool(), WebFetchTool()]

    def max_new(self, inst: Any) -> int | None:
        return 512

    def prompt(self, inst: Any) -> str:
        query = inst.get("query") if isinstance(inst, dict) else str(inst)
        return (
            f"{query}\n"
            "Call web_search, use the hits, cite at least one URL, then stop. "
            "Do not repeat yourself. Do not keep requesting further court orders."
        )

    def initial_state(self, inst: Any) -> dict[str, Any]:
        return {"search_count": 0}

    def search(self, inst: Any, query: str) -> dict[str, Any]:
        """Return the frozen SERP. Override to filter by ``query``."""
        hits = inst.get("results", []) if isinstance(inst, dict) else []
        return {"results": hits}

    def fetch(self, inst: Any, url: str) -> str | None:
        if not isinstance(inst, dict):
            return None
        for hit in inst.get("results") or []:
            if str(hit.get("url")) == url:
                return str(hit.get("page") or hit.get("snippet") or "")
        return None

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        text = episode.text
        must = []
        results = []
        if isinstance(inst, dict):
            must = [str(s) for s in inst.get("must_mention") or []]
            results = list(inst.get("results") or [])
        hits = 0.0
        for needle in must:
            if needle.lower() in text.lower():
                hits += 1.0
        mention = hits / len(must) if must else (1.0 if text.strip() else 0.0)
        urls = [str(h.get("url", "")) for h in results]
        cited = 1.0 if any(u and u in text for u in urls) else 0.0
        score = 0.6 * mention + 0.4 * cited
        if not text.strip():
            score = -0.5
        return score, {"mention": mention, "cited": cited, "search": score}
