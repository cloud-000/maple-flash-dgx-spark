"""Frozen / in-process web search base. No HTTP in the ES inner loop.

Concrete datasets subclass this in ``maple_run.eggroll_envs``.
"""

from __future__ import annotations

import json
from typing import Any

from maple_run.eggroll.envs.base import Episode, ToolEnv, function_tool
from maple_run.eggroll.envs.registry import register

WEB_SEARCH_TOOL = function_tool(
    "web_search",
    "Search the web. Returns a frozen list of {title, url, snippet} hits.",
    {"query": {"type": "string"}},
    ["query"],
)
WEB_FETCH_TOOL = function_tool(
    "web_fetch",
    "Fetch a URL from the current search hits (frozen page or snippet). No live network.",
    {"url": {"type": "string"}},
    ["url"],
)


@register
class SearchEnv(ToolEnv):
    """Base search env. Override ``search`` / ``fetch`` / ``verify`` / ``sample``.

    Default tools: ``web_search`` (once) then optional ``web_fetch`` of a hit URL.
    """

    kind = "search"
    abstract = True

    def __init__(self) -> None:
        super().__init__()
        self._inst: Any = None
        self._search_count = 0

    def tools(self, inst: Any) -> list:
        return [WEB_SEARCH_TOOL, WEB_FETCH_TOOL]

    def max_new(self, inst: Any) -> int | None:
        return 512

    def prompt(self, inst: Any) -> str:
        query = inst.get("query") if isinstance(inst, dict) else str(inst)
        return (
            f"{query}\n"
            "Call web_search, use the hits, cite at least one URL, then stop. "
            "Do not repeat yourself. Do not keep requesting further court orders."
        )

    def reset(self, inst: Any) -> None:
        self._inst = inst
        self._search_count = 0

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

    def step(self, name: str, args: dict | None) -> tuple[str, bool]:
        if args is None:
            return "ERROR: arguments must be JSON object", False
        if name == "web_search":
            if self._search_count >= 1:
                return "ERROR: already searched; answer now from the hits", False
            self._search_count += 1
            query = str(args.get("query", ""))
            payload = self.search(self._inst, query)
            return json.dumps(payload, ensure_ascii=False), True
        if name == "web_fetch":
            url = str(args.get("url", ""))
            text = self.fetch(self._inst, url)
            if text is None:
                return "ERROR: url not in current search hits", False
            return text, True
        return f"ERROR: unknown tool {name}", False

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
