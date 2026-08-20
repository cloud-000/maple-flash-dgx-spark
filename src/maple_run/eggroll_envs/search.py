"""Synthetic SERP search env (frozen JSON, in-process verifier)."""

from __future__ import annotations

from typing import Any

from maple_run.eggroll.envs.registry import register
from maple_run.eggroll.envs.search import SearchEnv

_PROCEDURAL = (
    {
        "query": "Which court decided Maple v. Grove and in what year?",
        "must_mention": ["Kansas", "2008"],
        "results": [
            {
                "title": "Maple v. Grove, 177 P.3d 981 (Kan. App. 2008)",
                "url": "https://example.test/maple-v-grove",
                "snippet": (
                    "The Kansas Court of Appeals decided Maple v. Grove in 2008 "
                    "(177 P.3d 981)."
                ),
            },
            {
                "title": "Sugar maple - forestry notes",
                "url": "https://example.test/sugar-maple",
                "snippet": "Acer saccharum is native to eastern North America.",
            },
        ],
    },
    {
        "query": "When was HTTPS specified as RFC 2818?",
        "must_mention": ["2000"],
        "results": [
            {
                "title": "RFC 2818: HTTP Over TLS",
                "url": "https://example.test/rfc2818",
                "snippet": "RFC 2818 was published in May 2000 and specifies HTTPS.",
            },
            {
                "title": "TLS handshake overview",
                "url": "https://example.test/tls",
                "snippet": "TLS negotiates keys after the TCP handshake.",
            },
        ],
    },
    {
        "query": "Who wrote notes on Babbage's Analytical Engine?",
        "must_mention": ["Ada", "Lovelace"],
        "results": [
            {
                "title": "Ada Lovelace - notes on the Analytical Engine",
                "url": "https://example.test/ada-lovelace",
                "snippet": (
                    "Ada Lovelace wrote extensive notes on Babbage's Analytical Engine "
                    "in 1843."
                ),
            },
            {
                "title": "Charles Babbage biography",
                "url": "https://example.test/babbage",
                "snippet": "Babbage designed the Difference Engine and Analytical Engine.",
            },
        ],
    },
)


@register
class ProceduralSearch(SearchEnv):
    """Concrete search env: synthetic SERP JSON, in-process verifier."""

    def sample(self, rng) -> dict[str, Any]:
        row = rng.choice(_PROCEDURAL) if hasattr(rng, "choice") else _PROCEDURAL[0]
        return {
            "query": row["query"],
            "must_mention": list(row["must_mention"]),
            "results": [dict(h) for h in row["results"]],
        }

    def catalog(self) -> list[Any]:
        return [
            {
                "query": row["query"],
                "must_mention": list(row["must_mention"]),
                "results": [dict(h) for h in row["results"]],
            }
            for row in _PROCEDURAL
        ]
