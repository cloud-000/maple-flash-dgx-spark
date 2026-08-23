"""Generic EGGROLL tasks: immutable env + sampled instance + meta + weight."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def item_kind(item: Any) -> str:
    env = getattr(item, "env", None)
    if env is not None:
        return str(getattr(env, "kind", type(env).__name__))
    return str(getattr(item, "kind", type(item).__name__))


@dataclass(frozen=True)
class Task:
    """One rollout job. Domain data lives on ``inst`` / ``env``, not here."""

    env: Any
    inst: Any
    meta: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta or {})))
        object.__setattr__(self, "weight", float(self.weight))
        if self.env is None:
            raise ValueError("Task.env is required")

    @property
    def kind(self) -> str:
        return item_kind(self)


def sample_batch(suite: list, k: int, rng) -> list[Task]:
    """Draw ``k`` tasks, covering every kind when the suite allows.

    Items may be frozen :class:`Task` objects or env plugins with ``sample`` /
    ``bind`` (instantiated each draw).
    """
    if k <= 0:
        return []
    by_kind: dict[str, list] = {}
    for item in suite:
        by_kind.setdefault(item_kind(item), []).append(item)
    batch: list[Task] = []
    kinds = list(by_kind)
    i = 0
    while len(batch) < k:
        kind = kinds[i % len(kinds)]
        chosen = rng.choice(by_kind[kind])
        sample = getattr(chosen, "sample", None)
        bind = getattr(chosen, "bind", None)
        if callable(sample) and callable(bind):
            batch.append(bind(sample(rng)))
        else:
            batch.append(chosen)
        i += 1
    return batch


def default_suite() -> list:
    """Environments from the default training recipe (not frozen prompts)."""
    from maple_run.eggroll.recipes import get_recipe

    return get_recipe("default").environments()
