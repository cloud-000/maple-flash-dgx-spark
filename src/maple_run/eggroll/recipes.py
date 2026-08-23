"""Training-recipe plugin registry for EGGROLL.

A recipe constructs environments, samples tasks, and supplies hyperparameter
defaults. CLI flags override those defaults; the merged snapshot is what
``config.json`` records.

Bundled recipes (``default``, ``agent``, ``ipi``) live in
``maple_run.eggroll_envs.recipes``. This module is the protocol and loader only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from maple_run.eggroll.perturb import DEFAULT_MODULES, DEFAULT_RANK, DEFAULT_R_MAX, DEFAULT_SIGMA
from maple_run.eggroll.tasks import Task, sample_batch
from maple_run.generate import DEFAULT_TOP_K, DEFAULT_TOP_P

_RECIPES: dict[str, type] = {}
_LOADED_PATHS: set[str] = set()
_BUILTINS_LOADED = False

RECIPE_FIELDS = (
    "steps",
    "population",
    "rank",
    "r_max",
    "sigma",
    "lr",
    "seed",
    "max_tokens",
    "max_tool_rounds",
    "prompts_per_step",
    "max_batch",
    "temperature",
    "top_p",
    "top_k",
    "modules",
    "save_every",
)


def _base_defaults() -> dict[str, Any]:
    return {
        "steps": 50,
        "population": 16,
        "rank": DEFAULT_RANK,
        "r_max": DEFAULT_R_MAX,
        "sigma": DEFAULT_SIGMA,
        "lr": 0.001,
        "seed": 0,
        "max_tokens": 256,
        "max_tool_rounds": 6,
        "prompts_per_step": 4,
        "max_batch": 8,
        "temperature": 0.0,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "modules": DEFAULT_MODULES,
        "save_every": 10,
    }


def _coerce_modules(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_MODULES
    if isinstance(value, str):
        return tuple(m.strip() for m in value.split(",") if m.strip())
    return tuple(value)


class TrainingRecipe:
    """RL-style training plugin. Subclass and ``@register_recipe``."""

    name: str = "base"

    def defaults(self) -> dict[str, Any]:
        return _base_defaults()

    def environments(self) -> list:
        raise NotImplementedError

    def sample_tasks(self, rng, k: int) -> list[Task]:
        return sample_batch(self.environments(), k, rng)

    def eval_tasks(self) -> list[Task]:
        out: list[Task] = []
        for env in self.environments():
            out.extend(env.bind(inst) for inst in env.catalog())
        return out

    def resolve(self, args: Any, explicit: set[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Merge recipe defaults with CLI dests listed in ``explicit``.

        Returns ``(resolved, overrides)``. ``overrides`` is only what the CLI set.
        """
        resolved = dict(self.defaults())
        overrides: dict[str, Any] = {}
        explicit = explicit or set()
        for key in RECIPE_FIELDS:
            if key not in explicit:
                continue
            val = getattr(args, key, None)
            if key == "modules":
                val = _coerce_modules(val)
            resolved[key] = val
            overrides[key] = val
        if "modules" in resolved:
            resolved["modules"] = _coerce_modules(resolved["modules"])
        return resolved, overrides


class RecipeAPI:
    """Passed to a plugin's ``setup(api)`` hook."""

    def add_recipe(self, cls: type, *, name: str | None = None) -> type:
        return register_recipe(cls, name=name)


def register_recipe(cls: type | None = None, *, name: str | None = None) -> Any:
    def deco(c: type) -> type:
        key = name or getattr(c, "name", None) or c.__name__
        c.name = key
        _RECIPES[str(key)] = c
        return c

    if cls is None:
        return deco
    return deco(cls)


def registry() -> dict[str, type]:
    load_builtins()
    return dict(_RECIPES)


def list_recipes() -> list[str]:
    return sorted(registry())


def get_recipe(name: str) -> TrainingRecipe:
    load_builtins()
    cls = _RECIPES.get(name)
    if cls is None:
        available = ", ".join(list_recipes() or ["(none)"])
        raise KeyError(f"unknown recipe {name!r}. registered: {available}")
    return cls()


def load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    import maple_run.eggroll_envs  # noqa: F401  # bundled envs + recipes


def default_recipe_dirs() -> list[Path]:
    local = Path("eggroll_recipes")
    return [local] if local.is_dir() else []


def load_recipe_file(file: str | Path) -> None:
    load_builtins()
    path = Path(file).resolve()
    key = str(path)
    if key in _LOADED_PATHS:
        return
    if not path.is_file():
        raise FileNotFoundError(f"recipe plugin not found: {path}")
    mod_name = f"maple_run_eggroll_recipe_{path.stem}_{abs(hash(key)) & 0xFFFF}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load recipe plugin {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _LOADED_PATHS.add(key)
    setup: Callable[..., Any] | None = getattr(module, "setup", None)
    if callable(setup):
        setup(RecipeAPI())


def load_recipe_dir(directory: str | Path) -> None:
    load_builtins()
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"recipe dir not found: {path}")
    for file in sorted(path.glob("*.py")):
        if file.name.startswith("_"):
            continue
        load_recipe_file(file)


def load_recipe_plugins(directories: list[str | Path] | None = None) -> None:
    load_builtins()
    seen: set[str] = set()
    dirs: list[Path] = []
    dirs.extend(default_recipe_dirs())
    for raw in directories or []:
        dirs.append(Path(raw))
    for directory in dirs:
        resolved = str(directory.resolve()) if directory.exists() else str(directory)
        if resolved in seen:
            continue
        seen.add(resolved)
        load_recipe_dir(directory)
