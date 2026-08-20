"""Discover and register EGGROLL env plugins (Pi-style modules)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from maple_run.eggroll.envs.base import Env

_REGISTRY: dict[str, type[Env]] = {}
_LOADED_PATHS: set[str] = set()
_BUILTINS_LOADED = False


class EggrollAPI:
    """Passed to a plugin's ``setup(api)`` hook, analogous to Pi's ExtensionAPI."""

    def add_env(self, cls: type[Env], *, name: str | None = None) -> type[Env]:
        return register(cls, name=name)


def register(cls: type[Env] | None = None, *, name: str | None = None) -> Any:
    """Class decorator, or ``register(MyEnv)``. Bases with ``abstract=True`` stay listed."""

    def deco(c: type[Env]) -> type[Env]:
        key = name or c.__name__
        _REGISTRY[key] = c
        return c

    if cls is None:
        return deco
    return deco(cls)


def registry() -> dict[str, type[Env]]:
    load_builtins()
    return dict(_REGISTRY)


def load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    import maple_run.eggroll.envs  # noqa: F401  # SearchEnv, CodingEnv
    import maple_run.eggroll_envs  # noqa: F401


def load_env_dir(directory: str | Path) -> None:
    """Import ``*.py`` in ``directory`` (not recursive)."""
    load_builtins()
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"env dir not found: {path}")
    for file in sorted(path.glob("*.py")):
        if file.name.startswith("_"):
            continue
        load_env_file(file)


def load_env_file(file: str | Path) -> None:
    load_builtins()
    path = Path(file).resolve()
    key = str(path)
    if key in _LOADED_PATHS:
        return
    if not path.is_file():
        raise FileNotFoundError(f"env plugin not found: {path}")
    mod_name = f"maple_run_eggroll_plugin_{path.stem}_{abs(hash(key)) & 0xFFFF}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load env plugin {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _LOADED_PATHS.add(key)
    setup: Callable[..., Any] | None = getattr(module, "setup", None)
    if callable(setup):
        setup(EggrollAPI())


def default_env_dirs() -> list[Path]:
    local = Path("eggroll_envs")
    return [local] if local.is_dir() else []


def instantiate(name: str) -> Env:
    load_builtins()
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY) or ["(none)"])
        raise KeyError(f"unknown env {name!r}. registered: {available}")
    if cls.__dict__.get("abstract", False):
        raise TypeError(
            f"{name} is a base type; subclass it in a plugin "
            "(CodingEnv is a stub, SearchEnv needs a concrete subclass)."
        )
    return cls()


def get_envs(names: list[str]) -> list[Env]:
    return [instantiate(n) for n in names]


def load_plugins(directories: list[str | Path] | None = None) -> None:
    """Load bundled extensions, then ``./eggroll_envs`` (if present), then extra dirs."""
    load_builtins()
    seen: set[str] = set()
    dirs: list[Path] = []
    dirs.extend(default_env_dirs())
    for raw in directories or []:
        dirs.append(Path(raw))
    for directory in dirs:
        resolved = str(directory.resolve()) if directory.exists() else str(directory)
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_dir(directory)
