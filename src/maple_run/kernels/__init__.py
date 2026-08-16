"""GPU kernels. Decode GEMV first; fused MoE later. See docs/HANDOFF.md phase 2."""

from __future__ import annotations

__all__ = ["ternary_gemv"]


def __getattr__(name: str):
    if name == "ternary_gemv":
        from maple_run.kernels.ternary_gemv import ternary_gemv

        return ternary_gemv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
