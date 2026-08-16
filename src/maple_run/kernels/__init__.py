"""GPU kernels. Packed GEMV, fused expert dispatch, 4-bit RTN head."""

from __future__ import annotations

__all__ = ["ternary_gemv", "ternary_expert_gemv", "rtn4_gemv", "rtn4_embedding"]


def __getattr__(name: str):
    if name == "ternary_gemv":
        from maple_run.kernels.ternary_gemv import ternary_gemv

        return ternary_gemv
    if name == "ternary_expert_gemv":
        from maple_run.kernels.ternary_expert import ternary_expert_gemv

        return ternary_expert_gemv
    if name in {"rtn4_gemv", "rtn4_embedding"}:
        from maple_run.kernels import rtn4 as _rtn4

        return getattr(_rtn4, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
