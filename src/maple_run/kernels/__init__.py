"""GPU kernels. Packed GEMV, fused expert dispatch, 4-bit RTN head."""

from __future__ import annotations

__all__ = [
    "ternary_gemv",
    "ternary_expert_gemv",
    "ternary_expert_swiglu",
    "ternary_expert_down_sum",
    "rtn4_gemv",
    "rtn4_indexed_gemv",
    "rtn4_embedding",
    "rms_norm",
    "add_rms_norm",
]


def __getattr__(name: str):
    if name == "ternary_gemv":
        from maple_run.kernels.ternary_gemv import ternary_gemv

        return ternary_gemv
    if name in {
        "ternary_expert_gemv",
        "ternary_expert_swiglu",
        "ternary_expert_down_sum",
    }:
        from maple_run.kernels import ternary_expert as _exp

        return getattr(_exp, name)
    if name in {"rtn4_gemv", "rtn4_indexed_gemv", "rtn4_embedding"}:
        from maple_run.kernels import rtn4 as _rtn4

        return getattr(_rtn4, name)
    if name in {"rms_norm", "add_rms_norm"}:
        from maple_run.kernels import fused_norm as _norm

        return getattr(_norm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
