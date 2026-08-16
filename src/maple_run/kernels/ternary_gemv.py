"""Packed ternary GEMV: y = x @ W_packed, W never unpacked to bf16.

Target: DGX Spark GB10 (sm_121). Prefer Triton in this repo. A kernel that
dequantizes to a dense bf16 matrix and then matmuls has failed the design.
"""

from __future__ import annotations


def ternary_gemv(x, packed_weight, row_alpha):
    raise NotImplementedError(
        "Phase 2: Triton (or CUDA) packed ternary GEMV. See docs/HANDOFF.md."
    )
