"""Packed ternary linear layer. Holds codes + row_alpha; calls the GEMV kernel."""

from __future__ import annotations


class PackedTernaryLinear:
    def __init__(self, packed_weight, row_alpha):
        self.packed_weight = packed_weight
        self.row_alpha = row_alpha

    def forward(self, x):
        raise NotImplementedError(
            "Phase 2: call maple_run.kernels.ternary_gemv. See docs/HANDOFF.md."
        )
