"""Packed ternary linear layer. Holds codes + row_alpha; calls the GEMV kernel."""

from __future__ import annotations

from maple_run.kernels.ternary_gemv import ternary_gemv


class PackedTernaryLinear:
    def __init__(self, packed_weight, row_alpha):
        self.packed_weight = packed_weight
        self.row_alpha = row_alpha

    def forward(self, x):
        return ternary_gemv(x, self.packed_weight, self.row_alpha)

    __call__ = forward
