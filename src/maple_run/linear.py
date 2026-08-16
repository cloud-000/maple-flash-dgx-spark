"""Packed linear layers. Hold codes + scales; call GEMV kernels."""

from __future__ import annotations

from maple_run.kernels.rtn4 import rtn4_embedding, rtn4_gemv
from maple_run.kernels.ternary_expert import ternary_expert_gemv
from maple_run.kernels.ternary_gemv import ternary_gemv
from maple_run.pack import HEAD_GROUP_SIZE


class PackedTernaryLinear:
    def __init__(self, packed_weight, row_alpha):
        self.packed_weight = packed_weight
        self.row_alpha = row_alpha

    def forward(self, x):
        return ternary_gemv(x, self.packed_weight, self.row_alpha)

    __call__ = forward


class PackedTernaryExperts:
    """Stacked experts ``[E, N, K/16]``; one fused launch over selected ids."""

    def __init__(self, packed_weight, row_alpha):
        self.packed_weight = packed_weight
        self.row_alpha = row_alpha

    def forward(self, x, expert_ids):
        return ternary_expert_gemv(x, self.packed_weight, self.row_alpha, expert_ids)

    __call__ = forward


class PackedRTN4Linear:
    def __init__(self, packed_weight, scales, biases, group_size: int = HEAD_GROUP_SIZE):
        self.packed_weight = packed_weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size

    def forward(self, x):
        return rtn4_gemv(
            x, self.packed_weight, self.scales, self.biases, group_size=self.group_size
        )

    __call__ = forward


class PackedRTN4Embedding:
    def __init__(self, packed_weight, scales, biases, group_size: int = HEAD_GROUP_SIZE):
        self.packed_weight = packed_weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size

    def forward(self, input_ids):
        return rtn4_embedding(
            input_ids,
            self.packed_weight,
            self.scales,
            self.biases,
            group_size=self.group_size,
        )

    __call__ = forward
