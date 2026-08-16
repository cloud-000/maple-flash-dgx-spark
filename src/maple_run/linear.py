"""Packed linear layers. Hold codes + scales; call GEMV kernels."""

from __future__ import annotations

from maple_run.kernels.rtn4 import rtn4_embedding, rtn4_gemv
from maple_run.kernels.ternary_expert import (
    ternary_expert_down_sum,
    ternary_expert_gemv,
    ternary_expert_swiglu,
)
from maple_run.kernels.ternary_gemv import ternary_gemv
from maple_run.pack import HEAD_GROUP_SIZE


def _packed_kn_2d(packed_weight):
    """``[N, nwords]`` → ``[nwords, N]`` so consecutive output rows coalesce."""
    return packed_weight.transpose(0, 1).contiguous()


def _packed_kn_3d(packed_weight):
    """``[E, N, nwords]`` → ``[E, nwords, N]``."""
    return packed_weight.transpose(1, 2).contiguous()


class PackedTernaryLinear:
    def __init__(self, packed_weight, row_alpha):
        self.packed_weight = _packed_kn_2d(packed_weight)
        self.row_alpha = row_alpha

    def forward(self, x, rms_weight=None, rms_eps: float = 1e-6):
        return ternary_gemv(
            x,
            self.packed_weight,
            self.row_alpha,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            packed_kn=True,
        )

    __call__ = forward


class PackedTernaryExperts:
    """Stacked experts ``[E, nwords, N]`` after load; one fused launch over selected ids."""

    def __init__(self, packed_weight, row_alpha):
        self.packed_weight = _packed_kn_3d(packed_weight)
        self.row_alpha = row_alpha

    def forward(self, x, expert_ids):
        return ternary_expert_gemv(
            x, self.packed_weight, self.row_alpha, expert_ids, packed_kn=True
        )

    def swiglu(self, x, expert_ids):
        return ternary_expert_swiglu(
            x, self.packed_weight, self.row_alpha, expert_ids, packed_kn=True
        )

    def down_sum(self, x, expert_ids, topk_weight, residual=None):
        return ternary_expert_down_sum(
            x,
            self.packed_weight,
            self.row_alpha,
            expert_ids,
            topk_weight,
            residual=residual,
            packed_kn=True,
        )

    __call__ = forward


class PackedRTN4Linear:
    def __init__(self, packed_weight, scales, biases, group_size: int = HEAD_GROUP_SIZE):
        self.packed_weight = _packed_kn_2d(packed_weight)
        self.scales = scales.transpose(0, 1).contiguous()
        self.biases = biases.transpose(0, 1).contiguous()
        self.group_size = group_size

    def forward(self, x):
        return rtn4_gemv(
            x,
            self.packed_weight,
            self.scales,
            self.biases,
            group_size=self.group_size,
            packed_kn=True,
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
