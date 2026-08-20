"""Packed linear layers. Hold codes + scales; call GEMV kernels."""

from __future__ import annotations

from maple_run.eggroll.perturb import add_adapter_delta
from maple_run.kernels.rtn4 import rtn4_embedding, rtn4_gemv
from maple_run.kernels.ternary_expert import (
    ternary_expert_down_sum,
    ternary_expert_gemv,
    ternary_expert_swiglu,
)
from maple_run.kernels.ternary_gemv import ternary_gemv
from maple_run.pack import HEAD_GROUP_SIZE


class PackedTernaryLinear:
    """Codes stay in the checkpoint's ``[N, nwords]`` order.

    Transposing to ``[nwords, N]`` puts consecutive output rows next to each
    other, but it also scatters one CTA's K tile across ``BLOCK_K_WORDS``
    addresses ``N * 4`` bytes apart. Read cold — which is the only way decode
    ever reads them — the untransposed layout keeps each row's K tile
    contiguous and measured faster on every projection: QKV 141 vs 125 GB/s,
    O 191 vs 165, expert up/gate 178 vs 170, expert down 174 vs 167.
    """

    def __init__(self, packed_weight, row_alpha):
        self.packed_weight = packed_weight.contiguous()
        self.row_alpha = row_alpha

    def forward(self, x, rms_weight=None, rms_eps: float = 1e-6):
        y = ternary_gemv(
            x,
            self.packed_weight,
            self.row_alpha,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
        )
        return add_adapter_delta(self, y, x, rms_weight=rms_weight, rms_eps=rms_eps)

    __call__ = forward


class PackedTernaryExperts:
    """Stacked experts ``[E, N, nwords]``; one fused launch over selected ids."""

    def __init__(self, packed_weight, row_alpha):
        self.packed_weight = packed_weight.contiguous()
        self.row_alpha = row_alpha

    def forward(self, x, expert_ids):
        y = ternary_expert_gemv(x, self.packed_weight, self.row_alpha, expert_ids)
        return add_adapter_delta(self, y, x, expert_ids=expert_ids)

    def swiglu(self, x, expert_ids):
        y = ternary_expert_swiglu(x, self.packed_weight, self.row_alpha, expert_ids)
        return add_adapter_delta(self, y, x, expert_ids=expert_ids)

    def down_sum(self, x, expert_ids, topk_weight, residual=None):
        return ternary_expert_down_sum(
            x,
            self.packed_weight,
            self.row_alpha,
            expert_ids,
            topk_weight,
            residual=residual,
        )

    __call__ = forward


class PackedRTN4Linear:
    """4-bit head, also left in the checkpoint's ``[N, nwords]`` order (215 vs
    194 GB/s transposed; see ``PackedTernaryLinear``)."""

    def __init__(self, packed_weight, scales, biases, group_size: int = HEAD_GROUP_SIZE):
        self.packed_weight = packed_weight.contiguous()
        self.scales = scales.contiguous()
        self.biases = biases.contiguous()
        self.group_size = group_size

    def forward(self, x):
        y = rtn4_gemv(
            x,
            self.packed_weight,
            self.scales,
            self.biases,
            group_size=self.group_size,
        )
        return add_adapter_delta(self, y, x)

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
