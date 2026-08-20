"""EGGROLL post-training on packed Maple.

Rank-1 (or rank-r) perturbations sit on top of the frozen 2-bit GEMV: the
packed codes never unpack. See arXiv:2511.16652.
"""

from maple_run.eggroll.es import EggrollRuntime, centered_advantages, rank_advantages
from maple_run.eggroll.perturb import Rank1Adapter, mix_seed, rank1_apply, sample_factors

__all__ = [
    "EggrollRuntime",
    "Rank1Adapter",
    "centered_advantages",
    "mix_seed",
    "rank1_apply",
    "rank_advantages",
    "sample_factors",
]
