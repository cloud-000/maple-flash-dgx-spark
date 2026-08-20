"""Rank-1 (or rank-r) residuals on packed GEMVs.

A population member is ``y = W_packed x + (σ / √r) A (Bᵀ x)`` with
``A ∈ R^{N×r}``, ``B ∈ R^{K×r}`` reconstructed from a seed. Fused ES
updates accumulate in the same low-rank factors (SVD-truncated to ``r_max``)
so the ternary codes stay packed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

DEFAULT_RANK = 1
DEFAULT_R_MAX = 32
DEFAULT_SIGMA = 0.001
# Attention + expert down + lm_head. Router stays frozen (MoE routing stability).
DEFAULT_MODULES = ("qkv", "o_proj", "down", "lm_head")


def mix_seed(base: int, *parts: int) -> int:
    """Splitmix64-style seed mix so (step, layer, member) reconstructs noise."""
    s = int(base) & 0xFFFFFFFFFFFFFFFF
    for part in parts:
        s ^= (int(part) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        s = (s * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        s = (s ^ (s >> 30)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 31
    return s & 0x7FFFFFFFFFFFFFFF


def _generator(seed: int, device: torch.device | str | None) -> torch.Generator:
    # CUDA generators are per-device; CPU is enough to reconstruct and copy.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def sample_factors(
    out_features: int,
    in_features: int,
    rank: int,
    seed: int,
    *,
    n_exp: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``A ~ N(0,1)^{[E,] N, r}`` and ``B ~ N(0,1)^{[E,] K, r}``."""
    gen = _generator(seed, device)
    if n_exp is None:
        a = torch.randn(out_features, rank, generator=gen, dtype=torch.float32)
        b = torch.randn(in_features, rank, generator=gen, dtype=torch.float32)
    else:
        a = torch.randn(n_exp, out_features, rank, generator=gen, dtype=torch.float32)
        b = torch.randn(n_exp, in_features, rank, generator=gen, dtype=torch.float32)
    return a.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


def rank1_apply(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(x @ B) @ Aᵀ`` with ``A [N, r]``, ``B [K, r]``, ``x [..., K]``."""
    orig = x.shape
    x2 = x.reshape(-1, orig[-1]).to(dtype=torch.float32)
    a32 = a.to(dtype=torch.float32, device=x.device)
    b32 = b.to(dtype=torch.float32, device=x.device)
    y = (x2 @ b32) @ a32.T
    return y.to(dtype=x.dtype).reshape(*orig[:-1], a.shape[-2])


def expert_rank1_apply(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Indexed ``(x @ B[e]) @ A[e]ᵀ`` for selected experts.

    ``a`` is ``[E, N, r]``, ``b`` is ``[E, K, r]``. ``x`` is either one row per
    token (``[..., K]`` with ``expert_ids [..., topk]``) or one row per slot
    (``x.shape[:-1] == expert_ids.shape``).
    """
    ids = expert_ids.long()
    a32 = a.to(dtype=torch.float32, device=x.device)
    b32 = b.to(dtype=torch.float32, device=x.device)
    xf = x.to(dtype=torch.float32)
    b_sel = b32[ids]
    a_sel = a32[ids]
    if xf.shape[:-1] != ids.shape:
        xf = xf.unsqueeze(-2).expand(*ids.shape, xf.shape[-1])
    z = torch.matmul(xf.unsqueeze(-2), b_sel).squeeze(-2)
    y = torch.matmul(z.unsqueeze(-2), a_sel.transpose(-1, -2)).squeeze(-2)
    return y.to(dtype=x.dtype)


def rms_normalize(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim. Torch path so CPU tests and CUDA both work."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    w = weight.float().reshape(*([1] * (xf.ndim - 1)), -1)
    return (xf * rstd * w).to(dtype=x.dtype)


def compress_factors(
    a: torch.Tensor,
    b: torch.Tensor,
    r_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated SVD of ``A Bᵀ`` back into factors of rank ``≤ r_max``.

    ``a`` is ``[..., N, r]``, ``b`` is ``[..., K, r]``. Experts are compressed
    independently (leading dims flattened).
    """
    r = int(a.shape[-1])
    if r <= r_max:
        return a, b
    *batch, n, _ = a.shape
    k = int(b.shape[-2])
    a_f = a.reshape(-1, n, r).float()
    b_f = b.reshape(-1, k, r).float()
    keep = min(int(r_max), r)
    out_a = a.new_zeros(a_f.shape[0], n, keep)
    out_b = b.new_zeros(b_f.shape[0], k, keep)
    for i in range(a_f.shape[0]):
        qa, ra = torch.linalg.qr(a_f[i], mode="reduced")
        qb, rb = torch.linalg.qr(b_f[i], mode="reduced")
        u, s, vh = torch.linalg.svd(ra @ rb.mT, full_matrices=False)
        r_i = min(keep, int(s.numel()))
        out_a[i, :, :r_i] = qa @ (u[:, :r_i] * s[:r_i])
        out_b[i, :, :r_i] = qb @ vh[:r_i].T
    shape_a = (*batch, n, keep)
    shape_b = (*batch, k, keep)
    return out_a.to(dtype=a.dtype).reshape(shape_a), out_b.to(dtype=b.dtype).reshape(
        shape_b
    )


def concat_rank(left: torch.Tensor | None, right: torch.Tensor) -> torch.Tensor:
    if left is None or left.numel() == 0:
        return right
    return torch.cat([left, right], dim=-1)


@dataclass
class Rank1Adapter:
    """Low-rank residual on one packed linear (2-D) or stacked experts (3-D)."""

    name: str
    out_features: int
    in_features: int
    rank: int = DEFAULT_RANK
    r_max: int = DEFAULT_R_MAX
    n_exp: int | None = None
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    residual_a: torch.Tensor | None = None
    residual_b: torch.Tensor | None = None
    noise_a: torch.Tensor | None = None
    noise_b: torch.Tensor | None = None
    noise_scale: float = 0.0
    seed: int | None = None

    @property
    def active(self) -> bool:
        return (
            (self.residual_a is not None and self.residual_a.numel() > 0)
            or self.noise_a is not None
        )

    @property
    def residual_rank(self) -> int:
        if self.residual_a is None:
            return 0
        return int(self.residual_a.shape[-1])

    def clear_noise(self) -> None:
        self.noise_a = None
        self.noise_b = None
        self.noise_scale = 0.0
        self.seed = None

    def set_noise(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        scale: float,
        seed: int | None = None,
    ) -> None:
        self.noise_a = a.to(device=self.device, dtype=self.dtype)
        self.noise_b = b.to(device=self.device, dtype=self.dtype)
        self.noise_scale = float(scale)
        self.seed = seed

    def delta(
        self,
        x: torch.Tensor,
        *,
        rms_weight: torch.Tensor | None = None,
        rms_eps: float = 1e-6,
        expert_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_eff = rms_normalize(x, rms_weight, rms_eps) if rms_weight is not None else x
        y = None
        if self.residual_a is not None:
            y = self._apply(x_eff, self.residual_a, self.residual_b, expert_ids)
        if self.noise_a is not None:
            noise = self._apply(x_eff, self.noise_a, self.noise_b, expert_ids)
            noise = noise * self.noise_scale
            y = noise if y is None else y + noise
        if y is None:
            return torch.zeros(
                *x.shape[:-1],
                self.out_features,
                device=x.device,
                dtype=x.dtype,
            )
        return y

    def _apply(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor | None,
        expert_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        assert b is not None
        if self.n_exp is None:
            return rank1_apply(x, a, b)
        if expert_ids is None:
            raise ValueError(f"adapter {self.name} is expert-shaped; need expert_ids")
        return expert_rank1_apply(x, expert_ids, a, b)

    def fuse(self, a_noise: torch.Tensor, b_noise: torch.Tensor, weight: float) -> None:
        """Add ``weight * A Bᵀ`` into the residual and SVD-truncate to ``r_max``."""
        self.fuse_many([(a_noise, b_noise, weight)])

    def fuse_many(
        self, terms: list[tuple[torch.Tensor, torch.Tensor, float]]
    ) -> None:
        a_cat = self.residual_a
        b_cat = self.residual_b
        added = False
        for a_noise, b_noise, weight in terms:
            if weight == 0.0:
                continue
            scale = math.sqrt(abs(float(weight)))
            sign = 1.0 if weight >= 0 else -1.0
            a_new = a_noise.to(device=self.device, dtype=self.dtype) * (scale * sign)
            b_new = b_noise.to(device=self.device, dtype=self.dtype) * scale
            a_cat = concat_rank(a_cat, a_new)
            b_cat = concat_rank(b_cat, b_new)
            added = True
        if not added:
            return
        self.residual_a, self.residual_b = compress_factors(a_cat, b_cat, self.r_max)

    def state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.residual_a is not None:
            out[f"{self.name}.a"] = self.residual_a.detach().cpu().contiguous()
            out[f"{self.name}.b"] = self.residual_b.detach().cpu().contiguous()
        return out

    def load_state_dict(self, tensors: dict[str, torch.Tensor]) -> None:
        a = tensors.get(f"{self.name}.a")
        b = tensors.get(f"{self.name}.b")
        if a is None or b is None:
            self.residual_a = None
            self.residual_b = None
            return
        self.residual_a = a.to(device=self.device, dtype=self.dtype)
        self.residual_b = b.to(device=self.device, dtype=self.dtype)


def add_adapter_delta(
    module,
    y: torch.Tensor,
    x: torch.Tensor,
    *,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    expert_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """No-op unless ``module.eggroll`` is an active :class:`Rank1Adapter`."""
    adapter = getattr(module, "eggroll", None)
    if adapter is None or not adapter.active:
        return y
    return y + adapter.delta(
        x, rms_weight=rms_weight, rms_eps=rms_eps, expert_ids=expert_ids
    )


def packed_nk(module) -> tuple[int, int, int | None]:
    """``(N, K, n_exp or None)`` from a packed linear / expert / RTN4 head."""
    w = module.packed_weight
    if w.ndim == 2:
        n_out, nwords = w.shape
        codes = 8 if hasattr(module, "scales") else 16
        return int(n_out), int(nwords) * codes, None
    n_exp, n_out, nwords = w.shape
    return int(n_out), int(nwords) * 16, int(n_exp)
