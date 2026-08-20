"""EGGROLL population, antithetic fitness, fused low-rank update.

Paper (arXiv:2511.16652) Algorithm 1 with the Gaussian score
``µ ← µ + (α / (N σ)) Σ_i â_i E_i``, ``E_i = (1/√r) A_i B_iᵀ``.
Antithetic pairs match the official transformer trainer
(``ESHyperscale/eggroll-vllm``): one noise, ``+σE`` and ``−σE``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from maple_run.eggroll.perturb import (
    DEFAULT_MODULES,
    DEFAULT_RANK,
    DEFAULT_R_MAX,
    DEFAULT_SIGMA,
    Rank1Adapter,
    mix_seed,
    packed_nk,
    sample_factors,
)

ADAPTER_CONFIG = "eggroll.json"
ADAPTER_WEIGHTS = "adapters.safetensors"


def rank_advantages(fitness: list[float]) -> list[float]:
    """Centered ranks in ``(-0.5, 0.5)`` (Salimans et al. 2017)."""
    n = len(fitness)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: fitness[i])
    ranks = [0.0] * n
    for rank, i in enumerate(order):
        ranks[i] = rank / (n - 1) - 0.5 if n > 1 else 0.0
    return ranks


def centered_advantages(fitness: list[float], *, zscore: bool = False) -> list[float]:
    """Z-score or leave as rank-shaped values already centered."""
    if not fitness:
        return []
    if not zscore:
        return rank_advantages(fitness)
    t = torch.tensor(fitness, dtype=torch.float64)
    std = t.std(unbiased=False).clamp_min(1e-8)
    return ((t - t.mean()) / std).tolist()


class EggrollRuntime:
    """Adapters attached to a packed Maple (or any object with the same layout)."""

    def __init__(
        self,
        *,
        rank: int = DEFAULT_RANK,
        r_max: int = DEFAULT_R_MAX,
        sigma: float = DEFAULT_SIGMA,
        modules: tuple[str, ...] = DEFAULT_MODULES,
        base_seed: int = 0,
    ):
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if r_max < rank:
            raise ValueError("r_max must be >= rank")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.rank = int(rank)
        self.r_max = int(r_max)
        self.sigma = float(sigma)
        self.modules = tuple(modules)
        self.base_seed = int(base_seed)
        self.adapters: dict[str, Rank1Adapter] = {}
        self.noise_active = False
        self._step = 0
        self._member: int | None = None
        self._sign = 1.0

    def attach(self, model) -> EggrollRuntime:
        """Bind adapters onto packed linears. Default generate is unchanged until then."""
        device = getattr(model, "device", torch.device("cpu"))
        dtype = torch.float32
        wanted = set(self.modules)
        if "qkv" in wanted or "o_proj" in wanted or "down" in wanted:
            for i, layer in enumerate(getattr(model, "layers", [])):
                attn = layer.self_attn
                if "qkv" in wanted:
                    self._bind(attn.qkv_proj, f"layers.{i}.qkv", device, dtype)
                if "o_proj" in wanted:
                    self._bind(attn.o_proj, f"layers.{i}.o_proj", device, dtype)
                if "down" in wanted:
                    self._bind(layer.mlp.down, f"layers.{i}.down", device, dtype)
                if "up_gate" in wanted:
                    self._bind(layer.mlp.up_gate, f"layers.{i}.up_gate", device, dtype)
        if "lm_head" in wanted and getattr(model, "lm_head", None) is not None:
            self._bind(model.lm_head, "lm_head", device, dtype)
            # FlashHead scores a cluster subset; the residual is on the full head.
            model.lm_head_flash = None
        pending = getattr(self, "_pending", None)
        if pending:
            for adapter in self.adapters.values():
                adapter.load_state_dict(pending)
            self._pending = None
        model.eggroll = self
        return self

    def _bind(self, module, name: str, device, dtype) -> None:
        n_out, k_in, n_exp = packed_nk(module)
        adapter = Rank1Adapter(
            name=name,
            out_features=n_out,
            in_features=k_in,
            rank=self.rank,
            r_max=self.r_max,
            n_exp=n_exp,
            device=device,
            dtype=dtype,
        )
        if name in self.adapters and self.adapters[name].residual_a is not None:
            adapter.residual_a = self.adapters[name].residual_a.to(device=device, dtype=dtype)
            adapter.residual_b = self.adapters[name].residual_b.to(device=device, dtype=dtype)
        module.eggroll = adapter
        self.adapters[name] = adapter

    def set_member(self, step: int, member: int, *, sign: float = 1.0) -> None:
        """Sample (or reconstruct) rank-r noise for one population member."""
        self.noise_active = True
        self._step = int(step)
        self._member = int(member)
        self._sign = float(sign)
        scale = self._sign * self.sigma / math.sqrt(self.rank)
        for i, adapter in enumerate(self.adapters.values()):
            seed = mix_seed(self.base_seed, step, member, i)
            a, b = sample_factors(
                adapter.out_features,
                adapter.in_features,
                self.rank,
                seed,
                n_exp=adapter.n_exp,
                device=adapter.device,
                dtype=adapter.dtype,
            )
            adapter.set_noise(a, b, scale=scale, seed=seed)

    def clear_member(self) -> None:
        self.noise_active = False
        self._member = None
        self._sign = 1.0
        for adapter in self.adapters.values():
            adapter.clear_noise()

    def fuse_antithetic(
        self,
        step: int,
        advantages: list[float],
        *,
        lr: float,
        population: int,
    ) -> None:
        """Fuse ``Σ_pair (f⁺ − f⁻) E_pair`` into residuals.

        ``advantages`` is one scalar per antithetic pair (already ``â⁺ − â⁻``).
        """
        n_pairs = len(advantages)
        if n_pairs == 0:
            return
        # Official vLLM trainer: (lr / (N σ)) * Σ (f+ − f−) (√σ A)(√(σ/r) B)ᵀ
        # With E = A Bᵀ / √r the same coefficient is lr / (N σ) * σ/√r wait:
        # noise factors already include 1/√r in the forward via noise_scale.
        # Update uses E_i = A Bᵀ / √r, coefficient α/(Nσ).
        coeff = float(lr) / (float(population) * self.sigma)
        inv_sqrt_r = 1.0 / math.sqrt(self.rank)
        for i, adapter in enumerate(self.adapters.values()):
            terms = []
            for pair, adv in enumerate(advantages):
                if adv == 0.0:
                    continue
                seed = mix_seed(self.base_seed, step, pair, i)
                a, b = sample_factors(
                    adapter.out_features,
                    adapter.in_features,
                    self.rank,
                    seed,
                    n_exp=adapter.n_exp,
                    device=adapter.device,
                    dtype=adapter.dtype,
                )
                terms.append((a, b, coeff * inv_sqrt_r * float(adv)))
            adapter.fuse_many(terms)

    def config_dict(self) -> dict:
        return {
            "format": "maple-run-eggroll-v1",
            "rank": self.rank,
            "r_max": self.r_max,
            "sigma": self.sigma,
            "modules": list(self.modules),
            "base_seed": self.base_seed,
            "residual_ranks": {
                name: adapter.residual_rank for name, adapter in self.adapters.items()
            },
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        for adapter in self.adapters.values():
            tensors.update(adapter.state_dict())
        if tensors:
            save_file(tensors, directory / ADAPTER_WEIGHTS)
        (directory / ADAPTER_CONFIG).write_text(
            json.dumps(self.config_dict(), indent=2) + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path, *, device=None) -> EggrollRuntime:
        directory = Path(directory)
        cfg = json.loads((directory / ADAPTER_CONFIG).read_text())
        rt = cls(
            rank=int(cfg.get("rank", DEFAULT_RANK)),
            r_max=int(cfg.get("r_max", DEFAULT_R_MAX)),
            sigma=float(cfg.get("sigma", DEFAULT_SIGMA)),
            modules=tuple(cfg.get("modules") or DEFAULT_MODULES),
            base_seed=int(cfg.get("base_seed", 0)),
        )
        weights_path = directory / ADAPTER_WEIGHTS
        if weights_path.is_file():
            rt._pending = load_file(str(weights_path), device="cpu")
        return rt

    def attach_and_restore(self, model) -> EggrollRuntime:
        return self.attach(model)
