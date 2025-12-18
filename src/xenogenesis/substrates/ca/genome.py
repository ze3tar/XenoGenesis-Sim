"""Genome encoding and decoding for CA parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from .kernels import KernelParams


@dataclass(frozen=True)
class CAParams:
    """Decoded CA parameters ready for simulation."""

    grid_size: int
    kernel_params: KernelParams
    mu: float
    sigma: float
    dt: float
    growth_alpha: float
    decay_lambda: float
    regen_rate: float
    consumption_rate: float
    resource_diffusion: float
    biomass_diffusion: float
    noise_std: float
    polarity_gain: float
    polarity_decay: float
    polarity_mobility: float
    polarity_noise: float
    max_mass: float
    death_factor: float
    elongation_trigger: float
    fission_assist: float
    render_gamma: float
    contour_level: float
    dominant_band: int

    def as_dict(self) -> dict:
        return {
            "grid_size": self.grid_size,
            "kernel_params": {
                "size": self.kernel_params.size,
                "rings": [list(r) for r in self.kernel_params.rings],
                "ring_weights": list(self.kernel_params.ring_weights),
            },
            "mu": self.mu,
            "sigma": self.sigma,
            "dt": self.dt,
            "growth_alpha": self.growth_alpha,
            "decay_lambda": self.decay_lambda,
            "regen_rate": self.regen_rate,
            "consumption_rate": self.consumption_rate,
            "resource_diffusion": self.resource_diffusion,
            "biomass_diffusion": self.biomass_diffusion,
            "noise_std": self.noise_std,
            "polarity_gain": self.polarity_gain,
            "polarity_decay": self.polarity_decay,
            "polarity_mobility": self.polarity_mobility,
            "polarity_noise": self.polarity_noise,
            "max_mass": self.max_mass,
            "death_factor": self.death_factor,
            "elongation_trigger": self.elongation_trigger,
            "fission_assist": self.fission_assist,
            "render_gamma": self.render_gamma,
            "contour_level": self.contour_level,
            "dominant_band": self.dominant_band,
        }


@dataclass(frozen=True)
class Genome:
    genes: np.ndarray

    @classmethod
    def random(cls, length: int, rng: np.random.Generator) -> "Genome":
        return cls(genes=rng.uniform(-1.0, 1.0, size=length).astype(np.float32))


def _scale(val: float, lo: float, hi: float) -> float:
    return float(lo + (val + 1.0) * 0.5 * (hi - lo))


def decode(genome: Genome, *, grid_size: int, dt: float, base_noise: float = 0.002) -> CAParams:
    """Decode a genome vector into CA parameters.

    The mapping is deterministic given the provided genes.
    """

    g = genome.genes.astype(np.float32)
    if g.shape[0] < 24:
        raise ValueError("Genome must have at least 24 genes")

    r1 = _scale(g[0], 1.5, 8.0)
    r2 = max(_scale(g[1], r1 + 1.0, 20.0), r1 + 1.0)
    r3 = max(_scale(g[2], r2 + 1.0, 40.0), r2 + 1.0)
    radii = (r1, r2, r3)
    raw_weights = np.array([
        _scale(g[3], -1.5, 1.5),
        _scale(g[4], -1.5, 1.5),
        _scale(g[5], -1.5, 1.5),
    ], dtype=np.float32)
    norm = float(np.abs(raw_weights).sum())
    if norm < 1e-6:
        raw_weights = np.array([1.0, -0.5, 0.25], dtype=np.float32)
        norm = float(np.abs(raw_weights).sum())
    weights = tuple((raw_weights / norm).tolist())

    rings: tuple[tuple[float, float], ...] = tuple((0.0 if i == 0 else radii[i - 1], r) for i, r in enumerate(radii))
    kernel_params = KernelParams(size=grid_size, rings=rings, ring_weights=weights)
    sigma = _scale(g[7], 0.5, 6.0)
    mu = _scale(g[6], -0.5, 0.5)
    growth_alpha = _scale(g[8], 0.01, 0.35)
    decay_lambda = _scale(g[9], 0.0, 0.08)
    consumption_rate = _scale(g[10], 0.0, 0.25)
    regen_rate = _scale(g[11], 0.0, 0.08)
    resource_diffusion = _scale(g[12], 0.0, 0.35)
    biomass_diffusion = _scale(g[13], 0.0, 0.15)
    polarity_decay = _scale(g[14], 0.8, 0.999)
    polarity_gain = _scale(g[15], 0.0, 2.0)
    polarity_mobility = _scale(g[16], 0.0, 1.0)
    polarity_noise = _scale(g[17], 0.0, 0.02)
    max_mass = _scale(g[18], 0.4, 1.2)
    death_factor = _scale(g[19], 0.0, 0.5)
    elongation_trigger = _scale(g[20], 1.2, 3.0)
    fission_assist = _scale(g[21], 0.0, 1.0)
    gamma = _scale(g[22], 0.6, 2.2)
    contour = _scale(g[23], 0.1, 0.9)
    dominant_band = int(np.argmax(np.abs(weights)))

    return CAParams(
        grid_size=grid_size,
        kernel_params=kernel_params,
        mu=mu,
        sigma=sigma,
        dt=dt,
        growth_alpha=growth_alpha,
        decay_lambda=decay_lambda,
        regen_rate=regen_rate,
        consumption_rate=consumption_rate,
        resource_diffusion=resource_diffusion,
        biomass_diffusion=biomass_diffusion,
        noise_std=_scale(g[17], 0.0, 0.02) + base_noise,
        polarity_gain=polarity_gain,
        polarity_decay=polarity_decay,
        polarity_mobility=polarity_mobility,
        polarity_noise=polarity_noise,
        max_mass=max_mass,
        death_factor=death_factor,
        elongation_trigger=elongation_trigger,
        fission_assist=fission_assist,
        render_gamma=gamma,
        contour_level=contour,
        dominant_band=dominant_band,
    )


def mutate(genome: Genome, rng: np.random.Generator, *, sigma: float = 0.05, structural_prob: float = 0.05) -> Genome:
    """Gaussian mutation with occasional structural perturbations."""

    mutated = genome.genes.astype(np.float32) + rng.normal(0.0, sigma, size=genome.genes.shape)
    mutated = np.clip(mutated, -1.0, 1.0)
    if rng.random() < structural_prob:
        choice = rng.choice(["swap_rings", "widen", "flip_weight"])
        if choice == "swap_rings":
            mutated[[0, 1]] = mutated[[1, 0]]
        elif choice == "widen":
            idx = rng.integers(0, 3)
            mutated[idx] = np.clip(mutated[idx] + abs(rng.normal(0.0, sigma)), -1.0, 1.0)
        elif choice == "flip_weight":
            idx = rng.integers(3, 6)
            mutated[idx] = -mutated[idx]
    return Genome(mutated.astype(np.float32))


def crossover(parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Genome:
    """Blend crossover with occasional ring swaps."""

    alpha = rng.uniform(0.2, 0.8, size=parent_a.genes.shape)
    child_genes = alpha * parent_a.genes + (1.0 - alpha) * parent_b.genes
    if rng.random() < 0.2:
        # swap ring radii or weights to encourage structural jumps
        swap_idx = rng.choice([(0, 1), (1, 2), (3, 4), (4, 5)])
        child_genes[list(swap_idx)] = child_genes[list(swap_idx[::-1])]
    child_genes = np.clip(child_genes, -1.0, 1.0)
    return Genome(child_genes.astype(np.float32))


def phenotype_summary(params: CAParams) -> dict:
    rings = [{"inner": float(r[0]), "outer": float(r[1])} for r in params.kernel_params.rings]
    weights = list(params.kernel_params.ring_weights)
    return {
        "rings": rings,
        "weights": weights,
        "growth": {"mu": params.mu, "sigma": params.sigma, "alpha": params.growth_alpha, "decay": params.decay_lambda},
        "metabolism": {
            "regen_rate": params.regen_rate,
            "consumption_rate": params.consumption_rate,
            "resource_diffusion": params.resource_diffusion,
            "biomass_diffusion": params.biomass_diffusion,
        },
        "motility": {
            "polarity_gain": params.polarity_gain,
            "polarity_decay": params.polarity_decay,
            "polarity_mobility": params.polarity_mobility,
            "polarity_noise": params.polarity_noise,
        },
        "division": {
            "max_mass": params.max_mass,
            "death_factor": params.death_factor,
            "elongation_trigger": params.elongation_trigger,
            "fission_assist": params.fission_assist,
        },
        "dominant_band": params.dominant_band,
        "render": {"gamma": params.render_gamma, "contour_level": params.contour_level},
    }


def params_from_config(ca_cfg) -> CAParams:
    """Build ``CAParams`` directly from a CA config (genome bypass)."""

    kernel_params = KernelParams(
        size=ca_cfg.grid_size,
        rings=tuple(tuple(r) for r in ca_cfg.rings),
        ring_weights=tuple(ca_cfg.ring_weights),
    )
    return CAParams(
        grid_size=ca_cfg.grid_size,
        kernel_params=kernel_params,
        mu=ca_cfg.mu,
        sigma=ca_cfg.sigma,
        dt=ca_cfg.dt,
        growth_alpha=ca_cfg.growth_alpha,
        decay_lambda=getattr(ca_cfg, "decay_lambda", 0.0),
        regen_rate=ca_cfg.regen_rate,
        consumption_rate=ca_cfg.consumption_rate,
        resource_diffusion=getattr(ca_cfg, "resource_diffusion", 0.0),
        biomass_diffusion=getattr(ca_cfg, "biomass_diffusion", 0.0),
        noise_std=getattr(ca_cfg, "noise_std", 0.0),
        polarity_gain=ca_cfg.polarity_gain,
        polarity_decay=ca_cfg.polarity_decay,
        polarity_mobility=ca_cfg.polarity_mobility,
        polarity_noise=getattr(ca_cfg, "polarity_noise", 0.0),
        max_mass=ca_cfg.max_mass,
        death_factor=ca_cfg.death_factor,
        elongation_trigger=getattr(ca_cfg, "elongation_trigger", 1.3),
        fission_assist=getattr(ca_cfg, "fission_assist", 0.0),
        render_gamma=ca_cfg.gamma,
        contour_level=getattr(ca_cfg, "contour_level", 0.5),
        dominant_band=int(np.argmax(np.abs(ca_cfg.ring_weights))),
    )
