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
    maintenance_cost: float
    competition_scale: float
    competition_radius: float
    resource_capacity: float
    resource_gradient: float
    polarity_diffusion: float
    polarity_mutation: float
    directional_gain: float
    division_threshold: float
    division_fraction: float
    reproduction_cost: float
    resource_affinity: float
    toxin_rate: float
    drift_rate: float

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
            "maintenance_cost": self.maintenance_cost,
            "competition_scale": self.competition_scale,
            "competition_radius": self.competition_radius,
            "resource_capacity": self.resource_capacity,
            "resource_gradient": self.resource_gradient,
            "polarity_diffusion": self.polarity_diffusion,
            "polarity_mutation": self.polarity_mutation,
            "directional_gain": self.directional_gain,
            "division_threshold": self.division_threshold,
            "division_fraction": self.division_fraction,
            "reproduction_cost": self.reproduction_cost,
            "resource_affinity": self.resource_affinity,
            "toxin_rate": self.toxin_rate,
            "drift_rate": self.drift_rate,
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
    if g.shape[0] < 8:
        raise ValueError("Genome must have at least 8 genes")

    inner_radius = _scale(g[0], 1.2, 7.5)
    outer_radius = max(_scale(g[1], inner_radius + 1.0, 20.0), inner_radius + 1.0)
    ring_ratio = _scale(g[2], 0.2, 0.95)
    tertiary_radius = outer_radius * (1.0 + 0.35 * ring_ratio)
    radii = (inner_radius, outer_radius, tertiary_radius)
    weights = np.array([1.0, -ring_ratio, 0.25 + 0.15 * ring_ratio], dtype=np.float32)
    weights = tuple((weights / np.abs(weights).sum()).tolist())

    rings: tuple[tuple[float, float], ...] = tuple((0.0 if i == 0 else radii[i - 1], r) for i, r in enumerate(radii))
    kernel_params = KernelParams(size=grid_size, rings=rings, ring_weights=weights)
    growth_alpha = _scale(g[3], 0.2, 1.6)
    maintenance_cost = _scale(g[4], 0.02, 0.45)
    polarity_gain = _scale(g[5], 0.05, 1.6)
    division_threshold = _scale(g[6], 0.25, 0.85)
    resource_affinity = _scale(g[7], 0.1, 0.9)

    # Optional extended genes keep backward compatibility with older genomes.
    sigma = _scale(g[8] if g.shape[0] > 8 else 0.0, 0.2, 4.5)
    mu = _scale(g[9] if g.shape[0] > 9 else 0.0, -0.25, 0.45)
    decay_lambda = _scale(g[10] if g.shape[0] > 10 else 0.0, 0.0, 0.08)
    consumption_rate = _scale(g[11] if g.shape[0] > 11 else 0.0, 0.0, 0.25)
    regen_rate = _scale(g[12] if g.shape[0] > 12 else 0.0, 0.02, 0.12)
    resource_diffusion = _scale(g[13] if g.shape[0] > 13 else 0.0, 0.0, 0.35)
    biomass_diffusion = _scale(g[14] if g.shape[0] > 14 else 0.0, 0.0, 0.15)
    polarity_decay = _scale(g[15] if g.shape[0] > 15 else 0.0, 0.85, 0.999)
    polarity_mobility = _scale(g[16] if g.shape[0] > 16 else 0.0, 0.02, 0.45)
    polarity_noise = _scale(g[17] if g.shape[0] > 17 else 0.0, 0.0, 0.02)
    max_mass = _scale(g[18] if g.shape[0] > 18 else 0.0, 0.4, 1.15)
    death_factor = _scale(g[19] if g.shape[0] > 19 else 0.0, 0.0, 0.5)
    elongation_trigger = _scale(g[20] if g.shape[0] > 20 else 0.0, 1.1, 3.0)
    fission_assist = _scale(g[21] if g.shape[0] > 21 else 0.0, 0.0, 1.0)
    gamma = _scale(g[22] if g.shape[0] > 22 else 0.0, 0.6, 2.2)
    contour = _scale(g[23] if g.shape[0] > 23 else 0.0, 0.1, 0.9)
    competition_scale = _scale(g[24] if g.shape[0] > 24 else 0.0, 0.02, 0.6)
    directional_gain = _scale(g[25] if g.shape[0] > 25 else 0.0, 0.2, 2.5)
    competition_radius = _scale(g[26] if g.shape[0] > 26 else 0.0, 1.0, 5.0)
    division_fraction = _scale(g[27] if g.shape[0] > 27 else 0.0, 0.25, 0.65)
    reproduction_cost = _scale(g[28] if g.shape[0] > 28 else 0.0, 0.05, 0.35)
    resource_capacity = _scale(g[29] if g.shape[0] > 29 else 0.0, 1.0, 1.5)
    toxin_rate = _scale(g[30] if g.shape[0] > 30 else 0.0, 0.0, 0.25)
    drift_rate = _scale(g[31] if g.shape[0] > 31 else 0.0, 0.0, 0.05)

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
        noise_std=_scale(g[17] if g.shape[0] > 17 else 0.0, 0.0, 0.02) + base_noise,
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
        maintenance_cost=maintenance_cost,
        competition_scale=competition_scale,
        competition_radius=competition_radius,
        resource_capacity=resource_capacity,
        resource_gradient=resource_affinity,
        polarity_diffusion=0.05,
        polarity_mutation=0.02,
        directional_gain=directional_gain,
        division_threshold=division_threshold,
        division_fraction=division_fraction,
        reproduction_cost=reproduction_cost,
        resource_affinity=resource_affinity,
        toxin_rate=toxin_rate,
        drift_rate=drift_rate,
    )


def mutate(genome: Genome, rng: np.random.Generator, *, sigma: float = 0.05, structural_prob: float = 0.05) -> Genome:
    """Gaussian mutation with occasional structural perturbations."""

    mutated = genome.genes.astype(np.float32) + rng.normal(0.0, sigma, size=genome.genes.shape)
    mutated = np.clip(mutated, -1.0, 1.0)
    if rng.random() < structural_prob:
        choices = ["swap_rings", "widen", "flip_weight"]
        if mutated.shape[0] > 24:
            choices.append("division_shift")
        choice = rng.choice(choices)
        if choice == "swap_rings" and mutated.shape[0] >= 2:
            mutated[[0, 1]] = mutated[[1, 0]]
        elif choice == "widen":
            idx = rng.integers(0, min(3, mutated.shape[0]))
            mutated[idx] = np.clip(mutated[idx] + abs(rng.normal(0.0, sigma)), -1.0, 1.0)
        elif choice == "flip_weight" and mutated.shape[0] >= 6:
            idx = rng.integers(3, 6)
            mutated[idx] = -mutated[idx]
        elif choice == "division_shift" and mutated.shape[0] > 27:
            mutated[27] = np.clip(mutated[27] + rng.normal(0.0, sigma), -1.0, 1.0)
    return Genome(mutated.astype(np.float32))


def crossover(parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Genome:
    """Blend crossover with occasional ring swaps."""

    alpha = rng.uniform(0.2, 0.8, size=parent_a.genes.shape)
    child_genes = alpha * parent_a.genes + (1.0 - alpha) * parent_b.genes
    if rng.random() < 0.2:
        # swap ring radii or weights to encourage structural jumps
        swap_idx = rng.choice([(0, 1), (1, 2), (3, 4), (4, 5)])
        swap_idx = [i for i in swap_idx if i < len(child_genes)]
        if len(swap_idx) == 2:
            child_genes[swap_idx] = child_genes[list(reversed(swap_idx))]
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
            "division_threshold": params.division_threshold,
            "division_fraction": params.division_fraction,
        },
        "metabolic_costs": {
            "maintenance": params.maintenance_cost,
            "competition_scale": params.competition_scale,
            "resource_capacity": params.resource_capacity,
            "resource_affinity": params.resource_affinity,
            "reproduction_cost": params.reproduction_cost,
        },
        "environment": {
            "resource_gradient": params.resource_gradient,
            "toxicity": params.toxin_rate,
            "drift_rate": params.drift_rate,
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
        maintenance_cost=getattr(ca_cfg, "maintenance_cost", 0.05),
        competition_scale=getattr(ca_cfg, "competition_scale", 0.2),
        competition_radius=getattr(ca_cfg, "competition_radius", 3.0),
        resource_capacity=getattr(ca_cfg, "resource_capacity", 1.2),
        resource_gradient=getattr(ca_cfg, "resource_gradient", 0.25),
        polarity_diffusion=getattr(ca_cfg, "polarity_diffusion", 0.02),
        polarity_mutation=getattr(ca_cfg, "polarity_mutation", 0.02),
        directional_gain=getattr(ca_cfg, "directional_gain", 1.0),
        division_threshold=getattr(ca_cfg, "division_threshold", 0.5),
        division_fraction=getattr(ca_cfg, "division_fraction", 0.5),
        reproduction_cost=getattr(ca_cfg, "reproduction_cost", 0.15),
        resource_affinity=getattr(ca_cfg, "resource_affinity", 0.35),
        toxin_rate=getattr(ca_cfg, "toxin_rate", 0.01),
        drift_rate=getattr(ca_cfg, "drift_rate", 0.01),
    )
