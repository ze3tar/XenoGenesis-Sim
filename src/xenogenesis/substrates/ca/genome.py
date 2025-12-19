"""Genome encoding and decoding for CA parameters."""
from __future__ import annotations

from dataclasses import dataclass
import json
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


# --- Structured genome schema -------------------------------------------------


def _normalize_weights(weights: Iterable[float]) -> tuple[float, ...]:
    weights_arr = np.asarray(list(weights), dtype=np.float32)
    if np.allclose(weights_arr, 0):
        weights_arr[:] = 1.0 / max(len(weights_arr), 1)
    weights_arr /= np.sum(np.abs(weights_arr))
    return tuple(float(w) for w in weights_arr)


STRUCTURED_BOUNDS: dict[str, tuple[float, float]] = {
    "kernel.inner": (1.0, 10.0),
    "kernel.outer": (2.5, 24.0),
    "kernel.ring_ratio": (0.2, 0.95),
    "growth.mu": (-0.5, 0.6),
    "growth.sigma": (0.1, 5.0),
    "growth.alpha": (0.1, 2.5),
    "growth.decay": (0.0, 0.2),
    "metabolism.regen_rate": (0.01, 0.2),
    "metabolism.consumption_rate": (0.0, 0.5),
    "metabolism.resource_diffusion": (0.0, 0.5),
    "metabolism.biomass_diffusion": (0.0, 0.3),
    "motility.polarity_gain": (0.01, 2.0),
    "motility.polarity_decay": (0.8, 0.999),
    "motility.polarity_mobility": (0.01, 0.6),
    "motility.polarity_noise": (0.0, 0.05),
    "division.division_threshold": (0.1, 0.95),
    "division.division_fraction": (0.1, 0.9),
    "division.elongation_trigger": (1.0, 4.0),
    "division.reproduction_cost": (0.01, 0.45),
    "environment_response.toxicity_resistance": (0.0, 1.0),
    "environment_response.resource_affinity": (0.0, 1.0),
    "environment_response.drift_sensitivity": (0.0, 0.08),
}


def _bounded_mutation(val: float, rng: np.random.Generator, *, key: str, sigma: float) -> float:
    lo, hi = STRUCTURED_BOUNDS[key]
    scale = (hi - lo) * sigma
    mutated = float(val + rng.normal(0.0, scale))
    return float(np.clip(mutated, lo, hi))


def _normalized_distance(a: float, b: float, *, key: str) -> float:
    lo, hi = STRUCTURED_BOUNDS[key]
    scale = hi - lo
    if scale <= 0:
        return 0.0
    return (a - b) / scale


@dataclass(frozen=True)
class KernelSection:
    inner: float
    outer: float
    weights: tuple[float, float, float]
    ring_ratio: float

    def to_dict(self) -> dict:
        return {
            "rings": [self.inner, self.outer],
            "weights": list(self.weights),
            "ring_ratio": self.ring_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KernelSection":
        inner, outer = data.get("rings", [2.0, 6.0])
        weights = data.get("weights", [1.0, -0.5, 0.2])
        ring_ratio = data.get("ring_ratio", 0.5)
        weights = weights if len(weights) >= 3 else list(weights) + [0.25 + 0.15 * ring_ratio]
        return cls(inner=float(inner), outer=float(outer), weights=_normalize_weights(weights[:3]), ring_ratio=float(ring_ratio))

    @classmethod
    def random(cls, rng: np.random.Generator) -> "KernelSection":
        inner = rng.uniform(*STRUCTURED_BOUNDS["kernel.inner"])
        outer = max(inner + 0.5, rng.uniform(*STRUCTURED_BOUNDS["kernel.outer"]))
        ring_ratio = rng.uniform(*STRUCTURED_BOUNDS["kernel.ring_ratio"])
        weights = _normalize_weights([1.0, -ring_ratio, 0.25 + 0.15 * ring_ratio])
        return cls(inner=inner, outer=outer, weights=weights, ring_ratio=ring_ratio)

    def mutate(self, rng: np.random.Generator, sigma: float) -> "KernelSection":
        inner = _bounded_mutation(self.inner, rng, key="kernel.inner", sigma=sigma)
        outer = _bounded_mutation(self.outer, rng, key="kernel.outer", sigma=sigma)
        outer = max(outer, inner + 0.25)
        ring_ratio = _bounded_mutation(self.ring_ratio, rng, key="kernel.ring_ratio", sigma=sigma)
        mutated_weights = [np.clip(w + rng.normal(0.0, sigma), -2.0, 2.0) for w in self.weights]
        return KernelSection(inner=inner, outer=outer, weights=_normalize_weights(mutated_weights), ring_ratio=ring_ratio)

    def distance(self, other: "KernelSection") -> float:
        dist = _normalized_distance(self.inner, other.inner, key="kernel.inner") ** 2
        dist += _normalized_distance(self.outer, other.outer, key="kernel.outer") ** 2
        dist += _normalized_distance(self.ring_ratio, other.ring_ratio, key="kernel.ring_ratio") ** 2
        weight_diff = np.linalg.norm(np.asarray(self.weights) - np.asarray(other.weights))
        dist += (weight_diff / len(self.weights)) ** 2
        return dist


@dataclass(frozen=True)
class GrowthSection:
    mu: float
    sigma: float
    alpha: float
    decay: float

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "alpha": self.alpha, "decay": self.decay}

    @classmethod
    def from_dict(cls, data: dict) -> "GrowthSection":
        return cls(
            mu=float(data.get("mu", 0.0)),
            sigma=float(data.get("sigma", 1.0)),
            alpha=float(data.get("alpha", 1.0)),
            decay=float(data.get("decay", 0.05)),
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> "GrowthSection":
        return cls(
            mu=rng.uniform(*STRUCTURED_BOUNDS["growth.mu"]),
            sigma=rng.uniform(*STRUCTURED_BOUNDS["growth.sigma"]),
            alpha=rng.uniform(*STRUCTURED_BOUNDS["growth.alpha"]),
            decay=rng.uniform(*STRUCTURED_BOUNDS["growth.decay"]),
        )

    def mutate(self, rng: np.random.Generator, sigma: float) -> "GrowthSection":
        return GrowthSection(
            mu=_bounded_mutation(self.mu, rng, key="growth.mu", sigma=sigma),
            sigma=_bounded_mutation(self.sigma, rng, key="growth.sigma", sigma=sigma),
            alpha=_bounded_mutation(self.alpha, rng, key="growth.alpha", sigma=sigma),
            decay=_bounded_mutation(self.decay, rng, key="growth.decay", sigma=sigma),
        )

    def distance(self, other: "GrowthSection") -> float:
        return sum(
            _normalized_distance(getattr(self, field), getattr(other, field), key=f"growth.{field}") ** 2
            for field in ("mu", "sigma", "alpha", "decay")
        )


@dataclass(frozen=True)
class MetabolismSection:
    regen_rate: float
    consumption_rate: float
    resource_diffusion: float
    biomass_diffusion: float

    def to_dict(self) -> dict:
        return {
            "regen_rate": self.regen_rate,
            "consumption_rate": self.consumption_rate,
            "resource_diffusion": self.resource_diffusion,
            "biomass_diffusion": self.biomass_diffusion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetabolismSection":
        return cls(
            regen_rate=float(data.get("regen_rate", 0.05)),
            consumption_rate=float(data.get("consumption_rate", 0.1)),
            resource_diffusion=float(data.get("resource_diffusion", 0.1)),
            biomass_diffusion=float(data.get("biomass_diffusion", 0.05)),
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> "MetabolismSection":
        return cls(
            regen_rate=rng.uniform(*STRUCTURED_BOUNDS["metabolism.regen_rate"]),
            consumption_rate=rng.uniform(*STRUCTURED_BOUNDS["metabolism.consumption_rate"]),
            resource_diffusion=rng.uniform(*STRUCTURED_BOUNDS["metabolism.resource_diffusion"]),
            biomass_diffusion=rng.uniform(*STRUCTURED_BOUNDS["metabolism.biomass_diffusion"]),
        )

    def mutate(self, rng: np.random.Generator, sigma: float) -> "MetabolismSection":
        return MetabolismSection(
            regen_rate=_bounded_mutation(self.regen_rate, rng, key="metabolism.regen_rate", sigma=sigma),
            consumption_rate=_bounded_mutation(self.consumption_rate, rng, key="metabolism.consumption_rate", sigma=sigma),
            resource_diffusion=_bounded_mutation(self.resource_diffusion, rng, key="metabolism.resource_diffusion", sigma=sigma),
            biomass_diffusion=_bounded_mutation(self.biomass_diffusion, rng, key="metabolism.biomass_diffusion", sigma=sigma),
        )

    def distance(self, other: "MetabolismSection") -> float:
        return sum(
            _normalized_distance(getattr(self, field), getattr(other, field), key=f"metabolism.{field}") ** 2
            for field in ("regen_rate", "consumption_rate", "resource_diffusion", "biomass_diffusion")
        )


@dataclass(frozen=True)
class MotilitySection:
    polarity_gain: float
    polarity_decay: float
    polarity_mobility: float
    polarity_noise: float

    def to_dict(self) -> dict:
        return {
            "polarity_gain": self.polarity_gain,
            "polarity_decay": self.polarity_decay,
            "polarity_mobility": self.polarity_mobility,
            "polarity_noise": self.polarity_noise,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MotilitySection":
        return cls(
            polarity_gain=float(data.get("polarity_gain", 1.0)),
            polarity_decay=float(data.get("polarity_decay", 0.95)),
            polarity_mobility=float(data.get("polarity_mobility", 0.2)),
            polarity_noise=float(data.get("polarity_noise", 0.01)),
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> "MotilitySection":
        return cls(
            polarity_gain=rng.uniform(*STRUCTURED_BOUNDS["motility.polarity_gain"]),
            polarity_decay=rng.uniform(*STRUCTURED_BOUNDS["motility.polarity_decay"]),
            polarity_mobility=rng.uniform(*STRUCTURED_BOUNDS["motility.polarity_mobility"]),
            polarity_noise=rng.uniform(*STRUCTURED_BOUNDS["motility.polarity_noise"]),
        )

    def mutate(self, rng: np.random.Generator, sigma: float) -> "MotilitySection":
        return MotilitySection(
            polarity_gain=_bounded_mutation(self.polarity_gain, rng, key="motility.polarity_gain", sigma=sigma),
            polarity_decay=_bounded_mutation(self.polarity_decay, rng, key="motility.polarity_decay", sigma=sigma),
            polarity_mobility=_bounded_mutation(self.polarity_mobility, rng, key="motility.polarity_mobility", sigma=sigma),
            polarity_noise=_bounded_mutation(self.polarity_noise, rng, key="motility.polarity_noise", sigma=sigma),
        )

    def distance(self, other: "MotilitySection") -> float:
        return sum(
            _normalized_distance(getattr(self, field), getattr(other, field), key=f"motility.{field}") ** 2
            for field in ("polarity_gain", "polarity_decay", "polarity_mobility", "polarity_noise")
        )


@dataclass(frozen=True)
class DivisionSection:
    division_threshold: float
    division_fraction: float
    elongation_trigger: float
    reproduction_cost: float

    def to_dict(self) -> dict:
        return {
            "division_threshold": self.division_threshold,
            "division_fraction": self.division_fraction,
            "elongation_trigger": self.elongation_trigger,
            "reproduction_cost": self.reproduction_cost,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DivisionSection":
        return cls(
            division_threshold=float(data.get("division_threshold", 0.5)),
            division_fraction=float(data.get("division_fraction", 0.5)),
            elongation_trigger=float(data.get("elongation_trigger", 1.5)),
            reproduction_cost=float(data.get("reproduction_cost", 0.15)),
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> "DivisionSection":
        return cls(
            division_threshold=rng.uniform(*STRUCTURED_BOUNDS["division.division_threshold"]),
            division_fraction=rng.uniform(*STRUCTURED_BOUNDS["division.division_fraction"]),
            elongation_trigger=rng.uniform(*STRUCTURED_BOUNDS["division.elongation_trigger"]),
            reproduction_cost=rng.uniform(*STRUCTURED_BOUNDS["division.reproduction_cost"]),
        )

    def mutate(self, rng: np.random.Generator, sigma: float) -> "DivisionSection":
        return DivisionSection(
            division_threshold=_bounded_mutation(self.division_threshold, rng, key="division.division_threshold", sigma=sigma),
            division_fraction=_bounded_mutation(self.division_fraction, rng, key="division.division_fraction", sigma=sigma),
            elongation_trigger=_bounded_mutation(self.elongation_trigger, rng, key="division.elongation_trigger", sigma=sigma),
            reproduction_cost=_bounded_mutation(self.reproduction_cost, rng, key="division.reproduction_cost", sigma=sigma),
        )

    def distance(self, other: "DivisionSection") -> float:
        return sum(
            _normalized_distance(getattr(self, field), getattr(other, field), key=f"division.{field}") ** 2
            for field in ("division_threshold", "division_fraction", "elongation_trigger", "reproduction_cost")
        )


@dataclass(frozen=True)
class EnvironmentSection:
    toxicity_resistance: float
    resource_affinity: float
    drift_sensitivity: float

    def to_dict(self) -> dict:
        return {
            "toxicity_resistance": self.toxicity_resistance,
            "resource_affinity": self.resource_affinity,
            "drift_sensitivity": self.drift_sensitivity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentSection":
        return cls(
            toxicity_resistance=float(data.get("toxicity_resistance", 0.5)),
            resource_affinity=float(data.get("resource_affinity", 0.5)),
            drift_sensitivity=float(data.get("drift_sensitivity", 0.02)),
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> "EnvironmentSection":
        return cls(
            toxicity_resistance=rng.uniform(*STRUCTURED_BOUNDS["environment_response.toxicity_resistance"]),
            resource_affinity=rng.uniform(*STRUCTURED_BOUNDS["environment_response.resource_affinity"]),
            drift_sensitivity=rng.uniform(*STRUCTURED_BOUNDS["environment_response.drift_sensitivity"]),
        )

    def mutate(self, rng: np.random.Generator, sigma: float) -> "EnvironmentSection":
        return EnvironmentSection(
            toxicity_resistance=_bounded_mutation(self.toxicity_resistance, rng, key="environment_response.toxicity_resistance", sigma=sigma),
            resource_affinity=_bounded_mutation(self.resource_affinity, rng, key="environment_response.resource_affinity", sigma=sigma),
            drift_sensitivity=_bounded_mutation(self.drift_sensitivity, rng, key="environment_response.drift_sensitivity", sigma=sigma),
        )

    def distance(self, other: "EnvironmentSection") -> float:
        return sum(
            _normalized_distance(getattr(self, field), getattr(other, field), key=f"environment_response.{field}") ** 2
            for field in ("toxicity_resistance", "resource_affinity", "drift_sensitivity")
        )


@dataclass(frozen=True)
class StructuredGenome:
    """Versioned structured genome compatible with the Lenia-like CA."""

    kernel: KernelSection
    growth: GrowthSection
    metabolism: MetabolismSection
    motility: MotilitySection
    division: DivisionSection
    environment_response: EnvironmentSection
    version: str = "v1"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "kernel": self.kernel.to_dict(),
            "growth": self.growth.to_dict(),
            "metabolism": self.metabolism.to_dict(),
            "motility": self.motility.to_dict(),
            "division": self.division.to_dict(),
            "environment_response": self.environment_response.to_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredGenome":
        return cls(
            kernel=KernelSection.from_dict(data.get("kernel", {})),
            growth=GrowthSection.from_dict(data.get("growth", {})),
            metabolism=MetabolismSection.from_dict(data.get("metabolism", {})),
            motility=MotilitySection.from_dict(data.get("motility", {})),
            division=DivisionSection.from_dict(data.get("division", {})),
            environment_response=EnvironmentSection.from_dict(data.get("environment_response", {})),
            version=data.get("version", "v1"),
        )

    @classmethod
    def from_json(cls, text: str) -> "StructuredGenome":
        return cls.from_dict(json.loads(text))

    @classmethod
    def random(cls, rng: np.random.Generator) -> "StructuredGenome":
        return cls(
            kernel=KernelSection.random(rng),
            growth=GrowthSection.random(rng),
            metabolism=MetabolismSection.random(rng),
            motility=MotilitySection.random(rng),
            division=DivisionSection.random(rng),
            environment_response=EnvironmentSection.random(rng),
        )

    def mutate(self, rng: np.random.Generator, *, sigma: float = 0.05) -> "StructuredGenome":
        return StructuredGenome(
            kernel=self.kernel.mutate(rng, sigma),
            growth=self.growth.mutate(rng, sigma),
            metabolism=self.metabolism.mutate(rng, sigma),
            motility=self.motility.mutate(rng, sigma),
            division=self.division.mutate(rng, sigma),
            environment_response=self.environment_response.mutate(rng, sigma),
            version=self.version,
        )

    def distance(self, other: "StructuredGenome") -> float:
        if self.version != other.version:
            return float("inf")
        dist = 0.0
        dist += self.kernel.distance(other.kernel)
        dist += self.growth.distance(other.growth)
        dist += self.metabolism.distance(other.metabolism)
        dist += self.motility.distance(other.motility)
        dist += self.division.distance(other.division)
        dist += self.environment_response.distance(other.environment_response)
        return float(np.sqrt(dist))

    def to_ca_params(self, *, grid_size: int, dt: float, base_noise: float = 0.002) -> CAParams:
        tertiary_radius = self.kernel.outer * (1.0 + 0.35 * self.kernel.ring_ratio)
        rings = (
            (0.0, self.kernel.inner),
            (self.kernel.inner, self.kernel.outer),
            (self.kernel.outer, tertiary_radius),
        )
        ring_weights = _normalize_weights(self.kernel.weights)
        kernel_params = KernelParams(size=grid_size, rings=rings, ring_weights=ring_weights)
        dominant_band = int(np.argmax(np.abs(ring_weights)))
        toxin_rate = max(0.0, 1.0 - self.environment_response.toxicity_resistance)
        return CAParams(
            grid_size=grid_size,
            kernel_params=kernel_params,
            mu=self.growth.mu,
            sigma=self.growth.sigma,
            dt=dt,
            growth_alpha=self.growth.alpha,
            decay_lambda=self.growth.decay,
            regen_rate=self.metabolism.regen_rate,
            consumption_rate=self.metabolism.consumption_rate,
            resource_diffusion=self.metabolism.resource_diffusion,
            biomass_diffusion=self.metabolism.biomass_diffusion,
            noise_std=base_noise,
            polarity_gain=self.motility.polarity_gain,
            polarity_decay=self.motility.polarity_decay,
            polarity_mobility=self.motility.polarity_mobility,
            polarity_noise=self.motility.polarity_noise,
            max_mass=1.0,
            death_factor=0.05,
            elongation_trigger=self.division.elongation_trigger,
            fission_assist=0.0,
            render_gamma=1.0,
            contour_level=0.5,
            dominant_band=dominant_band,
            maintenance_cost=max(0.01, self.metabolism.consumption_rate * 0.5),
            competition_scale=0.25,
            competition_radius=3.0,
            resource_capacity=1.2,
            resource_gradient=self.environment_response.resource_affinity,
            polarity_diffusion=0.05,
            polarity_mutation=0.02,
            directional_gain=max(0.1, self.motility.polarity_gain * 0.5),
            division_threshold=self.division.division_threshold,
            division_fraction=self.division.division_fraction,
            reproduction_cost=self.division.reproduction_cost,
            resource_affinity=self.environment_response.resource_affinity,
            toxin_rate=toxin_rate,
            drift_rate=self.environment_response.drift_sensitivity,
        )
