"""Pydantic config schema and loader."""
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationInfo


class EnvironmentConfig(BaseModel):
    gravity: float = 9.81
    temperature: float = 288.0
    radiation: float = 0.1
    resource_regen: float = 0.01
    resource_diffusion: float = 0.15
    resource_cap: float = 1.0


class CAConfig(BaseModel):
    grid_size: int = 256
    mu: float = 0.15
    sigma: float = 0.015
    dt: float = 0.1
    growth_alpha: float = 0.8
    rings: list[list[float]] = Field(default_factory=lambda: [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0]])
    ring_weights: list[float] = Field(default_factory=lambda: [1.0, -0.5, 0.2])
    regen_rate: float = 0.05
    consumption_rate: float = 0.02
    noise_std: float = 0.002
    decay_lambda: float = 0.0
    resource_diffusion: float = 0.0
    biomass_diffusion: float = 0.0
    polarity_gain: float = 0.35
    polarity_decay: float = 0.94
    polarity_mobility: float = 0.05
    polarity_noise: float = 0.0005
    max_mass: float = 0.8
    death_factor: float = 0.55
    elongation_trigger: float = 1.3
    fission_assist: float = 0.0
    mass_threshold: float = 0.05
    active_threshold: float = 0.01
    maintenance_cost: float = 0.08
    competition_scale: float = 0.25
    competition_radius: float = 3.0
    resource_capacity: float = 1.2
    resource_gradient: float = 0.25
    polarity_diffusion: float = 0.02
    polarity_mutation: float = 0.02
    directional_gain: float = 1.0
    division_threshold: float = 0.55
    division_fraction: float = 0.45
    reproduction_cost: float = 0.15
    resource_affinity: float = 0.35
    toxin_rate: float = 0.01
    drift_rate: float = 0.01
    split_threshold: float = 0.5
    split_gain: float = 0.6
    motility_gain: float = 0.2
    death_threshold: float = 0.02
    gamma: float = 1.0
    contour_level: float = 0.5
    show_contours: bool = False
    steps: int = 256
    record_interval: int = 8
    render_stride: int = 4
    render_cmap: str = "magma"
    novelty_enabled: bool = False

    @field_validator("grid_size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("grid_size must be positive")
        return v

    @field_validator("rings", "ring_weights", mode="before")
    @classmethod
    def validate_lists(cls, v):
        if isinstance(v, tuple):
            return list(v)
        return v

    @field_validator("ring_weights")
    @classmethod
    def validate_ring_weights(cls, v, info: ValidationInfo):
        rings = info.data.get("rings", [])
        if len(v) != len(rings):
            raise ValueError("ring_weights must match rings length")
        return v

    @field_validator("rings")
    @classmethod
    def validate_rings(cls, v):
        for r in v:
            if len(r) != 2 or r[0] < 0 or r[1] <= r[0]:
                raise ValueError("rings must be [inner, outer] with outer>inner>=0")
        return v

    @field_validator(
        "growth_alpha",
        "polarity_gain",
        "polarity_mobility",
        "regen_rate",
        "consumption_rate",
        "maintenance_cost",
        "competition_scale",
        "resource_capacity",
        "resource_gradient",
        "directional_gain",
        "division_threshold",
        "division_fraction",
        "reproduction_cost",
        "resource_affinity",
        "split_threshold",
        "split_gain",
        "motility_gain",
        "death_threshold",
    )
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("rates must be non-negative")
        return v

    @field_validator("polarity_decay", "death_factor")
    @classmethod
    def validate_unit_interval(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("parameters must be within [0, 1]")
        return v

    @field_validator("drift_rate", "toxin_rate", mode="before")
    @classmethod
    def validate_small(cls, v: float) -> float:
        return float(v)


class GenomeConfig(BaseModel):
    enabled: bool = False
    length: int = 36
    mutation_sigma: float = 0.05
    structural_prob: float = 0.05


class EvolutionConfig(BaseModel):
    population: int = 64
    generations: int = 40
    selection: str = "nsga2"
    mutation_rate: float = 0.1
    workers: int = 4
    checkpoint_interval: int = 5


class OutputConfig(BaseModel):
    run_dir: Path = Path("runs")
    render: bool = True
    archive_limit: int = 1000
    summarize: bool = True


class ConfigSchema(BaseModel):
    seed: int = 0
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    ca: CAConfig = Field(default_factory=CAConfig)
    genome: GenomeConfig = Field(default_factory=GenomeConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path) -> ConfigSchema:
    data = yaml.safe_load(Path(path).read_text())
    return ConfigSchema(**data)
