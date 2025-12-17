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
    resource_regen: float = 0.05


class CAConfig(BaseModel):
    grid_size: int = 256
    mu: float = 0.15
    sigma: float = 0.015
    dt: float = 0.1
    rings: list[list[float]] = Field(default_factory=lambda: [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0]])
    ring_weights: list[float] = Field(default_factory=lambda: [1.0, -0.5, 0.2])
    regen_rate: float = 0.05
    consumption_rate: float = 0.02
    noise_std: float = 0.002
    mass_threshold: float = 0.05
    active_threshold: float = 0.01
    gamma: float = 1.0
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
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path) -> ConfigSchema:
    data = yaml.safe_load(Path(path).read_text())
    return ConfigSchema(**data)
