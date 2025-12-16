"""Pydantic config schema and loader."""
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field, validator


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
    inner_radius: float = 3.0
    outer_radius: float = 6.0
    ring_ratio: float = 0.5
    steps: int = 256
    novelty_enabled: bool = False

    @validator("grid_size")
    def validate_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("grid_size must be positive")
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


class ConfigSchema(BaseModel):
    seed: int = 0
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    ca: CAConfig = Field(default_factory=CAConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path) -> ConfigSchema:
    data = yaml.safe_load(Path(path).read_text())
    return ConfigSchema(**data)
