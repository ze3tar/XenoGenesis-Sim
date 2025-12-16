"""Planet environment parameters."""
from __future__ import annotations
from dataclasses import dataclass
from xenogenesis.config import EnvironmentConfig


@dataclass
class PlanetEnv:
    config: EnvironmentConfig

    def hazards(self) -> float:
        return self.config.radiation

    def resource_field(self) -> float:
        return self.config.resource_regen
