"""Hazard field placeholder."""
from __future__ import annotations
import numpy as np


class HazardField:
    def __init__(self, size: int, intensity: float):
        self.grid = np.full((size, size), intensity, dtype=np.float32)

    def sample(self) -> float:
        return float(np.mean(self.grid))
