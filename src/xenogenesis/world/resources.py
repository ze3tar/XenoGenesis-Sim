"""Resource field model."""
from __future__ import annotations
import numpy as np


class ResourceField:
    def __init__(self, size: int, regen_rate: float):
        self.grid = np.zeros((size, size), dtype=np.float32)
        self.regen_rate = regen_rate

    def step(self):
        self.grid += self.regen_rate
        np.clip(self.grid, 0, 1, out=self.grid)
        return self.grid
