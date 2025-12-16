"""Terrain stub."""
from __future__ import annotations
import numpy as np


class Terrain:
    def __init__(self, size: int):
        self.heightmap = np.zeros((size, size), dtype=np.float32)

    def slope(self) -> float:
        return float(np.mean(np.abs(np.gradient(self.heightmap))))
