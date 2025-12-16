"""Voxel morphology representation."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VoxelMorphology:
    shape: Tuple[int, int] = (4, 4)

    def to_array(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random(self.shape, dtype=np.float32)
