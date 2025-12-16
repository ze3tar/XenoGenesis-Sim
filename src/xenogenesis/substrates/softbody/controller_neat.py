"""Minimal controller placeholder (NEAT-lite)."""
from __future__ import annotations
import numpy as np


class Controller:
    def __init__(self, weights=None):
        self.weights = np.array(weights if weights is not None else [0.5, -0.2, 0.1], dtype=np.float32)

    def act(self, obs: np.ndarray) -> float:
        obs = obs.flatten()
        return float((obs[: len(self.weights)] * self.weights).sum())
