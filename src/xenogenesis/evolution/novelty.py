"""Novelty search helper."""
from __future__ import annotations
from typing import List
import numpy as np


def novelty_score(descriptor: List[float], archive: List[List[float]], k: int = 3) -> float:
    if not archive:
        return float(np.mean(descriptor))
    dists = [np.linalg.norm(np.array(descriptor) - np.array(a)) for a in archive]
    dists = sorted(dists)[:k]
    return float(np.mean(dists))
