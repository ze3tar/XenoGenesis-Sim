"""Speciation placeholder using distance threshold."""
from __future__ import annotations
from typing import List
import numpy as np


def simple_speciation(descriptors: List[List[float]], threshold: float = 0.5) -> int:
    if not descriptors:
        return 0
    clusters = 1
    for i in range(1, len(descriptors)):
        dist = np.linalg.norm(np.array(descriptors[i]) - np.array(descriptors[i-1]))
        if dist > threshold:
            clusters += 1
    return clusters
