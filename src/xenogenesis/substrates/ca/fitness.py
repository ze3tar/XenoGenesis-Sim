"""Fitness metrics for CA patterns."""
from __future__ import annotations
import importlib.util
import importlib
import numpy as np
from xenogenesis.core.rng import make_rng
native_entropy = None
_native_spec = importlib.util.find_spec("xenogenesis_native")
if _native_spec:
    native_entropy = importlib.import_module("xenogenesis_native").entropy  # type: ignore


def _entropy(arr: np.ndarray) -> float:
    if native_entropy:
        return float(native_entropy(arr.astype(np.float32).ravel().tolist()))
    hist, _ = np.histogram(arr, bins=10, range=(0, 1), density=True)
    hist = hist + 1e-9
    return float(-(hist * np.log2(hist)).sum())


def ca_fitness(states: list[np.ndarray]) -> dict:
    # states: list of time-series arrays
    if not states:
        return {"persistence": 0.0, "complexity": 0.0, "motility": 0.0, "energy": 0.0}
    last = states[-1]
    persistence = float(np.mean([np.mean(s > 0.01) for s in states]))
    entropy = _entropy(last)
    edges = np.mean(np.abs(np.gradient(last)))
    com = np.array(np.nonzero(last > 0.1)).mean(axis=1) if np.any(last > 0.1) else np.zeros(2)
    motility = float(np.linalg.norm(com) / max(len(states), 1))
    energy = float(np.mean(last))
    complexity = float(entropy + edges)
    return {"persistence": persistence, "complexity": complexity, "motility": motility, "energy": energy}
