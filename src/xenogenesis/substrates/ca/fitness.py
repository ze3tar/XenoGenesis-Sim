"""Fitness metrics for CA patterns."""
from __future__ import annotations
import importlib.util
import importlib
import numpy as np

native_entropy = None
_native_spec = importlib.util.find_spec("xenogenesis_native")
if _native_spec:
    native_entropy = importlib.import_module("xenogenesis_native").entropy  # type: ignore


def _entropy(arr: np.ndarray) -> float:
    if native_entropy:
        return float(native_entropy(arr.astype(np.float32).ravel().tolist()))
    hist, _ = np.histogram(arr, bins=32, range=(0, 1), density=True)
    hist = hist + 1e-9
    return float(-(hist * np.log2(hist)).sum())


def _center_of_mass(state: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    mask = state > threshold
    if not np.any(mask):
        return np.zeros(2)
    coords = np.argwhere(mask)
    weights = state[mask]
    weighted = (coords * weights[:, None]).sum(axis=0) / weights.sum()
    return weighted


def behavior_descriptor(states: list[np.ndarray]) -> np.ndarray:
    if not states:
        return np.zeros(4, dtype=np.float32)
    entropy_series = np.array([_entropy(s) for s in states], dtype=np.float32)
    edge_series = np.array([np.mean(np.abs(np.gradient(s))) for s in states], dtype=np.float32)
    com_series = np.stack([_center_of_mass(s) for s in states])
    displacement = np.linalg.norm(com_series[-1] - com_series[0]) if len(com_series) > 1 else 0.0
    mass_series = np.array([float(np.mean(s)) for s in states], dtype=np.float32)
    energy_period = float(np.argmax(np.correlate(mass_series - mass_series.mean(), mass_series - mass_series.mean(), mode="full")) - (len(mass_series) - 1)) if len(mass_series) > 4 else 0.0
    com_speed = float(displacement / max(len(states) - 1, 1))
    return np.array([
        float(entropy_series.mean()),
        float(edge_series.mean()),
        com_speed,
        energy_period,
    ], dtype=np.float32)


def ca_fitness(states: list[np.ndarray], resource_penalty: float = 0.25) -> dict:
    """Compute multi-objective scores plus a novelty descriptor."""
    if not states:
        return {
            "persistence": 0.0,
            "complexity": 0.0,
            "motility": 0.0,
            "energy_efficiency": 0.0,
            "descriptor": [0.0, 0.0, 0.0, 0.0],
        }
    descriptor = behavior_descriptor(states)
    entropy_mean, edge_mean, com_speed, energy_period = descriptor.tolist()
    persistence = float(np.mean([np.mean(s > 0.05) for s in states]))
    energy = float(np.mean([np.mean(s) for s in states]))
    motility = com_speed
    complexity = float(entropy_mean + edge_mean)
    energy_efficiency = float(complexity * np.clip(1.0 - resource_penalty * energy, 0.0, 1.0))
    return {
        "persistence": persistence,
        "complexity": complexity,
        "motility": motility,
        "energy_efficiency": energy_efficiency,
        "descriptor": descriptor.tolist(),
        "energy": energy,
        "edge_density": float(edge_mean),
        "entropy": float(entropy_mean),
        "periodicity": energy_period,
    }
