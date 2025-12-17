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


def ca_fitness(
    states: list[np.ndarray],
    *,
    resource_penalty: float = 0.25,
    mass_threshold: float = 0.05,
    active_threshold: float = 0.01,
) -> dict:
    """Compute multi-objective scores plus a novelty descriptor."""

    if not states:
        return {
            "persistence": 0.0,
            "complexity": 0.0,
            "motility": 0.0,
            "energy_efficiency": 0.0,
            "descriptor": [0.0, 0.0, 0.0, 0.0],
            "reproduction_events": 0.0,
            "component_count": 0.0,
            "longevity": 0.0,
        }

    descriptor = behavior_descriptor(states)
    entropy_mean, edge_mean, com_speed, energy_period = descriptor.tolist()
    persistence_series = np.array([np.mean(s > 0.05) for s in states], dtype=np.float32)
    energy_series = np.array([np.mean(s) for s in states], dtype=np.float32)

    def _component_count(mask: np.ndarray) -> int:
        visited = np.zeros_like(mask, dtype=bool)
        count = 0
        h, w = mask.shape
        for i in range(h):
            for j in range(w):
                if mask[i, j] and not visited[i, j]:
                    count += 1
                    stack = [(i, j)]
                    visited[i, j] = True
                    while stack:
                        x, y = stack.pop()
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))
        return count

    component_counts = np.array([_component_count(s > 0.2) for s in states], dtype=np.int32)
    reproduction_events = float(np.maximum(np.diff(component_counts), 0).sum())
    longevity = float(np.sum(energy_series > energy_series.max() * 0.05))

    motility = float(com_speed)
    complexity = float(entropy_mean + edge_mean)
    energy = float(energy_series.mean())
    energy_efficiency = float(complexity * np.clip(1.0 - resource_penalty * energy, 0.0, 1.0))
    persistence = float(persistence_series.mean())

    survival_ok = float(energy_series[-1] >= mass_threshold and persistence_series[-1] >= active_threshold)
    if not survival_ok:
        persistence = 0.0
        complexity = 0.0
        motility = 0.0
        energy_efficiency = 0.0

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
        "reproduction_events": reproduction_events,
        "component_count": float(component_counts[-1]),
        "longevity": longevity,
        "survived": bool(survival_ok),
    }
