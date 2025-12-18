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


def _components(state: np.ndarray, threshold: float = 0.1) -> list[dict]:
    mask = state > threshold
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    comps: list[dict] = []
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            pixels = []
            while stack:
                x, y = stack.pop()
                pixels.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
            coords = np.array(pixels)
            masses = state[coords[:, 0], coords[:, 1]]
            centroid = (coords * masses[:, None]).sum(axis=0) / max(masses.sum(), 1e-6)
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            elongation = float(max(height, width) / max(min(height, width), 1.0))
            comps.append(
                {
                    "mass": float(masses.sum()),
                    "area": float(len(pixels)),
                    "centroid": centroid.astype(np.float32),
                    "bbox": (height, width),
                    "elongation": elongation,
                }
            )
    return comps


def _track_longevity(component_series: list[list[dict]], match_radius: float = 5.0) -> float:
    tracks: list[dict] = []
    for comps in component_series:
        updated_tracks: list[dict] = []
        for comp in comps:
            centroid = comp["centroid"]
            best_idx = None
            best_dist = 1e9
            for idx, track in enumerate(tracks):
                dist = float(np.linalg.norm(track["centroid"] - centroid))
                if dist < best_dist and dist <= match_radius:
                    best_idx = idx
                    best_dist = dist
            if best_idx is not None:
                track = tracks[best_idx]
                track["centroid"] = centroid
                track["age"] += 1
                updated_tracks.append(track)
            else:
                updated_tracks.append({"centroid": centroid, "age": 1})
        tracks = updated_tracks
    if not tracks:
        return 0.0
    return float(np.mean([t["age"] for t in tracks]))


def behavior_descriptor(states: list[np.ndarray]) -> np.ndarray:
    if not states:
        return np.zeros(6, dtype=np.float32)
    entropy_series = np.array([_entropy(s) for s in states], dtype=np.float32)
    edge_series = np.array([np.mean(np.abs(np.gradient(s))) for s in states], dtype=np.float32)
    com_series = np.stack([_center_of_mass(s) for s in states])
    displacement = np.linalg.norm(com_series[-1] - com_series[0]) if len(com_series) > 1 else 0.0
    mass_series = np.array([float(np.mean(s)) for s in states], dtype=np.float32)
    energy_period = float(np.argmax(np.correlate(mass_series - mass_series.mean(), mass_series - mass_series.mean(), mode="full")) - (len(mass_series) - 1)) if len(mass_series) > 4 else 0.0
    com_speed = float(displacement / max(len(states) - 1, 1))
    elongations = np.array([np.mean([c["elongation"] for c in _components(s, 0.15)]) if np.any(s > 0.15) else 0.0 for s in states], dtype=np.float32)
    return np.array(
        [
            float(entropy_series.mean()),
            float(edge_series.mean()),
            com_speed,
            energy_period,
            float(elongations.mean()),
            float(np.median(elongations)),
        ],
        dtype=np.float32,
    )


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
            "descriptor": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reproduction_events": 0.0,
            "component_count": 0.0,
            "longevity": 0.0,
            "component_longevity": 0.0,
            "morphological_complexity": 0.0,
            "reproduction_rate": 0.0,
            "motility_score": 0.0,
            "elongation": 0.0,
            "active_fraction": 0.0,
        }

    descriptor = behavior_descriptor(states)
    entropy_mean, edge_mean, com_speed, energy_period, elongation_mean, elongation_median = descriptor.tolist()
    persistence_series = np.array([np.mean(s > 0.05) for s in states], dtype=np.float32)
    energy_series = np.array([np.mean(s) for s in states], dtype=np.float32)

    component_series = [_components(s, 0.2) for s in states]
    component_counts = np.array([len(c) for c in component_series], dtype=np.int32)
    reproduction_events = 0
    for prev, new in zip(component_series[:-1], component_series[1:]):
        if not prev:
            continue
        if len(new) > len(prev):
            elongated_parent = max((c["elongation"] for c in prev), default=1.0)
            parent_mass = sum(c["mass"] for c in prev)
            child_mass = sum(c["mass"] for c in new)
            if elongated_parent >= 1.35 and 0.4 * parent_mass <= child_mass <= 1.6 * parent_mass:
                reproduction_events += len(new) - len(prev)
    longevity = _track_longevity(component_series)

    motility = float(com_speed)
    complexity = float(entropy_mean + edge_mean + 0.2 * elongation_mean)
    energy = float(energy_series.mean())
    energy_efficiency = float(complexity * np.clip(1.0 - resource_penalty * energy, 0.0, 1.0))
    persistence = float(persistence_series.mean())
    morphological_complexity = float(entropy_mean + edge_mean + elongation_median)

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
        "component_longevity": longevity,
        "morphological_complexity": morphological_complexity,
        "reproduction_rate": float(reproduction_events / max(len(states), 1)),
        "motility_score": motility,
        "active_fraction": float(persistence_series.mean()),
        "elongation": float(elongation_mean),
        "survived": bool(survival_ok),
    }
