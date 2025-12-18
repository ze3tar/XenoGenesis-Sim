"""Species clustering and taxonomy utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:  # optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    hdbscan = None

from xenogenesis.substrates.ca.fitness import _components


@dataclass
class OrganismTrack:
    organism_id: str
    frames: list[int]
    masses: list[float]
    areas: list[float]
    elongations: list[float]
    perimeters: list[float]
    centroids: list[np.ndarray]
    polarity: list[float]

    def to_descriptor(self, *, dominant_band: int, reproduction_rate: float, resource_series: list[float]) -> dict:
        if not self.frames:
            return {}
        centroid_arr = np.stack(self.centroids)
        velocities = np.linalg.norm(np.diff(centroid_arr, axis=0), axis=1) if len(centroid_arr) > 1 else np.array([0.0])
        displacement = float(np.linalg.norm(centroid_arr[-1] - centroid_arr[0])) if len(centroid_arr) > 1 else 0.0
        path_length = float(velocities.sum()) if len(velocities) else 0.0
        migration_persistence = displacement / max(path_length, 1e-6)
        boundary_roughness = float(np.mean([(p ** 2) / max(a, 1e-6) for p, a in zip(self.perimeters, self.areas)]))
        compactness = float(np.mean([a / max(p, 1e-6) for a, p in zip(self.areas, self.perimeters)]))
        resource_mean = float(np.mean(resource_series)) if resource_series else 0.0
        polarity_coherence = float(np.mean(self.polarity)) if self.polarity else 0.0
        masses = np.array(self.masses, dtype=np.float32)
        descriptor = {
            "organism_id": self.organism_id,
            "lifespan": len(self.frames),
            "mean_mass": float(masses.mean()),
            "std_mass": float(masses.std()),
            "mean_speed": float(np.mean(velocities)) if len(velocities) else 0.0,
            "max_speed": float(np.max(velocities)) if len(velocities) else 0.0,
            "compactness": compactness,
            "elongation": float(np.mean(self.elongations)) if self.elongations else 0.0,
            "boundary_roughness": boundary_roughness,
            "energy_efficiency": float(np.mean(masses) * max(1.0 - resource_mean, 0.0)),
            "reproduction_frequency": reproduction_rate,
            "mean_resource_intake": resource_mean,
            "migration_persistence": migration_persistence,
            "shape_entropy": float(np.std(self.elongations)) if self.elongations else 0.0,
            "component_stability": float(1.0 / (1.0 + np.var(masses))) if len(masses) else 0.0,
            "dominant_kernel_band": dominant_band,
            "polarity_coherence": polarity_coherence,
        }
        return descriptor


def _load_states(run_dir: Path) -> np.ndarray | None:
    npz_path = run_dir / "states.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    return data.get("states")


def _assign_tracks(tracks: list[OrganismTrack], comps: list[dict], frame_idx: int, state_slice: np.ndarray | None):
    used = set()
    for comp in comps:
        centroid = comp.get("centroid")
        best_idx = None
        best_dist = 1e9
        for idx, tr in enumerate(tracks):
            if not tr.centroids:
                continue
            dist = float(np.linalg.norm(tr.centroids[-1] - centroid))
            if dist < best_dist and dist <= 6.0:
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            track = tracks[best_idx]
        else:
            track = OrganismTrack(
                organism_id=f"org_{len(tracks):04d}",
                frames=[],
                masses=[],
                areas=[],
                elongations=[],
                perimeters=[],
                centroids=[],
                polarity=[],
            )
            tracks.append(track)
        track.frames.append(frame_idx)
        track.masses.append(comp.get("mass", 0.0))
        track.areas.append(comp.get("area", 0.0))
        track.elongations.append(comp.get("elongation", 0.0))
        track.perimeters.append(comp.get("perimeter", 0.0))
        track.centroids.append(centroid)
        if state_slice is not None and state_slice.shape[0] > 3:
            mask = state_slice[0] > 0.2
            polarity_mag = np.clip(np.hypot(state_slice[2], state_slice[3]), 0.0, 1.0)
            track.polarity.append(float(np.mean(polarity_mag[mask])))
        used.add(id(comp))
    return tracks


def extract_organism_descriptors(run_dir: Path) -> pd.DataFrame:
    states = _load_states(run_dir)
    metrics_path = run_dir / "metrics.csv"
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if states is None:
        return pd.DataFrame()

    tracks: list[OrganismTrack] = []
    resource_series: list[float] = []
    for idx, frame in enumerate(states):
        biomass = frame[0]
        comps = _components(biomass, 0.2)
        resource_mean = float(frame[1].mean()) if frame.shape[0] > 1 else 0.0
        resource_series.append(resource_mean)
        tracks = _assign_tracks(tracks, comps, idx, frame)

    dominant_band = 0
    phenotype_path = run_dir / "phenotype_summary.json"
    if phenotype_path.exists():
        dom = json.loads(phenotype_path.read_text()).get("dominant_band")
        if dom is not None:
            dominant_band = int(dom)
    if not metrics_df.empty and "reproduction_events" in metrics_df:
        reproduction_rate = float(metrics_df["reproduction_events"].diff().fillna(0).clip(lower=0).mean())
    else:
        reproduction_rate = 0.0

    descriptors = []
    for track in tracks:
        desc = track.to_descriptor(dominant_band=dominant_band, reproduction_rate=reproduction_rate, resource_series=resource_series)
        if desc:
            descriptors.append(desc)
    return pd.DataFrame(descriptors)


def _cluster_with_kmeans(features: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_score = -1
    best_labels = np.zeros(len(features), dtype=int)
    for k in range(2, min(5, len(features) + 1)):
        model = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = model.fit_predict(features)
        if len(set(labels)) <= 1:
            continue
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
    return best_labels


def cluster_species(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    feature_cols = [
        "mean_mass",
        "std_mass",
        "mean_speed",
        "max_speed",
        "compactness",
        "elongation",
        "boundary_roughness",
        "energy_efficiency",
        "reproduction_frequency",
        "lifespan",
        "mean_resource_intake",
        "migration_persistence",
        "shape_entropy",
        "component_stability",
        "dominant_kernel_band",
        "polarity_coherence",
    ]
    data = df[feature_cols].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    labels = None
    if hdbscan is not None:
        labels = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(scaled)
    if labels is None or (labels == -1).all():
        labels = DBSCAN(eps=0.8, min_samples=2).fit_predict(scaled)
    if (labels == -1).all():
        labels = _cluster_with_kmeans(scaled)
    df = df.copy()
    df["species_id"] = labels
    # Stabilize IDs: replace noise label -1 with unique species numbers
    if (labels == -1).any():
        noise_mask = labels == -1
        start = int(df["species_id"].max()) + 1 if len(df) else 0
        df.loc[noise_mask, "species_id"] = [start + i for i in range(noise_mask.sum())]
    return df


def species_summary(df: pd.DataFrame) -> dict:
    if df.empty or "species_id" not in df.columns:
        return {"species": [], "counts": {}}
    summary = []
    for sid, group in df.groupby("species_id"):
        exemplar = group.iloc[0].to_dict()
        summary.append({"species_id": int(sid), "count": len(group), "traits": group.mean(numeric_only=True).to_dict(), "exemplar": exemplar})
    counts = {int(k): int(v) for k, v in df["species_id"].value_counts().to_dict().items()}
    return {"species": summary, "counts": counts}


def _render_gallery(run_dir: Path, states: np.ndarray, df: pd.DataFrame):
    gallery_dir = run_dir / "species_gallery"
    gallery_dir.mkdir(exist_ok=True)
    cmap = plt.get_cmap("magma")
    for sid, group in df.groupby("species_id"):
        exemplar = group.iloc[0]
        frame_idx = int(exemplar.get("lifespan", 1) // 2)
        frame_idx = min(frame_idx, states.shape[0] - 1)
        frame = states[frame_idx][0]
        fig, ax = plt.subplots(figsize=(3, 3), dpi=120)
        ax.imshow(frame, cmap=cmap)
        ax.set_title(f"Species {sid}")
        ax.axis("off")
        out_path = gallery_dir / f"species_{sid}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def annotate_species(run_dir: Path):
    states = _load_states(run_dir)
    if states is None:
        return None
    df = extract_organism_descriptors(run_dir)
    df = cluster_species(df)
    summary = species_summary(df)
    (run_dir / "species.csv").write_text(df.to_csv(index=False))
    (run_dir / "species_summary.json").write_text(json.dumps(summary, indent=2))
    _render_gallery(run_dir, states, df)
    return df
