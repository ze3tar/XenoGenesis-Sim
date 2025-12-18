"""Simulation runner for CA/softbody/digital substrates."""
from __future__ import annotations
import json
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from xenogenesis.config import ConfigSchema, load_config
from xenogenesis.core.rng import make_rng
from xenogenesis.engine.checkpointing import load_checkpoint
from xenogenesis.engine.lineage import LineageTracker
from xenogenesis.engine.metrics import save_metrics
from xenogenesis.substrates.ca import CAStepper, ca_fitness, render_frames
from xenogenesis.substrates.ca.genome import Genome, decode, phenotype_summary, CAParams, params_from_config, mutate
from xenogenesis.substrates.ca.fitness import _components, _center_of_mass
from xenogenesis.substrates.softbody import VoxelMorphology, Controller, softbody_fitness
from xenogenesis.substrates.digital import InstructionGenome, digital_fitness

console = Console()


def _moran_i(field: np.ndarray) -> float:
    mean_val = float(field.mean())
    diff = field - mean_val
    weights = (
        np.roll(diff, 1, axis=0)
        + np.roll(diff, -1, axis=0)
        + np.roll(diff, 1, axis=1)
        + np.roll(diff, -1, axis=1)
    )
    denom = float((diff ** 2).sum()) + 1e-6
    return float((diff * weights).sum() / (4.0 * denom))


def _frame_stats(state: np.ndarray, resource: np.ndarray | None = None, *, elongation_trigger: float = 1.3, prev_centroid: np.ndarray | None = None) -> dict:
    sample = state[::2, ::2] if max(state.shape) > 96 else state
    grad = np.gradient(sample)
    entropy_hist, _ = np.histogram(sample, bins=32, range=(0, 1), density=True)
    entropy_hist = entropy_hist + 1e-9
    entropy = float(-(entropy_hist * np.log2(entropy_hist)).sum())
    comps = _components(sample, 0.2)
    centroid = _center_of_mass(sample)
    centroid_speed = float(np.linalg.norm(centroid - prev_centroid)) if prev_centroid is not None else 0.0
    autocorr = _moran_i(sample)
    return {
        "mass": float(sample.mean()),
        "active_fraction": float(np.mean(sample > 0.05)),
        "edge_density": float(np.mean(np.abs(grad))),
        "entropy": entropy,
        "component_count": len(comps),
        "max_elongation": max((c["elongation"] for c in comps), default=0.0),
        "resource_mean": float(resource.mean()) if resource is not None else 0.0,
        "resource_var": float(np.var(resource)) if resource is not None else 0.0,
        "centroid_speed": centroid_speed,
        "spatial_autocorrelation": autocorr,
        "centroid_yx": centroid.tolist(),
    }


def run_ca(config: ConfigSchema) -> Path:
    rng = make_rng(config.seed)
    run_dir = Path(config.outputs.run_dir) / f"ca_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.genome.enabled:
        genome = Genome.random(config.genome.length, rng)
        ca_params = decode(genome, grid_size=config.ca.grid_size, dt=config.ca.dt, base_noise=config.ca.noise_std)
        genome_schema = {
            "kernel_inner_radius": float(ca_params.kernel_params.rings[0][1]),
            "kernel_outer_radius": float(ca_params.kernel_params.rings[1][1]),
            "ring_ratio": float(abs(ca_params.kernel_params.ring_weights[1])),
            "growth_gain": float(ca_params.growth_alpha),
            "maintenance_cost": float(ca_params.maintenance_cost),
            "polarity_strength": float(ca_params.polarity_gain),
            "division_threshold": float(ca_params.division_threshold),
            "resource_affinity": float(ca_params.resource_affinity),
        }
        (run_dir / "genome.json").write_text(json.dumps({"genes": genome.genes.tolist(), "decoded": ca_params.as_dict()}, indent=2))
        (run_dir / "genome_schema.json").write_text(json.dumps(genome_schema, indent=2))
        (run_dir / "phenotype_summary.json").write_text(json.dumps(phenotype_summary(ca_params), indent=2))
    else:
        genome = None
        ca_params = params_from_config(config.ca)
    biomass = rng.random((config.ca.grid_size, config.ca.grid_size), dtype=np.float32)
    resource = np.ones_like(biomass, dtype=np.float32)
    polarity = np.zeros_like(biomass, dtype=np.float32)
    state = np.stack((biomass, resource, polarity, polarity))
    stepper = CAStepper()
    tracker = LineageTracker(threshold=config.ca.active_threshold, match_radius=6.0, persistence=3)
    history: List[np.ndarray] = []
    history_full: List[np.ndarray] = []
    frame_metrics: List[dict] = []
    records = []
    prev_component_count = 0
    reproduction_seen = 0
    prev_snapshot: np.ndarray | None = None
    prev_centroid: np.ndarray | None = None
    lineage_records: list[dict] = []
    genome_archive: list[dict] = []
    root_id = hashlib.sha1((genome.genes if genome is not None else np.array([config.seed])).tobytes()).hexdigest()[:12]
    root_entry = {
        "individual_id": root_id,
        "parent_id": None,
        "generation": 0,
        "parents": [],
        "fitness": {},
        "species_id": None,
        "phenotype_descriptor": [],
        "genome": genome.genes.tolist() if genome is not None else [],
        "birth_step": 0,
        "death_step": None,
    }
    lineage_records.append(root_entry)
    genome_archive.append({"id": root_id, "genome": root_entry["genome"], "parent": None, "step": 0})
    for step_idx in range(config.ca.steps):
        result = stepper.step(state, ca_params, rng=rng)
        state = result.state
        if step_idx % config.ca.record_interval == 0:
            snapshot_full = state.copy()
            history_full.append(snapshot_full)
            snapshot = snapshot_full[0].copy()
            if max(snapshot.shape) > 96:
                history.append(snapshot[::2, ::2])
            else:
                history.append(snapshot)
            resource_snapshot = snapshot_full[1] if snapshot_full.shape[0] > 1 else None
            lineage_summary = tracker.update(snapshot, step_idx)
            stats = _frame_stats(
                snapshot,
                resource_snapshot,
                elongation_trigger=ca_params.elongation_trigger,
                prev_centroid=prev_centroid,
            )
            stats["step"] = step_idx
            stats["reproduction_events_step"] = result.stats.reproduction_events + lineage_summary["reproduction_events"]
            stats["resource_mean"] = result.stats.resource
            stats["component_count"] = lineage_summary["component_count"]
            stats["active_lineages"] = lineage_summary["active_tracks"]
            stats["reproduction_lineage"] = lineage_summary["reproduction_events"]
            reproduction_seen += result.stats.reproduction_events + lineage_summary["reproduction_events"]
            stats["reproduction_events"] = reproduction_seen
            prev_component_count = stats["component_count"]
            prev_snapshot = snapshot
            prev_centroid = np.array(stats.get("centroid_yx", prev_centroid if prev_centroid is not None else [0.0, 0.0]))
            frame_metrics.append(stats)
            records.append(stats)
            if genome is not None and result.stats.reproduction_events > 0:
                for _ in range(result.stats.reproduction_events):
                    child_genome = mutate(
                        genome,
                        rng,
                        sigma=config.genome.mutation_sigma,
                        structural_prob=config.genome.structural_prob,
                    )
                    child_id = hashlib.sha1(child_genome.genes.tobytes()).hexdigest()[:12]
                    lineage_records.append(
                        {
                            "individual_id": child_id,
                            "parent_id": root_id,
                            "generation": step_idx,
                            "parents": [root_id],
                            "fitness": {},
                            "species_id": None,
                            "phenotype_descriptor": [],
                            "genome": child_genome.genes.tolist(),
                            "birth_step": step_idx,
                            "death_step": None,
                        }
                    )
                    genome_archive.append({"id": child_id, "genome": child_genome.genes.tolist(), "parent": root_id, "step": step_idx})
    fitness = ca_fitness(
        history,
        mass_threshold=config.ca.mass_threshold,
        active_threshold=config.ca.active_threshold,
    )
    fitness["lineage_reproduction_events"] = reproduction_seen
    records.append({"step": config.ca.steps, **fitness, "seed": config.seed})
    metrics_path = run_dir / "metrics.csv"
    save_metrics(records, metrics_path)
    with open(run_dir / "best_individual.json", "w") as f:
        json.dump({"fitness": fitness}, f)
    if history_full:
        buf = BytesIO()
        np.savez_compressed(buf, states=np.stack(history_full))
        (run_dir / "states.npz").write_bytes(buf.getvalue())
    species_df = None
    if config.outputs.render:
        render_frames(
            history_full[:: config.ca.render_stride],
            run_dir / "renders",
            cmap=config.ca.render_cmap,
            gamma=ca_params.render_gamma,
            show_contours=config.ca.show_contours,
            contour_level=ca_params.contour_level,
            metric_history=frame_metrics[:: config.ca.render_stride],
            overlay_delta=True,
            snapshot_name="organism_snapshot.png",
            video_name="alien_life.mp4",
            show_polarity_vectors=True,
            multi_panel=True,
        )
    try:
        from xenogenesis.analysis import annotate_species

        species_df = annotate_species(run_dir)
    except Exception:
        species_df = None
    lineage_records[0]["fitness"] = fitness
    lineage_records[0]["phenotype_descriptor"] = fitness.get("descriptor", [])
    lineage_records[0]["death_step"] = config.ca.steps
    if species_df is not None and not species_df.empty:
        dominant_species = int(species_df["species_id"].value_counts().idxmax())
        for entry in lineage_records:
            entry["species_id"] = dominant_species
    for entry in lineage_records[1:]:
        entry["death_step"] = entry.get("death_step") or config.ca.steps
    lineage_event_log = tracker.events
    if lineage_event_log:
        (run_dir / "lineage_events.jsonl").write_text("\n".join(json.dumps(ev) for ev in lineage_event_log) + "\n")
    (run_dir / "lineage.jsonl").write_text("\n".join(json.dumps(entry) for entry in lineage_records) + "\n")
    (run_dir / "genome_archive.jsonl").write_text("\n".join(json.dumps(entry) for entry in genome_archive) + "\n")
    config_dict = config.model_dump()
    if isinstance(config_dict.get("outputs", {}).get("run_dir"), Path):
        config_dict["outputs"]["run_dir"] = str(config_dict["outputs"]["run_dir"])
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config_dict, f)
    if config.outputs.summarize:
        table = Table(title="CA objectives", show_lines=True)
        table.add_column("metric")
        table.add_column("value")
        for key in (
            "persistence",
            "complexity",
            "motility",
            "energy_efficiency",
            "entropy",
            "edge_density",
            "reproduction_events",
            "reproduction_rate",
            "component_longevity",
        ):
            if key in fitness:
                table.add_row(key, f"{fitness[key]:.4f}")
        console.print(table)
    console.print(f"CA run complete -> {run_dir}")
    return run_dir


def run_softbody(config: ConfigSchema) -> Path:
    morph = VoxelMorphology()
    controller = Controller()
    fitness = softbody_fitness(morph, controller)
    run_dir = Path(config.outputs.run_dir) / f"softbody_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_metrics([{**fitness, "seed": config.seed}], run_dir / "metrics.csv")
    return run_dir


def run_digital(config: ConfigSchema) -> Path:
    genome = InstructionGenome()
    fitness = digital_fitness(genome)
    run_dir = Path(config.outputs.run_dir) / f"digital_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_metrics([{**fitness, "seed": config.seed}], run_dir / "metrics.csv")
    return run_dir


def resume_checkpoint(path: Path) -> Dict[str, Any]:
    return load_checkpoint(path)
