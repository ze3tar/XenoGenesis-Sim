"""Simulation runner for CA/softbody/digital substrates."""
from __future__ import annotations
import json
import hashlib
import os
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
from xenogenesis.engine.metrics import save_metrics
from xenogenesis.substrates.ca import CAStepper, ca_fitness, render_frames
from xenogenesis.substrates.ca.genome import Genome, decode, phenotype_summary, CAParams, params_from_config, mutate
from xenogenesis.substrates.ca.fitness import _components
from xenogenesis.substrates.softbody import VoxelMorphology, Controller, softbody_fitness
from xenogenesis.substrates.digital import InstructionGenome, digital_fitness

console = Console()


def _frame_stats(state: np.ndarray, resource: np.ndarray | None = None, *, elongation_trigger: float = 1.3) -> dict:
    sample = state[::2, ::2] if max(state.shape) > 96 else state
    grad = np.gradient(sample)
    entropy_hist, _ = np.histogram(sample, bins=32, range=(0, 1), density=True)
    entropy_hist = entropy_hist + 1e-9
    entropy = float(-(entropy_hist * np.log2(entropy_hist)).sum())
    comps = _components(sample, 0.2)
    return {
        "mass": float(sample.mean()),
        "active_fraction": float(np.mean(sample > 0.05)),
        "edge_density": float(np.mean(np.abs(grad))),
        "entropy": entropy,
        "component_count": len(comps),
        "max_elongation": max((c["elongation"] for c in comps), default=0.0),
        "resource_mean": float(resource.mean()) if resource is not None else 0.0,
    }


def run_ca(config: ConfigSchema) -> Path:
    skip_analysis = os.getenv("XG_SKIP_ANALYSIS") == "1"
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
    grid_size = config.ca.grid_size
    biomass = rng.normal(0.05, 0.02, (grid_size, grid_size)).clip(0.0, 1.0).astype(np.float32)
    yy, xx = np.ogrid[:grid_size, :grid_size]
    center = grid_size // 2
    seed_radius = max(4, grid_size // 8)
    core_mask = (yy - center) ** 2 + (xx - center) ** 2 <= seed_radius**2
    biomass[core_mask] += 0.6
    for _ in range(3):
        cy = int(rng.integers(grid_size))
        cx = int(rng.integers(grid_size))
        radius = int(rng.integers(max(3, grid_size // 16), max(4, grid_size // 12)))
        patch = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        biomass[patch] += rng.uniform(0.3, 0.55)
    biomass = biomass.clip(0.0, 1.0)
    resource = np.ones_like(biomass, dtype=np.float32)
    polarity = np.zeros_like(biomass, dtype=np.float32)
    state = np.stack((biomass, resource, polarity, polarity))
    stepper = CAStepper()
    history: List[np.ndarray] = []
    history_full: List[np.ndarray] = []
    frame_metrics: List[dict] = []
    records = []
    prev_component_count = 0
    reproduction_seen = 0
    prev_snapshot: np.ndarray | None = None
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
    progress_interval = config.ca.progress_interval or max(10, config.ca.steps // 20)
    last_stats: dict | None = None
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
            stats = _frame_stats(snapshot, resource_snapshot, elongation_trigger=ca_params.elongation_trigger)
            stats["step"] = step_idx
            stats["reproduction_events_step"] = result.stats.reproduction_events
            stats["resource_mean"] = result.stats.resource
            reproduction_seen += result.stats.reproduction_events
            stats["reproduction_events"] = reproduction_seen
            prev_component_count = stats["component_count"]
            prev_snapshot = snapshot
            frame_metrics.append(stats)
            records.append(stats)
            last_stats = stats
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
        if (step_idx + 1) % progress_interval == 0:
            current = last_stats or {"mass": float(state[0].mean())}
            console.log(
                f"step {step_idx + 1}/{config.ca.steps} | mass={current.get('mass', 0.0):.3f} | components={prev_component_count} | reproduction={reproduction_seen}"
            )
    def _sample_frames(frames: list[np.ndarray], *, max_frames: int = 160) -> list[np.ndarray]:
        if len(frames) <= max_frames:
            return frames
        stride = max(1, math.ceil(len(frames) / max_frames))
        return frames[::stride]

    def _shrink(frame: np.ndarray) -> np.ndarray:
        if max(frame.shape) <= 32:
            return frame
        stride = 2 if max(frame.shape) <= 96 else max(2, math.ceil(max(frame.shape) / 64))
        return frame[::stride, ::stride]

    fitness_frames = [_shrink(f) for f in _sample_frames(history)]
    console.log(f"computing CA fitness on {len(fitness_frames)} frames (source={len(history)})")
    fitness_start = time.time()
    fitness = ca_fitness(
        fitness_frames,
        mass_threshold=config.ca.mass_threshold,
        active_threshold=config.ca.active_threshold,
    )
    console.log(f"fitness computed in {time.time() - fitness_start:.2f}s")
    records.append({"step": config.ca.steps, **fitness, "seed": config.seed})
    metrics_path = run_dir / "metrics.csv"
    save_metrics(records, metrics_path)
    with open(run_dir / "best_individual.json", "w") as f:
        json.dump({"fitness": fitness}, f)
    if history_full:
        console.log(f"saving {len(history_full)} recorded states")
        buf = BytesIO()
        np.savez_compressed(buf, states=np.stack(history_full))
        (run_dir / "states.npz").write_bytes(buf.getvalue())
    species_df = None
    if config.outputs.render:
        render_frameset = history_full[:: config.ca.render_stride]
        console.log(f"rendering {len(render_frameset)} frames (stride={config.ca.render_stride})")
        render_start = time.time()
        render_frames(
            render_frameset,
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
            track_ids=config.ca.render_track_ids,
            smooth_sigma=config.ca.render_smoothing,
            show_metrics=config.ca.render_show_metrics,
            metric_keys=config.ca.render_metric_keys,
            show_ids=config.ca.render_show_ids,
        )
    if not skip_analysis:
        try:
            from xenogenesis.analysis import annotate_species

            species_df = annotate_species(run_dir, frame_stride=max(config.ca.render_stride, 2), max_frames=400)
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
