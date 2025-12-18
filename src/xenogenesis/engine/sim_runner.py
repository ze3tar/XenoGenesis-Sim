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
from xenogenesis.engine.metrics import save_metrics
from xenogenesis.substrates.ca import CAStepper, ca_fitness, render_frames
from xenogenesis.substrates.ca.genome import Genome, decode, phenotype_summary, CAParams, params_from_config
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
    rng = make_rng(config.seed)
    run_dir = Path(config.outputs.run_dir) / f"ca_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.genome.enabled:
        genome = Genome.random(config.genome.length, rng)
        ca_params = decode(genome, grid_size=config.ca.grid_size, dt=config.ca.dt, base_noise=config.ca.noise_std)
        (run_dir / "genome.json").write_text(json.dumps({"genes": genome.genes.tolist(), "decoded": ca_params.as_dict()}, indent=2))
        (run_dir / "phenotype_summary.json").write_text(json.dumps(phenotype_summary(ca_params), indent=2))
    else:
        genome = None
        ca_params = params_from_config(config.ca)
    biomass = rng.random((config.ca.grid_size, config.ca.grid_size), dtype=np.float32)
    resource = np.ones_like(biomass, dtype=np.float32)
    state = np.stack((biomass, resource))
    stepper = CAStepper()
    history: List[np.ndarray] = []
    history_full: List[np.ndarray] = []
    frame_metrics: List[dict] = []
    records = []
    prev_component_count = 0
    reproduction_seen = 0
    prev_snapshot: np.ndarray | None = None
    for step_idx in range(config.ca.steps):
        state = stepper.step(state, ca_params, rng=rng)
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
            if prev_snapshot is not None:
                delta_components = stats["component_count"] - prev_component_count
                if delta_components > 0 and stats.get("max_elongation", 0.0) >= ca_params.elongation_trigger:
                    reproduction_seen += delta_components
            stats["reproduction_events"] = reproduction_seen
            prev_component_count = stats["component_count"]
            prev_snapshot = snapshot
            frame_metrics.append(stats)
            records.append(stats)
    fitness = ca_fitness(
        history,
        mass_threshold=config.ca.mass_threshold,
        active_threshold=config.ca.active_threshold,
    )
    records.append({"step": config.ca.steps, **fitness, "seed": config.seed})
    metrics_path = run_dir / "metrics.csv"
    save_metrics(records, metrics_path)
    with open(run_dir / "best_individual.json", "w") as f:
        json.dump({"fitness": fitness}, f)
    if history_full:
        buf = BytesIO()
        np.savez_compressed(buf, states=np.stack(history_full))
        (run_dir / "states.npz").write_bytes(buf.getvalue())
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
        )
    if genome is not None:
        lineage_id = hashlib.sha1(genome.genes.tobytes()).hexdigest()[:12]
    else:
        lineage_id = hashlib.sha1(str(config.seed).encode()).hexdigest()[:12]
    lineage_entry = {
        "individual_id": lineage_id,
        "generation": 0,
        "parents": [],
        "fitness": fitness,
        "species_id": None,
        "phenotype_descriptor": fitness.get("descriptor", []),
        "genome": genome.genes.tolist() if genome is not None else [],
    }
    (run_dir / "lineage.jsonl").write_text(json.dumps(lineage_entry) + "\n")
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
