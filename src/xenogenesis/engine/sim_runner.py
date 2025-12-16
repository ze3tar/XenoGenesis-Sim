"""Simulation runner for CA/softbody/digital substrates."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from rich.console import Console

from xenogenesis.config import ConfigSchema, load_config
from xenogenesis.core.rng import make_rng
from xenogenesis.engine.checkpointing import save_checkpoint, load_checkpoint
from xenogenesis.engine.parallel import parallel_map
from xenogenesis.engine.metrics import save_metrics
from xenogenesis.substrates.ca import CAStepper, ca_fitness, render_frames
from xenogenesis.substrates.softbody import VoxelMorphology, Controller, softbody_fitness
from xenogenesis.substrates.digital import InstructionGenome, digital_fitness

console = Console()


def run_ca(config: ConfigSchema) -> Path:
    rng = make_rng(config.seed)
    run_dir = Path(config.outputs.run_dir) / f"ca_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = rng.random((config.ca.grid_size, config.ca.grid_size), dtype=np.float32)
    stepper = CAStepper()
    history: List[np.ndarray] = []
    records = []
    for step_idx in range(config.ca.steps):
        state = stepper.step(state, mu=config.ca.mu, sigma=config.ca.sigma, dt=config.ca.dt, inner_radius=config.ca.inner_radius, outer_radius=config.ca.outer_radius, ring_ratio=config.ca.ring_ratio)
        if step_idx % 8 == 0:
            history.append(state.copy())
    fitness = ca_fitness(history)
    records.append({"step": config.ca.steps, **fitness, "seed": config.seed})
    metrics_path = run_dir / "metrics.parquet"
    save_metrics(records, metrics_path)
    with open(run_dir / "best_individual.json", "w") as f:
        json.dump({"fitness": fitness}, f)
    if config.outputs.render:
        render_frames(history, run_dir / "renders")
    with open(run_dir / "config.yaml", "w") as f:
        f.write(json.dumps(config.model_dump(), indent=2))
    console.print(f"CA run complete -> {run_dir}")
    return run_dir


def run_softbody(config: ConfigSchema) -> Path:
    morph = VoxelMorphology()
    controller = Controller()
    fitness = softbody_fitness(morph, controller)
    run_dir = Path(config.outputs.run_dir) / f"softbody_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_metrics([{**fitness, "seed": config.seed}], run_dir / "metrics.parquet")
    return run_dir


def run_digital(config: ConfigSchema) -> Path:
    genome = InstructionGenome()
    fitness = digital_fitness(genome)
    run_dir = Path(config.outputs.run_dir) / f"digital_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_metrics([{**fitness, "seed": config.seed}], run_dir / "metrics.parquet")
    return run_dir


def resume_checkpoint(path: Path) -> Dict[str, Any]:
    return load_checkpoint(path)
