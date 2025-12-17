# XenoGenesis-Sim

XenoGenesis-Sim is a Linux-first artificial-life sandbox for evolving digital organisms under customizable planetary conditions. It combines a production-grade continuous cellular automata (CA) engine, lightweight soft-body and digital-instruction substrates, evolutionary search tooling, and scripted workflows for analysis. The CA core uses FFT-accelerated update kernels with research-grade fitness instrumentation and cinematic renders for communicating emergent behaviors.

## Features
- **Continuous CA substrate** (Lenia/SmoothLife-inspired) with FFT-based stepping (FFTW3 preferred) and optional novelty search.
- **Soft-body voxel creatures** with minimal physics fallback and optional EvoGym integration; NEAT-lite controllers drive locomotion tasks.
- **Digital instruction organisms** using a small stack VM that can harvest resources and replicate.
- **Planetary environments** (gravity, temperature, radiation, resources) that influence substrates via shared interfaces.
- **Evolution loop** with DEAP-backed NSGA-II, checkpoint/resume, deterministic seeding, and parallel evaluation.
- **Outputs**: Parquet metrics, SQLite checkpoints plus compressed state blobs, plots, MP4/GIF renders with per-frame annotations, auto-generated reports.

## Repository layout
```
xenogenesis-sim/
  README.md  LICENSE  pyproject.toml  CMakeLists.txt  Makefile  .gitignore
  src/xenogenesis/...
  cpp/...
  scripts/...
  tests/...
  .github/workflows/ci.yml
```
Key modules:
- `src/xenogenesis/cli.py`: Typer-powered CLI (`xenogenesis`) for running, resuming, analyzing simulations, and environment checks.
- `src/xenogenesis/config/`: YAML configs validated by Pydantic models; `defaults.yaml` contains tuned demo defaults.
- `src/xenogenesis/engine/`: Simulation runner, checkpointing, metrics sink, and parallel evaluation helpers.
- `src/xenogenesis/substrates/ca`: CA model, kernels, FFT-backed stepping via pybind11, fitness metrics, and renders.
- `src/xenogenesis/substrates/softbody`: Morphology genome, simple controller, fallback physics, and locomotion fitness.
- `src/xenogenesis/substrates/digital`: Instruction VM, genome handling, and fitness evaluation.
- `src/xenogenesis/evolution/`: Selection, variation, DEAP bridge, novelty/speciation helpers.
- `src/xenogenesis/world/`: Planet/environment resources, hazards, terrain helpers.
- `src/xenogenesis/analysis/`: Plotting, reporting, summary utilities.
- `scripts/`: Install, demo, and profiling helpers.
- `tests/`: CI smoke, checkpoint/resume, determinism, and CA stepper equivalence tests.

## Installation (Linux)
Requirements: Python 3.10+ (3.11 recommended), CMake, a C++20 compiler, and FFTW3/ffmpeg for best performance.

```bash
bash scripts/install_ubuntu.sh
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

If FFTW3 is missing, the CA falls back to NumPy FFT (slower). EvoGym and NEAT extras are optional and auto-detected when installed.

## Quickstart demo (<5 minutes)
Run the bundled demo (headless) that evolves CA patterns and saves outputs under `runs/`:

```bash
make dev && make demo
```

This one-line pipeline installs the editable package, builds native extensions through `pyproject.toml`, runs `xenogenesis run ca` with the defaults in `src/xenogenesis/config/defaults.yaml`, generates Parquet metrics, summary plots, an MP4/GIF render, and prints the best genome/objectives. The latest run is auto-analyzed to produce plots and `report.md`.

### Mathematical model (CA substrate)
The CA is Lenia-inspired with a **biomass + resource** state and multi-ring excitation/inhibition kernels:

1. Build a normalized, signed multi-ring kernel \(K\) from concentric annuli (`ca.rings`) and weights (`ca.ring_weights`). Positive and negative bands are normalized separately for stability.
2. Compute smoothed activation \(A = K \ast B\) where \(B\) is the biomass channel.
3. Signed growth: \(G = 2\exp(-0.5 ((A - \mu)/\sigma)^2) - 1\), gated by local resource \(R\): \(\Delta B = \Delta t\, G \cdot R + \text{motion bias}\).
4. Resource dynamics: \(R' = \mathrm{clip}(R + r_\mathrm{regen}(1-R) - r_\mathrm{use} B R, 0, 1)\).
5. Noise, contour-friendly clipping, and mild high-density suppression encourage reproduction/splitting.

`mu`, `sigma`, `dt`, `rings`, `ring_weights`, and resource/fitness thresholds live in `config/defaults.yaml` and are validated by Pydantic. Kernels are cached so long sweeps reuse FFTs and avoid recomputation overhead.

### Fitness and diagnostics
- **Persistence**: mean active fraction across recorded frames.
- **Complexity proxy**: entropy of the state histogram plus edge density.
- **Motility**: center-of-mass speed estimated from tracked centroids.
- **Energy efficiency**: complexity scaled by a penalty on excess mass.
- **Reproduction/Longevity**: connected-component splits, surviving mass, and lifetime above activity thresholds.
- **Behavior descriptor**: `[entropy, edge_density, com_speed, energy_period]` for novelty/archive use.

Frame-level metrics (entropy, edge density, active fraction) are stored alongside summary objectives in `metrics.csv`, plotted in `plots/fitness.png`, and rendered on top of MP4 frames with optional gamma/contour overlays.

## Configuration reference
Configs are YAML validated by Pydantic (`ConfigSchema`). Key sections:
- `seed`: Deterministic base seed (PCG64DXSM).
- `environment`: Gravity, temperature, radiation, resource regeneration.
- `ca`: `grid_size`, `mu`, `sigma`, `dt`, `rings`, `ring_weights`, resource rates (`regen_rate`, `consumption_rate`), thresholds (`mass_threshold`, `active_threshold`), `noise_std`, render controls (`render_stride`, `gamma`, `show_contours`).
- `evolution`: `population`, `generations`, `selection` (`nsga2`), `mutation` rates, `workers`, `checkpoint_interval`.
- `outputs`: run directory, render toggles, archive sizes.

See `src/xenogenesis/config/defaults.yaml` for tuned demo values.

## Adding a new substrate
1. Implement the substrate protocol (`reset/step/render/serialize/deserialize/metrics`) in a new module under `src/xenogenesis/substrates/`.
2. Wire the substrate into `engine.sim_runner` and CLI registration.
3. Provide basic fitness functions and optional rendering.
4. Add a tiny config example and, if applicable, a test comparing Python vs native stepping.

## Performance tips
- Install FFTW3 (`libfftw3-dev`) and ffmpeg for fast CA stepping and video export.
- Use `--workers` to parallelize evaluations; OpenMP is enabled for native CA kernels. The FFT path caches kernels and plans for repeatability and speed.
- For debugging, use `make dev` for editable installs and `make profile` or `scripts/profile_ca.sh` to profile CA stepping.
- Headless rendering avoids GUI dependencies; set `XDG_RUNTIME_DIR` as needed on servers.

## Testing
Run the fast suite:
```bash
pytest -q
```
The smoke tests cover CLI execution, checkpoint/resume, determinism, and Python/C++ CA equivalence on a tiny grid.
