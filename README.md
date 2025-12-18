# XenoGenesis-Sim

XenoGenesis-Sim is a Linux-first artificial-life sandbox for evolving digital organisms inside configurable planetary environments. It pairs a continuous cellular-automata (CA) substrate with lightweight soft-body and digital-instruction organisms, all driven by an evolutionary search loop and supported by native (pybind11/C++) acceleration.

## What you can do
- **Run fast Lenia-style CA worlds** with FFT-accelerated kernels, resource coupling, and cinematic renders.
- **Evolve behaviors** using DEAP-backed NSGA-II with novelty/behavior descriptors, checkpoints, and deterministic seeds.
- **Explore multiple substrates**: continuous CA, voxel soft-bodies with controller genomes, and digital stack-VM organisms that harvest resources.
- **Analyze results** via Parquet metrics, annotated MP4/GIF renders, plots, and auto-generated markdown reports.

## Mathematical model (CA substrate)
The default substrate is Lenia-inspired but extended with metabolism and polarity memory. A biomass \(B\) channel is coupled to a resource \(R\) field and a two-plane polarity \(P=(P_x,P_y)\). Let \(K\) be a normalized three-ring kernel (`ca.rings`, `ca.ring_weights`). With timestep \(\Delta t\):

1. **Convolution**: \(A = K * B\) via cached FFTs.
2. **Signed growth**: \(G = \alpha \tanh((A-\mu)/\sigma)\) where \(\alpha\) is `growth_alpha`. Positive/negative bands compete while the kernel mass is normalized to prevent runaway growth.
3. **Metabolism**: resources regenerate and are consumed by growth: \(R' = \mathrm{clip}(R + r_\text{regen}(1-R) - r_\text{use}\,B\max(G,0), 0, 1)\).
4. **Polarity / motion memory**: gradients reinforce polarity \(P' = \lambda P + \gamma \nabla B\); biomass advects along polarity: \(B' = B + \Delta t \,(G \cdot R + m\,\mathrm{sum}(P))\).
5. **Density-dependent death**: regions above `max_mass` are damped by `death_factor`, encouraging necking and fission.
6. **Stability extras**: small Gaussian noise, clipping, and cached FFT kernels for long runs.

### Fitness and diagnostics
Frame-level metrics (entropy, edge density, active fraction, component elongation) feed into multi-objective scores:
- **Persistence**: mean active fraction across recorded frames.
- **Morphological complexity**: entropy + edges + elongation.
- **Motility**: center-of-mass speed from tracked centroids.
- **Energy efficiency**: complexity penalized by excess mass.
- **Reproduction/Longevity**: split detections from component tracking and centroid lifetimes.
A behavior descriptor `[entropy, edge_density, com_speed, energy_period, elongation_mean, elongation_median]` supports novelty search and archiving.

## Project layout
```
xenogenesis-sim/
  README.md  LICENSE  pyproject.toml  CMakeLists.txt  Makefile
  src/xenogenesis/...
  cpp/...
  scripts/...
  tests/...
```
Key entry points:
- `src/xenogenesis/cli.py`: Typer-powered CLI (`xenogenesis`) for running/resuming simulations and analysis.
- `src/xenogenesis/config/`: YAML configs validated by Pydantic (`defaults.yaml` contains tuned demos).
- `src/xenogenesis/engine/`: Simulation runner, checkpointing, metrics sinks, and parallel evaluation.
- `src/xenogenesis/substrates/`: CA, soft-body, and digital substrates with shared protocol.
- `scripts/`: install, demo, profiling helpers.  
- `tests/`: CI smoke, checkpoint/resume, determinism, and CA equivalence tests.

## Install (Linux)
Requirements: Python 3.10+ (3.11 recommended), CMake, a C++20 compiler, and FFTW3/ffmpeg for best performance.

```bash
bash scripts/install_ubuntu.sh
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
If FFTW3 is missing, NumPy FFT is used (slower). EvoGym/NEAT extras are optional and auto-detected when available.

## Run the quick demo (<5 minutes)
Runs headless and writes outputs under `runs/`:
```bash
make dev && make demo
```
This installs in editable mode, builds native extensions, runs `xenogenesis run ca` with defaults from `src/xenogenesis/config/defaults.yaml`, and produces Parquet metrics, plots, MP4/GIF renders, and a `report.md` for the latest run.

## Custom runs
Invoke the CLI directly (after activation):
```bash
xenogenesis run ca --config path/to/config.yaml --generations 50 --workers 4
```
Useful flags: `--resume CHECKPOINT`, `--render/--no-render`, `--seed`, and substrate-specific options in the YAML config (grid size, ring radii/weights, resource rates, mutation rates, etc.).

## Outputs
Each run directory contains:
- `metrics.parquet` / `metrics.csv`: frame and summary metrics.
- `checkpoints/`: serialized population + substrate state for resume.
- `renders/`: MP4/GIF with overlays (entropy, contours, objectives).
- `report.md`: auto-generated summary with plots under `plots/`.

## Testing
Run the fast suite:
```bash
pytest -q
```
Smoke tests cover the CLI, checkpoint/resume, deterministic seeding, and CA Python/C++ equivalence on tiny grids.
