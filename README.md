# XenoGenesis-Sim

A blunt description for people who actually want to run code instead of collecting buzzwords. XenoGenesis-Sim is a Linux-first artificial-life playground: a chunky Lenia-inspired CA core, some soft-body and digital critters, and an evolutionary loop glued together with Python and a bit of C++/pybind11. It is built to run, fail loudly, and render what happened.

## What this thing does
- Evolves alien-looking organisms on a big continuous CA grid with FFT kernels and metabolic quirks.
- Renders those runs into MP4/PNG so humans can *see* the organisms instead of staring at logs.
- Supports novelty search and multi-objective scoring without a graduate seminar.
- Ships with sane defaults; you can still tweak every knob if you enjoy YAML.

## Quick start (Linux)
Dependencies: Python 3.10+, CMake, a C++20 compiler, FFmpeg, FFTW3. On Ubuntu, `scripts/install_ubuntu.sh` pulls the basics.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
make demo  # runs the CA demo with renders turned on
```

If your network blocks wheels, build from source; the code does not ask for sympathy.

## Demo expectations
- Grid: 192×192 by default—big enough to see structure without a microscope.
- Rendering: on by default; uses a mellow `cividis` palette, light smoothing, and minimal text overlays so shapes stay recognizable.
- Output: `runs/<timestamp>/renders/alien_life.mp4` (GIF fallback if FFmpeg is missing) plus a snapshot PNG.
- Runtime: a few hundred steps; CPU-only containers may take a minute. Bring a GPU if you want bragging rights.

Toggle behavior with environment variables: `DEMO_STEPS=800` for longer runs, `DEMO_RENDER=0` if you *really* want headless.

## Demo visuals
Run `make demo` to generate a clip and snapshot under `runs/<timestamp>/renders/`. If you want to keep a copy outside the run folder, `demo/media/` is the expected parking spot, but binaries are ignored in git on purpose.

## Running your own worlds
The CLI is in `src/xenogenesis/cli.py`. Example:

```bash
python -m xenogenesis.cli run ca \
  --config configs/alien_life.yaml \
  --steps 600 \
  --render \
  --analyze
```

Configs live in `configs/` and `src/xenogenesis/config/`. `ca.grid_size`, ring radii, diffusion rates, and mutation knobs are all there. Keep your configs under version control; reproducibility beats folklore.

## Repository map
```
CMakeLists.txt  Makefile  pyproject.toml
configs/        demo/     scripts/      tests/
src/xenogenesis/    # CLI, engine, substrates, analysis
cpp/                # native kernels and bindings
```

## Testing
Use pytest; it is fast enough that you have no excuse.

```bash
PYTHONPATH=src pytest -q
```

## Contributing
Send patches, not manifestos. Keep imports clean (no try/except around them), write tests, and make the demo look better than before.
