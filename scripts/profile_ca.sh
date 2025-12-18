#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
python -m xenogenesis.cli doctor
python - <<'PY'
import time
import numpy as np
from xenogenesis.substrates.ca.ca_model import CAStepper
from xenogenesis.substrates.ca.kernels import KernelParams
from xenogenesis.substrates.ca.genome import CAParams

size = 128
steps = 50
state = np.stack(
    (
        np.random.default_rng(0).random((size, size), dtype=np.float32),
        np.ones((size, size), dtype=np.float32),
        np.zeros((size, size), dtype=np.float32),
        np.zeros((size, size), dtype=np.float32),
    )
)
stepper = CAStepper()
params = KernelParams(size=size, rings=((1.0, 4.0), (4.0, 8.0), (8.0, 12.0)), ring_weights=(1.0, -0.6, 0.2))
ca_params = CAParams(
    grid_size=size,
    kernel_params=params,
    mu=0.15,
    sigma=0.015,
    dt=0.1,
    growth_alpha=0.9,
    decay_lambda=0.0,
    regen_rate=0.05,
    consumption_rate=0.02,
    resource_diffusion=0.0,
    biomass_diffusion=0.0,
    noise_std=0.002,
    polarity_gain=0.35,
    polarity_decay=0.94,
    polarity_mobility=0.05,
    polarity_noise=0.0005,
    max_mass=0.8,
    death_factor=0.55,
    elongation_trigger=1.3,
    fission_assist=0.0,
    render_gamma=1.0,
    contour_level=0.5,
    dominant_band=0,
    maintenance_cost=0.08,
    competition_scale=0.25,
    competition_radius=3.0,
    resource_capacity=1.2,
    resource_gradient=0.25,
    polarity_diffusion=0.02,
    polarity_mutation=0.02,
    directional_gain=1.0,
    division_threshold=0.5,
    division_fraction=0.45,
    reproduction_cost=0.15,
    resource_affinity=0.35,
    toxin_rate=0.0,
    drift_rate=0.0,
)
start = time.time()
for _ in range(steps):
    state = stepper.step(state, ca_params).state
elapsed = time.time() - start
print(f"Ran {steps} steps on {size}x{size} grid in {elapsed:.3f}s ({steps/elapsed:.2f} steps/sec)")
PY
