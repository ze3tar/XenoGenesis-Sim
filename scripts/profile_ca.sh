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

size = 128
steps = 50
state = np.stack((np.random.default_rng(0).random((size, size), dtype=np.float32), np.ones((size, size), dtype=np.float32)))
stepper = CAStepper()
params = KernelParams(size=size, rings=((1.0, 4.0), (4.0, 8.0), (8.0, 12.0)), ring_weights=(1.0, -0.6, 0.2))
start = time.time()
for _ in range(steps):
    state = stepper.step(
        state,
        mu=0.15,
        sigma=0.015,
        dt=0.1,
        kernel_params=params,
        regen_rate=0.05,
        consumption_rate=0.02,
        noise_std=0.002,
        growth_alpha=0.9,
        polarity_gain=0.35,
        polarity_decay=0.94,
        polarity_mobility=0.05,
        max_mass=0.8,
        death_factor=0.55,
        polarity_noise=0.0005,
    )
elapsed = time.time() - start
print(f"Ran {steps} steps on {size}x{size} grid in {elapsed:.3f}s ({steps/elapsed:.2f} steps/sec)")
PY
