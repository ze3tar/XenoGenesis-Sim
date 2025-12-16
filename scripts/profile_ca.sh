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

size = 128
steps = 50
state = np.random.default_rng(0).random((size, size), dtype=np.float32)
stepper = CAStepper()
start = time.time()
for _ in range(steps):
    state = stepper.step(state, mu=0.15, sigma=0.015, dt=0.1, inner_radius=3.0, outer_radius=6.0, ring_ratio=0.5)
elapsed = time.time() - start
print(f"Ran {steps} steps on {size}x{size} grid in {elapsed:.3f}s ({steps/elapsed:.2f} steps/sec)")
PY
