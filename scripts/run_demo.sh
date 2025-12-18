#!/usr/bin/env bash
set -euo pipefail

SEED=${1:-42}
CONFIG=${2:-configs/demo_life.yaml}

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "[demo] Running CA demo with seed=${SEED} config=${CONFIG}"
python -m xenogenesis.cli run ca --config "${CONFIG}" --seed "${SEED}" --render True

RUN_DIR="runs/ca_${SEED}"
echo "[demo] Analyzing run at ${RUN_DIR}"
python -m xenogenesis.cli analyze --run "${RUN_DIR}" --species --phylogeny

echo "[demo] Outputs ready in ${RUN_DIR}"
