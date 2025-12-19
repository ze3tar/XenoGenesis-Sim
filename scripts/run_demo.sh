#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
CONFIG_PATH=${1:-${PROJECT_ROOT}/configs/alien_life.yaml}

echo "Running CA demo..."
python -m xenogenesis.cli run ca \
  --config ${CONFIG_PATH} \
  --generations 30 --pop 48 --workers 4

LATEST_RUN=$(ls -dt ${PROJECT_ROOT}/runs/* 2>/dev/null | head -n1 || true)
if [ -n "${LATEST_RUN}" ]; then
  python -m xenogenesis.cli analyze --run ${LATEST_RUN} --species --phylogeny --frame-stride 4 --max-frames 300
fi
