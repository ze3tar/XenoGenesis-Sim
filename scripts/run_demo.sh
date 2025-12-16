#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}

echo "Running CA demo..."
python -m xenogenesis.cli run ca \
  --config ${PROJECT_ROOT}/src/xenogenesis/config/defaults.yaml \
  --generations 40 --pop 64 --steps 256 --workers 4

LATEST_RUN=$(ls -dt ${PROJECT_ROOT}/runs/* 2>/dev/null | head -n1 || true)
if [ -n "${LATEST_RUN}" ]; then
  python -m xenogenesis.cli analyze --run ${LATEST_RUN}
fi
