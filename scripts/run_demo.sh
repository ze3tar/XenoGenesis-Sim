#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH:-}
CONFIG_PATH=${1:-${PROJECT_ROOT}/configs/alien_life.yaml}

echo "Running CA demo..."
STEP_ARGS=()
if [ -n "${DEMO_STEPS:-}" ]; then
  STEP_ARGS=(--steps ${DEMO_STEPS})
fi
if [ "${DEMO_ANALYZE:-0}" = "0" ]; then
  export XG_SKIP_ANALYSIS=1
else
  unset XG_SKIP_ANALYSIS
fi

python -m xenogenesis.cli run ca \
  --config ${CONFIG_PATH} \
  --no-render \
  "${STEP_ARGS[@]}"

LATEST_RUN=$(ls -dt ${PROJECT_ROOT}/runs/* 2>/dev/null | head -n1 || true)
if [ -n "${LATEST_RUN}" ] && [ "${DEMO_ANALYZE:-0}" != "0" ]; then
  python -m xenogenesis.cli analyze \
    --run ${LATEST_RUN} \
    --species \
    --phylogeny \
    --frame-stride ${DEMO_FRAME_STRIDE:-6} \
    --max-frames ${DEMO_MAX_FRAMES:-120}
fi
