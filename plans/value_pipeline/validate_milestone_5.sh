#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/scripts/lerobot_value_viz.py \
  tests/value_function/test_value_viz_data.py \
  tests/value_function/test_value_viz_api.py \
  tests/value_function/test_value_viz_package.py \
  tests/value_function/test_value_viz_real_sample.py

if command -v node >/dev/null 2>&1; then
  node --check src/lerobot/scripts/value_viz/app.js
fi

"${PYTHON}" -m pytest \
  tests/value_function/test_value_viz_data.py \
  tests/value_function/test_value_infer_writeback.py \
  tests/value_function/test_raw_value_io.py \
  -q

if [[ "${LEROBOT_RUN_SOCKET_TESTS:-0}" == "1" ]]; then
  "${PYTHON}" -m pytest \
    tests/value_function/test_value_viz_api.py \
    tests/value_function/test_value_viz_package.py \
    tests/value_function/test_advantage_labeler_api.py \
    tests/value_function/test_advantage_labeler_package.py \
    tests/value_function/test_advantage_weight_viz_api.py \
    tests/value_function/test_advantage_weight_viz_package.py \
    -q
else
  echo "Skipping local HTTP/wheel service tests; set LEROBOT_RUN_SOCKET_TESTS=1 to run them."
fi

"${PYTHON}" -m pytest \
  tests/value_function \
  tests/scripts/test_subtask_progress_data_pipeline.py \
  --ignore=tests/value_function/test_value_viz_api.py \
  --ignore=tests/value_function/test_value_viz_package.py \
  --ignore=tests/value_function/test_advantage_labeler_api.py \
  --ignore=tests/value_function/test_advantage_labeler_package.py \
  --ignore=tests/value_function/test_advantage_weight_viz_api.py \
  --ignore=tests/value_function/test_advantage_weight_viz_package.py \
  -q

if [[ "${LEROBOT_RUN_REAL_VALUE_VIZ_SMOKE:-0}" == "1" ]]; then
  LEROBOT_RUN_REAL_VALUE_VIZ_SMOKE=1 "${PYTHON}" -m pytest \
    tests/value_function/test_value_viz_real_sample.py \
    -q
fi

"${PYTHON}" -m lerobot.scripts.lerobot_value_viz --help >/dev/null
git diff --check
