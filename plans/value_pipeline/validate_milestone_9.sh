#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/scripts/lerobot_build_dataset.py \
  src/lerobot/datasets/factory.py \
  src/lerobot/processor/converters.py \
  tests/scripts/test_value_extras_build_dataset.py \
  tests/value_function/test_milestone_9_shadow_smoke.py

"${PYTHON}" -m pytest \
  tests/scripts/test_value_extras_build_dataset.py \
  tests/scripts/test_subtask_progress_data_pipeline.py \
  tests/processor/test_converters.py \
  -q

"${PYTHON}" -m pytest \
  tests/value_function/test_raw_value_io.py \
  tests/value_function/test_value_targets.py \
  tests/value_function/test_mock_predictions.py \
  tests/value_function/test_advantage.py \
  tests/value_function/test_advantage_labeling.py \
  tests/value_function/test_advantage_labeler_api.py \
  tests/value_function/test_advantage_labeler_package.py \
  tests/value_function/test_advantage_weights.py \
  tests/value_function/test_advantage_weight_viz_api.py \
  tests/value_function/test_advantage_weight_viz_package.py \
  -q

LEROBOT_RUN_REAL_VALUE_PIPELINE_SMOKE=1 "${PYTHON}" -m pytest \
  tests/value_function/test_milestone_8_shadow_smoke.py \
  tests/value_function/test_milestone_9_shadow_smoke.py \
  -q

git diff --check
