#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"
PI0_CHECKPOINT="${LEROBOT_PI0_CHECKPOINT:-/home/zenbot-robot/models/lerobot/pi0_strike_match_3_relative_bs192_20260630_172828/checkpoints/012000/pretrained_model/model.safetensors}"
PI05_CHECKPOINT="${LEROBOT_PI05_CHECKPOINT:-/home/zenbot-robot/models/lerobot/pi05_strike_match_3_subtask_relative_bs192_20260706_200321/checkpoints/010000/pretrained_model/model.safetensors}"

cd "${REPO_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/value_function/configuration.py \
  src/lerobot/value_function/modeling_pi0_value.py \
  tests/value_function/test_value_model_shapes.py \
  tests/value_function/test_value_model_checkpoint.py \
  tests/value_function/test_value_model_real_checkpoint.py

"${PYTHON}" -m pytest \
  tests/value_function/test_value_model_shapes.py \
  tests/value_function/test_value_model_checkpoint.py \
  -q

"${PYTHON}" -m pytest \
  tests/value_function \
  tests/scripts/test_subtask_progress_data_pipeline.py \
  -q

if [[ ! -f "${PI0_CHECKPOINT}" ]]; then
  echo "Missing required PI0 checkpoint smoke input: ${PI0_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${PI05_CHECKPOINT}" ]]; then
  echo "Missing required PI0.5 checkpoint smoke input: ${PI05_CHECKPOINT}" >&2
  exit 1
fi

LEROBOT_RUN_REAL_VALUE_MODEL_SMOKE=1 \
LEROBOT_PI0_CHECKPOINT="${PI0_CHECKPOINT}" \
LEROBOT_PI05_CHECKPOINT="${PI05_CHECKPOINT}" \
  "${PYTHON}" -m pytest \
  tests/value_function/test_value_model_real_checkpoint.py \
  -q

git diff --check
