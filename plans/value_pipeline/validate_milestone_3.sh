#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"
RAW_RUN="${LEROBOT_RAW_RUN:-/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3}"
PI0_CHECKPOINT="${LEROBOT_PI0_CHECKPOINT:-/home/zenbot-robot/models/lerobot/pi0_strike_match_3_relative_bs192_20260630_172828/checkpoints/012000/pretrained_model/model.safetensors}"

cd "${REPO_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/value_function/dataset.py \
  src/lerobot/value_function/training.py \
  src/lerobot/scripts/lerobot_train_value_function.py \
  tests/value_function/test_value_dataset.py \
  tests/value_function/test_train_value_smoke.py \
  tests/value_function/test_value_train_real_sample.py

"${PYTHON}" -m pytest \
  tests/value_function/test_value_dataset.py \
  tests/value_function/test_train_value_smoke.py \
  -q

"${PYTHON}" -m pytest \
  tests/value_function \
  tests/scripts/test_subtask_progress_data_pipeline.py \
  -q

if [[ ! -d "${RAW_RUN}" ]]; then
  echo "Missing required real raw-run smoke input: ${RAW_RUN}" >&2
  exit 1
fi
if [[ ! -f "${PI0_CHECKPOINT}" ]]; then
  echo "Missing required PI0 checkpoint smoke input: ${PI0_CHECKPOINT}" >&2
  exit 1
fi

LEROBOT_RUN_REAL_VALUE_TRAIN_DATA_SMOKE=1 \
LEROBOT_RAW_RUN="${RAW_RUN}" \
LEROBOT_PI0_CHECKPOINT="${PI0_CHECKPOINT}" \
  "${PYTHON}" -m pytest \
  tests/value_function/test_value_train_real_sample.py \
  -q

"${PYTHON}" -m lerobot.scripts.lerobot_train_value_function --help >/dev/null
git diff --check
