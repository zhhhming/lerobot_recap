#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/value_function/inference.py \
  src/lerobot/value_function/modeling_pi0_value.py \
  src/lerobot/value_function/dataset.py \
  src/lerobot/value_function/training.py \
  src/lerobot/scripts/lerobot_value_infer.py \
  tests/value_function/test_value_infer_writeback.py

"${PYTHON}" -m pytest \
  tests/value_function/test_value_infer_writeback.py \
  tests/value_function/test_raw_value_io.py \
  tests/value_function/test_value_model_shapes.py \
  tests/value_function/test_value_dataset.py \
  tests/value_function/test_train_value_smoke.py \
  tests/value_function/test_advantage.py \
  -q

if [[ "${LEROBOT_RUN_SOCKET_TESTS:-0}" == "1" ]]; then
  "${PYTHON}" -m pytest \
    tests/value_function \
    tests/scripts/test_subtask_progress_data_pipeline.py \
    -q
else
  "${PYTHON}" -m pytest \
    tests/value_function \
    tests/scripts/test_subtask_progress_data_pipeline.py \
    --ignore=tests/value_function/test_advantage_labeler_api.py \
    --ignore=tests/value_function/test_advantage_labeler_package.py \
    --ignore=tests/value_function/test_advantage_weight_viz_api.py \
    --ignore=tests/value_function/test_advantage_weight_viz_package.py \
    -q
fi

"${PYTHON}" -m lerobot.scripts.lerobot_value_infer --help >/dev/null

if [[ -n "${LEROBOT_VALUE_CHECKPOINT:-}" || -n "${LEROBOT_VALUE_INFER_SHADOW_RUN:-}" ]]; then
  if [[ -z "${LEROBOT_VALUE_CHECKPOINT:-}" || -z "${LEROBOT_VALUE_INFER_SHADOW_RUN:-}" ]]; then
    echo "Set both LEROBOT_VALUE_CHECKPOINT and LEROBOT_VALUE_INFER_SHADOW_RUN" >&2
    exit 1
  fi
  if [[ ! -f "${LEROBOT_VALUE_CHECKPOINT}" ]]; then
    echo "Missing value checkpoint: ${LEROBOT_VALUE_CHECKPOINT}" >&2
    exit 1
  fi
  if [[ ! -d "${LEROBOT_VALUE_INFER_SHADOW_RUN}" ]]; then
    echo "Missing writable shadow raw run: ${LEROBOT_VALUE_INFER_SHADOW_RUN}" >&2
    exit 1
  fi
  "${PYTHON}" -m lerobot.scripts.lerobot_value_infer \
    --root "${LEROBOT_VALUE_INFER_SHADOW_RUN}" \
    --checkpoint "${LEROBOT_VALUE_CHECKPOINT}" \
    --mode "${LEROBOT_VALUE_INFER_MODE:-both}" \
    --subtask_inference_path "${LEROBOT_VALUE_INFER_PATH:-both}" \
    --batch_size "${LEROBOT_VALUE_INFER_BATCH_SIZE:-8}" \
    --device "${LEROBOT_VALUE_INFER_DEVICE:-auto}"
fi

git diff --check
