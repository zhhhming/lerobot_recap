#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export LEROBOT_PYTHON="${PYTHON}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/utils/advantage_weights.py \
  src/lerobot/utils/memory_conditioning.py \
  src/lerobot/scripts/lerobot_train.py \
  src/lerobot/datasets/memory_history.py \
  tests/scripts/test_advantage_weighted_train.py \
  tests/datasets/test_memory_history.py

plans/memory_pipeline/validate_milestone_4.sh

"${PYTHON}" -m pytest \
  tests/scripts/test_advantage_weighted_train.py \
  tests/utils/test_advantage_weights.py \
  tests/utils/test_memory_conditioning.py \
  tests/scripts/test_memory_train.py \
  tests/datasets/test_memory_history.py \
  tests/processor/test_advantage_processor.py \
  tests/processor/test_memory_processor.py \
  tests/processor/test_converters.py \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi05_subtask_training.py \
  tests/policies/pi0_pi05/test_memory_modeling.py \
  -q

if "${PYTHON}" -m ruff --version >/dev/null 2>&1; then
  "${PYTHON}" -m ruff check \
    tests/scripts/test_advantage_weighted_train.py \
    tests/datasets/test_memory_history.py
  "${PYTHON}" -m ruff format --check \
    tests/scripts/test_advantage_weighted_train.py \
    tests/datasets/test_memory_history.py
else
  echo "ruff: skipped (not installed in ${PYTHON})"
fi

git diff --check
