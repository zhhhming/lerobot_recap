#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export LEROBOT_PYTHON="${PYTHON}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/policies/pi0/modeling_pi0.py \
  src/lerobot/policies/pi05/modeling_pi05.py \
  tests/policies/pi0_pi05/test_memory_modeling.py \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi05_subtask_training.py \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py

plans/memory_pipeline/validate_milestone_3.sh

"${PYTHON}" -m pytest \
  tests/policies/pi0_pi05/test_memory_modeling.py \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi05_subtask_training.py \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py \
  tests/processor/test_memory_processor.py \
  tests/processor/test_memory_disabled_baseline.py \
  -q

if "${PYTHON}" -m ruff --version >/dev/null 2>&1; then
  "${PYTHON}" -m ruff check \
    src/lerobot/policies/pi0/modeling_pi0.py \
    src/lerobot/policies/pi05/modeling_pi05.py \
    tests/policies/pi0_pi05/test_memory_modeling.py \
    tests/policies/pi0_pi05/test_pi0_subtask_inference.py \
    tests/policies/pi0_pi05/test_pi05_subtask_inference.py
  "${PYTHON}" -m ruff format --check \
    src/lerobot/policies/pi0/modeling_pi0.py \
    src/lerobot/policies/pi05/modeling_pi05.py \
    tests/policies/pi0_pi05/test_memory_modeling.py \
    tests/policies/pi0_pi05/test_pi0_subtask_inference.py \
    tests/policies/pi0_pi05/test_pi05_subtask_inference.py
else
  echo "ruff: skipped (not installed in ${PYTHON})"
fi

git diff --check
