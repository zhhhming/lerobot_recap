#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export LEROBOT_PYTHON="${PYTHON}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/utils/memory_conditioning.py \
  src/lerobot/configs/train.py \
  src/lerobot/scripts/lerobot_train.py \
  tests/utils/test_memory_conditioning.py \
  tests/scripts/test_memory_train.py

plans/memory_pipeline/validate_milestone_2.sh

"${PYTHON}" -m pytest \
  tests/utils/test_memory_conditioning.py \
  tests/scripts/test_memory_train.py \
  tests/utils/test_advantage_weights.py \
  tests/scripts/test_advantage_weighted_train.py \
  -q

if "${PYTHON}" -m ruff --version >/dev/null 2>&1; then
  "${PYTHON}" -m ruff check \
    src/lerobot/utils/memory_conditioning.py \
    src/lerobot/configs/train.py \
    src/lerobot/scripts/lerobot_train.py \
    tests/utils/test_memory_conditioning.py \
    tests/scripts/test_memory_train.py
  "${PYTHON}" -m ruff format --check \
    src/lerobot/utils/memory_conditioning.py \
    src/lerobot/configs/train.py \
    src/lerobot/scripts/lerobot_train.py \
    tests/utils/test_memory_conditioning.py \
    tests/scripts/test_memory_train.py
else
  echo "ruff: skipped (not installed in ${PYTHON})"
fi

git diff --check
