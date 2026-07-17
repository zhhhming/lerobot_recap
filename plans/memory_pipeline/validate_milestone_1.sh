#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export LEROBOT_PYTHON="${PYTHON}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/datasets/memory_history.py \
  src/lerobot/datasets/factory.py \
  tests/datasets/test_memory_history.py

plans/memory_pipeline/validate_milestone_0.sh

"${PYTHON}" -m pytest \
  tests/datasets/test_dataset_reader.py \
  tests/datasets/test_lerobot_dataset.py \
  -q

if "${PYTHON}" -m ruff --version >/dev/null 2>&1; then
  "${PYTHON}" -m ruff check \
    src/lerobot/datasets/memory_history.py \
    src/lerobot/datasets/factory.py \
    tests/datasets/test_memory_history.py
  "${PYTHON}" -m ruff format --check \
    src/lerobot/datasets/memory_history.py \
    src/lerobot/datasets/factory.py \
    tests/datasets/test_memory_history.py
else
  echo "ruff: skipped (not installed in ${PYTHON})"
fi

git diff --check
