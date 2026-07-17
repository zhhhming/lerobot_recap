#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export LEROBOT_PYTHON="${PYTHON}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/processor/memory_processor.py \
  src/lerobot/processor/subtask_processor.py \
  src/lerobot/processor/converters.py \
  src/lerobot/processor/__init__.py \
  src/lerobot/policies/pi0/configuration_pi0.py \
  src/lerobot/policies/pi0/processor_pi0.py \
  src/lerobot/policies/pi05/configuration_pi05.py \
  src/lerobot/policies/pi05/processor_pi05.py \
  tests/processor/test_memory_processor.py \
  tests/processor/test_converters.py

plans/memory_pipeline/validate_milestone_1.sh

if "${PYTHON}" -m ruff --version >/dev/null 2>&1; then
  "${PYTHON}" -m ruff check \
    src/lerobot/processor/memory_processor.py \
    src/lerobot/processor/subtask_processor.py \
    src/lerobot/processor/converters.py \
    src/lerobot/processor/__init__.py \
    src/lerobot/policies/pi0/configuration_pi0.py \
    src/lerobot/policies/pi0/processor_pi0.py \
    src/lerobot/policies/pi05/configuration_pi05.py \
    src/lerobot/policies/pi05/processor_pi05.py \
    tests/processor/test_memory_processor.py \
    tests/processor/test_converters.py
  "${PYTHON}" -m ruff format --check \
    src/lerobot/processor/memory_processor.py \
    src/lerobot/processor/subtask_processor.py \
    src/lerobot/processor/converters.py \
    src/lerobot/processor/__init__.py \
    src/lerobot/policies/pi0/configuration_pi0.py \
    src/lerobot/policies/pi0/processor_pi0.py \
    src/lerobot/policies/pi05/configuration_pi05.py \
    src/lerobot/policies/pi05/processor_pi05.py \
    tests/processor/test_memory_processor.py \
    tests/processor/test_converters.py
else
  echo "ruff: skipped (not installed in ${PYTHON})"
fi

git diff --check
