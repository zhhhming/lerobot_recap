#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export LEROBOT_PYTHON="${PYTHON}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  src/lerobot/inference_engines/rtc.py \
  tests/inference_engines/test_rtc_memory.py \
  tests/policies/rtc/test_action_queue.py \
  tests/policies/rtc/test_latency_tracker.py

plans/memory_pipeline/validate_milestone_5.sh

"${PYTHON}" -m pytest \
  tests/inference_engines/test_rtc_memory.py \
  tests/policies/rtc \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py \
  -q

if "${PYTHON}" -m ruff --version >/dev/null 2>&1; then
  "${PYTHON}" -m ruff check \
    src/lerobot/inference_engines/rtc.py \
    tests/inference_engines/test_rtc_memory.py \
    tests/policies/rtc/test_action_queue.py \
    tests/policies/rtc/test_latency_tracker.py
  "${PYTHON}" -m ruff format --check \
    src/lerobot/inference_engines/rtc.py \
    tests/inference_engines/test_rtc_memory.py \
    tests/policies/rtc/test_action_queue.py \
    tests/policies/rtc/test_latency_tracker.py
else
  echo "ruff: skipped (not installed in ${PYTHON})"
fi

git diff --check
