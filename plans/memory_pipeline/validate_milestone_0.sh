#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LEROBOT_PYTHON:-/home/zenbot-robot/.conda/envs/lerobot-main/bin/python}"

cd "${REPO_ROOT}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"${PYTHON}" -m py_compile \
  tests/processor/test_memory_disabled_baseline.py \
  tests/datasets/test_memory_history.py \
  tests/utils/test_memory_conditioning.py \
  tests/processor/test_memory_processor.py

"${PYTHON}" -m pytest \
  tests/processor/test_memory_disabled_baseline.py \
  tests/datasets/test_memory_history.py \
  tests/utils/test_memory_conditioning.py \
  tests/processor/test_memory_processor.py \
  tests/processor/test_subtask_ar_processors.py \
  tests/processor/test_advantage_processor.py \
  tests/processor/test_tokenizer_processor.py \
  tests/processor/test_converters.py \
  tests/utils/test_advantage_weights.py \
  tests/scripts/test_advantage_weighted_train.py \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi05_subtask_training.py \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py \
  -q

git diff --check
