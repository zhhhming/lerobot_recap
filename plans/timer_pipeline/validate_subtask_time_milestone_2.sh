#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
CONDA_ENV="${LEROBOT_CONDA_ENV:-lerobot-main}"

cd "${REPO_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

run_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

contract_validation() {
  run_python -m py_compile \
    src/lerobot/utils/subtask_time_conditioning.py \
    src/lerobot/configs/train.py \
    src/lerobot/scripts/lerobot_train.py \
    tests/utils/test_subtask_time_conditioning.py \
    tests/scripts/test_subtask_time_train.py

  run_python -m pytest \
    tests/utils/test_subtask_time_conditioning.py \
    tests/scripts/test_subtask_time_train.py \
    -q

  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_1.sh contract

  if run_python -m ruff --version >/dev/null 2>&1; then
    run_python -m ruff check \
      src/lerobot/utils/subtask_time_conditioning.py \
      src/lerobot/configs/train.py \
      src/lerobot/scripts/lerobot_train.py \
      tests/utils/test_subtask_time_conditioning.py \
      tests/scripts/test_subtask_time_train.py
    run_python -m ruff format --check \
      src/lerobot/utils/subtask_time_conditioning.py \
      src/lerobot/configs/train.py \
      src/lerobot/scripts/lerobot_train.py \
      tests/utils/test_subtask_time_conditioning.py \
      tests/scripts/test_subtask_time_train.py
  else
    echo "ruff: skipped (not installed in Conda environment ${CONDA_ENV})"
  fi
}

smoke_validation() {
  run_python -m pytest \
    tests/scripts/test_subtask_time_train.py \
    -q \
    -k "cpu_single_step"
}

regression_validation() {
  run_python -m pytest \
    tests/utils/test_memory_conditioning.py \
    tests/utils/test_advantage_weights.py \
    tests/scripts/test_memory_train.py \
    tests/scripts/test_advantage_weighted_train.py \
    tests/policies/pi0_pi05/test_memory_modeling.py \
    tests/policies/pi0_pi05/test_pi0_subtask_training.py \
    tests/policies/pi0_pi05/test_pi05_subtask_training.py \
    -q

  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_1.sh regression
}

case "${MODE}" in
  contract)
    contract_validation
    ;;
  smoke)
    smoke_validation
    ;;
  regression)
    regression_validation
    ;;
  all)
    contract_validation
    smoke_validation
    regression_validation
    ;;
  *)
    echo "Usage: $0 [contract|smoke|regression|all]" >&2
    exit 2
    ;;
esac

bash -n "$0"
git diff --check
