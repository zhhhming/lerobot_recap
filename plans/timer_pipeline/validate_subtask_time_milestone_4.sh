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
    src/lerobot/datasets/subtask_timing.py \
    src/lerobot/inference_engines/subtask_time_tracker.py \
    tests/datasets/test_subtask_timing.py \
    tests/inference_engines/test_subtask_time_tracker.py

  run_python -m pytest \
    tests/inference_engines/test_subtask_time_tracker.py \
    -q

  run_python -m pytest \
    tests/datasets/test_subtask_timing.py \
    -q \
    -k "scanner and not workers"

  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_3.sh contract

  if run_python -m ruff --version >/dev/null 2>&1; then
    run_python -m ruff check \
      src/lerobot/inference_engines/subtask_time_tracker.py \
      tests/inference_engines/test_subtask_time_tracker.py
    run_python -m ruff format --check \
      src/lerobot/inference_engines/subtask_time_tracker.py \
      tests/inference_engines/test_subtask_time_tracker.py
  else
    echo "ruff: skipped (not installed in Conda environment ${CONDA_ENV})"
  fi
}

data_validation() {
  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_1.sh data
}

regression_validation() {
  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_3.sh checkpoint
  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_3.sh regression
}

case "${MODE}" in
  contract)
    contract_validation
    ;;
  data)
    data_validation
    ;;
  regression)
    regression_validation
    ;;
  all)
    contract_validation
    data_validation
    regression_validation
    ;;
  *)
    echo "Usage: $0 [contract|data|regression|all]" >&2
    exit 2
    ;;
esac

bash -n "$0"
git diff --check
