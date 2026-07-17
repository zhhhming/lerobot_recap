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
    src/lerobot/inference_engines/rtc.py \
    src/lerobot/scripts/lerobot_policy_deploy.py \
    src/lerobot/utils/terminal_status.py \
    tests/inference_engines/test_rtc_subtask_time.py \
    tests/scripts/test_subtask_time_deploy.py \
    tests/scripts/test_lerobot_policy_deploy_status.py

  run_python -m pytest \
    tests/inference_engines/test_rtc_subtask_time.py \
    tests/inference_engines/test_subtask_time_tracker.py \
    tests/inference_engines/test_rtc_memory.py \
    tests/scripts/test_subtask_time_deploy.py \
    tests/scripts/test_lerobot_policy_deploy_status.py \
    -q

  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_5.sh contract

  if run_python -m ruff --version >/dev/null 2>&1; then
    run_python -m ruff check \
      src/lerobot/inference_engines/rtc.py \
      src/lerobot/scripts/lerobot_policy_deploy.py \
      src/lerobot/utils/terminal_status.py \
      tests/inference_engines/test_rtc_subtask_time.py \
      tests/scripts/test_subtask_time_deploy.py \
      tests/scripts/test_lerobot_policy_deploy_status.py
    run_python -m ruff format --check \
      src/lerobot/inference_engines/rtc.py \
      src/lerobot/scripts/lerobot_policy_deploy.py \
      src/lerobot/utils/terminal_status.py \
      tests/inference_engines/test_rtc_subtask_time.py \
      tests/scripts/test_subtask_time_deploy.py \
      tests/scripts/test_lerobot_policy_deploy_status.py
  else
    echo "ruff: skipped (not installed in Conda environment ${CONDA_ENV})"
  fi
}

data_validation() {
  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_5.sh data
}

regression_validation() {
  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_5.sh regression

  run_python -m pytest \
    tests/inference_engines/test_rtc_memory.py \
    tests/inference_engines/test_rtc_subtask_time.py \
    tests/scripts/test_subtask_time_deploy.py \
    tests/scripts/test_lerobot_policy_deploy_status.py \
    -q

  run_python -m pytest \
    tests/scripts/test_lerobot_hil_record.py \
    -k terminal_keyboard \
    -q
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
