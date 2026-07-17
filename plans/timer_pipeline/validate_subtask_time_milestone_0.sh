#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
CONDA_ENV="${LEROBOT_CONDA_ENV:-lerobot-main}"
MATCH_ROOT="${LEROBOT_TIMER_MATCH_ROOT:-/home/zenbot-robot/.cache/huggingface/lerobot/ming326/strike_match_3_subtask}"
EGG_ROOT="${LEROBOT_TIMER_EGG_ROOT:-/home/zenbot-robot/.cache/huggingface/lerobot/ming326/nero_egg_subtask}"

cd "${REPO_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

run_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

contract_validation() {
  run_python -m py_compile \
    tests/processor/test_subtask_time_disabled_baseline.py \
    tests/datasets/test_subtask_timing.py \
    tests/utils/test_subtask_time_conditioning.py \
    tests/processor/test_subtask_time_processor.py \
    tests/inference_engines/test_subtask_time_tracker.py \
    plans/timer_pipeline/subtask_time_m0_data_audit.py

  run_python -m pytest \
    tests/processor/test_subtask_time_disabled_baseline.py \
    tests/datasets/test_subtask_timing.py \
    tests/utils/test_subtask_time_conditioning.py \
    tests/processor/test_subtask_time_processor.py \
    tests/inference_engines/test_subtask_time_tracker.py \
    -q

  run_python -m pytest \
    tests/processor/test_memory_disabled_baseline.py \
    tests/processor/test_subtask_ar_processors.py \
    tests/processor/test_memory_processor.py \
    tests/processor/test_advantage_processor.py \
    tests/utils/test_memory_conditioning.py \
    tests/utils/test_advantage_weights.py \
    tests/scripts/test_memory_train.py \
    tests/scripts/test_advantage_weighted_train.py \
    tests/policies/pi0_pi05/test_memory_modeling.py \
    tests/policies/pi0_pi05/test_pi0_subtask_training.py \
    tests/policies/pi0_pi05/test_pi05_subtask_training.py \
    tests/policies/pi0_pi05/test_pi0_subtask_inference.py \
    tests/policies/pi0_pi05/test_pi05_subtask_inference.py \
    tests/inference_engines/test_rtc_memory.py \
    tests/scripts/test_lerobot_policy_deploy_status.py \
    -q

  if run_python -m ruff --version >/dev/null 2>&1; then
    run_python -m ruff check \
      tests/processor/test_subtask_time_disabled_baseline.py \
      tests/datasets/test_subtask_timing.py \
      tests/utils/test_subtask_time_conditioning.py \
      tests/processor/test_subtask_time_processor.py \
      tests/inference_engines/test_subtask_time_tracker.py \
      plans/timer_pipeline/subtask_time_m0_data_audit.py
    run_python -m ruff format --check \
      tests/processor/test_subtask_time_disabled_baseline.py \
      tests/datasets/test_subtask_timing.py \
      tests/utils/test_subtask_time_conditioning.py \
      tests/processor/test_subtask_time_processor.py \
      tests/inference_engines/test_subtask_time_tracker.py \
      plans/timer_pipeline/subtask_time_m0_data_audit.py
  else
    echo "ruff: skipped (not installed in Conda environment ${CONDA_ENV})"
  fi
}

data_validation() {
  run_python plans/timer_pipeline/subtask_time_m0_data_audit.py \
    --repo-id=ming326/strike_match_3_subtask \
    --root="${MATCH_ROOT}" \
    --expected-episodes=70 \
    --expected-frames=53794 \
    --expected-subtasks=6
  run_python plans/timer_pipeline/subtask_time_m0_data_audit.py \
    --repo-id=ming326/nero_egg_subtask \
    --root="${EGG_ROOT}" \
    --expected-episodes=61 \
    --expected-frames=350010 \
    --expected-subtasks=12
}

regression_validation() {
  LEROBOT_CONDA_ENV="${CONDA_ENV}" plans/memory_pipeline/validate_milestone_8.sh regression
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

git diff --check
