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
    src/lerobot/processor/subtask_time_processor.py \
    src/lerobot/processor/__init__.py \
    src/lerobot/processor/converters.py \
    src/lerobot/policies/pi0/configuration_pi0.py \
    src/lerobot/policies/pi0/processor_pi0.py \
    src/lerobot/policies/pi05/configuration_pi05.py \
    src/lerobot/policies/pi05/processor_pi05.py \
    src/lerobot/scripts/lerobot_train.py \
    tests/processor/test_subtask_time_processor.py \
    tests/processor/test_subtask_time_disabled_baseline.py \
    tests/processor/test_converters.py \
    tests/scripts/test_subtask_time_checkpoint.py

  run_python -m pytest \
    tests/processor/test_subtask_time_processor.py \
    tests/processor/test_subtask_time_disabled_baseline.py \
    tests/processor/test_converters.py \
    -q

  run_python -m pytest \
    tests/datasets/test_subtask_timing.py \
    tests/utils/test_subtask_time_conditioning.py \
    tests/scripts/test_subtask_time_train.py \
    tests/inference_engines/test_subtask_time_tracker.py \
    -q \
    -k "not workers"

  if run_python -m ruff --version >/dev/null 2>&1; then
    run_python -m ruff check \
      src/lerobot/processor/subtask_time_processor.py \
      src/lerobot/processor/__init__.py \
      src/lerobot/processor/converters.py \
      src/lerobot/policies/pi0/configuration_pi0.py \
      src/lerobot/policies/pi0/processor_pi0.py \
      src/lerobot/policies/pi05/configuration_pi05.py \
      src/lerobot/policies/pi05/processor_pi05.py \
      src/lerobot/scripts/lerobot_train.py \
      tests/processor/test_subtask_time_processor.py \
      tests/processor/test_subtask_time_disabled_baseline.py \
      tests/processor/test_converters.py \
      tests/scripts/test_subtask_time_checkpoint.py
    run_python -m ruff format --check \
      src/lerobot/processor/subtask_time_processor.py \
      src/lerobot/processor/__init__.py \
      src/lerobot/processor/converters.py \
      src/lerobot/policies/pi0/configuration_pi0.py \
      src/lerobot/policies/pi0/processor_pi0.py \
      src/lerobot/policies/pi05/configuration_pi05.py \
      src/lerobot/policies/pi05/processor_pi05.py \
      src/lerobot/scripts/lerobot_train.py \
      tests/processor/test_subtask_time_processor.py \
      tests/processor/test_subtask_time_disabled_baseline.py \
      tests/processor/test_converters.py \
      tests/scripts/test_subtask_time_checkpoint.py
  else
    echo "ruff: skipped (not installed in Conda environment ${CONDA_ENV})"
  fi
}

checkpoint_validation() {
  run_python -m pytest \
    tests/scripts/test_subtask_time_checkpoint.py \
    -q
}

regression_validation() {
  run_python -m pytest \
    tests/processor/test_memory_disabled_baseline.py \
    tests/processor/test_memory_processor.py \
    tests/processor/test_subtask_ar_processors.py \
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
    tests/datasets/test_subtask_timing.py \
    tests/utils/test_subtask_time_conditioning.py \
    tests/scripts/test_subtask_time_train.py \
    -q \
    -k "not workers"
}

case "${MODE}" in
  contract)
    contract_validation
    ;;
  checkpoint)
    checkpoint_validation
    ;;
  regression)
    regression_validation
    ;;
  all)
    contract_validation
    checkpoint_validation
    regression_validation
    ;;
  *)
    echo "Usage: $0 [contract|checkpoint|regression|all]" >&2
    exit 2
    ;;
esac

bash -n "$0"
git diff --check
