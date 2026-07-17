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
    src/lerobot/datasets/subtask_timing.py \
    src/lerobot/datasets/memory_history.py \
    src/lerobot/datasets/factory.py \
    tests/datasets/test_subtask_timing.py \
    plans/timer_pipeline/subtask_time_m1_real_validation.py

  plans/timer_pipeline/validate_subtask_time_milestone_0.sh contract

  run_python -m pytest \
    tests/datasets/test_memory_history.py \
    -q

  if run_python -m ruff --version >/dev/null 2>&1; then
    run_python -m ruff check \
      src/lerobot/datasets/subtask_timing.py \
      src/lerobot/datasets/memory_history.py \
      src/lerobot/datasets/factory.py \
      tests/datasets/test_subtask_timing.py \
      plans/timer_pipeline/subtask_time_m1_real_validation.py
    run_python -m ruff format --check \
      src/lerobot/datasets/subtask_timing.py \
      src/lerobot/datasets/memory_history.py \
      src/lerobot/datasets/factory.py \
      tests/datasets/test_subtask_timing.py \
      plans/timer_pipeline/subtask_time_m1_real_validation.py
  else
    echo "ruff: skipped (not installed in Conda environment ${CONDA_ENV})"
  fi
}

data_validation() {
  run_python plans/timer_pipeline/subtask_time_m1_real_validation.py \
    --repo-id=ming326/strike_match_3_subtask \
    --root="${MATCH_ROOT}" \
    --expected-episodes=70 \
    --expected-frames=53794 \
    --expected-subtasks=6

  run_python plans/timer_pipeline/subtask_time_m1_real_validation.py \
    --repo-id=ming326/nero_egg_subtask \
    --root="${EGG_ROOT}" \
    --expected-episodes=61 \
    --expected-frames=350010 \
    --expected-subtasks=12 \
    --expected-stat="Stir the beaten eggs.:43.9:48.9" \
    --expected-stat="Start frying the eggs.:95.766667:100.766667"
}

regression_validation() {
  run_python -m pytest \
    tests/datasets/test_dataset_reader.py \
    tests/datasets/test_lerobot_dataset.py \
    tests/datasets/test_memory_history.py \
    -q
  plans/timer_pipeline/validate_subtask_time_milestone_0.sh regression
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
