#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-automated}"
CONDA_ENV="${LEROBOT_CONDA_ENV:-lerobot-main}"
MATCH_REPO_ID="${LEROBOT_T7_MATCH_REPO_ID:-ming326/strike_match_3_subtask}"
MATCH_ROOT="${LEROBOT_T7_MATCH_ROOT:-/home/zenbot-robot/.cache/huggingface/lerobot/ming326/strike_match_3_subtask}"
EGG_REPO_ID="${LEROBOT_T7_EGG_REPO_ID:-ming326/nero_egg_subtask}"
EGG_ROOT="${LEROBOT_T7_EGG_ROOT:-/home/zenbot-robot/.cache/huggingface/lerobot/ming326/nero_egg_subtask}"
PI0_BASE="${LEROBOT_T7_PI0_BASE:-/home/zenbot-robot/models/lerobot/pi0_nero_egg_relative_bs256_20260715_155327/checkpoints/019000/pretrained_model}"
PI05_BASE="${LEROBOT_T7_PI05_BASE:-/home/zenbot-robot/models/lerobot/pi05_strike_match_3_subtask_relative_bs192_20260706_200321/checkpoints/010000/pretrained_model}"
PI0_HISTORY_BASE="${LEROBOT_T7_PI0_HISTORY_BASE:-${REPO_ROOT}/outputs/memory_m8_automated/pi0_memory/checkpoints/last/pretrained_model}"
PI05_HISTORY_BASE="${LEROBOT_T7_PI05_HISTORY_BASE:-${REPO_ROOT}/outputs/memory_m8_automated/pi05_memory/checkpoints/last/pretrained_model}"
OUTPUT_ROOT="${LEROBOT_T7_OUTPUT_ROOT:-${REPO_ROOT}/outputs/subtask_time_t7_automated}"

cd "${REPO_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export LEROBOT_T7_MATCH_REPO_ID="${MATCH_REPO_ID}"
export LEROBOT_T7_MATCH_ROOT="${MATCH_ROOT}"
export LEROBOT_T7_EGG_REPO_ID="${EGG_REPO_ID}"
export LEROBOT_T7_EGG_ROOT="${EGG_ROOT}"
export LEROBOT_T7_PI0_BASE="${PI0_BASE}"
export LEROBOT_T7_PI05_BASE="${PI05_BASE}"

run_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

preflight() {
  test -f "${MATCH_ROOT}/meta/info.json"
  test -f "${EGG_ROOT}/meta/info.json"
  test -f "${PI0_BASE}/model.safetensors"
  test -f "${PI05_BASE}/model.safetensors"
  test -f "${PI0_HISTORY_BASE}/model.safetensors"
  test -f "${PI05_HISTORY_BASE}/model.safetensors"
  run_python -c 'import torch; import sqlite3; import lerobot.scripts.lerobot_policy_deploy; print("deploy import: passed"); print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())'
  run_python -m py_compile \
    tests/datasets/test_subtask_time_t7_real_data.py \
    plans/timer_pipeline/subtask_time_t7_train_log_check.py \
    plans/timer_pipeline/subtask_time_t7_checkpoint_rtc_smoke.py
}

require_cuda() {
  run_python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)'
}

data_validation() {
  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_6.sh data
  run_python -m pytest tests/datasets/test_subtask_time_t7_real_data.py -q
}

regression_validation() {
  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_6.sh contract
  LEROBOT_CONDA_ENV="${CONDA_ENV}" \
    plans/timer_pipeline/validate_subtask_time_milestone_6.sh regression
  run_python -m pytest tests/processor/test_subtask_time_disabled_baseline.py -q
}

train_policy() {
  local policy_type="$1"
  local base_checkpoint="$2"
  local output_dir="$3"
  local log_file="$4"
  local steps="$5"
  local save_checkpoint="$6"
  local memory_enabled="$7"
  local advantage_enabled="$8"
  local dataset_root="$9"
  shift 9

  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing T7 output: ${output_dir}" >&2
    return 2
  fi

  local episodes='[0]'
  local advantage_args=()
  if [[ "${advantage_enabled}" == true ]]; then
    episodes='[0,1,2]'
    advantage_args=(
      --policy.use_advantage_conditioning=true
      --policy.advantage_label_key=advantage_label_subtask
      --policy.advantage_loss_weight_key=advantage_loss_weight_subtask
      --use_advantage_weighting=true
      --advantage_label_key=advantage_label_subtask
      --advantage_loss_weight_key=advantage_loss_weight_subtask
      --advantage_condition_dropout_prob=0.1
    )
  fi

  local memory_args=(--policy.use_memory_conditioning=false)
  if [[ "${memory_enabled}" == true ]]; then
    memory_args=(
      --policy.use_memory_conditioning=true
      --policy.memory_tokenizer_max_length=128
      --memory_lookback_min_frames=1
      --memory_lookback_max_frames=12
      --memory_dropout_prob=0.2
    )
  fi

  run_python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${EGG_REPO_ID}" \
    --dataset.root="${dataset_root}" \
    --dataset.episodes="${episodes}" \
    --dataset.streaming=false \
    --dataset.use_imagenet_stats=true \
    --policy.path="${base_checkpoint}" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.predict_subtask=true \
    --policy.subtask_max_tokens=48 \
    --policy.subtask_max_decode_tokens=48 \
    --policy.subtask_ce_loss_weight=0.25 \
    --policy.subtask_dropout_prob=0.2 \
    --policy.subtask_generate_at_inference=true \
    --policy.use_subtask_time_conditioning=true \
    --policy.subtask_time_tokenizer_max_length=128 \
    --subtask_time_noise_ratio=0.4 \
    --subtask_time_noise_max_seconds=5.0 \
    --subtask_time_dropout_prob=0.2 \
    --batch_size=1 \
    --num_workers=0 \
    --steps="${steps}" \
    --log_freq=1 \
    --eval_freq=0 \
    --save_checkpoint="${save_checkpoint}" \
    --save_freq="${steps}" \
    --output_dir="${output_dir}" \
    --job_name="subtask_time_t7_${policy_type}" \
    --wandb.enable=false \
    --use_policy_training_preset=false \
    --optimizer.type=sgd \
    --optimizer.lr=0.00001 \
    --optimizer.momentum=0.0 \
    --optimizer.weight_decay=0.0 \
    --optimizer.grad_clip_norm=1.0 \
    --scheduler.type=diffuser \
    --scheduler.name=constant \
    --scheduler.num_warmup_steps=0 \
    "${memory_args[@]}" \
    "${advantage_args[@]}" \
    "$@" >"${log_file}" 2>&1

  run_python plans/timer_pipeline/subtask_time_t7_train_log_check.py \
    --log="${log_file}" \
    --expected-updates="${steps}" \
    --label="$(basename "${output_dir}")" \
    --report="${OUTPUT_ROOT}/reports/$(basename "${output_dir}")_metrics.json"
}

checkpoint_validation() {
  local pi0_checkpoint="${OUTPUT_ROOT}/pi0_both/checkpoints/last/pretrained_model"
  local pi05_checkpoint="${OUTPUT_ROOT}/pi05_both/checkpoints/last/pretrained_model"
  test -f "${pi0_checkpoint}/model.safetensors"
  test -f "${pi05_checkpoint}/model.safetensors"

  run_python plans/timer_pipeline/subtask_time_t7_checkpoint_rtc_smoke.py \
    --checkpoint="${pi0_checkpoint}" \
    --dataset-root="${EGG_ROOT}" \
    --dataset-repo-id="${EGG_REPO_ID}" \
    --expected-policy=pi0 \
    --report="${OUTPUT_ROOT}/reports/pi0_checkpoint_rtc.json"
  run_python plans/timer_pipeline/subtask_time_t7_checkpoint_rtc_smoke.py \
    --checkpoint="${pi05_checkpoint}" \
    --dataset-root="${EGG_ROOT}" \
    --dataset-repo-id="${EGG_REPO_ID}" \
    --expected-policy=pi05 \
    --report="${OUTPUT_ROOT}/reports/pi05_checkpoint_rtc.json"

  run_python plans/memory_pipeline/m8_checkpoint_rtc_smoke.py \
    --checkpoint="${PI0_BASE}" \
    --dataset-root="${EGG_ROOT}" \
    --dataset-repo-id="${EGG_REPO_ID}" \
    --expected-policy=pi0 \
    --memory-mode=off
  run_python plans/memory_pipeline/m8_checkpoint_rtc_smoke.py \
    --checkpoint="${PI05_BASE}" \
    --dataset-root="${EGG_ROOT}" \
    --dataset-repo-id="${EGG_REPO_ID}" \
    --expected-policy=pi05 \
    --memory-mode=off
}

gpu_validation() {
  require_cuda
  if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Refusing to overwrite existing T7 output root: ${OUTPUT_ROOT}" >&2
    return 2
  fi
  mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/reports"

  train_policy pi0 "${PI0_HISTORY_BASE}" "${OUTPUT_ROOT}/pi0_both" \
    "${OUTPUT_ROOT}/logs/pi0_both.log" 2 true true false "${EGG_ROOT}" \
    --policy.image_augmentation.enable=false
  train_policy pi05 "${PI05_HISTORY_BASE}" "${OUTPUT_ROOT}/pi05_both" \
    "${OUTPUT_ROOT}/logs/pi05_both.log" 2 true true false "${EGG_ROOT}"

  train_policy pi0 "${PI0_BASE}" "${OUTPUT_ROOT}/pi0_time_only" \
    "${OUTPUT_ROOT}/logs/pi0_time_only.log" 1 false false false "${EGG_ROOT}" \
    --policy.image_augmentation.enable=false
  train_policy pi05 "${PI05_BASE}" "${OUTPUT_ROOT}/pi05_time_only" \
    "${OUTPUT_ROOT}/logs/pi05_time_only.log" 1 false false false "${EGG_ROOT}"

  local fixture_parent
  fixture_parent="$(mktemp -d /tmp/lerobot-t7-advantage.XXXXXX)"
  local fixture_root="${fixture_parent}/nero_egg_subtask_advantage"
  run_python plans/memory_pipeline/m8_make_advantage_fixture.py \
    --source="${EGG_ROOT}" \
    --destination="${fixture_root}"
  printf '%s\n' "${fixture_root}" >"${OUTPUT_ROOT}/reports/advantage_fixture_path.txt"

  train_policy pi0 "${OUTPUT_ROOT}/pi0_both/checkpoints/last/pretrained_model" \
    "${OUTPUT_ROOT}/pi0_advantage" "${OUTPUT_ROOT}/logs/pi0_advantage.log" \
    1 false true true "${fixture_root}" --policy.image_augmentation.enable=false
  train_policy pi05 "${OUTPUT_ROOT}/pi05_both/checkpoints/last/pretrained_model" \
    "${OUTPUT_ROOT}/pi05_advantage" "${OUTPUT_ROOT}/logs/pi05_advantage.log" \
    1 false true true "${fixture_root}"

  checkpoint_validation
}

static_validation() {
  bash -n plans/timer_pipeline/validate_subtask_time_milestone_7.sh
  git diff --check
  if run_python -m ruff --version >/dev/null 2>&1; then
    run_python -m ruff check \
      tests/datasets/test_subtask_time_t7_real_data.py \
      plans/timer_pipeline/subtask_time_t7_train_log_check.py \
      plans/timer_pipeline/subtask_time_t7_checkpoint_rtc_smoke.py
    run_python -m ruff format --check \
      tests/datasets/test_subtask_time_t7_real_data.py \
      plans/timer_pipeline/subtask_time_t7_train_log_check.py \
      plans/timer_pipeline/subtask_time_t7_checkpoint_rtc_smoke.py
  else
    echo "ruff: skipped (not installed in Conda environment ${CONDA_ENV})"
  fi
}

preflight
case "${MODE}" in
  data)
    data_validation
    ;;
  regression)
    regression_validation
    ;;
  gpu)
    gpu_validation
    ;;
  checkpoints)
    require_cuda
    checkpoint_validation
    ;;
  automated)
    regression_validation
    data_validation
    gpu_validation
    ;;
  *)
    echo "Usage: $0 [data|regression|gpu|checkpoints|automated]" >&2
    exit 2
    ;;
esac
static_validation
