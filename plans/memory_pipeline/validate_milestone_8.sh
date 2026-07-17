#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-data}"
CONDA_ENV="${LEROBOT_CONDA_ENV:-lerobot-main}"
DATASET_REPO_ID="${LEROBOT_M8_DATASET_REPO_ID:-ming326/nero_egg_subtask}"
DATASET_ROOT="${LEROBOT_M8_DATASET_ROOT:-/home/zenbot-robot/.cache/huggingface/lerobot/ming326/nero_egg_subtask}"
PI0_BASE="${LEROBOT_M8_PI0_BASE:-/home/zenbot-robot/models/lerobot/pi0_nero_egg_relative_bs256_20260715_155327/checkpoints/019000/pretrained_model}"
PI05_BASE="${LEROBOT_M8_PI05_BASE:-/home/zenbot-robot/models/lerobot/pi05_strike_match_3_subtask_relative_bs192_20260706_200321/checkpoints/010000/pretrained_model}"
OUTPUT_ROOT="${LEROBOT_M8_OUTPUT_ROOT:-${REPO_ROOT}/outputs/memory_m8_automated}"

cd "${REPO_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export LEROBOT_M8_DATASET_REPO_ID="${DATASET_REPO_ID}"
export LEROBOT_M8_DATASET_ROOT="${DATASET_ROOT}"
export LEROBOT_M8_PI0_BASE="${PI0_BASE}"
export LEROBOT_M8_PI05_BASE="${PI05_BASE}"

run_python() {
  conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

preflight() {
  run_python -c 'import torch; import sqlite3; import lerobot.scripts.lerobot_policy_deploy; print("deploy import: passed"); print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())'
  test -f "${DATASET_ROOT}/meta/info.json"
  test -f "${PI0_BASE}/model.safetensors"
  test -f "${PI05_BASE}/model.safetensors"
  run_python -m py_compile \
    tests/datasets/test_memory_m8_real_dataset.py \
    plans/memory_pipeline/m8_make_advantage_fixture.py \
    plans/memory_pipeline/m8_checkpoint_rtc_smoke.py
}

data_validation() {
  run_python -m pytest tests/datasets/test_memory_m8_real_dataset.py -q
}

regression_validation() {
  CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_7.sh
}

train_policy() {
  local policy_type="$1"
  local base_checkpoint="$2"
  local output_dir="$3"
  local dataset_root="$4"
  local advantage_enabled="$5"
  shift 5

  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing M8 output: ${output_dir}" >&2
    return 2
  fi

  local save_checkpoint=true
  local steps=2
  local episodes='[0]'
  local advantage_args=()
  if [[ "${advantage_enabled}" == true ]]; then
    save_checkpoint=false
    steps=1
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

  run_python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
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
    --policy.use_memory_conditioning=true \
    --policy.memory_tokenizer_max_length=128 \
    --memory_lookback_min_frames=1 \
    --memory_lookback_max_frames=12 \
    --memory_dropout_prob=0.2 \
    --batch_size=1 \
    --num_workers=0 \
    --steps="${steps}" \
    --log_freq=1 \
    --eval_freq=0 \
    --save_checkpoint="${save_checkpoint}" \
    --save_freq="${steps}" \
    --output_dir="${output_dir}" \
    --job_name="memory_m8_${policy_type}" \
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
    "${advantage_args[@]}" \
    "$@"
}

gpu_validation() {
  mkdir -p "${OUTPUT_ROOT}"
  local pi0_output="${OUTPUT_ROOT}/pi0_memory"
  local pi05_output="${OUTPUT_ROOT}/pi05_memory"
  train_policy pi0 "${PI0_BASE}" "${pi0_output}" "${DATASET_ROOT}" false \
    --policy.image_augmentation.enable=false
  train_policy pi05 "${PI05_BASE}" "${pi05_output}" "${DATASET_ROOT}" false

  checkpoint_validation
  advantage_validation
}

checkpoint_validation() {
  local pi0_checkpoint="${OUTPUT_ROOT}/pi0_memory/checkpoints/last/pretrained_model"
  local pi05_checkpoint="${OUTPUT_ROOT}/pi05_memory/checkpoints/last/pretrained_model"
  test -f "${pi0_checkpoint}/model.safetensors"
  test -f "${pi05_checkpoint}/model.safetensors"
  run_python plans/memory_pipeline/m8_checkpoint_rtc_smoke.py \
    --checkpoint="${pi0_checkpoint}" \
    --dataset-root="${DATASET_ROOT}" \
    --dataset-repo-id="${DATASET_REPO_ID}" \
    --expected-policy=pi0
  run_python plans/memory_pipeline/m8_checkpoint_rtc_smoke.py \
    --checkpoint="${pi05_checkpoint}" \
    --dataset-root="${DATASET_ROOT}" \
    --dataset-repo-id="${DATASET_REPO_ID}" \
    --expected-policy=pi05
  run_python plans/memory_pipeline/m8_checkpoint_rtc_smoke.py \
    --checkpoint="${PI0_BASE}" \
    --dataset-root="${DATASET_ROOT}" \
    --dataset-repo-id="${DATASET_REPO_ID}" \
    --expected-policy=pi0 \
    --memory-mode=off
  run_python plans/memory_pipeline/m8_checkpoint_rtc_smoke.py \
    --checkpoint="${PI05_BASE}" \
    --dataset-root="${DATASET_ROOT}" \
    --dataset-repo-id="${DATASET_REPO_ID}" \
    --expected-policy=pi05 \
    --memory-mode=off
}

advantage_validation() {
  local pi0_checkpoint="${OUTPUT_ROOT}/pi0_memory/checkpoints/last/pretrained_model"
  local pi05_checkpoint="${OUTPUT_ROOT}/pi05_memory/checkpoints/last/pretrained_model"
  local fixture_parent
  fixture_parent="$(mktemp -d /tmp/lerobot-m8-advantage.XXXXXX)"
  local fixture_root="${fixture_parent}/nero_egg_subtask_advantage"
  run_python plans/memory_pipeline/m8_make_advantage_fixture.py \
    --source="${DATASET_ROOT}" \
    --destination="${fixture_root}"
  train_policy pi0 "${pi0_checkpoint}" "${OUTPUT_ROOT}/pi0_memory_advantage" "${fixture_root}" true \
    --policy.image_augmentation.enable=false
  train_policy pi05 "${pi05_checkpoint}" "${OUTPUT_ROOT}/pi05_memory_advantage" "${fixture_root}" true
  echo "Temporary advantage fixture retained for audit: ${fixture_root}"
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

git diff --check
