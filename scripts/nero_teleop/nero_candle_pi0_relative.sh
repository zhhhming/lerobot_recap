#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATASTORE_ROOT="${DATASTORE_ROOT:-/datastore01/hongming}"
DISABLE_PROXY="${DISABLE_PROXY:-1}"
if [[ "${DISABLE_PROXY}" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

export HF_HOME="${HF_HOME:-${DATASTORE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${DATASTORE_ROOT}/lerobot}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_REPO_ID="${DATASET_REPO_ID:-ming326/nero_candle_3}"
DATASET_SLUG="${DATASET_REPO_ID##*/}"
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_HOME}/${DATASET_REPO_ID}}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
RELATIVE_EXCLUDE_JOINTS="${RELATIVE_EXCLUDE_JOINTS:-['gripper']}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HF_SNAPSHOT_MAX_WORKERS="${HF_SNAPSHOT_MAX_WORKERS:-1}"
POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-lerobot/pi0_base}"
TOKENIZER_NAME="${TOKENIZER_NAME:-google/paligemma-3b-pt-224}"
NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-50}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-pi0_${DATASET_SLUG}_relative_bs${GLOBAL_BATCH_SIZE}_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASTORE_ROOT}/lerobot_outputs/${JOB_NAME}}"
POLICY_COMPILE="${POLICY_COMPILE:-true}"
POLICY_DTYPE="${POLICY_DTYPE:-float32}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
PREDICT_SUBTASK="${PREDICT_SUBTASK:-false}"
SUBTASK_MAX_TOKENS="${SUBTASK_MAX_TOKENS:-48}"
SUBTASK_CE_LOSS_WEIGHT="${SUBTASK_CE_LOSS_WEIGHT:-0.25}"
SUBTASK_DROPOUT_PROB="${SUBTASK_DROPOUT_PROB:-0.2}"
SUBTASK_GENERATE_AT_INFERENCE="${SUBTASK_GENERATE_AT_INFERENCE:-true}"
SUBTASK_MAX_DECODE_TOKENS="${SUBTASK_MAX_DECODE_TOKENS:-48}"
SUBTASK_DECODE_TEMPERATURE="${SUBTASK_DECODE_TEMPERATURE:-0.0}"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  env             Print the resolved environment and dataset paths.
  check           Check that the local Python environment can import LeRobot.
  download        Download/materialize ${DATASET_REPO_ID} into ${DATASET_ROOT}.
  download-policy Download/cache ${POLICY_PRETRAINED_PATH}.
  download-tokenizer Download/cache ${TOKENIZER_NAME}.
  download-all    Run download, download-policy, and download-tokenizer.
  info            Show LeRobot dataset metadata/features.
  stats           Recompute stats with relative actions for pi0.
  verify-stats    Print the action stats and action feature names from meta/.
  train-command   Print the matching lerobot-train command template.
  train           Train pi0 with Accelerate on ${NUM_GPUS} GPUs.

Environment overrides:
  DATASTORE_ROOT, HF_HOME, HF_LEROBOT_HOME, DATASET_REPO_ID, DATASET_ROOT,
  CHUNK_SIZE, RELATIVE_EXCLUDE_JOINTS, NUM_WORKERS, HF_SNAPSHOT_MAX_WORKERS, DISABLE_PROXY,
  POLICY_PRETRAINED_PATH, NUM_GPUS, GLOBAL_BATCH_SIZE, PER_DEVICE_BATCH_SIZE,
  STEPS, SAVE_FREQ, LOG_FREQ, RUN_ID, JOB_NAME, OUTPUT_DIR, POLICY_COMPILE,
  POLICY_DTYPE, MIXED_PRECISION, GRADIENT_CHECKPOINTING, WANDB_ENABLE,
  PREDICT_SUBTASK, SUBTASK_MAX_TOKENS, SUBTASK_CE_LOSS_WEIGHT,
  SUBTASK_DROPOUT_PROB, SUBTASK_GENERATE_AT_INFERENCE,
  SUBTASK_MAX_DECODE_TOKENS, SUBTASK_DECODE_TEMPERATURE
EOF
}

print_env() {
    cat <<EOF
REPO_ROOT=${REPO_ROOT}
DISABLE_PROXY=${DISABLE_PROXY}
HF_HOME=${HF_HOME}
HF_HUB_CACHE=${HF_HUB_CACHE}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME}
DATASET_REPO_ID=${DATASET_REPO_ID}
DATASET_SLUG=${DATASET_SLUG}
DATASET_ROOT=${DATASET_ROOT}
CHUNK_SIZE=${CHUNK_SIZE}
RELATIVE_EXCLUDE_JOINTS=${RELATIVE_EXCLUDE_JOINTS}
NUM_WORKERS=${NUM_WORKERS}
HF_SNAPSHOT_MAX_WORKERS=${HF_SNAPSHOT_MAX_WORKERS}
POLICY_PRETRAINED_PATH=${POLICY_PRETRAINED_PATH}
TOKENIZER_NAME=${TOKENIZER_NAME}
NUM_GPUS=${NUM_GPUS}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-auto}
STEPS=${STEPS}
SAVE_FREQ=${SAVE_FREQ}
LOG_FREQ=${LOG_FREQ}
RUN_ID=${RUN_ID}
JOB_NAME=${JOB_NAME}
OUTPUT_DIR=${OUTPUT_DIR}
POLICY_COMPILE=${POLICY_COMPILE}
POLICY_DTYPE=${POLICY_DTYPE}
MIXED_PRECISION=${MIXED_PRECISION}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}
WANDB_ENABLE=${WANDB_ENABLE}
PREDICT_SUBTASK=${PREDICT_SUBTASK}
SUBTASK_MAX_TOKENS=${SUBTASK_MAX_TOKENS}
SUBTASK_CE_LOSS_WEIGHT=${SUBTASK_CE_LOSS_WEIGHT}
SUBTASK_DROPOUT_PROB=${SUBTASK_DROPOUT_PROB}
SUBTASK_GENERATE_AT_INFERENCE=${SUBTASK_GENERATE_AT_INFERENCE}
SUBTASK_MAX_DECODE_TOKENS=${SUBTASK_MAX_DECODE_TOKENS}
SUBTASK_DECODE_TEMPERATURE=${SUBTASK_DECODE_TEMPERATURE}
EOF
}

check_imports() {
    python -c "import lerobot, huggingface_hub, draccus, transformers; from transformers.models.auto import CONFIG_MAPPING; assert 'paligemma' in CONFIG_MAPPING and 'gemma' in CONFIG_MAPPING; print('imports ok')"
}

download_dataset() {
    mkdir -p "$(dirname "${DATASET_ROOT}")"
    python -c "import sys; from huggingface_hub import snapshot_download; from lerobot.datasets.lerobot_dataset import LeRobotDataset; repo_id, root, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]); snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=root, max_workers=workers); ds=LeRobotDataset(repo_id, root=root); print(f'downloaded/materialized {repo_id} at {root} ({len(ds)} frames, {ds.meta.total_episodes} episodes)')" \
        "${DATASET_REPO_ID}" "${DATASET_ROOT}" "${HF_SNAPSHOT_MAX_WORKERS}"
}

download_policy() {
    python -c "import sys; from huggingface_hub import snapshot_download; path=snapshot_download(repo_id=sys.argv[1], repo_type='model'); print(f'downloaded/cached {sys.argv[1]} at {path}')" \
        "${POLICY_PRETRAINED_PATH}"
}

download_tokenizer() {
    python -c "import sys; from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained(sys.argv[1]); print(f'downloaded/cached tokenizer {sys.argv[1]} as {tok.__class__.__name__}')" \
        "${TOKENIZER_NAME}"
}

per_device_batch_size() {
    if [[ -n "${PER_DEVICE_BATCH_SIZE}" ]]; then
        echo "${PER_DEVICE_BATCH_SIZE}"
        return
    fi
    if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
        echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}; set PER_DEVICE_BATCH_SIZE explicitly to override." >&2
        exit 2
    fi
    echo $(( GLOBAL_BATCH_SIZE / NUM_GPUS ))
}

show_info() {
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id "${DATASET_REPO_ID}" \
        --root "${DATASET_ROOT}" \
        --operation.type info \
        --operation.show_features true
}

recompute_relative_stats() {
    python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id "${DATASET_REPO_ID}" \
        --root "${DATASET_ROOT}" \
        --operation.type recompute_stats \
        --operation.relative_action true \
        --operation.chunk_size "${CHUNK_SIZE}" \
        --operation.relative_exclude_joints "${RELATIVE_EXCLUDE_JOINTS}" \
        --operation.num_workers "${NUM_WORKERS}"
}

verify_stats() {
    python -c "import json, sys; from pathlib import Path; root=Path(sys.argv[1]); stats=json.loads((root/'meta/stats.json').read_text()); info=json.loads((root/'meta/info.json').read_text()); action=stats.get('action'); names=info.get('features', {}).get('action', {}).get('names'); print('dataset_root=', root); print('action_names=', names); print('action_stats_keys=', sorted(action) if action else None); print('action_mean_dim=', len(action.get('mean', [])) if action else None); print('action_std_dim=', len(action.get('std', [])) if action else None); print('action_mean=', action.get('mean') if action else None); print('action_std=', action.get('std') if action else None)" \
        "${DATASET_ROOT}"
}

print_train_command() {
    local per_device_bs
    per_device_bs="$(per_device_batch_size)"
    cat <<EOF
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME}"
export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

accelerate launch \\
  --multi_gpu \\
  --num_processes=${NUM_GPUS} \\
  --num_machines=1 \\
  --mixed_precision=${MIXED_PRECISION} \\
  --dynamo_backend=no \\
  -m lerobot.scripts.lerobot_train \\
  --dataset.repo_id=${DATASET_REPO_ID} \\
  --dataset.root=${DATASET_ROOT} \\
  --policy.type=pi0 \\
  --policy.pretrained_path=${POLICY_PRETRAINED_PATH} \\
  --policy.use_relative_actions=true \\
  --policy.relative_exclude_joints='["gripper"]' \\
  --policy.compile_model=${POLICY_COMPILE} \\
  --policy.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \\
  --policy.dtype=${POLICY_DTYPE} \\
  --policy.freeze_vision_encoder=false \\
  --policy.train_expert_only=false \\
  --policy.predict_subtask=${PREDICT_SUBTASK} \\
  --policy.subtask_max_tokens=${SUBTASK_MAX_TOKENS} \\
  --policy.subtask_ce_loss_weight=${SUBTASK_CE_LOSS_WEIGHT} \\
  --policy.subtask_dropout_prob=${SUBTASK_DROPOUT_PROB} \\
  --policy.subtask_generate_at_inference=${SUBTASK_GENERATE_AT_INFERENCE} \\
  --policy.subtask_max_decode_tokens=${SUBTASK_MAX_DECODE_TOKENS} \\
  --policy.subtask_decode_temperature=${SUBTASK_DECODE_TEMPERATURE} \\
  --policy.push_to_hub=false \\
  --policy.device=cuda \\
  --batch_size=${per_device_bs} \\
  --num_workers=${NUM_WORKERS} \\
  --steps=${STEPS} \\
  --save_freq=${SAVE_FREQ} \\
  --log_freq=${LOG_FREQ} \\
  --job_name=${JOB_NAME} \\
  --output_dir=${OUTPUT_DIR} \\
  --wandb.enable=${WANDB_ENABLE}
EOF
}

train_pi0() {
    mkdir -p "$(dirname "${OUTPUT_DIR}")"
    local per_device_bs
    per_device_bs="$(per_device_batch_size)"
    download_tokenizer
    accelerate launch \
        --multi_gpu \
        --num_processes="${NUM_GPUS}" \
        --num_machines=1 \
        --mixed_precision="${MIXED_PRECISION}" \
        --dynamo_backend=no \
        -m lerobot.scripts.lerobot_train \
        --dataset.repo_id="${DATASET_REPO_ID}" \
        --dataset.root="${DATASET_ROOT}" \
        --policy.type=pi0 \
        --policy.pretrained_path="${POLICY_PRETRAINED_PATH}" \
        --policy.use_relative_actions=true \
        --policy.relative_exclude_joints='["gripper"]' \
        --policy.compile_model="${POLICY_COMPILE}" \
        --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
        --policy.dtype="${POLICY_DTYPE}" \
        --policy.freeze_vision_encoder=false \
        --policy.train_expert_only=false \
        --policy.predict_subtask="${PREDICT_SUBTASK}" \
        --policy.subtask_max_tokens="${SUBTASK_MAX_TOKENS}" \
        --policy.subtask_ce_loss_weight="${SUBTASK_CE_LOSS_WEIGHT}" \
        --policy.subtask_dropout_prob="${SUBTASK_DROPOUT_PROB}" \
        --policy.subtask_generate_at_inference="${SUBTASK_GENERATE_AT_INFERENCE}" \
        --policy.subtask_max_decode_tokens="${SUBTASK_MAX_DECODE_TOKENS}" \
        --policy.subtask_decode_temperature="${SUBTASK_DECODE_TEMPERATURE}" \
        --policy.push_to_hub=false \
        --policy.device=cuda \
        --batch_size="${per_device_bs}" \
        --num_workers="${NUM_WORKERS}" \
        --steps="${STEPS}" \
        --save_freq="${SAVE_FREQ}" \
        --log_freq="${LOG_FREQ}" \
        --job_name="${JOB_NAME}" \
        --output_dir="${OUTPUT_DIR}" \
        --wandb.enable="${WANDB_ENABLE}"
}

command="${1:-}"
case "${command}" in
    env) print_env ;;
    check) print_env; check_imports ;;
    download) print_env; check_imports; download_dataset ;;
    download-policy) print_env; check_imports; download_policy ;;
    download-tokenizer) print_env; check_imports; download_tokenizer ;;
    download-all) print_env; check_imports; download_dataset; download_policy; download_tokenizer ;;
    info) print_env; check_imports; show_info ;;
    stats) print_env; check_imports; recompute_relative_stats ;;
    verify-stats) print_env; verify_stats ;;
    train-command) print_train_command ;;
    train) print_env; check_imports; train_pi0 ;;
    -h|--help|help|"") usage ;;
    *) usage; exit 2 ;;
esac
