# PI0 / PI0.5 Subtask Dataset Annotation and Training Workflow

Date: 2026-07-03

This runbook covers the next step after M1-M5: annotate a raw run, build a LeRobot dataset with `subtask` and `subtask_progress`, upload it to Hugging Face, sync the current code to a remote server, recompute relative-action stats there, and start PI0 subtask training.

## 0. Current Local Raw Dataset

Candidate raw run:

```bash
/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3
```

It contains 70 episodes and is about 55G.

A local backup was started at:

```bash
/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3_backup_20260703_200123
```

## 1. Annotate Subtasks

Run this on the local machine that has access to the raw images:

```bash
conda run -n lerobot-main python -m lerobot.scripts.lerobot_annotate_subtask \
  --root /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3 \
  --host 127.0.0.1 \
  --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Recommended label style:

- Use short English subtask names, e.g. `reach match`, `grasp match`, `strike`, `withdraw`.
- Keep labels consistent across episodes.
- Mark every meaningful task segment; unannotated frames become empty subtask with progress `0.0`.
- Click Export after annotation. Export writes each episode's `extras.parquet` with:
  - `subtask`: string label.
  - `subtask_progress`: float progress inside each contiguous label segment, from near `0.0` to `1.0`.

Quick local check after export:

```bash
conda run -n lerobot-main python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
root = Path("/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3")
paths = sorted(root.glob("ep_*/extras.parquet"))
print("extras files:", len(paths))
table = pq.read_table(paths[0])
print(table.schema)
print(table.slice(0, 5).to_pydict())
PY
```

## 2. Build LeRobot Dataset

Choose a new dataset repo id, for example:

```bash
export DATASET_REPO_ID=ming326/strike_match_3_subtask
export RAW_ROOT=/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3
export DATASET_ROOT=/home/zenbot-robot/.cache/huggingface/lerobot/${DATASET_REPO_ID}
```

Build locally:

```bash
conda run -n lerobot-main python -m lerobot.scripts.lerobot_build_dataset \
  --runs "${RAW_ROOT}" \
  --output_repo_id "${DATASET_REPO_ID}" \
  --output_root "${DATASET_ROOT}" \
  --video true \
  --vcodec libsvtav1 \
  --push_to_hub false \
  --force true
```

Inspect dataset features:

```bash
conda run -n lerobot-main python -m lerobot.scripts.lerobot_edit_dataset \
  --repo_id "${DATASET_REPO_ID}" \
  --root "${DATASET_ROOT}" \
  --operation.type info \
  --operation.show_features true
```

Confirm `subtask` and `subtask_progress` are present.

## 3. Upload Dataset to Hugging Face

Login first if needed:

```bash
conda run -n lerobot-main huggingface-cli login
```

With proxy:

```bash
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080
```

Dry run:

```bash
conda run -n lerobot-main python -m lerobot.scripts.lerobot_push_dataset \
  --repo_id "${DATASET_REPO_ID}" \
  --root "${DATASET_ROOT}" \
  --private \
  --dry-run
```

Upload:

```bash
conda run -n lerobot-main python -m lerobot.scripts.lerobot_push_dataset \
  --repo_id "${DATASET_REPO_ID}" \
  --root "${DATASET_ROOT}" \
  --private \
  --proxy http://127.0.0.1:1080 \
  --upload-large-folder \
  --num-workers 1
```

If upload is unstable, retry the same command. The large-folder path keeps upload state under `.cache/huggingface`.

## 4. Sync Current Code to Remote Server

The remote server must use the code containing M1-M5 subtask changes.

Recommended options:

1. Commit/push this repo branch, then pull it on the server.
2. If not committing yet, use `rsync` from local to remote, excluding caches and outputs.

Example:

```bash
rsync -av \
  --exclude .git \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude outputs \
  /home/zenbot-robot/repos/lerobot/ \
  USER@REMOTE:/path/to/lerobot/
```

On the remote server:

```bash
cd /path/to/lerobot
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
conda run -n lerobot-main python -m pytest \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py -q
```

## 5. Remote Download and Recompute Stats

The existing helper script was updated to include subtask AR training parameters:

```bash
scripts/nero_teleop/nero_candle_pi0_relative.sh
```

On the remote server, set paths and dataset id:

```bash
cd /path/to/lerobot
export DATASTORE_ROOT=/datastore01/hongming
export DATASET_REPO_ID=ming326/strike_match_3_subtask
export DATASET_ROOT=/datastore01/hongming/lerobot/${DATASET_REPO_ID}
export DISABLE_PROXY=0
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080
```

Download dataset, base policy, and tokenizer:

```bash
bash scripts/nero_teleop/nero_candle_pi0_relative.sh download-all
```

Check features:

```bash
bash scripts/nero_teleop/nero_candle_pi0_relative.sh info
```

Recompute relative-action stats for PI0:

```bash
export CHUNK_SIZE=50
export RELATIVE_EXCLUDE_JOINTS="['gripper']"
export NUM_WORKERS=8
bash scripts/nero_teleop/nero_candle_pi0_relative.sh stats
bash scripts/nero_teleop/nero_candle_pi0_relative.sh verify-stats
```

Re-upload after stats only if you want the Hub copy to contain the recomputed stats:

```bash
conda run -n lerobot-main python -m lerobot.scripts.lerobot_push_dataset \
  --repo_id "${DATASET_REPO_ID}" \
  --root "${DATASET_ROOT}" \
  --private \
  --upload-large-folder \
  --num-workers 1
```

## 6. Start Training

Recommended first subtask PI0 run:

```bash
export DATASTORE_ROOT=/datastore01/hongming
export DATASET_REPO_ID=ming326/strike_match_3_subtask
export DATASET_ROOT=/datastore01/hongming/lerobot/${DATASET_REPO_ID}

export PREDICT_SUBTASK=true
export SUBTASK_MAX_TOKENS=48
export SUBTASK_CE_LOSS_WEIGHT=0.25
export SUBTASK_DROPOUT_PROB=0.2
export SUBTASK_GENERATE_AT_INFERENCE=true
export SUBTASK_MAX_DECODE_TOKENS=48
export SUBTASK_DECODE_TEMPERATURE=0.0

export POLICY_PRETRAINED_PATH=lerobot/pi0_base
export POLICY_DTYPE=float32
export MIXED_PRECISION=bf16
export GRADIENT_CHECKPOINTING=true
export POLICY_COMPILE=false

export NUM_GPUS=8
export GLOBAL_BATCH_SIZE=128
export STEPS=20000
export SAVE_FREQ=1000
export LOG_FREQ=50
export WANDB_ENABLE=true
export JOB_NAME=pi0_strike_match_3_subtask_relative_ar

bash scripts/nero_teleop/nero_candle_pi0_relative.sh train-command
bash scripts/nero_teleop/nero_candle_pi0_relative.sh train
```

Notes:

- Start with `POLICY_COMPILE=false` for the first subtask run. After a clean smoke run, you can try `POLICY_COMPILE=true`.
- `train_expert_only` must stay false when `PREDICT_SUBTASK=true`, because the VLM path needs to learn the subtask CE objective.
- `freeze_vision_encoder=false` is the current script default. If memory is tight, consider `--policy.freeze_vision_encoder=true` later, but keep the language model trainable.

## 7. Useful Subtask Parameters

Training/inference behavior:

- `PREDICT_SUBTASK=true`: enables subtask AR training and inference support.
- `SUBTASK_MAX_TOKENS=48`: max token length of teacher-forced subtask text during training.
- `SUBTASK_CE_LOSS_WEIGHT=0.25`: CE loss weight. If flow matching gets worse, try `0.1`; if generated subtask is poor, try `0.5`.
- `SUBTASK_DROPOUT_PROB=0.2`: probability that state/action suffix cannot attend to subtask text during training. Higher means more robust no-AR inference, lower means stronger reliance on generated subtask.
- `SUBTASK_GENERATE_AT_INFERENCE=true`: default deploy mode. Set false to skip AR at inference while using the dropout-trained branch.
- `SUBTASK_MAX_DECODE_TOKENS=48`: max AR decoding steps at inference.
- `SUBTASK_DECODE_TEMPERATURE=0.0`: greedy decode. Keep `0.0` for robotics deployment.

Training scale:

- `GLOBAL_BATCH_SIZE`: global batch size across GPUs.
- `NUM_GPUS`: number of accelerate processes.
- `PER_DEVICE_BATCH_SIZE`: override if global batch is not divisible by GPU count.
- `STEPS`: total train steps.
- `SAVE_FREQ`: checkpoint interval.
- `LOG_FREQ`: logging interval.

Relative action stats:

- `CHUNK_SIZE=50`: must match policy chunk size.
- `RELATIVE_EXCLUDE_JOINTS="['gripper']"`: keeps gripper absolute, makes other action dims relative.

## 8. Minimal Smoke Before Long Training

Before a 20k-step run, do:

```bash
export STEPS=20
export SAVE_FREQ=20
export LOG_FREQ=1
export WANDB_ENABLE=false
export JOB_NAME=pi0_strike_match_3_subtask_smoke
bash scripts/nero_teleop/nero_candle_pi0_relative.sh train
```

Expected:

- Training starts without missing `subtask`/`subtask_progress`.
- Logs include `fm_loss` and `ce_loss`.
- `ce_loss` is nonzero when annotated frames are sampled.

Then restore the full training settings.
