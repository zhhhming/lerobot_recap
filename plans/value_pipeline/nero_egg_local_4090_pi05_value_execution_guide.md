# Nero Egg：本地 4090 + PI0.5 Value / Advantage 顺序执行指南

日期：2026-07-24

适用代码：

```text
/home/zenbot-robot/repos/lerobot
```

适用 raw dataset：

```text
/data1/lerobot/raw/ming326/nero_egg_jpeg95
```

本指南是一份按顺序执行的 runbook，覆盖：

1. 环境和数据只读检查；
2. 下载完整 `lerobot/pi05_base`；
3. 创建互不重叠的 train / validation / old-holdout raw view；
4. 生成 subtask value target；
5. 在单张 RTX 4090 上进行 value smoke、测速和正式训练；
6. validation、old holdout 和 train 的 value inference；
7. value 曲线可视化；
8. 正式 model-predicted advantage；
9. advantage label 检查与导出；
10. group-relative advantage weight；
11. 使用正确 task prompt 构建 PI0.5 训练所需的 LeRobotDataset；
12. 最终 artifact 和 provenance 检查。

本文档不会修改 raw dataset 中已有的错误 task 文本。Value model 明确不使用 task text；构建最终
LeRobotDataset 时通过 `--task_override` 写入正确指令：

```text
Prepare and cook scrambled eggs by adding oil and salt, pouring and stirring the beaten eggs in the pan, serving them in a bowl.
```

## 0. 已固定的实验设计

### 0.1 Value 模式和 backbone

本轮固定：

```text
value mode:             subtask
value backbone:         PI0.5
pretrained source:      lerobot/pi05_base
precision:              bfloat16
image cameras:          left_wrist, right_wrist, third_person
state input:            enabled, independent normalized MLP branch
num bins:               128
VLM layers:             3
vision encoder:         frozen
backbone:               frozen
unfrozen VLM layers:    0
elapsed auxiliary head: disabled
```

PI0 和 PI0.5 在当前 value 实现中使用相同的截断视觉/VLM结构，主要区别是预训练权重来源。下游
VLA 也计划使用 PI0.5，因此本轮选择 PI0.5。

当前 value model 不把 `observation.state` 或 task text 编码进 prompt。它完全不读取 task text，实际
结构为：

```text
三路 224×224 RGB
  -> PI0.5 vision tower + multimodal projector
  -> 截取 PaliGemma language model 的前 3 层（layer 0、1、2）
  -> 三路 visual tokens 拼接并 mean pool
  -> visual feature

16-D observation.state
  -> 仅使用 train split 计算 mean/std
  -> normalization
  -> LayerNorm + Linear(16, 256) + GELU
  -> state feature

visual feature + state feature
  -> concat
  -> LayerNorm + Linear(..., 512) + GELU + Dropout
  -> fusion feature
  -> 12 类 subtask classifier
  -> 12 × 128-bin distributional remaining-value head
```

本轮固定完全冻结 vision tower、multimodal projector 和 3 层 VLM，只训练：

```text
state encoder
fusion MLP
subtask classifier
subtask distributional remaining-value head
```

虽然代码支持：

```text
--freeze_backbone true
--num_unfrozen_backbone_layers 1
```

以只解冻截断 backbone 的最后一层（原始 layer 2），第一轮不启用。原因是当前 trainer 没有 value
backbone gradient checkpointing、分层 learning rate、best checkpoint 或正式 resume；在 4090 上解冻
一个 Gemma layer 会额外保存其 activation、gradient 和 AdamW state，显著增加显存、耗时和中断风险。
78 条 episode、364,989 帧对于先训练上述非线性 fusion/head 已经足够。

只有第一轮同时出现以下现象，才另建 v3 实验解冻 1 层：

```text
train loss 和 val loss 都长期不下降
train subtask accuracy 也明显偏低
排除数据、target、scale 和 learning rate 问题
```

如果 train 指标很好而 val 差，是过拟合或 domain shift，不应通过解冻更多 backbone 解决。

### 0.2 Episode 划分

Raw dataset 实际有 122 条 episode：

```text
ep_000000 ... ep_000122
```

其中 `ep_000051` 不存在。

旧数据中按时间均匀选 20 条训练：

```text
0 3 6 9 13 16 19 22 25 28 32 35 38 41 44 47 52 55 58 61
```

新数据中固定 3 条 validation：

```text
72 92 112
```

剩余 `62..122` 新数据全部训练。其余旧数据作为独立 old holdout，不参与训练，也不在每轮 validation
中运行。

最终统计：

| Split | Episodes | Frames | 30 FPS 时长 |
|---|---:|---:|---:|
| Value train | 78 | 364,989 | 3.380 h |
| Value validation | 3 | 12,798 | 0.119 h |
| Old holdout | 41 | 235,336 | 2.179 h |

三个 split 互不重叠，继续用于 value model 的训练和验证。正式 value inference、advantage、label 和
weight 在 `ALL_ROOT` 上覆盖全部 122 条 episode；按数据所有者最终决定，VLA LeRobotDataset 也从
`ALL_ROOT` 构建全部 122 条 episode。

### 0.3 成功轨迹契约

本轮所有 episode 均由数据所有者确认为成功示范。当前 `info.json` 没有 `success/outcome` 字段，因此
target metadata 会记录：

```text
success_validation = declared_no_outcome_field
```

这是当前代码的正常行为，但它依赖上述人工确认。未来失败、超时或中止 episode 不能直接加入本 pipeline。

## 1. 打开终端并固定所有路径

后续命令默认在同一个 Bash 终端依次执行。先运行：

```bash
cd /home/zenbot-robot/repos/lerobot

export PY=/home/zenbot-robot/.conda/envs/lerobot-main/bin/python
export HF=/home/zenbot-robot/.conda/envs/lerobot-main/bin/hf
export PYTHONPATH=$PWD/src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

export RAW=/data1/lerobot/raw/ming326/nero_egg_jpeg95
export VIEW_BASE=/data1/lerobot/raw_views/ming326
export TRAIN_ROOT=$VIEW_BASE/nero_egg_value_train_v2
export VAL_ROOT=$VIEW_BASE/nero_egg_value_val_ep72_92_112_v2
export OLD_TEST_ROOT=$VIEW_BASE/nero_egg_value_old_holdout_v2
export VAL_E1_ROOT=$VIEW_BASE/nero_egg_value_val_ep72_92_112_epoch1_v2
export VAL_E2_ROOT=$VIEW_BASE/nero_egg_value_val_ep72_92_112_epoch2_v2
export ALL_ROOT=$VIEW_BASE/nero_egg_value_all_ep000_122_v2

export PI05_INIT=/data1/lerobot/models/pi05_base
export VALUE_OUT=/data1/lerobot/outputs/value_nero_egg_subtask_pi05_v2
export VALUE_SMOKE=/data1/lerobot/outputs/value_nero_egg_subtask_pi05_v2_smoke
export VALUE_SPEED=/data1/lerobot/outputs/value_nero_egg_subtask_pi05_v2_speed

export DATASET_REPO_ID=ming326/nero_egg_adv_pi05_v2
export DATASET_ROOT=/data1/lerobot/datasets/ming326/nero_egg_adv_pi05_v2

export LOG_DIR=/data1/lerobot/logs/nero_egg_value_pi05_v2
export TASK='Prepare and cook scrambled eggs by adding oil and salt, pouring and stirring the beaten eggs in the pan, serving them in a bowl.'

mkdir -p \
  "$VIEW_BASE" \
  "$PI05_INIT" \
  "$(dirname "$VALUE_OUT")" \
  "$(dirname "$DATASET_ROOT")" \
  "$LOG_DIR"
```

确认变量没有展开为空：

```bash
printf '%s\n' \
  "RAW=$RAW" \
  "TRAIN_ROOT=$TRAIN_ROOT" \
  "VAL_ROOT=$VAL_ROOT" \
  "VAL_E1_ROOT=$VAL_E1_ROOT" \
  "VAL_E2_ROOT=$VAL_E2_ROOT" \
  "ALL_ROOT=$ALL_ROOT" \
  "OLD_TEST_ROOT=$OLD_TEST_ROOT" \
  "PI05_INIT=$PI05_INIT" \
  "VALUE_OUT=$VALUE_OUT" \
  "DATASET_ROOT=$DATASET_ROOT" \
  "TASK=$TASK"
```

检查 Python 和关键依赖：

```bash
"$PY" -c '
import torch, pyarrow, transformers, huggingface_hub
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("pyarrow:", pyarrow.__version__)
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
'
```

检查 4090：

```bash
nvidia-smi
```

放行条件：

```text
torch.cuda.is_available() == True
nvidia-smi 能看到 RTX 4090
GPU 没有被其他进程占用大量显存
```

如果 CUDA 不可用，在这里停止，不要启动训练。

保存代码状态用于实验追溯：

```bash
git rev-parse HEAD | tee "$LOG_DIR/git_head.txt"
git status --short | tee "$LOG_DIR/git_status.txt"
```

仓库当前可能存在未提交修改；本轮实验的 `git_head.txt` 和 `git_status.txt` 必须保留。

## 2. 只读核验 raw dataset

运行：

```bash
"$PY" - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/data1/lerobot/raw/ming326/nero_egg_jpeg95")
episodes = {
    int(path.name.split("_")[1]): path
    for path in root.glob("ep_*")
    if path.is_dir()
}
ids = sorted(episodes)
missing = [index for index in range(ids[0], ids[-1] + 1) if index not in episodes]
assert len(ids) == 122, len(ids)
assert ids[0] == 0 and ids[-1] == 122
assert missing == [51], missing

expected_order = (
    "Pick up the oil bottle and pour in the oil.",
    "Pick up the salt shaker and add some salt.",
    "Bring the bowl of beaten eggs to the pan.",
    "Pick up the fork.",
    "Stir the beaten eggs.",
    "Pour in the beaten eggs and put the bowl back.",
    "Place the serving bowl in front of the pan.",
    "Pick up the pan and the spatula.",
    "Start frying the eggs.",
    "Pour the eggs into the bowl.",
    "Put down the bowl and the spatula.",
    "Place the bowl of eggs on the left, then return to the starting position.",
)

total_frames = 0
reference_schema = None
for index in ids:
    episode = episodes[index]
    frame_rows = pq.read_metadata(episode / "frames.parquet").num_rows
    extras = pq.read_table(
        episode / "extras.parquet",
        columns=["subtask", "subtask_progress"],
    )
    assert extras.num_rows == frame_rows, (index, frame_rows, extras.num_rows)
    assert extras["subtask"].null_count == 0, index
    assert extras["subtask_progress"].null_count == 0, index

    order = tuple(dict.fromkeys(extras["subtask"].to_pylist()))
    assert order == expected_order, (index, order)

    schema = pq.read_schema(episode / "extras.parquet")
    if reference_schema is None:
        reference_schema = schema
    assert schema == reference_schema, index
    total_frames += frame_rows

assert total_frames == 613123, total_frames
print("raw verification passed")
print("episodes:", len(ids))
print("frames:", total_frames)
print("missing episode:", missing)
print("subtasks:", len(expected_order))
PY
```

预期最后输出：

```text
raw verification passed
episodes: 122
frames: 613123
missing episode: [51]
subtasks: 12
```

任何 assertion 失败都应停止，不要继续生成 target。

## 3. 下载完整 PI0.5 base

先看下载内容，不实际传输：

```bash
"$HF" download lerobot/pi05_base \
  --local-dir "$PI05_INIT" \
  --dry-run
```

正式下载完整 snapshot：

```bash
"$HF" download lerobot/pi05_base \
  --local-dir "$PI05_INIT" \
  2>&1 | tee "$LOG_DIR/pi05_download.log"
```

如果 Hub 要求认证，先执行：

```bash
"$HF" auth login
```

再重跑下载命令。

检查完整模型：

```bash
test -f "$PI05_INIT/model.safetensors"
test -f "$PI05_INIT/config.json"
du -sh "$PI05_INIT"
sha256sum "$PI05_INIT/model.safetensors" | tee "$LOG_DIR/pi05_model_sha256.txt"
```

Value loader 只选择性读取 `model.safetensors` 中所需的 vision/VLM 权重；保留完整 snapshot 是为了后续
PI0.5 VLA 训练复用。

## 4. 创建 train / validation / old-holdout raw view

这些 view 使用 symlink，不复制 172 GB 图片。

创建根目录和 `run_meta.json`：

```bash
mkdir -p "$TRAIN_ROOT" "$VAL_ROOT" "$OLD_TEST_ROOT"

for root in "$TRAIN_ROOT" "$VAL_ROOT" "$OLD_TEST_ROOT"; do
  test -e "$root/run_meta.json" || ln -s "$RAW/run_meta.json" "$root/run_meta.json"
done
```

固定 episode 数组：

```bash
OLD_TRAIN=(0 3 6 9 13 16 19 22 25 28 32 35 38 41 44 47 52 55 58 61)

NEW_TRAIN=(
  $(seq 62 71)
  $(seq 73 91)
  $(seq 93 111)
  $(seq 113 122)
)

VALIDATION=(72 92 112)

OLD_HOLDOUT=(
  1 2 4 5 7 8 10 11 12 14 15 17 18 20 21
  23 24 26 27 29 30 31 33 34 36 37 39 40 42 43
  45 46 48 49 50 53 54 56 57 59 60
)
```

链接训练 episode：

```bash
for i in "${OLD_TRAIN[@]}" "${NEW_TRAIN[@]}"; do
  ep=$(printf 'ep_%06d' "$i")
  test -e "$TRAIN_ROOT/$ep" || ln -s "$RAW/$ep" "$TRAIN_ROOT/$ep"
done
```

链接 validation episode：

```bash
for i in "${VALIDATION[@]}"; do
  ep=$(printf 'ep_%06d' "$i")
  test -e "$VAL_ROOT/$ep" || ln -s "$RAW/$ep" "$VAL_ROOT/$ep"
done
```

链接 old holdout：

```bash
for i in "${OLD_HOLDOUT[@]}"; do
  ep=$(printf 'ep_%06d' "$i")
  test -e "$OLD_TEST_ROOT/$ep" || ln -s "$RAW/$ep" "$OLD_TEST_ROOT/$ep"
done
```

验证 split 数量、交集和帧数：

```bash
"$PY" - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

roots = {
    "train": Path("/data1/lerobot/raw_views/ming326/nero_egg_value_train_v2"),
    "val": Path("/data1/lerobot/raw_views/ming326/nero_egg_value_val_ep72_92_112_v2"),
    "old_holdout": Path("/data1/lerobot/raw_views/ming326/nero_egg_value_old_holdout_v2"),
}
expected = {
    "train": (78, 364989),
    "val": (3, 12798),
    "old_holdout": (41, 235336),
}

ids_by_split = {}
for name, root in roots.items():
    episodes = sorted(path for path in root.glob("ep_*") if path.is_dir())
    ids = {int(path.name.split("_")[1]) for path in episodes}
    frames = sum(pq.read_metadata(path / "frames.parquet").num_rows for path in episodes)
    assert (len(ids), frames) == expected[name], (name, len(ids), frames)
    ids_by_split[name] = ids
    print(name, "episodes=", len(ids), "frames=", frames)

assert not (ids_by_split["train"] & ids_by_split["val"])
assert not (ids_by_split["train"] & ids_by_split["old_holdout"])
assert not (ids_by_split["val"] & ids_by_split["old_holdout"])
assert ids_by_split["val"] == {72, 92, 112}
print("split verification passed")
PY
```

预期：

```text
train episodes= 78 frames= 364989
val episodes= 3 frames= 12798
old_holdout episodes= 41 frames= 235336
split verification passed
```

`v2` 是本轮唯一 canonical view。若之前创建过 v1 view，后续不要混用 v1 metadata。

## 5. 固定 canonical subtask 顺序

从已经核验过的 ep0 提取顺序：

```bash
export ORDER="$("$PY" -c '
import json, sys
import pyarrow.parquet as pq
values = pq.read_table(sys.argv[1], columns=["subtask"])["subtask"].to_pylist()
print(json.dumps(list(dict.fromkeys(values))))
' "$RAW/ep_000000/extras.parquet")"
```

检查：

```bash
printf '%s\n' "$ORDER" | tee "$LOG_DIR/subtask_order.json"

"$PY" -c '
import json, sys
order=json.loads(sys.argv[1])
assert len(order) == 12, len(order)
assert len(set(order)) == 12
print("canonical order passed:", len(order), "subtasks")
' "$ORDER"
```

## 6. 生成 TRAIN_ROOT subtask value target

本轮只训练 subtask value，不生成 global target。

### 6.1 Train target dry-run

```bash
"$PY" -m lerobot.scripts.lerobot_value_prepare_targets \
  --root "$TRAIN_ROOT" \
  --mode subtask \
  --num_bins 128 \
  --subtask_scale p95 \
  --elapsed_aux false \
  --subtask_order_json "$ORDER" \
  --require_all_subtasks true \
  --require_single_segment_per_subtask true \
  --require_success_only true \
  --dry_run \
  2>&1 | tee "$LOG_DIR/train_targets_dry_run.log"
```

放行条件：

```text
没有 unlabeled frame
没有 missing/repeated subtask
没有 subtask order regression
没有 failure/timeout/abort
total_frames = 364989
```

### 6.2 正式写入 train target

```bash
"$PY" -m lerobot.scripts.lerobot_value_prepare_targets \
  --root "$TRAIN_ROOT" \
  --mode subtask \
  --num_bins 128 \
  --subtask_scale p95 \
  --elapsed_aux false \
  --subtask_order_json "$ORDER" \
  --require_all_subtasks true \
  --require_single_segment_per_subtask true \
  --require_success_only true \
  2>&1 | tee "$LOG_DIR/train_targets_write.log"
```

这一步会给 `TRAIN_ROOT` 指向的原始 episode `extras.parquet` 增加 value GT 列，并在 view 根目录生成：

```text
value_function_meta.json
```

提取仅由训练集确定的 scale：

```bash
export SCALES="$("$PY" -c '
import json, sys
metadata=json.load(open(sys.argv[1]))
print(json.dumps(metadata["subtask_scale"]["frames_by_subtask"]))
' "$TRAIN_ROOT/value_function_meta.json")"

printf '%s\n' "$SCALES" | tee "$LOG_DIR/subtask_scales.json"
```

检查 target metadata：

```bash
"$PY" - <<'PY'
import json
from pathlib import Path

path = Path("/data1/lerobot/raw_views/ming326/nero_egg_value_train_v2/value_function_meta.json")
meta = json.loads(path.read_text())
assert meta["value_mode"] == "subtask"
assert meta["num_bins"] == 128
assert meta["all_episodes_successful"] is True
assert meta["success_validation"] == "declared_no_outcome_field"
assert len(meta["subtask_order"]) == 12
assert set(meta["subtask_scale"]["frames_by_subtask"]) == set(meta["subtask_order"])

rates = meta["clip_summary"]["subtask_clip_rate_by_subtask"]
print("max clip rate:", max(rates.values()))
for name, rate in rates.items():
    print(f"{rate:8.4%}  {name}")
assert max(rates.values()) < 0.05, rates
print("train target metadata passed")
PY
```

本轮要求每个 subtask 的 p95 clip rate 小于 5%。如果 assertion 失败，在这里停止。

## 7. 使用 train scale 生成 validation 和 old-holdout target

Validation/test 不能自己计算 p95，必须使用第 6 步的 `SCALES`。

### 7.1 Dry-run

```bash
for root in "$VAL_ROOT" "$OLD_TEST_ROOT"; do
  "$PY" -m lerobot.scripts.lerobot_value_prepare_targets \
    --root "$root" \
    --mode subtask \
    --num_bins 128 \
    --subtask_scale manual \
    --subtask_scale_frames_json "$SCALES" \
    --elapsed_aux false \
    --subtask_order_json "$ORDER" \
    --require_all_subtasks true \
    --require_single_segment_per_subtask true \
    --require_success_only true \
    --dry_run
done 2>&1 | tee "$LOG_DIR/eval_targets_dry_run.log"
```

### 7.2 正式写入

```bash
for root in "$VAL_ROOT" "$OLD_TEST_ROOT"; do
  "$PY" -m lerobot.scripts.lerobot_value_prepare_targets \
    --root "$root" \
    --mode subtask \
    --num_bins 128 \
    --subtask_scale manual \
    --subtask_scale_frames_json "$SCALES" \
    --elapsed_aux false \
    --subtask_order_json "$ORDER" \
    --require_all_subtasks true \
    --require_single_segment_per_subtask true \
    --require_success_only true
done 2>&1 | tee "$LOG_DIR/eval_targets_write.log"
```

验证三个 root 的 scale 完全相同：

```bash
"$PY" - <<'PY'
import json
from pathlib import Path

roots = [
    Path("/data1/lerobot/raw_views/ming326/nero_egg_value_train_v2"),
    Path("/data1/lerobot/raw_views/ming326/nero_egg_value_val_ep72_92_112_v2"),
    Path("/data1/lerobot/raw_views/ming326/nero_egg_value_old_holdout_v2"),
]
metas = [json.loads((root / "value_function_meta.json").read_text()) for root in roots]
scales = [meta["subtask_scale"]["frames_by_subtask"] for meta in metas]
orders = [meta["subtask_order"] for meta in metas]
assert scales[0] == scales[1] == scales[2]
assert orders[0] == orders[1] == orders[2]
assert all(meta["num_bins"] == 128 for meta in metas)
print("cross-root target contract passed")
PY
```

## 8. PI0.5 value 两步 smoke

Smoke 只使用 `TRAIN_ROOT` 且 `val_fraction=0`，避免两步训练后意外完整推理 12,798 个 validation frame。
它验证：

```text
raw JPEG -> DataLoader
PI0.5 checkpoint selective load
三路相机 forward
state normalization
subtask classifier/value heads
backward
optimizer step
checkpoint save
```

运行：

```bash
/usr/bin/time -v \
"$PY" -m lerobot.scripts.lerobot_train_value_function \
  --root "$TRAIN_ROOT" \
  --output_dir "$VALUE_SMOKE" \
  --mode subtask \
  --backbone_type pi05 \
  --pretrained_path "$PI05_INIT" \
  --local_files_only true \
  --precision bfloat16 \
  --image_keys \
    observation.images.left_wrist \
    observation.images.right_wrist \
    observation.images.third_person \
  --num_vlm_layers 3 \
  --num_bins 128 \
  --use_elapsed_aux false \
  --use_state true \
  --freeze_vision_encoder true \
  --freeze_backbone true \
  --num_unfrozen_backbone_layers 0 \
  --val_fraction 0 \
  --epochs 1 \
  --max_steps 2 \
  --batch_size 1 \
  --num_workers 0 \
  --learning_rate 3e-5 \
  --augmentation false \
  --seed 42 \
  --device auto \
  2>&1 | tee "$LOG_DIR/value_smoke.log"
```

检查：

```bash
test -f "$VALUE_SMOKE/checkpoint.pt"
test -f "$VALUE_SMOKE/config.json"
test -f "$VALUE_SMOKE/value_function_meta.json"
test -f "$VALUE_SMOKE/train_metrics.jsonl"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/lerobot/outputs/value_nero_egg_subtask_pi05_v2_smoke")
config = json.loads((root / "config.json").read_text())
metrics = [json.loads(line) for line in (root / "train_metrics.jsonl").read_text().splitlines()]
assert config["model"]["backbone_type"] == "pi05"
assert config["model"]["precision"] == "bfloat16"
assert config["model"]["num_bins"] == 128
assert config["model"]["freeze_backbone"] is True
assert config["model"]["freeze_vision_encoder"] is True
assert metrics[-1]["step"] == 2
assert metrics[-1]["train"]["samples"] == 2
print("value smoke passed")
PY
```

如果这里 OOM，问题不是正式 batch size，而是基础模型/环境本身；不要继续正式训练。

## 9. Batch 8 短测速和显存检查

使用 100 个 train step、无 validation、无 augmentation 测试 batch 8：

```bash
/usr/bin/time -v \
"$PY" -m lerobot.scripts.lerobot_train_value_function \
  --root "$TRAIN_ROOT" \
  --output_dir "$VALUE_SPEED" \
  --mode subtask \
  --backbone_type pi05 \
  --pretrained_path "$PI05_INIT" \
  --local_files_only true \
  --precision bfloat16 \
  --image_keys \
    observation.images.left_wrist \
    observation.images.right_wrist \
    observation.images.third_person \
  --num_vlm_layers 3 \
  --num_bins 128 \
  --use_elapsed_aux false \
  --use_state true \
  --freeze_vision_encoder true \
  --freeze_backbone true \
  --num_unfrozen_backbone_layers 0 \
  --val_fraction 0 \
  --epochs 1 \
  --max_steps 100 \
  --batch_size 8 \
  --num_workers 8 \
  --learning_rate 3e-5 \
  --augmentation false \
  --seed 42 \
  --device auto \
  2>&1 | tee "$LOG_DIR/value_batch8_speed.log"
```
```bash
/usr/bin/time -v \
"$PY" -m lerobot.scripts.lerobot_train_value_function \
  --root "$TRAIN_ROOT" \
  --output_dir "$VALUE_SPEED_BS16" \
  --mode subtask \
  --backbone_type pi05 \
  --pretrained_path "$PI05_INIT" \
  --local_files_only true \
  --precision bfloat16 \
  --image_keys \
    observation.images.left_wrist \
    observation.images.right_wrist \
    observation.images.third_person \
  --num_vlm_layers 3 \
  --num_bins 128 \
  --use_elapsed_aux false \
  --use_state true \
  --freeze_vision_encoder true \
  --freeze_backbone true \
  --num_unfrozen_backbone_layers 0 \
  --val_fraction 0 \
  --epochs 1 \
  --max_steps 100 \
  --batch_size 128 \
  --num_workers 8 \
  --learning_rate 3e-5 \
  --augmentation false \
  --seed 42 \
  --device auto \
  2>&1 | tee "$LOG_DIR/value_batch128_speed.log"
```
同时可在另一个终端观察：

```bash
watch -n 1 nvidia-smi
```

本机实测 batch 8/16/32/64/128 后，正式训练使用 `batch_size=32`。batch 32
的吞吐约为 41.62 samples/s；batch 64 没有继续提升，batch 128 也只提升约
1.3%，但会明显增加内存占用并减少每轮参数更新次数。

粗略计算：

```text
batch 32: 11,406 train steps / epoch
不含 augmentation 和 validation 的纯训练时间约 2.44 小时 / epoch
```

正式训练包含 augmentation、validation 和 checkpoint 保存，预计约
2.6–2.9 小时 / epoch，3 epochs 约 7.8–8.7 小时。

```text
estimated_epoch_hours = average_step_seconds * 11406 / 3600
```

## 10. 正式 PI0.5 value 训练，并在 epoch 2 后停止

正式训练使用两个 root：

```text
root 0 = TRAIN_ROOT
root 1 = VAL_ROOT
```

因此 validation episode 必须写成：

```text
1:72 1:92 1:112
```

当前实验使用 `--epochs 3` 启动，因此 cosine scheduler 也是按 3 epochs 构造。现在决定只比较
epoch 1 和 epoch 2：不要重启训练，也不要把已经在跑的命令改成 `--epochs 2`。

当前正在运行的命令为：

```bash
/usr/bin/time -v \
"$PY" -m lerobot.scripts.lerobot_train_value_function \
  --root "$TRAIN_ROOT" "$VAL_ROOT" \
  --output_dir "$VALUE_OUT" \
  --mode subtask \
  --backbone_type pi05 \
  --pretrained_path "$PI05_INIT" \
  --local_files_only true \
  --precision bfloat16 \
  --image_keys \
    observation.images.left_wrist \
    observation.images.right_wrist \
    observation.images.third_person \
  --num_vlm_layers 3 \
  --num_bins 128 \
  --use_elapsed_aux false \
  --use_state true \
  --freeze_vision_encoder true \
  --freeze_backbone true \
  --num_unfrozen_backbone_layers 0 \
  --remaining_loss_weight 1.0 \
  --subtask_ce_loss_weight 0.2 \
  --elapsed_loss_weight 0.0 \
  --val_episodes 1:72 1:92 1:112 \
  --epochs 3 \
  --batch_size 32 \
  --num_workers 8 \
  --learning_rate 3e-5 \
  --weight_decay 1e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --progress true \
  --log_every_steps 50 \
  --save_every_epoch true \
  --augmentation true \
  --seed 42 \
  --device auto \
  2>&1 | tee "$LOG_DIR/value_train.log"
```

当前 trainer 的行为：

- 单 GPU，不使用 DDP/Accelerate；
- train 和 validation 都显示实时进度、ETA、loss、accuracy；train 额外显示 learning rate；
- 每个 epoch 结束后完整跑 3 条 validation episode；
- 每个 epoch 结束写一条 `train_metrics.jsonl`；
- 每个 epoch 保存为 `checkpoints/checkpoint_epoch_001.pt` 等；
- `checkpoint.pt` 是最新 epoch 的硬链接，后续推理命令保持不变且不额外重复占一份空间；
- 当前没有正式 resume、best checkpoint 或 early stopping。

### 10.1 安全停止时机

必须等待终端完整出现：

```text
Epoch 2/3 complete | step=... | train_loss=... | val_loss=... | val_acc=...
```

这行只会在 epoch 2 的 train、validation 和
`checkpoints/checkpoint_epoch_002.pt` 原子保存全部完成后出现。看到 epoch 3 的 train 进度条开始后，
按一次 `Ctrl-C`。退出状态 130、`KeyboardInterrupt` 或 `/usr/bin/time` 报 non-zero 都是本次人工停止的
正常现象。

不要在 `Epoch 2/3 val` 期间或 checkpoint 保存期间提前停止。

### 10.2 检查 epoch 1/2 checkpoint 和 validation metrics

```bash
export VALUE_CKPT_E1=$VALUE_OUT/checkpoints/checkpoint_epoch_001.pt
export VALUE_CKPT_E2=$VALUE_OUT/checkpoints/checkpoint_epoch_002.pt

test -f "$VALUE_CKPT_E1"
test -f "$VALUE_CKPT_E2"
test -f "$VALUE_OUT/config.json"
test -f "$VALUE_OUT/value_function_meta.json"
test -f "$VALUE_OUT/train_metrics.jsonl"

# 人工停止 epoch 3 后，latest 应仍然是 epoch 2 的硬链接。
test "$VALUE_OUT/checkpoint.pt" -ef "$VALUE_CKPT_E2"

sha256sum "$VALUE_CKPT_E1" | tee "$LOG_DIR/value_checkpoint_epoch1_sha256.txt"
sha256sum "$VALUE_CKPT_E2" | tee "$LOG_DIR/value_checkpoint_epoch2_sha256.txt"
```

打印两个 epoch 的同口径 validation 指标：

```bash
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/lerobot/outputs/value_nero_egg_subtask_pi05_v2")
records = [json.loads(line) for line in (root / "train_metrics.jsonl").read_text().splitlines()]
assert len(records) == 2, len(records)
for record in records:
    print(
        "epoch=", record["epoch"],
        "step=", record["step"],
        "train_loss=", record["train"]["losses"]["loss"],
        "val_loss=", record["val"]["losses"]["loss"],
        "val_frame_mae=", record["val"]["frame_mae"]["subtask"],
        "val_subtask_acc=", record["val"]["subtask_accuracy"],
        "val_monotonic_violation=",
        record["val"]["monotonic_violation_rate"]["subtask"],
    )
assert [record["epoch"] for record in records] == [1, 2]
assert all(record["val"]["samples"] == 12798 for record in records)
print("epoch 1/2 checkpoints and metrics passed")
PY
```

不要只凭 train loss 选择。先参考 val loss、val frame MAE、subtask accuracy 和 monotonic violation，
然后继续执行第 11–13 节的曲线对比。

## 11. 为 epoch 1/2 创建隔离的 validation 副本

`lerobot_value_infer` 会原地更新 episode 的 `extras.parquet`。不能让两个 checkpoint 直接依次写同一个
`VAL_ROOT`，否则 epoch 2 会覆盖 epoch 1 的结果。

下面创建两个轻量副本：每个副本只复制 3 个很小的 `extras.parquet` 和 root metadata；图片、
`frames.parquet`、`info.json` 等仍使用软链接，不复制原始图片。

要求 `VAL_E1_ROOT` 和 `VAL_E2_ROOT` 尚不存在。如果已经存在，不要覆盖；改一个新的目录后缀。

```bash
export VAL_E1_ROOT=/data1/lerobot/raw_views/ming326/nero_egg_value_val_ep72_92_112_epoch1_v2
export VAL_E2_ROOT=/data1/lerobot/raw_views/ming326/nero_egg_value_val_ep72_92_112_epoch2_v2

test -n "$VAL_E1_ROOT"
test -n "$VAL_E2_ROOT"
test ! -e "$VAL_E1_ROOT"
test ! -e "$VAL_E2_ROOT"

"$PY" - <<'PY'
import os
import shutil
from pathlib import Path

source_root = Path(os.environ["VAL_ROOT"])
destinations = [
    Path(os.environ["VAL_E1_ROOT"]),
    Path(os.environ["VAL_E2_ROOT"]),
]

for destination in destinations:
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; choose a fresh path instead of overwriting it"
        )
    destination.mkdir(parents=True)
    shutil.copy2(source_root / "run_meta.json", destination / "run_meta.json")
    shutil.copy2(
        source_root / "value_function_meta.json",
        destination / "value_function_meta.json",
    )

    for source_episode in sorted(source_root.glob("ep_*")):
        destination_episode = destination / source_episode.name
        destination_episode.mkdir()
        for source_entry in source_episode.iterdir():
            destination_entry = destination_episode / source_entry.name
            if source_entry.name == "extras.parquet":
                shutil.copy2(source_entry, destination_entry)
            else:
                destination_entry.symlink_to(
                    source_entry.resolve(),
                    target_is_directory=source_entry.is_dir(),
                )

print("created isolated validation roots:")
for destination in destinations:
    print(" ", destination)
PY
```

验证两个副本不会写回原始 validation：

```bash
"$PY" - <<'PY'
import os
from pathlib import Path

for variable in ("VAL_E1_ROOT", "VAL_E2_ROOT"):
    root = Path(os.environ[variable])
    episodes = sorted(root.glob("ep_*"))
    assert len(episodes) == 3, (variable, len(episodes))
    for episode in episodes:
        assert episode.is_dir() and not episode.is_symlink(), episode
        assert (episode / "extras.parquet").is_file()
        assert not (episode / "extras.parquet").is_symlink()
        assert (episode / "frames.parquet").is_symlink()
        assert (episode / "left_wrist").is_symlink()
        assert (episode / "right_wrist").is_symlink()
        assert (episode / "third_person").is_symlink()
    print(variable, "isolated extras passed")
PY

du -sh "$VAL_E1_ROOT" "$VAL_E2_ROOT"
```

## 12. 分别运行 epoch 1/2 validation inference

先用 epoch 1：

```bash
"$PY" -m lerobot.scripts.lerobot_value_infer \
  --root "$VAL_E1_ROOT" \
  --checkpoint "$VALUE_CKPT_E1" \
  --mode subtask \
  --subtask_inference_path both \
  --batch_size 16 \
  --num_workers 8 \
  --device auto \
  --progress true \
  --transition_penalty 0.0 \
  --allow_subtask_skip false \
  2>&1 | tee "$LOG_DIR/value_infer_val_epoch1.log"
```

再用 epoch 2：

```bash
"$PY" -m lerobot.scripts.lerobot_value_infer \
  --root "$VAL_E2_ROOT" \
  --checkpoint "$VALUE_CKPT_E2" \
  --mode subtask \
  --subtask_inference_path both \
  --batch_size 16 \
  --num_workers 8 \
  --device auto \
  --progress true \
  --transition_penalty 0.0 \
  --allow_subtask_skip false \
  2>&1 | tee "$LOG_DIR/value_infer_val_epoch2.log"
```

如果 batch 16 OOM，只把两个命令的 `--batch_size` 都改成 8；结果语义不变。

## 13. 并排可视化 epoch 1/2，并选择最终 checkpoint

在终端 A 启动 epoch 1 只读 UI：

```bash
cd /home/zenbot-robot/repos/lerobot
export PY=/home/zenbot-robot/.conda/envs/lerobot-main/bin/python
export PYTHONPATH=$PWD/src
export VAL_E1_ROOT=/data1/lerobot/raw_views/ming326/nero_egg_value_val_ep72_92_112_epoch1_v2

"$PY" -m lerobot.scripts.lerobot_value_viz \
  --root "$VAL_E1_ROOT" \
  --chunk_size 50 \
  --host 127.0.0.1 \
  --port 8003 \
  --no-browser
```

在终端 B 启动 epoch 2 只读 UI：

```bash
cd /home/zenbot-robot/repos/lerobot
export PY=/home/zenbot-robot/.conda/envs/lerobot-main/bin/python
export PYTHONPATH=$PWD/src
export VAL_E2_ROOT=/data1/lerobot/raw_views/ming326/nero_egg_value_val_ep72_92_112_epoch2_v2

"$PY" -m lerobot.scripts.lerobot_value_viz \
  --root "$VAL_E2_ROOT" \
  --chunk_size 50 \
  --host 127.0.0.1 \
  --port 8004 \
  --no-browser
```

浏览器并排打开：

```text
epoch 1: http://127.0.0.1:8003
epoch 2: http://127.0.0.1:8004
```

依次查看 `ep72`、`ep92`、`ep112`，在两个页面选取相同位置直接比较。优先选择：

- validation loss/frame MAE 更低且没有明显退化；
- raw classifier subtask id 和 `pred_smooth` boundary 更接近 GT；
- `pred_smooth` 按 0→11 单调经过全部 subtask；
- 每个 subtask 内 remaining value 总体更稳定地下行；
- 停顿、遮挡和模糊处没有异常大跳变；
- 不是单纯机械匀速倒计时；
- 三个 episode 表现一致，而不是只在其中一条更好。

检查完成后在两个 UI 终端分别按 `Ctrl-C`。

如果选择 epoch 1：

```bash
export VALUE_CKPT="$VALUE_CKPT_E1"
```

如果选择 epoch 2：

```bash
export VALUE_CKPT="$VALUE_CKPT_E2"
```

只执行上面二选一中的一条，然后记录选择：

```bash
printf '%s\n' "$VALUE_CKPT" | tee "$LOG_DIR/value_selected_checkpoint.txt"
sha256sum "$VALUE_CKPT" | tee "$LOG_DIR/value_selected_checkpoint_sha256.txt"
```

如果两个 checkpoint 在三个 validation episode 上都明显失败，在这里停止，不要执行全量 inference。

## 14. 使用选定 checkpoint 推理 ep00–ep122 全部 122 条 episode

这里的 `00–122` 实际共有 122 条 episode，因为 `ep51` 不存在，总帧数为 613,123。

先重新从训练 metadata 提取固定 order 和 scale，避免依赖旧终端变量：

```bash
export ORDER="$("$PY" -c '
import json, sys
meta = json.load(open(sys.argv[1]))
print(json.dumps(meta["subtask_order"]))
' "$TRAIN_ROOT/value_function_meta.json")"

export SCALES="$("$PY" -c '
import json, sys
meta = json.load(open(sys.argv[1]))
print(json.dumps(meta["subtask_scale"]["frames_by_subtask"]))
' "$TRAIN_ROOT/value_function_meta.json")"
```

创建全量 view。该 view 仍不复制图片；但后面的 target 和 inference 会有意把 GT/value prediction
写入 `$RAW` 中每条 episode 的 `extras.parquet`。

要求 `ALL_ROOT` 尚不存在：

```bash
export ALL_ROOT=/data1/lerobot/raw_views/ming326/nero_egg_value_all_ep000_122_v2

test -n "$ALL_ROOT"
test ! -e "$ALL_ROOT"
mkdir -p "$ALL_ROOT"
ln -s "$RAW/run_meta.json" "$ALL_ROOT/run_meta.json"

for source_episode in "$RAW"/ep_*; do
  test -d "$source_episode" || continue
  ln -s "$source_episode" "$ALL_ROOT/$(basename "$source_episode")"
done
```

使用训练集确定的 order/scale 为全量 view 建立一致的 target metadata：

```bash
"$PY" -m lerobot.scripts.lerobot_value_prepare_targets \
  --root "$ALL_ROOT" \
  --mode subtask \
  --num_bins 128 \
  --subtask_scale manual \
  --subtask_scale_frames_json "$SCALES" \
  --elapsed_aux false \
  --subtask_order_json "$ORDER" \
  --require_all_subtasks true \
  --require_single_segment_per_subtask true \
  --require_success_only true \
  2>&1 | tee "$LOG_DIR/all_targets_write.log"
```

确认 checkpoint 仍是刚才选择的 epoch 1 或 epoch 2：

```bash
case "$VALUE_CKPT" in
  "$VALUE_CKPT_E1"|"$VALUE_CKPT_E2") ;;
  *) echo "VALUE_CKPT is not epoch 1 or epoch 2" >&2; false ;;
esac
```

运行全量 inference：

```bash
"$PY" -m lerobot.scripts.lerobot_value_infer \
  --root "$ALL_ROOT" \
  --checkpoint "$VALUE_CKPT" \
  --mode subtask \
  --subtask_inference_path both \
  --batch_size 16 \
  --num_workers 8 \
  --device auto \
  --progress true \
  --transition_penalty 0.0 \
  --allow_subtask_skip false \
  2>&1 | tee "$LOG_DIR/value_infer_all.log"
```

这一步处理 613,123 帧 × 三路相机，是 inference 阶段最耗时的步骤。

验证 inference provenance：

```bash
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/lerobot/raw_views/ming326/nero_egg_value_all_ep000_122_v2")
meta = json.loads((root / "value_function_meta.json").read_text())
stage = meta["stages"]["value_inference.subtask"]
summary = meta["value_inference"]["subtask"]
assert stage["prediction_source"] == "model_pred"
assert stage["synthetic"] is False
assert stage["stale"] is False
required = {
    "value_subtask_id_pred",
    "value_subtask_confidence",
    "value_subtask_id_pred_smooth",
    "value_subtask_remaining_frames_pred_gt_head",
    "value_subtask_remaining_frames_pred_smooth_head",
}
assert required <= set(stage["output_columns"])
assert summary["frames"] == 613123, summary["frames"]
assert stage["config"]["checkpoint"] in {
    "/data1/lerobot/outputs/value_nero_egg_subtask_pi05_v2/checkpoints/checkpoint_epoch_001.pt",
    "/data1/lerobot/outputs/value_nero_egg_subtask_pi05_v2/checkpoints/checkpoint_epoch_002.pt",
}
print("all-episode model inference provenance passed")
PY
```

当前已经明确要为全部 122 条 episode 标注 advantage，因此第 15–17 节统一使用 `ALL_ROOT` 计算
advantage、导出 positive/negative 标签并计算 loss weight。第 18 节是否也用全部 122 条 episode 构建
VLA dataset，仍需在 advantage 检查完成后单独确认。

## 15. 为全部 122 条 episode 计算正式 subtask advantage

先固定并检查全量 root：

```bash
export ALL_ROOT=/data1/lerobot/raw_views/ming326/nero_egg_value_all_ep000_122_v2
test -f "$ALL_ROOT/value_function_meta.json"
```

本轮 action chunk 固定：

```text
chunk_size = 50 frames
```

在 30 FPS 下是约 1.67 秒。后续 PI0.5 policy 的 action chunk 必须保持 50；如果 policy action chunk
改变，必须从本步骤开始重跑。

### 15.1 Dry-run

```bash
"$PY" -m lerobot.scripts.lerobot_compute_advantage \
  --root "$ALL_ROOT" \
  --value_mode subtask \
  --value_source model_pred \
  --subtask_inference_path gt_conditioned \
  --chunk_size 50 \
  --boundary_transition_value 1.0 \
  --dry_run \
  2>&1 | tee "$LOG_DIR/advantage_dry_run.log"
```

正式实验必须满足：

```text
value_source = model_pred
subtask_inference_path = gt_conditioned
synthetic = false
```

### 15.2 正式写入

```bash
"$PY" -m lerobot.scripts.lerobot_compute_advantage \
  --root "$ALL_ROOT" \
  --value_mode subtask \
  --value_source model_pred \
  --subtask_inference_path gt_conditioned \
  --chunk_size 50 \
  --boundary_transition_value 1.0 \
  2>&1 | tee "$LOG_DIR/advantage_write.log"
```

核心输出：

```text
advantage_subtask_chunk
advantage_subtask_valid_horizon
advantage_subtask_is_valid
advantage_subtask_start_value
advantage_subtask_end_value
advantage_subtask_num_crossings
advantage_subtask_within_subtask_horizon
advantage_subtask_boundary_progress
```

当前 centered advantage 语义：

```text
约等于 0：按理想速度推进
大于 0：相对推进更好
小于 0：相对卡住、无效或进度不足
```

### 15.3 检查 advantage 分布

正式写入完成后，先打印 valid/invalid 数量、分位数，以及不同 positive 比例对应的 advantage cutoff：

```bash
"$PY" - <<'PY'
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path("/data1/lerobot/raw_views/ming326/nero_egg_value_all_ep000_122_v2")
values = []
invalid = 0
for episode in sorted(root.glob("ep_*")):
    table = pq.read_table(
        episode / "extras.parquet",
        columns=["advantage_subtask_chunk", "advantage_subtask_is_valid"],
    )
    advantage = np.asarray(table["advantage_subtask_chunk"].to_pylist(), dtype=np.float32)
    valid = np.asarray(table["advantage_subtask_is_valid"].to_pylist(), dtype=bool)
    values.append(advantage[valid])
    invalid += int((~valid).sum())

values = np.concatenate(values)
print("valid chunks:", len(values))
print("invalid/ignore chunks:", invalid)
for quantile in (0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100):
    print(f"q{quantile:03d}: {np.percentile(values, quantile):.6f}")

print("\nCandidate positive ratios:")
for positive_percent in (50, 60, 70, 80, 85, 90):
    cutoff = np.percentile(values, 100 - positive_percent)
    print(
        f"positive={positive_percent:2d}%  negative={100-positive_percent:2d}%"
        f"  cutoff≈{cutoff:.6f}"
    )
PY
```

## 16. 在 UI 中确定 positive/negative 比例并导出全量标签

这里的比例只作用于 valid chunk；每条 episode 末尾不足 action horizon 的 invalid chunk 会自动标成
`ignore`。所有 episode 成功不等于所有 chunk 都应该是 positive：negative 表示相对推进较差、停顿或
无效的 action chunk，不表示整条 episode 失败。

先以 80% positive / 20% negative 为候选起点：

```text
top_percent = 0.8
tie_policy = exact_count
```

### 16.1 Headless preview，不写入

```bash
"$PY" -m lerobot.scripts.lerobot_advantage_labeler \
  --root "$ALL_ROOT" \
  --value_mode subtask \
  --top_percent 0.8 \
  --sort_order desc \
  --tie_policy exact_count \
  --export \
  --dry_run \
  2>&1 | tee "$LOG_DIR/advantage_labels_dry_run.log"
```

注意：

```text
0.8 = 80% positive
不是 top 20%
```

上面只是 preview，不写 `advantage_label_subtask`。

### 16.2 启动 UI 并调节比例

```bash
"$PY" -m lerobot.scripts.lerobot_advantage_labeler \
  --root "$ALL_ROOT" \
  --value_mode subtask \
  --top_percent 0.8 \
  --sort_order desc \
  --tie_policy exact_count \
  --host 127.0.0.1 \
  --port 8001 \
  --no-browser
```

浏览器打开：

```text
http://127.0.0.1:8001
```

检查：

1. `Sort=high to low` 时抽查最高 advantage，确认确实是推进更好的动作；
2. `Sort=low to high` 时抽查最低 advantage，确认包含停顿、犹豫或无效动作；
3. 调节 `Positive %` 后点击 `Threshold page ...`，直接检查当前候选阈值附近的 chunk；
4. 12 个 subtask 和新旧 episode 都应有合理样本；
5. 末尾短 horizon/invalid chunk 应为 `ignore`；
6. 必要时手工 override 为 `positive`、`negative` 或 `ignore`。

比例选择原则：

```text
如果 positive 中仍有明显卡住/无效动作：降低 Positive %（例如 80% → 70%）。
如果 negative 中大量是正常、有效推进动作：提高 Positive %（例如 80% → 85% 或 90%）。
如果 80% 附近两侧语义基本合理：保留 80/20。
```

拖动 `Positive %` 只做实时 preview，不写文件。确定最终比例后再点击一次 UI 的 `Export`；确认弹窗后
才会正式写入全部 122 条 episode。Export 成功后记录最终比例，再按 `Ctrl-C` 关闭 server。

如果完全不做人工 override，也可以关闭 UI 后用以下命令正式导出：

```bash
# 按 UI 检查结果修改；例如 0.8 表示 80% positive。
export POSITIVE_PERCENT=0.8

"$PY" -m lerobot.scripts.lerobot_advantage_labeler \
  --root "$ALL_ROOT" \
  --value_mode subtask \
  --top_percent "$POSITIVE_PERCENT" \
  --sort_order desc \
  --tie_policy exact_count \
  --export \
  2>&1 | tee "$LOG_DIR/advantage_labels_write.log"
```

二选一：

```text
使用 UI Export
或
运行 headless 正式 export
```

不要两种都重复执行。

验证 label：

```bash
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/lerobot/raw_views/ming326/nero_egg_value_all_ep000_122_v2")
meta = json.loads((root / "value_function_meta.json").read_text())
stage = meta["stages"]["advantage_labeling.subtask"]
assert stage["prediction_source"] == "model_pred"
assert stage["synthetic"] is False
assert stage["stale"] is False
assert "advantage_label_subtask" in stage["output_columns"]
print("advantage label provenance passed")
PY
```

## 17. 为全部 122 条 episode 计算 group-relative advantage weight

固定第一版参数：

```text
group source:               progress
group bin width:            0.1
q:                          0.8
tau:                        0.08
w_min / w_max:              0.1 / 2.0
positive group max weight:  2.0
minimum group size:         4
negative weight:            1.0
ignore weight:              0.0（内部固定）
```

### 17.1 Dry-run

```bash
"$PY" -m lerobot.scripts.lerobot_compute_advantage_weights \
  --root "$ALL_ROOT" \
  --value_mode subtask \
  --group_source progress \
  --group_bin_width 0.1 \
  --q 0.8 \
  --tau 0.08 \
  --w_min 0.1 \
  --w_max 2.0 \
  --positive_group_max_weight 2.0 \
  --min_group_size 4 \
  --negative_weight 1.0 \
  --dry_run \
  2>&1 | tee "$LOG_DIR/advantage_weights_dry_run.log"
```

### 17.2 正式写入

```bash
"$PY" -m lerobot.scripts.lerobot_compute_advantage_weights \
  --root "$ALL_ROOT" \
  --value_mode subtask \
  --group_source progress \
  --group_bin_width 0.1 \
  --q 0.8 \
  --tau 0.08 \
  --w_min 0.1 \
  --w_max 2.0 \
  --positive_group_max_weight 2.0 \
  --min_group_size 4 \
  --negative_weight 1.0 \
  2>&1 | tee "$LOG_DIR/advantage_weights_write.log"
```

### 17.3 Weight UI

```bash
"$PY" -m lerobot.scripts.lerobot_compute_advantage_weights \
  --root "$ALL_ROOT" \
  --value_mode subtask \
  --serve \
  --host 127.0.0.1 \
  --port 8002 \
  --no-browser
```

浏览器打开：

```text
http://127.0.0.1:8002
```

检查：

1. 各 subtask/progress group 的大小；
2. 是否大量 group 小于 4；
3. positive weight 是否随 advantage 单调；
4. 最大 positive weight 是否为 2；
5. negative 是否为 1；
6. ignore 是否为 0；
7. 是否大量 positive 堆积在 0.1 或 2.0。

检查完成按 `Ctrl-C`。

如果大量 group 小于 4，可把 `group_bin_width` 从 `0.1` 改为 `0.2`；一旦修改，必须重新运行本步骤的
dry-run、正式写入和 UI 检查。

验证最终 provenance：

```bash
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/data1/lerobot/raw_views/ming326/nero_egg_value_all_ep000_122_v2")
meta = json.loads((root / "value_function_meta.json").read_text())
stages = meta["stages"]
for name in (
    "targets",
    "value_inference.subtask",
    "advantage.subtask",
    "advantage_labeling.subtask",
    "advantage_weights.subtask",
):
    assert name in stages, name
    assert stages[name]["stale"] is False, name
assert stages["value_inference.subtask"]["synthetic"] is False
assert stages["advantage.subtask"]["synthetic"] is False
assert stages["advantage_labeling.subtask"]["synthetic"] is False
assert stages["advantage_weights.subtask"]["synthetic"] is False
print("complete real-model pipeline provenance passed")
PY
```

## 18. 构建 PI0.5 VLA 使用的 LeRobotDataset

Value inference、advantage、label 和 weight 的完整 provenance 记录在 `ALL_ROOT`。最终
LeRobotDataset 也从 `ALL_ROOT` 构建：

```text
包含全部 122 条 episode
包含 ep72、ep92、ep112
包含原先的 41 条 old holdout
总帧数 613123
```

Raw 中错误的 match/candle task 不需要原地修改。本步骤通过：

```text
--task_override "$TASK"
```

把最终 dataset 的 task 统一写成：

```text
Prepare and cook scrambled eggs by adding oil and salt, pouring and stirring the beaten eggs in the pan, serving them in a bowl.
```

本轮只保留 VLA 训练需要的字段，排除 value/advantage debug 字段：

```text
保留：
  subtask
  subtask_progress
  advantage_label_subtask
  advantage_loss_weight_subtask

排除：
  value_*
  advantage_subtask_*
  advantage_group_id_subtask
```

### 18.1 Build dry-run

```bash
"$PY" -m lerobot.scripts.lerobot_build_dataset \
  --runs "['$ALL_ROOT']" \
  --output_repo_id "$DATASET_REPO_ID" \
  --output_root "$DATASET_ROOT" \
  --video true \
  --vcodec libsvtav1 \
  --exclude_features 'value_*,advantage_subtask_*,advantage_group_id_subtask' \
  --task_override "$TASK" \
  --push_to_hub false \
  --force false \
  --dry_run true \
  2>&1 | tee "$LOG_DIR/build_dataset_dry_run.log"
```

Dry-run 输出必须包含：

```text
subtask
subtask_progress
advantage_label_subtask
advantage_loss_weight_subtask
```

Dry-run 必须输出 `Discovered 122 episodes across 1 runs.`。

### 18.2 正式 build

首次构建必须保证 `$DATASET_ROOT` 不存在：

```bash
test ! -e "$DATASET_ROOT"
```

如果上面命令返回非零，停止并检查现有目录；不要直接使用 `--force true`。

正式构建：

```bash
"$PY" -m lerobot.scripts.lerobot_build_dataset \
  --runs "['$ALL_ROOT']" \
  --output_repo_id "$DATASET_REPO_ID" \
  --output_root "$DATASET_ROOT" \
  --video true \
  --vcodec libsvtav1 \
  --exclude_features 'value_*,advantage_subtask_*,advantage_group_id_subtask' \
  --task_override "$TASK" \
  --push_to_hub false \
  --force false \
  --dry_run false \
  2>&1 | tee "$LOG_DIR/build_dataset.log"
```

视频编码可能耗时较长。不要中断进程，不要并行启动第二个 build。

## 19. 验证最终 LeRobotDataset

```bash
"$PY" - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "ming326/nero_egg_adv_pi05_v2"
root = Path("/data1/lerobot/datasets/ming326/nero_egg_adv_pi05_v2")
task = (
    "Prepare and cook scrambled eggs by adding oil and salt, pouring and stirring "
    "the beaten eggs in the pan, serving them in a bowl."
)

dataset = LeRobotDataset(repo_id, root=root)
assert dataset.num_episodes == 122, dataset.num_episodes
assert dataset.num_frames == 613123, dataset.num_frames

required = {
    "subtask",
    "subtask_progress",
    "advantage_label_subtask",
    "advantage_loss_weight_subtask",
}
assert required <= set(dataset.features), sorted(dataset.features)
assert not any(name.startswith("value_") for name in dataset.features)
assert "advantage_group_id_subtask" not in dataset.features

item = dataset[0]
assert item["task"] == task, item["task"]
assert item["advantage_label_subtask"] in {"positive", "negative", "ignore"}
weight = float(item["advantage_loss_weight_subtask"])
assert 0.0 <= weight <= 2.0, weight

print("dataset root:", root)
print("episodes:", dataset.num_episodes)
print("frames:", dataset.num_frames)
print("task:", item["task"])
print("subtask:", item["subtask"])
print("advantage label:", item["advantage_label_subtask"])
print("advantage weight:", weight)
print("final LeRobotDataset verification passed")
PY
```

检查磁盘占用：

```bash
du -sh "$DATASET_ROOT"
```

## 20. 最终产物

完成后应保留：

### PI0.5 base

```text
/data1/lerobot/models/pi05_base/
```

至少包含：

```text
model.safetensors
config.json
```

### Value model

```text
/data1/lerobot/outputs/value_nero_egg_subtask_pi05_v2/
```

包含：

```text
checkpoint.pt
config.json
value_function_meta.json
train_metrics.jsonl
```

### Raw-derived pipeline 字段

`TRAIN_ROOT` 对应 episode 的 `extras.parquet` 包含：

```text
subtask
subtask_progress
value_subtask_*_gt
value_subtask_*_pred_*
advantage_subtask_*
advantage_label_subtask
advantage_group_id_subtask
advantage_loss_weight_subtask
```

### 最终 VLA dataset

```text
/data1/lerobot/datasets/ming326/nero_egg_adv_pi05_v2/
```

包含全部 122 条 episode 和正确 task prompt，不包含 value debug 字段。

### 实验日志

```text
/data1/lerobot/logs/nero_egg_value_pi05_v2/
```

包含代码状态、下载哈希、target、训练、inference、advantage 和 build 日志。

## 21. 下游 PI0.5 VLA 训练所需固定参数

本指南完成后，PI0.5 VLA 的数据入口固定为：

```text
dataset repo id:  ming326/nero_egg_adv_pi05_v2
dataset root:     /data1/lerobot/datasets/ming326/nero_egg_adv_pi05_v2
base model:       /data1/lerobot/models/pi05_base
```

Advantage conditioning/weighting 固定字段：

```text
policy type:                         pi05
use_advantage_conditioning:          true
advantage_label_key:                 advantage_label_subtask
use_advantage_weighting:             true
advantage_loss_weight_key:           advantage_loss_weight_subtask
advantage_condition_dropout_prob:    0.1
```

PI0.5 VLA 全参数训练和 value head 训练是两个不同任务。Value head 在单张 4090 上可运行；完整 PI0.5
VLA 是否适合单张 4090 取决于 gradient checkpointing、train_expert_only、batch size 和显存峰值，应先
另做真实 2-step smoke，不能直接套用本指南的 value batch size。

## 22. 重跑和 stale 规则

任何上游内容变化后，按以下依赖顺序重跑：

```text
subtask annotation / split
  -> value targets
  -> value training
  -> value inference
  -> advantage
  -> advantage labels
  -> advantage weights
  -> LeRobotDataset build
```

具体规则：

- 改 subtask boundary：从 target 开始全部重跑；
- 改 train episode：重新计算 p95 scale，三个 root 的 target 全部重跑；
- 改 value checkpoint：三个 root inference 重跑，之后 advantage/label/weight/build 重跑；
- 改 action chunk size：advantage/label/weight/build 重跑；
- 改 top percent/override：label/weight/build 重跑；
- 改 weight 参数：weight/build 重跑；
- 只改 task prompt：只需用新 dataset 目录重新 build；
- 不要在已有 dataset 目录上使用 `--force true`；创建 `v3` 新目录；
- 不要把 validation 或 old holdout 加入 advantage/VLA train dataset；
- 不要使用 `gt` 或 `mock_pred` advantage 作为正式实验 artifact。

每次进入下一阶段前，metadata 中当前 stage 必须：

```text
stale = false
synthetic = false
```
