# Nero Egg：本地标注、fcloud Value/VLA 训练与本地部署指南

日期：2026-07-14

适用仓库：`/home/zenbot-robot/repos/lerobot`

本指南把 `value_function_subtask_advantage_pipeline_plan.md` 中的 Milestone 12 落实为一套针对
`nero_egg` 的可执行流程，覆盖：

1. 本地 subtask 标注；
2. 严格隔离 value train/validation/test episode；
3. 向 `fcloud` 同步代码和约 232 GB raw 数据；
4. 在 `fcloud` 训练 value model、推理、可视化并确定 label/weight；
5. 在 2 张或 4 张 GPU 上完成 PI0/PI0.5 2-step smoke 和正式 VLA 训练；
6. 只把最终 VLA checkpoint 回传本地部署；
7. 后续追加 episode 时复用已有标注并生成新版本 dataset/checkpoint。

## 1. 已确认的机器、数据和路径

### 1.1 本地电脑

代码：

```text
/home/zenbot-robot/repos/lerobot
```

raw dataset：

```text
/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/nero_egg
```

当前数据统计：

```text
39 episodes: ep_000000 ... ep_000038
215,352 frames
30 FPS，约 2 小时
三路相机
约 232 GB
当前尚无 extras.parquet
```

本机磁盘目前仅余约 123 GB。不要在本机默认 Hugging Face cache 中再次构建完整 video dataset；
value/VLA 的正式 dataset 和训练输出放到 `fcloud` 的 `/datastore01`。

### 1.2 fcloud

SSH 配置名：

```bash
ssh fcloud
```

2026-07-14 检查结果：

```text
8 x NVIDIA H20-3e
每卡约 143,771 MiB
Conda: /home/hongming/miniconda3/envs/lerobot_recap
PyTorch: 2.10.0+cu128
Accelerate: 1.13.0
```

GPU 空闲情况会变化，每次训练前重新运行 `nvidia-smi`，不要永久写死本指南记录的 GPU 编号。

存储：

```text
/home/hongming             系统盘仅剩约 61 GB，只放代码
/datastore01               约 3.7 TB 可用，放 raw/dataset/output
```

推荐远程路径：

```text
代码          /home/hongming/repo/lerobot_value_pipeline
raw           /datastore01/hongming/lerobot_raw/ming326/nero_egg
raw views     /datastore01/hongming/lerobot_raw_views/ming326
dataset       /datastore01/hongming/lerobot/ming326
训练输出      /datastore01/hongming/lerobot_outputs
```

远程已有初始化 checkpoint，不需要从本地重复上传：

```text
PI0:
/datastore01/hongming/lerobot_outputs/pi0_strike_match_3_relative_bs192_20260630_172828/checkpoints/012000/pretrained_model

PI0.5:
/datastore01/hongming/lerobot_outputs/pi05_strike_match_3_subtask_relative_bs192_20260706_200321/checkpoints/010000/pretrained_model
```

## 2. 计算任务放在哪里

推荐分工：

| 工作 | 位置 | 原因 |
|---|---|---|
| raw 采集 | 本地 | 机器人和相机在本地 |
| subtask 标注 | 本地 | 原始图片已在本地，UI 不需要传输 |
| value target | fcloud | 和后续 provenance、训练 root 保持一致 |
| value model 训练 | fcloud 单卡 | 当前 value trainer 是单 GPU 实现 |
| 全量 value inference | fcloud 单卡 | 215k 帧三路图片，远程 H20 更合适 |
| value/weight UI | fcloud 运行，本地 SSH tunnel 查看 | 不必来回同步预测列 |
| VLA build/train | fcloud 2/4 卡 | dataset 和训练输出都在 `/datastore01` |
| 机器人部署 | 本地 | 只回传最终 `pretrained_model` |

当前 value function 是离线标注器，不参与机器人控制时的在线 forward。训练后的 VLA 在部署时若 batch
中没有 `advantage_label`，processor 会使用 `Advantage: positive`。因此第一版不需要把 value model
部署到本地控制环路；把它留在 `fcloud` 最省传输和显存。

若以后想做在线 value 监控，再单独复制 value checkpoint 或设计远程服务，不要把这件事和第一轮 VLA
部署绑定在一起。

## 3. Episode 划分

保留最后一条：

```text
ep_000038
```

作为严格 held-out value test episode。

划分如下：

```text
Value train candidates: ep_000000 ... ep_000037
Value validation:       ep_000000, ep_000010, ep_000020, ep_000030
Value train actual:     其余 34 条
Value final test:       ep_000038
VLA train v1:           ep_000000 ... ep_000037
```

不要仅把 `ep_000038` 放进 `--val_episodes`：validation 会被每轮查看和用于调参，不再是严格测试集。
本指南使用两个 raw view 隔离它。

`ep_000038` 仍然需要完整 subtask 标注，这样 UI 才能对比 GT subtask 边界、remaining GT 和 model
prediction。但它不能进入 value 训练或第一版 VLA dataset。

## 4. 本地标注

本地终端：

```bash
cd /home/zenbot-robot/repos/lerobot

export PY=/home/zenbot-robot/.conda/envs/lerobot-main/bin/python
export PYTHONPATH=$PWD/src
export LOCAL_RAW=/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/nero_egg
```

如果 `nero_egg` 的动作分解和 `strike_match_3` 完全相同，可以只复制 subtask 定义；不要复制旧的
`annotations.json`：

```bash
cp \
  /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3/annotation_config.json \
  "$LOCAL_RAW/annotation_config.json"
```

启动标注 UI：

```bash
$PY -m lerobot.scripts.lerobot_annotate_subtask \
  --root "$LOCAL_RAW" \
  --host 127.0.0.1 \
  --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

推荐 canonical 执行顺序：

```text
Pick up the match.
move the right arm to ready.
Pick up the matchbox.
move the left arm to ready.
Strike the match and light the candle.
Return to the home position.
```

每个 episode 必须满足：

- 六个 subtask 全部存在；
- 顺序完全一致；
- 每个 subtask 只出现一个连续 segment；
- 所有帧均有 label；
- 39 条 episode 都是成功示范，没有失败、超时或人工中止。

完成后点击 Export，生成每条 episode 的 `extras.parquet`。`ep_000038` 也要标注和 Export。

## 5. 同步代码到 fcloud

当前本地 value pipeline 修改尚未全部提交。正式训练前，推荐先创建一个可识别的 git commit/tag。
即使暂时不提交，也可以用 rsync 同步 working tree，但实验记录必须保存本地 commit hash 和
`git status --short`。

先保存状态：

```bash
cd /home/zenbot-robot/repos/lerobot
git rev-parse HEAD
git status --short
```

同步到一个独立远程目录，避免覆盖远程已有的 `lerobot_recap`：

```bash
ssh fcloud 'mkdir -p /home/hongming/repo/lerobot_value_pipeline'

rsync -a --info=progress2 \
  --exclude='.git/' \
  --exclude='outputs/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  /home/zenbot-robot/repos/lerobot/ \
  fcloud:/home/hongming/repo/lerobot_value_pipeline/
```

不要加 `--delete`，避免误删远程文件。

远程安装为 editable package，不改动已有 PyTorch/CUDA 依赖：

```bash
ssh fcloud

export REMOTE_REPO=/home/hongming/repo/lerobot_value_pipeline
export PY=/home/hongming/miniconda3/envs/lerobot_recap/bin/python

cd "$REMOTE_REPO"
$PY -m pip install -e . --no-deps
```

远程快速验证：

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

$PY -m pytest \
  tests/value_function/test_train_value_smoke.py \
  tests/value_function/test_value_infer_writeback.py \
  tests/scripts/test_advantage_weighted_train.py \
  tests/scripts/test_value_extras_build_dataset.py \
  -q
```

## 6. 同步 raw dataset 到 fcloud

直接同步到 `/datastore01`，不要传到 `/home/hongming`。

本地终端：

```bash
export LOCAL_RAW=/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/nero_egg
export REMOTE_RAW=/datastore01/hongming/lerobot_raw/ming326/nero_egg

ssh fcloud 'mkdir -p /datastore01/hongming/lerobot_raw/ming326/nero_egg'

rsync -aH \
  --info=progress2 \
  --partial \
  --partial-dir=.rsync-partial \
  "$LOCAL_RAW/" \
  "fcloud:$REMOTE_RAW/"
```

图片本身已经压缩，不建议加 `-z`，否则只会增加 CPU 开销。rsync 可断点续传；中断后直接重跑同一条
命令。

完成后进行只读差异检查：

```bash
rsync -aHn --itemize-changes \
  "$LOCAL_RAW/" \
  "fcloud:$REMOTE_RAW/"
```

理想结果是不输出任何待同步文件。不要用 `--delete`。

本机 NAS 上已有 `/mnt/nas-home/nero_egg.tar`，约 163 GB，但本指南没有把它当作 canonical 传输源，
因为归档内容和完整性尚未重新验证。只有在 `tar -tf` 完整通过并确认归档含最新标注后，才考虑传这个
单文件；常规流程优先使用可增量验证的 rsync。

## 7. 在 fcloud 创建 train/test raw view

以下操作只建立 symlink，不复制 232 GB 图片。

远程终端：

```bash
export REMOTE_RAW=/datastore01/hongming/lerobot_raw/ming326/nero_egg
export VIEW_BASE=/datastore01/hongming/lerobot_raw_views/ming326
export TRAIN_ROOT=$VIEW_BASE/nero_egg_value_train_v1
export TEST_ROOT=$VIEW_BASE/nero_egg_value_test_ep38_v1

mkdir -p "$TRAIN_ROOT" "$TEST_ROOT"

test -e "$TRAIN_ROOT/run_meta.json" || \
  ln -s "$REMOTE_RAW/run_meta.json" "$TRAIN_ROOT/run_meta.json"
test -e "$TEST_ROOT/run_meta.json" || \
  ln -s "$REMOTE_RAW/run_meta.json" "$TEST_ROOT/run_meta.json"

for i in $(seq 0 37); do
  ep=$(printf 'ep_%06d' "$i")
  test -e "$TRAIN_ROOT/$ep" || ln -s "$REMOTE_RAW/$ep" "$TRAIN_ROOT/$ep"
done

test -e "$TEST_ROOT/ep_000038" || \
  ln -s "$REMOTE_RAW/ep_000038" "$TEST_ROOT/ep_000038"
```

检查：

```bash
find "$TRAIN_ROOT" -maxdepth 1 -type l | wc -l
find "$TEST_ROOT" -maxdepth 1 -type l | wc -l
```

预期分别是 39（38 episode + run_meta）和 2（1 episode + run_meta）。

重要：从此以后，value/VLA v1 都以 `TRAIN_ROOT` 为输入，不要直接从完整 `REMOTE_RAW` build VLA
dataset，否则 `ep_000038` 会泄漏进训练。

## 8. 生成 train target 和严格 test target

远程环境：

```bash
cd /home/hongming/repo/lerobot_value_pipeline

export PY=/home/hongming/miniconda3/envs/lerobot_recap/bin/python
export PYTHONPATH=$PWD/src
export VIEW_BASE=/datastore01/hongming/lerobot_raw_views/ming326
export TRAIN_ROOT=$VIEW_BASE/nero_egg_value_train_v1
export TEST_ROOT=$VIEW_BASE/nero_egg_value_test_ep38_v1

export ORDER='["Pick up the match.","move the right arm to ready.","Pick up the matchbox.","move the left arm to ready.","Strike the match and light the candle.","Return to the home position."]'
```

先对 train view dry-run：

```bash
$PY -m lerobot.scripts.lerobot_value_prepare_targets \
  --root "$TRAIN_ROOT" \
  --mode both \
  --num_bins 256 \
  --global_scale p95 \
  --subtask_scale p95 \
  --elapsed_aux false \
  --subtask_order_json "$ORDER" \
  --require_all_subtasks true \
  --require_single_segment_per_subtask true \
  --require_success_only true \
  --dry_run
```

确认无 unlabeled、缺失 subtask、重复 segment 或顺序错误后，去掉 `--dry_run` 正式执行。

严格 test 必须使用 train root 算出的相同 scale，不能在单条 test episode 上重新计算 p95：

```bash
export TRAIN_META=$TRAIN_ROOT/value_function_meta.json

export GLOBAL_SCALE=$($PY -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["global_scale"]["frames"])' \
  "$TRAIN_META")

export SUBTASK_SCALES=$($PY -c \
  'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["subtask_scale"]["frames_by_subtask"]))' \
  "$TRAIN_META")

printf 'GLOBAL_SCALE=%s\nSUBTASK_SCALES=%s\n' "$GLOBAL_SCALE" "$SUBTASK_SCALES"
```

准备 test target：

```bash
$PY -m lerobot.scripts.lerobot_value_prepare_targets \
  --root "$TEST_ROOT" \
  --mode both \
  --num_bins 256 \
  --global_scale manual \
  --global_scale_frames "$GLOBAL_SCALE" \
  --subtask_scale manual \
  --subtask_scale_frames_json "$SUBTASK_SCALES" \
  --elapsed_aux false \
  --subtask_order_json "$ORDER" \
  --require_all_subtasks true \
  --require_single_segment_per_subtask true \
  --require_success_only true
```

train target 的 clip rate 建议：

```text
< 5%       理想
5% - 10%   可以先训练，但要看 UI
> 10%      建议把相应 scale 改成 max 后重新生成 train/test target
```

## 9. Value model：单卡 smoke、正式训练和 held-out test

当前 value trainer 不使用 Accelerate/DDP，所以只选一张空闲 H20。先查看：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

选择一张卡，例如：

```bash
export CUDA_VISIBLE_DEVICES=2
```

`CUDA_VISIBLE_DEVICES` 后，程序中的 `cuda:0` 就是所选物理卡。

远程变量：

```bash
export PI0_INIT=/datastore01/hongming/lerobot_outputs/pi0_strike_match_3_relative_bs192_20260630_172828/checkpoints/012000/pretrained_model
export VALUE_OUT=/datastore01/hongming/lerobot_outputs/value_nero_egg_subtask_pi0_v1
```

### 9.1 两步 smoke

```bash
$PY -m lerobot.scripts.lerobot_train_value_function \
  --root "$TRAIN_ROOT" \
  --output_dir "${VALUE_OUT}_smoke" \
  --mode subtask \
  --backbone_type pi0 \
  --pretrained_path "$PI0_INIT" \
  --local_files_only true \
  --precision bfloat16 \
  --image_keys \
    observation.images.left_wrist \
    observation.images.right_wrist \
    observation.images.third_person \
  --num_bins 256 \
  --use_elapsed_aux false \
  --freeze_vision_encoder true \
  --freeze_backbone true \
  --batch_size 1 \
  --num_workers 0 \
  --max_steps 2 \
  --augmentation false
```

### 9.2 正式 value 训练

```bash
$PY -m lerobot.scripts.lerobot_train_value_function \
  --root "$TRAIN_ROOT" \
  --output_dir "$VALUE_OUT" \
  --mode subtask \
  --backbone_type pi0 \
  --pretrained_path "$PI0_INIT" \
  --local_files_only true \
  --precision bfloat16 \
  --image_keys \
    observation.images.left_wrist \
    observation.images.right_wrist \
    observation.images.third_person \
  --num_bins 256 \
  --use_elapsed_aux false \
  --freeze_vision_encoder true \
  --freeze_backbone true \
  --val_episodes 0 10 20 30 \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 8 \
  --learning_rate 3e-5 \
  --augmentation true \
  --seed 42
```

H20 显存充足，可以在 smoke 后逐步增大 `batch_size`；不要在第一次运行就直接把 batch 拉到极限。

当前 value trainer 没有正式 `--resume` CLI。训练中断时保留 checkpoint，但第一版按新 output 重新训练；
若后续确实需要长时间 value 训练，再单独实现 resume，不在数据 pipeline 中临时绕过 provenance。

### 9.3 对 train 和严格 test 推理

```bash
export VALUE_CKPT=$VALUE_OUT/checkpoint.pt

$PY -m lerobot.scripts.lerobot_value_infer \
  --root "$TRAIN_ROOT" \
  --checkpoint "$VALUE_CKPT" \
  --mode subtask \
  --subtask_inference_path both \
  --batch_size 16 \
  --num_workers 8 \
  --device auto

$PY -m lerobot.scripts.lerobot_value_infer \
  --root "$TEST_ROOT" \
  --checkpoint "$VALUE_CKPT" \
  --mode subtask \
  --subtask_inference_path both \
  --batch_size 16 \
  --num_workers 8 \
  --device auto
```

如果 OOM，把 inference batch 依次降为 8、4；这只影响速度，不改变结果。

## 10. 通过 SSH tunnel 查看 held-out test value

在 fcloud 启动 UI：

```bash
cd /home/hongming/repo/lerobot_value_pipeline
export PY=/home/hongming/miniconda3/envs/lerobot_recap/bin/python
export PYTHONPATH=$PWD/src
export TEST_ROOT=/datastore01/hongming/lerobot_raw_views/ming326/nero_egg_value_test_ep38_v1

$PY -m lerobot.scripts.lerobot_value_viz \
  --root "$TEST_ROOT" \
  --chunk_size 50 \
  --host 127.0.0.1 \
  --port 8003 \
  --no-browser
```

本地另开终端建立 tunnel：

```bash
ssh -N -L 8003:127.0.0.1:8003 fcloud
```

本地浏览器访问：

```text
http://127.0.0.1:8003
```

在 `ep_000038` 上重点判断：

- subtask classifier 是否识别出正确顺序；
- `pred_smooth` 边界和 GT 边界差多少；
- `gt_conditioned` remaining value 是否在每段总体下降；
- 是否在停顿、恢复动作、抓取失败后恢复等位置出现合理变化；
- 是否只是根据视频时间匀速倒计时；
- value entropy/top1 confidence 是否在视觉模糊或边界附近恶化。

不要根据 test episode 反复改参数再把它仍称为 test。若看过 ep38 并据此调参，它就变成了开发集；
下一批数据应再固定一条全新的最终 test episode。

## 11. 正式 advantage、label 和 weight：只处理 TRAIN_ROOT

test episode 只用于 value 泛化检查，不生成第一版 VLA 训练 label/weight。

### 11.1 Advantage

```bash
$PY -m lerobot.scripts.lerobot_compute_advantage \
  --root "$TRAIN_ROOT" \
  --value_mode subtask \
  --value_source model_pred \
  --subtask_inference_path gt_conditioned \
  --chunk_size 50 \
  --boundary_transition_value 1.0 \
  --dry_run
```

确认后去掉 `--dry_run`。

`chunk_size=50` 必须和当前 PI0/PI0.5 policy checkpoint 一致。正式实验禁止使用 `gt` 或
`mock_pred` advantage。

### 11.2 Label

先 headless preview：

```bash
$PY -m lerobot.scripts.lerobot_advantage_labeler \
  --root "$TRAIN_ROOT" \
  --value_mode subtask \
  --top_percent 0.8 \
  --tie_policy exact_count \
  --export \
  --dry_run
```

`top_percent=0.8` 表示 80% 的 valid chunks 为 positive，不是 top 20%。

需要人工看排序时，在 fcloud 运行：

```bash
$PY -m lerobot.scripts.lerobot_advantage_labeler \
  --root "$TRAIN_ROOT" \
  --value_mode subtask \
  --top_percent 0.8 \
  --tie_policy exact_count \
  --host 127.0.0.1 \
  --port 8001 \
  --no-browser
```

本地 tunnel：

```bash
ssh -N -L 8001:127.0.0.1:8001 fcloud
```

UI 完成 override/export，或确认 headless 阈值后去掉 preview 命令的 `--dry_run` 正式 export。

### 11.3 Weight

```bash
$PY -m lerobot.scripts.lerobot_compute_advantage_weights \
  --root "$TRAIN_ROOT" \
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
  --dry_run
```

确认后去掉 `--dry_run`。

参数调整原则：

| 参数 | 默认 | 何时调整 |
|---|---:|---|
| `group_bin_width` | 0.1 | 小组大量少于 4 条时增到 0.2 |
| `q` | 0.8 | 越大越强调组内最优样本 |
| `tau` | 0.08 | 权重过激时增大，区分不足时减小 |
| `w_max` | 2.0 | 第一轮不要超过 2 |
| `negative_weight` | 1.0 | 第一轮保持 1 |
| condition dropout | 0.1 | 第一轮保持 0.1 |

当前语义：positive 权重可低于 1 或最高为 2；negative/dropout 为 1；ignore 为 0；weight 只作用于
flow-matching loss。

## 12. 在 fcloud 构建 VLA train dataset

只从 `TRAIN_ROOT` 构建，因此结果包含 38 条 episode，不含 `ep_000038`。

```bash
export DATASET_REPO_ID=ming326/nero_egg_value_train_v1
export DATASET_ROOT=/datastore01/hongming/lerobot/ming326/nero_egg_value_train_v1

$PY -m lerobot.scripts.lerobot_build_dataset \
  --runs "$TRAIN_ROOT" \
  --output_repo_id "$DATASET_REPO_ID" \
  --output_root "$DATASET_ROOT" \
  --video true \
  --vcodec libsvtav1 \
  --push_to_hub false \
  --dry_run true
```

dry-run schema 必须包含：

```text
subtask
subtask_progress
advantage_label_subtask
advantage_loss_weight_subtask
```

确认后去掉 `--dry_run true` 正式构建。

不要首次就使用 `--force true`。后续重建使用 `v2`、`v3` 新目录；保留旧 dataset 和对应 metadata，
确保实验可复现。

## 13. GPU 选择和 Accelerate 约定

训练前查看：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

选择 4 张卡示例：

```bash
export CUDA_VISIBLE_DEVICES=2,3,4,5
export NUM_PROCESSES=4
```

选择 2 张卡示例：

```bash
export CUDA_VISIBLE_DEVICES=2,3
export NUM_PROCESSES=2
```

设置后 Accelerate 看到的是逻辑卡 `0..NUM_PROCESSES-1`。不要同时把物理卡号再传给程序。

VLA 的 `--batch_size` 是每进程 batch：

```text
global batch = batch_size_per_gpu * NUM_PROCESSES
```

建议正式起点：

```text
4 GPU: 32/GPU -> global batch 128
2 GPU: 32/GPU -> global batch 64
```

不要为了机械复现旧的 global batch 192，直接在 2 卡时把 per-GPU batch 拉到 96。先从已经验证过的
per-GPU 32 开始，再按显存和吞吐逐步增加。

当前 38 条 VLA train episode 约 209,997 帧。相同 epoch 数对应的 optimizer steps 应按下式计算：

```text
steps ~= target_epochs * 209997 / global_batch
```

例如 8 个数据 epoch：

```text
4 GPU x 32: 约 13,125 steps
2 GPU x 32: 约 26,250 steps
```

比较 2 卡和 4 卡实验时，应比较相同数据 epoch/seen samples，而不只是相同 steps。

## 14. Milestone 12：PI0 和 PI0.5 真实多卡 2-step smoke

远程变量：

```bash
cd /home/hongming/repo/lerobot_value_pipeline

export PY=/home/hongming/miniconda3/envs/lerobot_recap/bin/python
export ACCELERATE=/home/hongming/miniconda3/envs/lerobot_recap/bin/accelerate
export PYTHONPATH=$PWD/src

export DATASET_REPO_ID=ming326/nero_egg_value_train_v1
export DATASET_ROOT=/datastore01/hongming/lerobot/ming326/nero_egg_value_train_v1

export PI0_INIT=/datastore01/hongming/lerobot_outputs/pi0_strike_match_3_relative_bs192_20260630_172828/checkpoints/012000/pretrained_model
export PI05_INIT=/datastore01/hongming/lerobot_outputs/pi05_strike_match_3_subtask_relative_bs192_20260706_200321/checkpoints/010000/pretrained_model
```

先选择本次使用的 2 或 4 张卡，并设置 `CUDA_VISIBLE_DEVICES`、`NUM_PROCESSES`。

### 14.1 PI0 smoke

```bash
export POLICY_INIT=$PI0_INIT
export VLA_SMOKE=/datastore01/hongming/lerobot_outputs/nero_egg_pi0_adv_smoke_v1

$ACCELERATE launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id "$DATASET_REPO_ID" \
  --dataset.root "$DATASET_ROOT" \
  --policy.path "$POLICY_INIT" \
  --policy.dtype bfloat16 \
  --policy.use_advantage_conditioning true \
  --policy.advantage_label_key advantage_label_subtask \
  --policy.advantage_loss_weight_key advantage_loss_weight_subtask \
  --policy.compile_model false \
  --use_advantage_weighting true \
  --advantage_label_key advantage_label_subtask \
  --advantage_loss_weight_key advantage_loss_weight_subtask \
  --advantage_condition_dropout_prob 0.1 \
  --batch_size 1 \
  --num_workers 0 \
  --steps 2 \
  --log_freq 1 \
  --save_freq 2 \
  --eval_freq 1000000 \
  --output_dir "$VLA_SMOKE" \
  --wandb.enable false
```

### 14.2 PI0.5 smoke

使用完全相同命令，只修改：

```bash
export POLICY_INIT=$PI05_INIT
export VLA_SMOKE=/datastore01/hongming/lerobot_outputs/nero_egg_pi05_adv_smoke_v1
```

两种 policy 都必须完成：

- dataset/DataLoader；
- 完整真实模型 forward；
- backward；
- 多 rank weighted mean；
- optimizer step；
- 第 2 step checkpoint 保存；
- loss 无 NaN/Inf。

两者都通过才算完成 Milestone 12 的核心放行条件。

## 15. 正式 VLA 实验

第一轮至少保留两个可比实验：

1. 普通 BC continuation：conditioning=false、weighting=false；
2. 完整 advantage：conditioning=true、weighting=true。

若资源允许，再增加：

3. conditioning only；
4. weighting only。

所有实验保持：

- 相同 `POLICY_INIT`；
- 相同 dataset v1；
- 相同 seed；
- 相同 global batch 或相同 seen samples；
- 相同训练数据 epoch；
- 不同 output directory。

以 4 卡、每卡 batch 32、约 8 epoch 为例，把 smoke 命令改为：

```text
batch_size=32
num_workers=8（每进程；若 I/O 过载改为 4）
steps=13125
save_freq=1000
log_freq=50
compile_model=true（先用 false 跑稳定，再单独测试 compile）
```

第一轮可继续使用 policy preset 的学习率 `2.5e-5`。如果从 4 卡切到 2 卡且 global batch 减半，可做
两种选择：

1. 为保持和旧实验最接近，学习率仍用 `2.5e-5`，重点监控 loss/grad norm；
2. 线性缩放到约 `1.25e-5`，作为更保守设置。

不要同时改变 GPU 数、学习率、weight 参数和 value mode，否则无法判断提升来自哪里。

模型选择不能只看离线 loss。每 1,000-2,000 step 做真实机器人 rollout，记录：

- 总成功率；
- 每个 subtask 成功率；
- 完成耗时；
- 卡住率；
- 点火失败、抓取失败、恢复动作等失败类型；
- 与普通 BC baseline 的配对比较。

## 16. 回传 checkpoint 到本地部署

本地部署只需要选中 checkpoint 下的完整 `pretrained_model` 目录，其中包括：

- `model.safetensors`；
- `config.json`；
- `train_config.json`；
- policy preprocessor/postprocessor 配置和 normalization 权重。

本地终端示例：

```bash
export REMOTE_CKPT=/datastore01/hongming/lerobot_outputs/nero_egg_pi05_adv_v1/checkpoints/013000/pretrained_model
export LOCAL_DEPLOY=/home/zenbot-robot/models/lerobot/nero_egg_pi05_adv_v1_013000

mkdir -p "$LOCAL_DEPLOY"

rsync -a --info=progress2 --partial \
  "fcloud:$REMOTE_CKPT/" \
  "$LOCAL_DEPLOY/"
```

传输后比较文件大小，必要时校验大权重 SHA256：

```bash
sha256sum "$LOCAL_DEPLOY/model.safetensors"
ssh fcloud "sha256sum '$REMOTE_CKPT/model.safetensors'"
```

本地部署前先做一次 policy load/单 batch inference smoke，再连接机器人。

不需要把远程构建的 100+ GB VLA dataset 传回本地；本地已有 raw 图片，训练 dataset 只在服务器保存。

## 17. 后续增加 episode 的版本化流程

人工标注是可复用资产。新增 episode 后：

1. 本地 annotation UI 只标新 episode；
2. Export 会保留旧 episode 已有 subtask/derived 列；
3. rsync 会只上传新增或变化文件；
4. `ep_000038` 固定保留为 v1 benchmark，不加入训练；
5. 新 episode 加入 train view；
6. 全量重跑 target/value inference/advantage/label/weight；
7. build `nero_egg_value_train_v2`；
8. 从上一轮最佳 VLA `pretrained_model` 初始化一个新训练 run。

可复用与必须重跑的内容：

| 内容 | 新增 episode 后 |
|---|---|
| raw 图片/state/action | 复用 |
| 旧 annotations/subtask | 复用 |
| 旧 episode 的人工边界 | 复用，除非标注规范变化 |
| GT target/p95 scale | 全量重算 |
| value prediction | 全量重新推理 |
| advantage/label/weight | 全量重算 |
| 旧 VLA checkpoint | 作为新 run 的 `--policy.path` |
| 旧 VLA optimizer | 不跨 dataset 版本 resume |

不要对改变了 dataset 的实验使用旧 run 的 `--resume true`，因为 resume 会恢复旧 dataset/config。
正确做法是：

```text
old_best/checkpoints/XXXXXX/pretrained_model
    -> new run 的 --policy.path
    -> 新 output directory
    -> 新 dataset v2
```

如果依据 `ep_000038` 的表现调整过 value 参数，它已经成为开发集。下一批采集时应再固定一个新的严格
test episode，并在查看其结果前冻结本轮参数。

这套迭代更准确地说是：

```text
成功示范采集
-> offline value/advantage labeling
-> advantage-conditioned behavior cloning
-> rollout 评估
-> 再采集成功示范
```

它具有强化学习的数据迭代结构，但当前不是接收失败轨迹和在线 reward update 的传统 RL。若以后加入
机器人自主 rollout 的失败/超时 episode，必须先重新设计 terminal/outcome target，不能直接混入当前
success-only pipeline。

## 18. 每轮实验检查清单

### 数据与 provenance

- [ ] 全部训练 episode 明确成功；
- [ ] train/test view 中没有 episode 重叠；
- [ ] canonical subtask 顺序一致；
- [ ] test scale 完全来自 train metadata；
- [ ] downstream stage 没有 stale provenance；
- [ ] VLA dataset 不含 `ep_000038`。

### Value

- [ ] value 2-step smoke 通过；
- [ ] val frame MAE、subtask accuracy、monotonic violation 已记录；
- [ ] p95 clip rate 可接受；
- [ ] `ep_000038` UI 已检查；
- [ ] 正式 advantage 使用 `model_pred/gt_conditioned`。

### Weight

- [ ] 每个 subtask 有 positive/negative 覆盖；
- [ ] 小 group 数量可接受；
- [ ] positive weight 无异常集中到 0.1 或 2.0；
- [ ] negative=1、ignore=0、dropout fallback=1。

### VLA

- [ ] PI0 真实多卡 2-step 通过；
- [ ] PI0.5 真实多卡 2-step 通过；
- [ ] baseline 和 advantage 实验只改变目标变量；
- [ ] 按 seen samples/epoch 比较 2 卡和 4 卡；
- [ ] checkpoint、git 状态、dataset 版本、GPU 列表均写入实验记录；
- [ ] 本地 checkpoint SHA256 与远程一致；
- [ ] 本地部署前完成 load/inference smoke。
