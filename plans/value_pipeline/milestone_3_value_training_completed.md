# Value Pipeline Milestone 3 Completion Record

日期：2026-07-14

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 推荐实施顺序中的第十阶段：
Milestone 3（独立 value function 训练流程）。

上一阶段 Milestone 2 已固定 PI0/PI0.5 value model、distributional heads、two-hot loss、state fusion 和
checkpoint payload。本阶段在该接口上实现 raw run dataset adapter、多个 root 契约、episode-level
train/validation split、轻量多相机 augmentation、训练/验证指标、optimizer/scheduler、canonical
checkpoint 和训练 CLI。

本阶段没有实现 Milestone 4 的整库 value inference、单调 Viterbi、prediction extras 列或 raw dataset
写回，也没有用当前训练 smoke 产出正式 advantage/label/weight。正式 model prediction 仍需下一阶段完成。

## 修改文件

新增：

- `src/lerobot/value_function/dataset.py`
- `src/lerobot/value_function/training.py`
- `src/lerobot/scripts/lerobot_train_value_function.py`
- `tests/value_function/test_value_dataset.py`
- `tests/value_function/test_train_value_smoke.py`
- `tests/value_function/test_value_train_real_sample.py`
- `plans/value_pipeline/validate_milestone_3.sh`
- 本完成记录

修改：

- `src/lerobot/value_function/__init__.py`
- `pyproject.toml`

没有修改旧的 `src/lerobot/rl/algorithms/recap_train_value_network_pi0.py`。旧脚本围绕已构建
LeRobotDataset、单 global `[-1,0]` target 和旧 RECAP network；本阶段直接使用 raw run 中当前有效的
GT target columns 和 Milestone 2 model contract，避免两套 target/model 语义混合。

## Raw value dataset adapter

新增 `RawValueFrameDataset`，直接读取：

- run-level `run_meta.json` 和 `value_function_meta.json`；
- per-episode `frames.parquet`、`extras.parquet`；
- configured camera PNG；
- `observation.state`；
- global/subtask remaining、elapsed、clip mask 和 subtask id GT columns。

每次构建 dataset 都调用 target stage provenance/fingerprint 检查，并要求所需列出现在当前 target stage 的
`output_columns` manifest。即使旧 subtask columns 仍残留在 `extras.parquet`，global-only target 重跑后也
不能被 subtask training 误用。

图片按需加载为 RGB `float32 [0,1] CHW`。缺图、frame/extras 长度不一致、state shape/dtype 非法、target
列缺失或 GT subtask id 越界都会在训练前明确报错。dataset 另外返回 root/episode/frame id、global/subtask
frame scale、subtask progress 和 clip mask，供 frame-unit metrics、分组方差和单调性统计使用。

## 多 root compatibility contract

多个 raw root 默认严格要求以下项目一致：

- robot type 和 fps；
- 每个 configured image key 的 dtype/shape/names；
- state key、dtype/shape/names；
- target bin count；
- global scale；
- canonical subtask order；
- 每个 subtask 的 frame scale。

每个 root 还必须声明 `all_episodes_successful=true`，且 current target stage 不 stale。第一版不按整数 id
静默合并不同 subtask order，也不提供隐式 remap；不兼容时错误同时列出两个 root 和冲突值。

## Episode split 和 state normalization

默认用固定 seed 按 episode 做 90/10 split，不允许同一 episode 的 frame 同时进入 train 和 validation。
支持显式 `--val_episodes`：单 root 使用 episode id，多 root 使用
`ROOT_INDEX:EPISODE_INDEX`，从而避免不同 root 的同名 episode 冲突。

state mean/std 只从 train split 流式计算，validation frames 不参与 normalization statistics。零方差维度
使用 `1e-6` 下限；结果同时写入 model buffers、checkpoint 和训练 metadata。

## Image augmentation

训练默认启用轻量 augmentation：

- random resized crop，默认 scale `[0.9,1.0]`；
- 小角度 rotation，默认 `3` 度；
- brightness/contrast/saturation jitter，默认幅度 `0.1`。

Gaussian blur、random grayscale 和 additive Gaussian noise 可配置，默认关闭。相同分辨率的多路相机在
同一 frame 上共享一次采样的变换参数；不同分辨率相机按 shape 分组，不会因 `torch.stack` shape 不同而
无法训练。validation 永远不执行随机 augmentation。

## 训练 CLI 和循环

`pyproject.toml` 新增入口：

```text
lerobot-train-value-function
```

CLI 支持一个或多个 `--root`、global/subtask/both、PI0/PI0.5/vision-only、image keys、bin count、VLM
layer 数、freeze 策略、state/elapsed 开关、loss weights、显式 validation episodes、augmentation、optimizer
和 device 配置。

训练循环使用：

- 只包含 `requires_grad=true` parameters 的 AdamW；
- warmup + cosine decay scheduler；
- configurable gradient clipping；
- global `--max_steps` quick smoke；
- CPU、CUDA、MPS 或 auto device。

`elapsed_loss_weight > 0` 时强制要求 `use_elapsed_aux=true`，避免用户设置非零 loss weight 但模型没有
elapsed head。raw target bin metadata 与 model bin config 不一致时也在模型构建前报错。

推荐命令形态：

```bash
lerobot-train-value-function \
  --root /path/to/raw/run \
  --mode subtask \
  --image_keys observation.images.left_wrist observation.images.right_wrist observation.images.third_person \
  --pretrained_path /path/to/pi0/model.safetensors \
  --num_bins 256 \
  --output_dir outputs/value/task_subtask_pi0
```

## 训练和验证指标

每轮写一条 `train_metrics.jsonl`，包含：

- total/component CE losses；
- global/subtask decoded normalized MAE；
- 使用 per-root global scale 或 per-frame subtask scale 复原预测值后，与未截断 `*_frames_gt` 对比的
  frame MAE；p95 clipped 样本不会用 clipped norm GT 低估长尾误差；
- subtask classification accuracy；
- elapsed normalized MAE（启用时）；
- 同一 GT subtask/progress bin 内 selected-head prediction variance；
- global episode 内、subtask GT segment 内相邻 frame value 单调性违规率；
- global 和每个 canonical subtask 的实际 frame-weighted target clip rate。

空 comparison/group 使用 JSON `null`，不写 NaN/Inf，因此 metadata 和 JSONL 可被严格 JSON reader 读取。

## Checkpoint 和输出契约

输出目录固定包含：

```text
checkpoint.pt
config.json
value_function_meta.json
train_metrics.jsonl
```

本阶段选择单一 canonical `checkpoint.pt`，每轮 validation 后用同目录临时文件原子覆盖。它代表最新完成
训练轮次，而不是一个含糊的 best/last 别名。checkpoint 包含：

- Milestone 2 `model_config`、完整 model state 和 state normalization；
- normalized training config；
- raw roots、image/state schema、subtask order、scales 和 target stage fingerprints；
- train/validation episode split；
- epoch/global step 和最新 metrics；
- optimizer 和 scheduler state。

新增 `load_value_function_checkpoint()` 使用 `load_pretrained=false` 重建架构后 strict load checkpoint，
不会再次读取 16 GB policy checkpoint。测试已验证 config、state stats 和 global/subtask output shape round-trip。

## 测试覆盖

专项测试覆盖：

- raw PNG/state/target/metadata batch 构建；
- episode split 无 frame 泄漏；
- state stats 只来自 train episodes；
- compatible multi-root 合并以及 fps contract mismatch；
- 相同分辨率多相机共享 augmentation；
- target metadata 缺失和 inactive leftover target columns 拒绝；
- tiny backbone CPU 2-step training；
- 四个固定输出文件和严格 JSON metrics；
- frame/norm MAE、subtask accuracy、clip/monotonic metrics 存在；
- p95 clipped tail 的 frame MAE 使用未截断 frame GT；
- checkpoint strict reload、state stats 和 output shape；
- model/target bin mismatch 拒绝；
- 真实 raw sample shadow run -> target preparation -> DataLoader -> local PI0 forward。

真实样例测试只复制一条 episode 的 parquet/metadata 到 pytest 临时目录并 symlink 原始图片；target 写入仅
发生在 shadow run，未修改原始 `strike_match_3`。

## 一键验收和实际结果

验收脚本：

```bash
plans/value_pipeline/validate_milestone_3.sh
```

支持环境变量：

```text
LEROBOT_PYTHON
LEROBOT_RAW_RUN
LEROBOT_PI0_CHECKPOINT
```

实际结果：

```text
Milestone 3 dataset + CPU 2-step smoke:
10 passed in 2.02s

Milestone 2 config/shape/loss/checkpoint regression:
23 passed in 1.63s

完整 value pipeline + subtask data pipeline regression:
165 passed, 7 skipped in 19.50s

真实 strike_match_3 raw DataLoader + PI0 checkpoint forward:
1 passed in 4.98s

CLI --help: passed
py_compile: passed
git diff --check: passed
```

完整回归在受限 sandbox 首次执行时，7 个已有 UI/API/package tests 因禁止创建 localhost socket 得到
`PermissionError: Operation not permitted`；同一命令在允许 loopback socket 的验收环境中全部通过，最终
结果为上面的 `165 passed, 7 skipped`。这些 7 个 skipped 是需要显式真实 checkpoint/smoke 开关的测试。

当前环境没有安装 `ruff`（`No module named ruff`），因此未报告 ruff 结果；已执行 py_compile、专项
pytest、完整回归、真实模型 forward、CLI help 和 `git diff --check`。

## 剩余边界和下一阶段

- 本阶段验证训练机制和真实 raw/model batch 接口，不要求 CPU 2-step smoke 收敛，也没有执行正式长时训练。
- `checkpoint.pt` 已提供 Milestone 4 所需模型、scale、subtask order、state normalization 和 data contract；
  下一阶段应在推理前再次严格校验 checkpoint/raw run compatibility。
- Milestone 4 需要实现 batched inference、distribution expectation debug fields、GT-conditioned 与
  pred-smooth paired head selection、单调 Viterbi、atomic extras 写回及 inference provenance。
- 完成真实 value training 和 Milestone 4 inference 前，不得把 synthetic mock label/weight 当作正式实验
  artifact。正式预测生成后仍需重新运行 Milestone 6/7/8。
