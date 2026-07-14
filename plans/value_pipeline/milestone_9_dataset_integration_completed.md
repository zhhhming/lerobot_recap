# Value Pipeline Milestone 9 Completion Record

日期：2026-07-13

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 推荐实施顺序中的第六阶段：
Milestone 9（raw -> LeRobotDataset 集成验证）。

Milestone 8 之前只验证了 builder 能在 `--dry_run` 中识别 label、group 和 weight schema；本阶段实际
创建本地 LeRobotDataset，并验证 `dataset[0]`、真实 DataLoader、pi0 policy preprocessor 和
`policy.action_delta_indices` action chunk 均能保留正确的 advantage 训练字段和 shape。

本阶段不实现 Milestone 10 的 advantage prompt conditioning，也不实现 Milestone 11 的 FM-only loss
weighting。它固定的是两者共同依赖的数据入口契约。

## 修改文件

- 修改 `src/lerobot/scripts/lerobot_build_dataset.py`
- 修改 `src/lerobot/processor/converters.py`
- 新增 `tests/scripts/test_value_extras_build_dataset.py`
- 新增 `tests/value_function/test_milestone_9_shadow_smoke.py`
- 新增 `plans/value_pipeline/validate_milestone_9.sh`
- 新增本完成记录

`src/lerobot/datasets/factory.py` 经真实 action chunk 测试确认现有行为已经满足要求，因此没有为了匹配
计划文件列表而增加无意义修改。

## Builder schema 和 provenance gate

`lerobot-build-dataset` 继续复用通用 extras 合并路径，没有为 value pipeline 增加专用 dataset writer。

新增行为：

1. 所有 episode 的 `extras.parquet` 不再只比较列名，而是比较完整 Arrow schema（列顺序、类型和
   nullability；忽略非结构 metadata）。同名 weight 在某个 episode 为 `float32`、另一个 episode 为
   `float64` 时会在创建 dataset 前报错。
2. feature include/exclude 过滤完成后，仅当最终选择了以下字段时才启用 value-pipeline gate：
   - `advantage_label_global|subtask`
   - `advantage_group_id_global|subtask`
   - `advantage_loss_weight_global|subtask`
3. label 要求对应 `advantage_labeling.{mode}` stage 存在且 current。
4. group/weight 要求对应 `advantage_weights.{mode}` stage 存在且 current。
5. gate 重新计算 stage input/output fingerprint、检查 stale dependency，并校验选中字段确实在该 stage
   的 `output_columns` manifest 中。
6. 选择 pipeline 字段但缺少 `value_function_meta.json` 时直接拒绝 build。
7. global/subtask 两种 mode 以及多个 raw run 分别校验，不能用某一个 run 的 metadata 替另一个 run
   背书。
8. synthetic artifact 允许用于接口 smoke，但 builder 会打印：

   ```text
   SYNTHETIC / NOT FOR EXPERIMENT
   ```

没有选择 label/group/weight 时，普通 raw extras build 不依赖 value pipeline metadata，避免影响通用
dataset builder。

## LeRobotDataset 和 action chunk 契约

实际 build 后已确认：

- `dataset[0]` 能读到 `advantage_label_global|subtask`。
- `dataset[0]` 能读到 `advantage_loss_weight_global|subtask`。
- `value_*`、advantage debug 和 bool 字段可以进入 dataset。
- `--exclude_features "value_*,advantage_global_is_valid"` 可以删除 debug 字段，同时保留 label/weight
  训练字段。
- `resolve_delta_timestamps()` 只为 `action` 应用 `policy.action_delta_indices`。
- DataLoader 中 `action` 为 `[B, chunk_size, action_dim]`。
- advantage label/weight 保持 start-frame scalar，不随 action chunk 展开。

## Policy preprocessor complementary data 和 shape

`batch_to_transition()` 现在会把以下字段放入 complementary data，并由 processor pipeline 原样保留：

- `advantage_label_*`
- `advantage_loss_weight_*`
- `advantage_condition_kept`

数值 weight 和 condition mask 使用统一的 per-sample shape contract：

- `[B]`：原样保留。
- `[B, 1]`：转换为 `[B]`。
- 0-D 单样本 scalar：转换为 `[1]`。
- 其他二维/高维 shape：明确报错。

这避免 weight `[B, 1]` 和 per-sample loss `[B]` 静默广播成 `[B, B]`。label 保持 string list，不做
数值 shape 转换。

本机当前 LeRobotDataset/Hugging Face backend 会把声明为 shape `(1,)` 的 extras scalar 读成 0-D，
因此默认 DataLoader 结果已经是 `[B]`；测试仍显式构造 `[B, 1]` 输入，覆盖其他 backend/旧 dataset
的兼容路径。

## 新增测试覆盖

`tests/scripts/test_value_extras_build_dataset.py` 覆盖：

- 从小型 raw run 真正 build 本地 LeRobotDataset，不是 `--dry_run`。
- `dataset[0]` label/weight 数值和类型。
- 真实 DataLoader。
- 真实 pi0 preprocessor（mock tokenizer 只隔离外部模型依赖）。
- action 按 `PI0Config.action_delta_indices` 展开为 chunk。
- label、weight 和手工注入的 condition mask 经过 preprocessor 后仍存在。
- `[B, 1] -> [B]` 和非法 `[B, 2]` 拒绝。
- include/exclude debug columns。
- 缺失 metadata 拒绝。
- 修改已记录 weight 后 stale fingerprint 拒绝。
- 跨 episode Arrow dtype 不一致拒绝。
- bool/list extras schema 映射回归。

`tests/value_function/test_milestone_9_shadow_smoke.py` 覆盖真实样例的只读 shadow 流程：

```text
targets -> mock predictions -> advantage -> labels -> weights
        -> build LeRobotDataset -> DataLoader/action chunk -> converter
```

shadow run 复制 `strike_match_3` 的前 2 个完整 episode 到 pytest `/tmp`：

- `ep_000000`: 885 frames
- `ep_000001`: 844 frames
- 合计：1,729 frames

build 时排除三路图片，使用真实 frames/extras、action/state schema 和完整 global/subtask pipeline 字段。
测试前后比较原始 `ep_000000/extras.parquet` bytes，确认原始样例没有被修改。

## 一键验收和结果

验收脚本：

```bash
plans/value_pipeline/validate_milestone_9.sh
```

脚本使用 `/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`，依次执行：

1. `py_compile`。
2. Milestone 9、subtask extras、processor converter 专项回归。
3. Milestone 0/1/1.5/6/7/8 核心、UI API 和 wheel package 回归。
4. Milestone 8 全 70 episode shadow pipeline。
5. Milestone 9 两 episode actual-build shadow pipeline。
6. `git diff --check`。

2026-07-13 实际结果：

```text
Milestone 9 / converter / dataset integration: 26 passed
existing value pipeline / API / wheel regression: 127 passed
real sample Milestone 8 + Milestone 9 shadow smoke: 2 passed
total: 155 passed
py_compile: passed
git diff --check: passed
```

UI/API/wheel 测试会在 `127.0.0.1` 创建临时 HTTP socket；受限 sandbox 内会得到
`PermissionError: Operation not permitted`。允许本机测试 socket 后完整脚本通过，这不是源码失败。

当前环境仍未安装 `ruff`，因此没有虚构 ruff 结果。

## 已知 warning 和剩余边界

- Hugging Face `datasets` 当前会对 shape `(1,)` 的 NumPy scalar extras 打印 NumPy 1.25
  deprecation warning；最终 dataset item、DataLoader shape 和数值均正确。本阶段没有扩大范围修改公共
  DatasetWriter。
- builder 在 build 时验证 raw provenance，但没有把完整 `value_function_meta.json` 复制到最终
  LeRobotDataset metadata；后续正式训练 artifact 管理如需跨目录审计，应单独设计 manifest 复制策略。
- synthetic label/weight 只允许接口 smoke；正式实验仍必须等待 Milestone 2/3/4 的 model prediction，
  然后重新运行 Milestone 6/7/8 并重新 build dataset。
- Milestone 10 才负责根据 label 生成 pi0/pi0.5 prompt 和 train-only condition mask。
- Milestone 11 才负责读取 `[B]` weight、dropout fallback、ignore 行为和 FM-only weighted mean。
