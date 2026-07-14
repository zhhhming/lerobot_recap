# Value Pipeline Milestone 6 Completion Record

日期：2026-07-09

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 实施顺序中的第三阶段：
Milestone 6，基于已写入 raw dataset `extras.parquet` 的 frame-unit value 列，
为每个 frame/action chunk 计算 advantage 并写回 raw dataset。

按照总计划第 6 节，本阶段先使用 GT value 跑通 advantage 公式和写回路径，
不依赖 value model 训练或推理。

## 修改文件

- 新增 `src/lerobot/value_function/advantage.py`
- 新增 `src/lerobot/scripts/lerobot_compute_advantage.py`
- 新增 `tests/value_function/test_advantage.py`
- 修改 `src/lerobot/value_function/schema.py`
- 修改 `pyproject.toml`

## 新增 CLI

注册了命令：

```bash
lerobot-compute-advantage
```

主要参数：

- `--root`
- `--value_mode global|subtask`
- `--value_source gt|pred`
- `--chunk_size`
- `--subtask_source gt|pred_smooth`
- `--boundary_bonus`
- `--dry_run`

## 新增字段

global advantage：

- `advantage_global_chunk`
- `advantage_global_valid_horizon`
- `advantage_global_is_valid`
- `advantage_global_start_value`
- `advantage_global_end_value`

subtask advantage：

- `advantage_subtask_chunk`
- `advantage_subtask_valid_horizon`
- `advantage_subtask_is_valid`
- `advantage_subtask_start_value`
- `advantage_subtask_end_value`

## 行为定义

global value 使用 frame-unit remaining value：

```text
valid_horizon = min(chunk_size, episode_len - 1 - frame_index)
advantage = V_remaining_start - V_remaining_end - valid_horizon
```

- episode 最后一帧 `valid_horizon=0`，`advantage_is_valid=false`。
- 靠近 episode 末尾时按真实剩余 transition 数截断，不把 padded last frame 算作真实推进。
- `--value_source gt` 读取 `value_global_remaining_frames_gt`。
- `--value_source pred` 读取 `value_global_remaining_frames_pred`。

subtask value 使用同一 subtask 内的相对差分：

```text
chunk [t, t+h] 按 subtask 边界切成同-subtask segment
segment_progress = V_subtask_remaining(a) - V_subtask_remaining(b)
segment_expected = b - a
advantage = sum(segment_progress - segment_expected) + boundary_bonus * num_crossings
```

- 默认 `boundary_bonus=0.0`。
- 默认 `--subtask_source gt` 读取 `value_subtask_id_gt` 作为边界。
- `--subtask_source pred_smooth` 读取 `value_subtask_id_pred_smooth`。
- 不使用跨 subtask 的绝对累计尺度。
- `advantage_subtask_start_value` 和 `advantage_subtask_end_value` 是参与计算的 segment start/end
  frame-unit value 之和，仅作为 debug 列；跨 subtask 时不表示单个可直接比较的全局 value。
- subtask id 为 `-1` 或 value 为 NaN 的 chunk 标为 invalid，advantage 写 `0.0`。

## Metadata

写回时会更新 `value_function_meta.json`：

- 在 `advantage.global` 或 `advantage.subtask` 下记录：
  - `value_source`
  - `chunk_size`
  - `subtask_source`
  - `boundary_bonus`
  - `columns_written`
  - valid/invalid chunk summary
- 同时把新增 advantage 字段并入 top-level `columns_written`。

如果 raw run 尚未有 `value_function_meta.json`，会写入一个包含 advantage 信息的最小 metadata。

## 测试

新增测试覆盖：

- 线性理想 episode：每帧推进 1 时 global advantage 为 `0`。
- 卡住片段：end value 没有下降足够时 advantage 为负。
- episode 末尾 padding：`valid_horizon` 正确截断。
- subtask 跨边界 chunk：按同-subtask segment 拆分计算。
- `boundary_bonus` 可配置并影响跨边界 chunk。
- 未标注 subtask id `-1` 的 chunk 标为 invalid。
- CLI `--dry_run` 不写回 advantage 列。
- `lerobot_build_dataset.py --dry_run` 能识别新增 advantage extras 列。

已通过：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest tests/value_function/test_advantage.py -q
```

结果：`8 passed`

已通过相关回归：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest \
  tests/value_function/test_raw_value_io.py \
  tests/value_function/test_value_targets.py \
  tests/value_function/test_advantage.py \
  tests/scripts/test_subtask_progress_data_pipeline.py -q
```

结果：`33 passed`

## 使用示例

用 GT global value 计算 action chunk advantage：

```bash
lerobot-compute-advantage \
  --root /path/to/raw/run \
  --value_mode global \
  --value_source gt \
  --chunk_size 50
```

用 GT subtask value 和 GT subtask 边界计算：

```bash
lerobot-compute-advantage \
  --root /path/to/raw/run \
  --value_mode subtask \
  --value_source gt \
  --subtask_source gt \
  --chunk_size 50 \
  --boundary_bonus 0.0
```

## 剩余风险

- 当前只计算 advantage，不做 positive/negative label，也不生成 loss weight。
- subtask 跨边界时 start/end debug value 是 segment value 之和，不能作为跨 subtask 绝对 value 解释。
- `--value_source pred` 路径已按列名支持，但需要后续 Milestone 4 写入 prediction 列后做真实数据验证。
- 如果使用 `allow_unlabeled=skip` 生成了 `value_subtask_id_gt=-1`，相关 chunk 会被标为 invalid；
  后续 label/weight 阶段需要决定 invalid chunk 是 `ignore` 还是过滤。

## 2026-07-10 review remediation（完成于 2026-07-13）

根据 `completed_milestones_improvement_plan.md` 对 Milestone 6 进行了 boundary-transition 公式、paired
subtask path、value source 和 provenance 返修。本节保留原始完成历史，只追加新行为和验证结果。

### 旧行为与问题

- `value_source=pred` 无法区分 mock prediction 和真实 model prediction。
- `subtask_source` 与 value column 可以独立组合，允许 GT-head value 配 smooth boundary 等不一致路径。
- 旧公式只减同-subtask horizon，再额外加 `boundary_bonus`；参数含义容易被误解为 centered
  advantage 之后的额外奖励。
- 默认 `boundary_bonus=0`，但没有显式记录真实 boundary transition 的推进量。
- 缺少 crossing、within-subtask horizon 和 boundary progress debug 字段。
- NaN、unknown id 等 invalid chunk 没有原因统计。
- high-level advantage 可以在缺少 source stage provenance 时直接写回，无法区分 stale 或 synthetic
  输入。

### 修改文件

- `src/lerobot/value_function/advantage.py`
- `src/lerobot/scripts/lerobot_compute_advantage.py`
- `src/lerobot/value_function/schema.py`
- `tests/value_function/test_advantage.py`
- `plans/value_pipeline/milestone_6_advantage_completed.md`

### Value source 和 paired inference path

CLI/config 的 value source 改为：

```text
gt | mock_pred | model_pred
```

subtask 的两个独立选择被单一 paired path 替代：

```text
subtask_inference_path = gt_conditioned | pred_smooth
```

字段映射：

| mode/source/path | value column | boundary column |
|---|---|---|
| global/gt | `value_global_remaining_frames_gt` | - |
| global/mock_pred | `value_global_remaining_frames_mock_pred` | - |
| global/model_pred | `value_global_remaining_frames_pred` | - |
| subtask/gt/gt_conditioned | `value_subtask_remaining_frames_gt` | `value_subtask_id_gt` |
| subtask/mock_pred/gt_conditioned | `value_subtask_remaining_frames_mock_pred` | `value_subtask_id_gt` |
| subtask/model_pred/gt_conditioned | `value_subtask_remaining_frames_pred_gt_head` | `value_subtask_id_gt` |
| subtask/model_pred/pred_smooth | `value_subtask_remaining_frames_pred_smooth_head` | `value_subtask_id_pred_smooth` |

新增 canonical model prediction 字段常量：

- `value_subtask_remaining_norm_pred_gt_head`
- `value_subtask_remaining_frames_pred_gt_head`
- `value_subtask_remaining_norm_pred_smooth_head`
- `value_subtask_remaining_frames_pred_smooth_head`

原模糊的 `value_subtask_remaining_*_pred` 常量暂时保留兼容，但 advantage 不再读取。`gt` 和
`mock_pred` 只允许 `gt_conditioned`；model path 缺少对应 value head 或 boundary 时在写回前报 paired
column mismatch。

### Boundary transition v2 公式

global 保持：

```text
advantage = V_start - V_end - valid_horizon
```

subtask 改为：

```text
within_subtask_progress = sum(V_k(a_j) - V_k(b_j))
boundary_progress = boundary_transition_value * num_crossings
total_progress = within_subtask_progress + boundary_progress
advantage = total_progress - valid_horizon
```

默认 `boundary_transition_value=1.0`。理想 chunk 中：

```text
within_subtask_horizon + num_crossings = valid_horizon
```

同-subtask remaining 每个 transition 减 1，boundary transition 贡献 1，因此跨 0、1 或多个 boundary
的理想 chunk advantage 均为 0。

metadata 公式版本：

```text
global:  global_centered_v1
subtask: subtask_boundary_transition_v2
```

### boundary_bonus 迁移

正式参数改为：

```text
--boundary_transition_value
```

CLI/Python config 暂时保留 deprecated `boundary_bonus`。为了保持旧公式语义，转换为：

```text
boundary_transition_value = boundary_bonus + 1.0
```

因此旧 `boundary_bonus=0` 等价于新 `boundary_transition_value=1`。使用 alias 会发出
`FutureWarning`；新旧参数同时设置会拒绝执行。

### 新增 debug 字段和 invalid 统计

subtask extras 新增：

- `advantage_subtask_num_crossings`
- `advantage_subtask_within_subtask_horizon`
- `advantage_subtask_boundary_progress`

chunk 检查完整 value window，而不是只检查 endpoint。当前 invalid reason：

- `zero_horizon`
- `nonfinite_value`
- `unknown_subtask_id`

invalid chunk 写 `is_valid=false`、advantage 0；无法定义的 endpoint debug value 保持 NaN。每个 episode
和 aggregate summary/metadata 都保存 `invalid_reason_counts`。

subtask boundary path 会根据 metadata canonical `subtask_order` 校验 ID 范围、完整顺序、单调和单段。
负 unknown ID 不参与 order 压缩，但覆盖它的 chunk 标为 invalid；超范围正 ID、回退、重复 segment 或
遗漏 canonical subtask 会拒绝 episode。

### Provenance gate

high-level compute 和 dry-run 都要求 source stage 存在且 current：

- GT 依赖 `targets`；
- mock 依赖 `mock_predictions`；
- model prediction 依赖 `value_inference.global|subtask`；
- GT-conditioned mock/model subtask 还同时依赖 current `targets` boundary。

检查 prediction source、synthetic flag、active output columns 和 output fingerprint。source stale、GT
被外部修改、model inference stage 缺失或 prediction source 错误均拒绝。

advantage stage 保存自身 input/output fingerprint 和 source dependency fingerprints。source 语义：

| source | synthetic | experiment_eligible |
|---|---:|---:|
| gt | false | false |
| mock_pred | true | false |
| model_pred | false | true |

GT 是真实监督 target，不标 synthetic，但只允许公式 sanity check，不能作为正式排序实验输入。

### 新增测试覆盖

- global 线性、卡住和 tail valid horizon。
- subtask 理想 chunk 跨 0、1、2 个 boundary 均为 0。
- transition value 0 时按 crossing 数降低 advantage。
- within horizon、crossing 和 boundary progress debug 字段。
- frame-unit 公式不读取 normalized scale。
- deprecated bonus 的 `+1` 映射、warning 和双参数冲突。
- GT/mock + pred_smooth 路径拒绝。
- model GT-head + GT boundary 和 smooth-head + smooth boundary 分别通过。
- 两种 head/boundary mismatch 都拒绝。
- NaN 和 unknown ID invalid reason 统计。
- predicted ID 回退、重复/额外 ID 拒绝。
- stale mock、外部修改 GT、缺失 model stage、错误 prediction source 拒绝。
- target -> mock -> advantage synthetic smoke 可复现且产生非零 advantage。
- dry-run、metadata output fingerprint 和 dataset builder 新字段回归。

### 实际验证

专项测试：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest tests/value_function/test_advantage.py -q
```

结果：

```text
28 passed in 0.70s
```

完整相关回归：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest \
  tests/value_function/test_raw_value_io.py \
  tests/value_function/test_value_targets.py \
  tests/value_function/test_mock_predictions.py \
  tests/value_function/test_advantage.py \
  tests/value_function/test_advantage_labeling.py \
  tests/scripts/test_subtask_progress_data_pipeline.py -q
```

结果：

```text
107 passed in 3.09s
```

### 真实样例 shadow-run smoke

原始样例：

```text
/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3
```

只复制 `run_meta.json`、`annotation_config.json`、episode `info.json`、`frames.parquet` 和
`extras.parquet` 到 `/tmp/lerobot-value-shadow.npiWRg`，不复制图片。在 shadow run 上实际执行 strict
targets 和 mock prediction，再执行 global/subtask mock advantage dry-run。

结果：

```text
episodes: 70
frames: 53,794
canonical subtasks: 6
target success validation: declared_no_outcome_field
mock prediction source: mock_pred
mock synthetic: true

global mock advantage:
  valid: 53,724
  invalid: 70
  invalid reasons: zero_horizon=70
  experiment_eligible: false

subtask mock advantage:
  valid: 53,724
  invalid: 70
  invalid reasons: zero_horizon=70
  formula: subtask_boundary_transition_v2
  experiment_eligible: false
```

检查原始样例后确认：没有 `value_function_meta.json`，`extras.parquet` 仍只有 `subtask` 和
`subtask_progress`，没有 value/advantage pipeline 字段。

### 迁移影响和剩余边界

- 旧 CLI `--value_source pred` 必须改为 `model_pred`；mock smoke 使用 `mock_pred`。
- 旧 `--subtask_source` 必须改为 `--subtask_inference_path`。
- 旧 `boundary_bonus` 暂时兼容但应迁移到 boundary transition value；后续可删除 alias。
- model prediction 路径已经按 canonical schema 和 provenance 实现，但真实 value inference stage 仍需
  Milestone 4 产出 checkpoint/inference metadata 后做真实模型验证。
- 本返修不修改 positive/negative label、loss weight 或 UI；这些属于后续 milestone。
