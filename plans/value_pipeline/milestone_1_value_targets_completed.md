# Value Pipeline Milestone 1 Completion Record

日期：2026-07-09

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 中的 Milestone 1：
从 raw dataset 和 subtask 标注生成 value function 训练所需的 GT target，
并写回每个 episode 的 `extras.parquet`。

## 修改文件

- 新增 `src/lerobot/value_function/targets.py`
- 新增 `src/lerobot/scripts/lerobot_value_prepare_targets.py`
- 新增 `tests/value_function/test_value_targets.py`
- 修改 `src/lerobot/value_function/schema.py`
- 修改 `pyproject.toml`

## 新增 CLI

注册了命令：

```bash
lerobot-value-prepare-targets
```

主要参数：

- `--root`
- `--mode global|subtask|both`
- `--num_bins`
- `--global_num_bins`
- `--subtask_num_bins`
- `--global_scale max|p95|manual`
- `--subtask_scale max|p95|manual`
- `--global_scale_frames`
- `--subtask_scale_frames_json`
- `--elapsed_aux true|false`
- `--allow_unlabeled error|skip|default_subtask`
- `--default_subtask`
- `--subtask_column`
- `--dry_run`

## 新增字段

global target：

- `value_global_remaining_frames_gt`
- `value_global_remaining_norm_gt`
- `value_global_remaining_norm_gt_is_clipped`
- `value_global_elapsed_frames_gt`
- `value_global_elapsed_norm_gt`

subtask target：

- `value_subtask_id_gt`
- `value_subtask_name_gt`
- `value_subtask_remaining_frames_gt`
- `value_subtask_remaining_norm_gt`
- `value_subtask_remaining_norm_gt_is_clipped`
- `value_subtask_elapsed_frames_gt`
- `value_subtask_elapsed_norm_gt`

## 行为定义

- 长度为 `10` 的 episode，global remaining 为 `[9, 8, ..., 0]`。
- global elapsed 为 `[0, 1, ..., 9]`。
- subtask segment `[0..3]` 的 remaining 为 `[3, 2, 1, 0]`。
- subtask elapsed 为 `[0, 1, 2, 3]`。
- normalized target 使用 `frames / scale`。
- 超过 `1.0` 的 norm target 会 clip 到 `1.0`，并写入 clipped mask。
- global scale：
  - `max` 使用所有 episode 的最大 `episode_len - 1`。
  - `p95` 使用所有 episode 的 `episode_len - 1` 的 p95。
  - `manual` 使用 `--global_scale_frames`。
- subtask scale：
  - `max` 使用该 subtask 所有 segment 的最大 `segment_len - 1`。
  - `p95` 使用该 subtask 所有 segment 的 `segment_len - 1` 的 p95。
  - `manual` 使用 `--subtask_scale_frames_json`。

## Subtask 顺序

`annotation_config.json` 中的 `subtasks` 只作为合法名称集合，不作为任务顺序来源。
原因是样例 raw run 中该文件的顺序是 UI/调色板顺序，不是实际任务顺序。

实际 subtask order 默认从 `extras.parquet` 中首次出现的非空 subtask label 推导。
随后仍会检查每个 episode 的 subtask id 单调不降；如果出现回退，会报错。

## Metadata

写入 `value_function_meta.json`，包含：

- `value_mode`
- `num_bins`
- `global_num_bins`
- `subtask_num_bins`
- `global_scale`
- `subtask_names`
- `subtask_scale`
- `elapsed_aux`
- `allow_unlabeled`
- `image_keys`
- `columns_written`
- `clip_summary`
- `created_at`

## 测试

新增测试覆盖：

- global remaining / elapsed。
- subtask remaining / elapsed。
- 从 annotation config 校验合法 label。
- 从 extras 首次出现顺序推导 subtask order。
- annotation config 顺序不同于实际任务顺序时不会误报。
- subtask 回退时报错。
- 未标注 frame 默认报错。
- `allow_unlabeled=skip` 写 sentinel id 和 NaN target。
- manual scale clip 和 metadata。
- `mode=global` 只写 global 字段。
- CLI `--dry_run` 不写 extras 和 metadata。
- `lerobot_build_dataset.py --dry_run` 能识别新增 value target 字段。

已通过：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest tests/value_function/test_value_targets.py -q
```

结果：`12 passed`

已通过相关回归：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest tests/value_function/test_raw_value_io.py \
  tests/value_function/test_value_targets.py \
  tests/scripts/test_subtask_progress_data_pipeline.py -q
```

结果：`25 passed`

## 真实样例 dry-run

只读运行：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -c "from pathlib import Path; from lerobot.value_function.targets import ValueTargetConfig, prepare_value_targets; root=Path('/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3'); s=prepare_value_targets(ValueTargetConfig(root=root, mode='both', dry_run=True)); print({'episodes': len(s['episodes']), 'total_frames': s['total_frames'], 'subtasks': len(s['subtask_names']), 'subtask_names': s['subtask_names'], 'global_scale': s['global_scale']})"
```

结果：

```text
episodes: 70
total_frames: 53794
subtasks: 6
subtask_names:
  - Pick up the match.
  - move the right arm to ready.
  - Pick up the matchbox.
  - move the left arm to ready.
  - Strike the match and light the candle.
  - Return to the home position.
global_scale:
  strategy: p95
  frames: 842.0999755859375
```

## 环境说明

直接运行 pytest 时，当前系统会自动加载 ROS 的 pytest 插件，插件依赖缺失：

```text
ModuleNotFoundError: No module named 'lark'
```

因此测试使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 禁用外部 pytest 插件自动加载。

`lerobot-main` 环境当前没有安装 ruff，未运行 ruff。

## 剩余风险

- 当前只生成 GT target，不训练 value model。
- `allow_unlabeled=skip` 会写 `value_subtask_id_gt=-1` 和 NaN target；后续训练 dataset
  需要显式跳过这类 frame。
- 当前 subtask order 由数据首次出现顺序推导。如果某个 run 的前几个 episode 缺少中间
  subtask，但后续 episode 包含它，仍会得到一个全局首次出现顺序，并用单调校验发现回退。
- p95 scale 会 clip 长尾 target；clip rate 已写 metadata，后续如 clip rate 偏高应改用
  `max` 或 manual scale。

## 2026-07-10 review remediation（完成于 2026-07-13）

根据 `completed_milestones_improvement_plan.md` 对 Milestone 1 进行了严格数据契约、准确统计和
synthetic mock prediction 返修。本节保留原始完成历史，仅追加返修后的行为和验证结果。

### 旧行为与问题

- canonical subtask order 通过跨全 run 的首次出现顺序拼接；第一批 episode 缺少 subtask 时可能产生
  错误的全局 order。
- 只校验 subtask id 不回退，没有要求每个 episode 包含完整集合，也没有拒绝同一 subtask 的多个
  segment。
- global/subtask aggregate clip rate 是 episode rate 的算术平均，短 episode 与长 episode 权重相同。
- `global_num_bins or num_bins` 会把显式 `0` 静默替换为默认值；manual subtask scale 的多余 key
  不会报错。
- `mode=both` 后重跑 `mode=global` 会保留旧 subtask 列，但 metadata 没有明确标记它们已不是当前
  target stage 的有效输出。
- 没有真实 value model 时只有 GT，centered advantage 恒为零，无法验证后续排序数据流。

### 修改文件

- `src/lerobot/value_function/targets.py`
- `src/lerobot/value_function/mock_predictions.py`（新增）
- `src/lerobot/value_function/raw_io.py`
- `src/lerobot/value_function/schema.py`
- `src/lerobot/scripts/lerobot_value_prepare_targets.py`
- `src/lerobot/scripts/lerobot_value_mock_predictions.py`（新增）
- `tests/value_function/test_value_targets.py`
- `tests/value_function/test_mock_predictions.py`（新增）
- `pyproject.toml`

### 严格 subtask contract

`ValueTargetConfig` 和 CLI 新增默认开启的配置：

```text
require_all_subtasks=true
require_single_segment_per_subtask=true
require_success_only=true
```

canonical order 来源优先级改为：

1. config `subtask_order` / CLI `--subtask_order_json` 显式顺序；
2. annotation config 或全 run labels 确定完整 subtask 集合；
3. 第一条包含完整集合的 episode 的实际 segment 顺序。

不再跨 episode 拼接首次出现顺序。每条 episode 会压缩成带 frame boundary 的 segment sequence，
并与 canonical order 对照。严格模式会拒绝缺失、额外、重复 segment、顺序交换或回退。

错误包含 episode index、canonical/observed sequence、每个 segment 的起止 frame、missing、extra 和
repeated 详情。annotation palette 的排列仍不被当作执行顺序，只用于确定合法完整名称集合。显式
order 必须名称非空、无重复，并与完整 subtask 集合精确一致。

### Success-only contract

- 当前阶段仍只允许成功 episode；`require_success_only=false` 会明确拒绝，因为 failure/timeout/abort
  terminal target 尚未设计。
- 如果 episode `info.json` 包含 `success`、`successful`、`outcome` 或 `termination_reason`，会拒绝
  明确失败、超时或人工中止的 episode。
- 当前 raw 样例没有 outcome 字段，因此接受用户已确认的数据假设，并写入：

```text
all_episodes_successful: true
success_validation: declared_no_outcome_field
```

- 存在并通过 outcome 字段检查时记录 `success_validation=validated_outcome_fields`。

### Bin、scale 和 clip 统计

- `num_bins`、有效 `global_num_bins` 和有效 `subtask_num_bins` 全部要求 `>=2`；显式 0 不再回退。
- manual subtask scale keys 必须与 canonical subtask names 精确一致，missing/extra key 都会报错。
- aggregate clip rate 改为：

```text
sum(clipped_eligible_frames) / sum(eligible_frames)
```

- summary/metadata 同时保存 clipped frame count、eligible frame count 和 frame-weighted clip rate。
- subtask 统计按 name 独立累计；缺少 subtask 不再用 0 rate 稀释统计，严格模式会更早报结构错误。
- 为兼容现有消费者，原 `global_clip_rate` 和 `subtask_clip_rate_by_subtask` 字段仍保留，同时增加
  包含计数的 `global` 和 `subtask_by_name` 结构。

### Target columns manifest 和 provenance

metadata 新增：

```text
target_columns:
  active: [...]
  inactive_present: [...]
```

`active` 与 `stages.targets.output_columns` 是当前有效 target 字段；`inactive_present` 是 parquet 中仍然
存在、但不属于本次 mode 的旧 target 字段。旧列不会自动删除，但 downstream 不应把它们当作当前
stage 输出。

target stage 还保存 active output columns 的内容 fingerprint。绕过 pipeline API 直接修改 GT 输出后，
provenance check 会报 `outputs changed`；target 重跑会递归标记已有 advantage/mock stage stale。

### Synthetic mock prediction

新增 CLI：

```bash
lerobot-value-mock-predictions \
  --root /path/to/raw/run \
  --mode both \
  --seed 42 \
  --noise_std_frames 3.0 \
  --temporal_smoothing_sigma_frames 0
```

输出独立 frame-unit 字段：

- `value_global_remaining_frames_mock_pred`
- `value_subtask_remaining_frames_mock_pred`

不会覆盖 GT 或真实 model prediction canonical 列。实现行为：

- 对 GT frame-unit value 加固定 seed Gaussian noise；
- RNG stream 按 seed、episode index 和 value mode 分离；global-only 与 both 中的 global 结果一致；
- 可选 temporal Gaussian smoothing；subtask 只在各连续 GT segment 内平滑，不跨边界；
- NaN/unlabeled frame 保持 NaN；
- `noise_std_frames=0` 且不平滑时逐值恢复 GT；
- seed、noise sigma 和 smoothing sigma 必须非负。

mock generation 要求 target stage 存在、非 stale、所需 GT 列属于 active manifest，且当前 GT output
fingerprint 与 target provenance 一致。

metadata 明确保存：

```text
prediction_source: mock_pred
generator: synthetic_gt_gaussian_noise
synthetic: true
experiment_eligible: false
warning: SYNTHETIC / NOT FOR EXPERIMENT
seed: ...
noise_std_frames: ...
temporal_smoothing_sigma_frames: ...
source_gt_columns: [...]
source_gt_fingerprint: ...
```

mock stage 依赖 target stage；target 重跑后旧 mock stage 会被标记 stale。正式 label/weight export 的
synthetic gate 仍由 Milestone 7 返修完成。

### 新增测试覆盖

- 完整、同序、单 segment subtask 通过。
- 缺失、重复 segment、交换顺序、未知 label 均明确报错。
- 第一条 episode 缺 subtask、后续 episode 完整时，从完整 episode 得到 order，再报告前者缺失。
- annotation palette 顺序与执行顺序不同仍通过；显式 order 优先。
- bin count 0、1、负数和 manual scale missing/extra keys 均报错。
- global 和 subtask aggregate clip rate 使用 frame count 加权。
- 明确 failure marker 拒绝；无 outcome marker 写 declared assumption。
- both -> global 重跑后旧 subtask target 列进入 inactive manifest。
- mock 相同 seed bitwise equal，不同 seed 不同，zero noise 恢复 GT。
- global RNG stream 不受 mode 影响；subtask smoothing 不跨 boundary。
- mock 不覆盖 model prediction 列。
- target inactive mode、外部篡改 GT、负 seed/sigma 均拒绝。
- target 重跑将 mock stage 标记 stale。
- mock dry-run、CLI、metadata 和 dataset builder schema 回归通过。

### 实际验证

专项测试：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest \
  tests/value_function/test_value_targets.py \
  tests/value_function/test_mock_predictions.py -q
```

结果：

```text
48 passed in 10.03s
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
87 passed in 10.88s
```

### 真实样例只读 strict dry-run

对 `/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3` 执行严格 `mode=both`
dry-run，没有写回样例数据。结果：

```text
episodes: 70
total_frames: 53,794
subtasks: 6
require_all_subtasks: true
require_single_segment_per_subtask: true
require_success_only: true
success_validation: declared_no_outcome_field
```

canonical order：

1. `Pick up the match.`
2. `move the right arm to ready.`
3. `Pick up the matchbox.`
4. `move the left arm to ready.`
5. `Strike the match and light the candle.`
6. `Return to the home position.`

所有 70 条 episode 均通过完整集合、固定顺序和单 segment 校验。自动测试没有写死这些真实样例数字。

### 迁移影响和剩余边界

- 默认 strict contract 会拒绝旧版本曾接受的缺失或重复 subtask 数据；正式第一阶段数据必须保持默认
  strict 配置。
- `allow_unlabeled=skip` 如果导致同一 subtask 被切成多个 segment，默认会被 single-segment contract
  拒绝；旧式调试流程需显式关闭该约束。
- mock prediction 只能用于 synthetic smoke，不得用于正式实验。
- 本返修没有修改 advantage 公式、paired inference path 或 label export gate；它们分别属于后续
  Milestone 6 和 7。
