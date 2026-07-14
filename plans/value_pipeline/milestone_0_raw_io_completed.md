# Value Pipeline Milestone 0 Completion Record

日期：2026-07-09

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 中的 Milestone 0：
定义 value pipeline 的 raw extras/schema 工具，固定 raw run 读取、`extras.parquet`
合并和 run-level metadata 写入行为。

## 修改文件

- 新增 `src/lerobot/value_function/__init__.py`
- 新增 `src/lerobot/value_function/schema.py`
- 新增 `src/lerobot/value_function/raw_io.py`
- 新增 `tests/value_function/test_raw_value_io.py`

## 新增能力

- 读取 raw run 的 `run_meta.json`。
- 枚举 `ep_XXXXXX` episode，并读取每个 episode 的 frame 数。
- 根据 run metadata 解析 image keys。
- 根据 image key 和 frame index 生成对应图片路径。
- 读取 episode 级 `frames.parquet` 和 `extras.parquet`。
- 向 episode 的 `extras.parquet` 合并新列：
  - 保留已有 `subtask`、`subtask_progress` 和其他 extras 列。
  - 同名列按原位置替换。
  - 新列追加到末尾。
  - 新增列长度必须等于 `frames.parquet` 行数。
- 对整个 raw run 批量合并 extras：
  - 要求每个 episode 都提供待写列。
  - 要求已有 extras schema 一致。
  - 写入前先构造并校验所有 episode 的最终 schema，避免部分写入后才发现 schema 不一致。
- 写入和读取 run-level `value_function_meta.json`。

## 涉及字段和文件名

集中定义在 `src/lerobot/value_function/schema.py`：

- `run_meta.json`
- `frames.parquet`
- `extras.parquet`
- `value_function_meta.json`
- global GT value 字段：
  - `value_global_remaining_frames_gt`
  - `value_global_remaining_norm_gt`
  - `value_global_remaining_norm_gt_is_clipped`
  - `value_global_elapsed_frames_gt`
  - `value_global_elapsed_norm_gt`
- subtask GT value 字段：
  - `value_subtask_id_gt`
  - `value_subtask_name_gt`
  - `value_subtask_remaining_frames_gt`
  - `value_subtask_remaining_norm_gt`
  - `value_subtask_remaining_norm_gt_is_clipped`
  - `value_subtask_elapsed_frames_gt`
  - `value_subtask_elapsed_norm_gt`
- prediction/debug 字段：
  - `value_global_remaining_norm_pred`
  - `value_global_remaining_frames_pred`
  - `value_subtask_id_pred`
  - `value_subtask_confidence`
  - `value_subtask_id_pred_smooth`
  - `value_subtask_name_pred_smooth`
  - `value_subtask_remaining_norm_pred`
  - `value_subtask_remaining_frames_pred`

## 测试

已通过：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest tests/value_function/test_raw_value_io.py -q
```

结果：`8 passed`

已通过回归：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest tests/scripts/test_subtask_progress_data_pipeline.py -q
```

结果：`5 passed`

合并运行也已通过：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest tests/value_function/test_raw_value_io.py \
  tests/scripts/test_subtask_progress_data_pipeline.py -q
```

结果：`13 passed`

## 环境说明

直接运行 pytest 时，当前系统会自动加载 ROS 的 pytest 插件，插件依赖缺失：

```text
ModuleNotFoundError: No module named 'lark'
```

因此测试使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 禁用外部 pytest 插件自动加载。

尝试运行 ruff：

```bash
/home/zenbot-robot/.conda/envs/lerobot-main/bin/python -m ruff check ...
```

当前环境没有安装 ruff：

```text
No module named ruff
```

## 剩余风险

- Milestone 0 只固定 raw IO 和 extras merge，不生成 value target。
- 新增字段类型由调用方提供的 pyarrow array 或 Python values 决定；后续 Milestone 1
  生成 GT target 时应显式使用稳定 dtype，例如 frame count 用 `int32` 或 `int64`，
  normalized value 用 `float32`。
- 当前不修改 `lerobot_build_dataset.py`，因为 dry-run 测试已确认 builder 能识别新增 extras
  列；如果后续出现更复杂 list/schema 类型，再局部扩展 builder。

## 2026-07-10 review remediation（完成于 2026-07-13）

根据 `completed_milestones_improvement_plan.md` 对 Milestone 0 进行了返修。本节保留原始完成记录，
只追加审阅后的新实现和验证结果。

### 旧行为与问题

- `extras.parquet` 和 `value_function_meta.json` 直接覆盖正式文件；写入中断时缺少原文件保护。
- run-level extras 虽然会先构造全部 Arrow table，但逐 episode 直接覆盖，第二个或更后 episode
  commit 失败时可能留下混合的新旧数据。
- metadata writer 默认整体覆盖；target 重跑可能丢失已有 advantage/labeling metadata。
- 没有统一的 stage config normalization、输入 fingerprint、stage fingerprint 和 stale dependency
  检查机制。
- 各 stage 没有独立的 provenance record，无法可靠判断上游重跑后下游是否仍然有效。

### 新实现

修改文件：

- `src/lerobot/value_function/raw_io.py`
- `src/lerobot/value_function/schema.py`
- `src/lerobot/value_function/targets.py`
- `src/lerobot/value_function/advantage.py`
- `src/lerobot/value_function/advantage_labeling.py`
- `tests/value_function/test_raw_value_io.py`
- `tests/value_function/test_value_targets.py`

实现内容：

1. 单 episode extras 使用同目录临时文件写入，完成 close/fsync 后通过 `os.replace` 原子替换；
   写入或 replace 失败时保留原 parquet，并清理临时文件。
2. run-level extras 在第一次 replace 前完成所有 episode 的长度、dtype、schema、table 构造和临时文件
   staging。commit 时保存原文件备份；任一 episode replace 失败后，回滚所有已替换 episode。
   原本不存在 extras 的 episode 会恢复为不存在。
3. metadata 使用同目录临时 JSON、flush/fsync 和 atomic replace。新增递归 merge API；更新 target、
   advantage 或 labeling 时保留其他 stage 和未知用户字段。
4. 新增 `PIPELINE_SCHEMA_VERSION=2`、canonical stage 名称和 prediction source 名称。
5. 新增递归 config normalization。mapping key 稳定排序，`Path` 规范为绝对字符串，tuple/list 使用
   同一 JSON 表示；NaN、Inf、非字符串 mapping key 和不支持类型明确拒绝。
6. 新增 SHA-256 input fingerprint。fingerprint 包含 episode/frame 结构、选定列 schema 和列内容，
   不依赖 mtime；未选择的 extras 列变化不影响该 fingerprint。
7. metadata 的 `stages` 中每个 stage 独立记录：
   - `created_at`
   - normalized `config`
   - `input_columns` / `input_fingerprint`
   - `output_columns`
   - `prediction_source` / `synthetic`
   - dependency stage fingerprints
   - `stage_fingerprint`
   - `stale` / `stale_reason`
8. upstream stage 每次成功重跑会产生新的 stage fingerprint，并递归标记依赖它的 downstream
   stage stale。检查 helper 同时验证保存的 dependency fingerprint 和当前实际输入列 fingerprint。
9. 现有 target、advantage、advantage labeling 写回已做最小接入，同时保留原来的顶层 metadata
   结构，避免破坏当前 UI 和已有测试。旧 run 没有 stage metadata 时仍可读取和写回；后续 milestone
   会逐步启用更严格的 provenance gate。
10. target stage 明确写入 `all_episodes_successful=true` 和当前 `subtask_order`。成功 episode 与严格
    subtask contract 的实际强校验仍属于 Milestone 1 返修。

### 事务保证边界

- 对正常 Python/文件系统异常，run-level commit 会尝试回滚，测试覆盖第二个 episode replace 失败。
- 单文件正式路径的每次替换都是 atomic replace。
- 当前没有实现跨多个目录的 durable transaction journal；进程被 `SIGKILL`、系统掉电或 rollback
  自身遭遇不可恢复文件系统错误，不承诺完整的跨文件 ACID recovery。rollback 自身失败会抛出包含
  episode 信息的明确错误，不会静默成功。

### 新增测试覆盖

- 单 episode parquet staging 写入失败后原文件字节不变。
- 单 episode atomic replace 失败后原文件字节不变。
- 多 episode 在 staging 阶段第二个文件失败时尚未 commit 任何 episode。
- 多 episode 在 commit 阶段第二个文件失败时，已替换 episode 完整回滚。
- 原本没有 extras 的 run 在 commit 失败后不残留新 extras。
- 失败路径不残留 `.tmp` / `.bak` 文件。
- metadata replace 失败后旧 JSON 字节不变且仍可读取。
- metadata 深度 merge 保留其他 stage、timestamp 和未知用户字段。
- config normalization 和 payload fingerprint 不受 mapping key 顺序影响。
- selected-column fingerprint 忽略无关列变化，并检测选定列变化。
- target → advantage → target 的实际调用链保留 advantage metadata，并将 advantage stage 标记 stale。
- stale 状态从 target 递归传播到 advantage labeling。
- stage check 可检测绕过 pipeline API 直接修改 extras 输入列的情况。
- 原有 extras 列顺序、dtype、builder dry-run 和 subtask progress 数据流继续通过。

### 实际验证

执行：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest \
  tests/value_function/test_raw_value_io.py \
  tests/value_function/test_value_targets.py \
  tests/value_function/test_advantage.py \
  tests/value_function/test_advantage_labeling.py \
  tests/scripts/test_subtask_progress_data_pipeline.py -q
```

结果：

```text
52 passed in 1.75s
```

其中 `tests/value_function/test_raw_value_io.py` 单独执行结果：

```text
19 passed
```

源码 `py_compile` 通过，`git diff --check` 通过。当前环境仍未安装 ruff：

```text
/home/zenbot-robot/.conda/envs/lerobot-main/bin/python: No module named ruff
```

### 迁移影响

- 新 metadata 会增加 `pipeline_schema_version` 和 `stages`，原顶层字段仍保留。
- 调用方应逐步改用 `update_stage_metadata`，并在消费 downstream artifact 前调用 stale 检查 helper。
- `write_value_function_metadata` 仍表示完整 metadata 写入；增量更新必须使用
  `merge_value_function_metadata` 或 `update_stage_metadata`。
- 后续 Milestone 1、6、7 返修分别负责更严格的 subtask/success contract、paired-path advantage gate
  和 label export provenance gate，本次没有提前改变其业务语义。
