# Completed Value-Pipeline Milestones Improvement Plan

日期：2026-07-10

适用仓库：`/home/zenbot-robot/repos/lerobot`

关联总体计划：`plans/value_function_subtask_advantage_pipeline_plan.md`

涉及已完成记录：

- `plans/value_pipeline/milestone_0_raw_io_completed.md`
- `plans/value_pipeline/milestone_1_value_targets_completed.md`
- `plans/value_pipeline/milestone_6_advantage_completed.md`
- `plans/value_pipeline/milestone_7_advantage_labeler_completed.md`

## 1. 目的和边界

本计划只返修已经完成的 Milestone 0、1、6、7，使它们满足 2026-07-10 审阅后确定的
数据契约、安全写回、subtask 约束、跨边界 advantage 和 UI 可交付要求。

本返修不实现真实 value model、value training、value inference、advantage weights 或 VLA training；
这些仍由总体计划后续 milestone 完成。允许增加 GT+Gaussian-noise mock prediction，但它只能用于
synthetic smoke，不能作为正式实验数据。

已确认的第一阶段数据假设：

- 所有 episode 都成功完成任务；不输入 failure、timeout 或人工中止 episode。
- 所有 episode 包含相同 subtask 集合。
- 每个 subtask 按同一固定顺序恰好出现一次，并且只有一个连续 segment。
- 正式 advantage 排序只能使用 value model prediction。
- GT 或 GT+noise 只用于公式和数据流 smoke。

## 2. 统一字段与 provenance 决策

### 2.1 Prediction source

所有 downstream stage 必须区分：

- `gt`：直接计算的监督标签；centered advantage 只用于零值 sanity check。
- `mock_pred`：GT frame-unit value 加 Gaussian noise 的 synthetic prediction。
- `model_pred`：真实 value model inference；只有它可以默认导出正式 label/weight。

mock prediction 使用独立 `*_mock_pred` 列，不覆盖 `*_pred_*` model prediction 列。

### 2.2 Subtask paired inference path

subtask 多头输出只允许两个成对 path：

- `gt_conditioned`：GT subtask boundary + GT id 对应 remaining head prediction。
- `pred_smooth`：smoothed predicted boundary + smoothed predicted id 对应 remaining head prediction。

Milestone 6 返修时先建立字段解析和一致性校验；真实 head prediction 要等 Milestone 4 产出。
禁止用 smooth-head value 搭配 GT boundary，或用 GT-head value搭配 predicted boundary。

### 2.3 Stage metadata

`value_function_meta.json` 增加：

- `pipeline_schema_version`
- `all_episodes_successful=true`
- canonical `subtask_order`
- 每个 stage 独立：
  - `created_at`
  - 完整 normalized config
  - input columns
  - input fingerprint
  - output columns
  - prediction source
  - synthetic flag

重跑上游后，旧下游 stage 必须被识别为 stale；第一版可以拒绝继续运行并给出重跑命令，不要求自动
删除用户列。

## 3. Milestone 0 返修：raw IO、atomic commit 和 provenance

### 3.1 需要修改

- `src/lerobot/value_function/raw_io.py`
- `src/lerobot/value_function/schema.py`
- `tests/value_function/test_raw_value_io.py`
- `plans/value_pipeline/milestone_0_raw_io_completed.md`

### 3.2 实现内容

1. 为 `extras.parquet` 和 `value_function_meta.json` 增加同目录临时文件写入与 atomic replace。
2. 所有 episode 先完成长度、dtype、schema 和可写性预检查，再进入 commit。
3. commit 中途失败时保留原始文件；如实现 run-level rollback，需测试恢复所有已替换 episode。
4. 增加统一 metadata merge API，禁止 target stage 整体覆盖已有 advantage/labeling metadata。
5. 增加 stage config normalization 和 input fingerprint helper。
6. 提供 stale dependency 检查 helper，供 advantage、labeling 和后续 weights/build 调用。
7. 保留所有非 pipeline extras 列及其顺序和 dtype。

### 3.3 验证

- 原有 raw IO 测试全部通过。
- 注入 `pq.write_table`/replace 失败后，原 parquet 仍可读取且内容不变。
- 多 episode commit 失败不会留下混合 schema 或无法识别的 stage 状态。
- 连续执行 target -> advantage -> target 时，metadata 不丢字段，且旧 advantage 被标为 stale。
- metadata 每个 stage 有独立 timestamp，更新 stage 不篡改其他 stage provenance。
- `lerobot-build-dataset --dry_run` 回归通过。

## 4. Milestone 1 返修：严格 subtask 结构与准确统计

### 4.1 需要修改

- `src/lerobot/value_function/targets.py`
- `src/lerobot/scripts/lerobot_value_prepare_targets.py`
- `tests/value_function/test_value_targets.py`
- `plans/value_pipeline/milestone_1_value_targets_completed.md`

### 4.2 实现内容

1. 新增默认开启的严格约束：
   - `require_all_subtasks=true`
   - `require_single_segment_per_subtask=true`
   - `require_success_only=true`
2. canonical order 来源优先级：
   - 显式 CLI/config order；
   - 第一条包含全部 subtask 的 episode；
   - 不再用跨全 run 的首次出现顺序拼接一个可能错误的 order。
3. 将每个 episode 的连续 labels 压缩成 sequence，要求与 canonical order 完全一致。
4. 重复、缺失、回退、额外 subtask 都给出 episode、label 和 frame boundary 的明确错误。
5. metadata 写入 `all_episodes_successful`、canonical order、严格校验配置。
6. aggregate clip rate 改为：

   ```text
   total_clipped_eligible_frames / total_eligible_frames
   ```

   某 episode 缺少某 subtask 时不得添加一个 0 rate；严格模式下缺失本身应先报错。
7. 校验所有 bin count >= 2；manual scale key 必须与 canonical subtask names 精确匹配。
8. 重跑 mode 时不静默保留可被误认为当前 stage 输出的旧 target；通过 provenance/columns manifest
   明确当前有效列。

### 4.3 Synthetic mock prediction

增加独立 CLI 或子命令，建议命令形态：

```bash
lerobot-value-mock-predictions \
  --root /path/to/raw/run \
  --mode both \
  --seed 42 \
  --noise_std_frames 3.0
```

要求：

- 输出独立 `*_mock_pred` frame-unit 列。
- 可选 temporal Gaussian smoothing，但参数必须写 metadata。
- metadata 显著标记 synthetic、seed、sigma、source GT fingerprint。
- 不覆盖真实 model prediction。

### 4.4 验证

- 每个 subtask 各出现一次且顺序一致：通过。
- 缺少一个 subtask：报错。
- 同一 subtask 出现两段：报错。
- 顺序交换：报错。
- annotation palette 顺序与执行顺序不同时，以显式/完整 episode canonical order 为准。
- 不同 episode 长度的 global/subtask clip rate 使用 frame-weighted 结果。
- bin count 0、1 或负数均报错。
- mock prediction 相同 seed bitwise/reproducibly equal，不同 seed 不同，sigma=0 恢复 GT。
- 真实样例 dry-run 仍得到 70 episodes、53,794 frames、6 个有序且单段 subtask，但测试代码不写死这些数。

## 5. Milestone 6 返修：paired path 与 boundary transition

### 5.1 需要修改

- `src/lerobot/value_function/advantage.py`
- `src/lerobot/scripts/lerobot_compute_advantage.py`
- `src/lerobot/value_function/schema.py`
- `tests/value_function/test_advantage.py`
- `plans/value_pipeline/milestone_6_advantage_completed.md`

### 5.2 新公式

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

默认 `boundary_transition_value=1.0`。它表示边界两侧相邻帧之间的真实 transition 贡献 1 单位
progress；不是在已经 centered 的 segment advantage 上额外奖励 1。

示例：chunk 有 4 个 transition，其中 1 个跨边界。同 subtask value difference 理想值合计 3，
boundary progress 为 1，总 progress=4，最终 advantage=4-4=0。

### 5.3 实现内容

1. 新增 `--value_source gt|mock_pred|model_pred`。
2. 新增 `--subtask_inference_path gt_conditioned|pred_smooth`，替代可任意组合的
   `--subtask_source`。
3. `model_pred` 下严格校验 paired value column 和 boundary column。
4. `gt`、`mock_pred` metadata 明确 synthetic/non-experiment 状态。
5. 将 `--boundary_bonus` 替换为 `--boundary_transition_value`；若保留 alias，打印 deprecation warning。
6. 新增列：
   - `advantage_subtask_num_crossings`
   - `advantage_subtask_within_subtask_horizon`
   - `advantage_subtask_boundary_progress`
7. 校验 subtask sequence 满足 canonical order、单段和非回退约束。
8. chunk 若存在 NaN、未知 id、head/boundary mismatch，标 invalid 并记录原因统计。
9. metadata 记录公式版本，例如 `advantage_formula_version=subtask_boundary_transition_v2`。

### 5.4 验证

- global 理想线性 episode advantage=0。
- subtask 理想线性 episode不跨边界 advantage=0。
- 理想 chunk 跨 1、2 个边界时 advantage 都为 0。
- boundary transition value 改成 0 时结果按 crossing 数减少。
- 卡住的同 subtask segment 为负。
- 末尾 padding 只使用真实 valid horizon。
- GT-head value + smooth boundary、smooth-head value + GT boundary 都应被拒绝。
- 改变 normalized scale 但保持 frame-unit prediction 不变时 advantage 不变。
- provenance stale 时拒绝写回。

## 6. Milestone 7 返修：label 语义、持久化、规模与打包

### 6.1 需要修改

- `src/lerobot/value_function/advantage_labeling.py`
- `src/lerobot/scripts/lerobot_advantage_labeler.py`
- `src/lerobot/scripts/advantage_labeler/index.html`
- `src/lerobot/scripts/advantage_labeler/app.js`
- `src/lerobot/scripts/advantage_labeler/style.css`
- `pyproject.toml`
- `tests/value_function/test_advantage_labeling.py`
- 新增 UI/API/package smoke test
- `plans/value_pipeline/milestone_7_advantage_labeler_completed.md`

### 6.2 实现内容

1. 展示 `sort_order` 与 label `positive_direction` 解耦：
   - 默认 positive 永远是 high advantage。
   - asc/desc 只改变列表展示顺序。
   - 若支持 low-as-positive，必须单独显式参数并在 UI 警告。
2. 定义 threshold tie policy，并在 metadata 保存 threshold value、tie count 和 exact-count/tie-inclusive 策略。
3. 校验 override key 必须属于当前 run；不存在的 key 直接报错，不能污染 summary count。
4. override 按 value mode 分开保存，不能从 global 模式泄漏到 subtask 模式。
5. metadata 保存具体 overrides 或独立 sidecar；重启 UI 后恢复。
6. UI 分别显示：
   - stored label
   - threshold preview label
   - manual override label/source
7. 导出旧 labels 前显示变更摘要；避免无意覆盖人工结果。
8. 增加 headless export：

   ```bash
   lerobot-advantage-labeler \
     --root /path/to/raw/run \
     --value_mode subtask \
     --top_percent 0.8 \
     --export \
     --dry_run
   ```

9. provenance gate：
   - `model_pred` 默认允许正式 export。
   - `gt/mock_pred` 默认拒绝；测试需 `--allow_synthetic true`。
   - UI 顶部持续显示 synthetic warning。
10. 规模优化：
    - server cache parquet-derived chunk metadata；extras 变化时按 fingerprint 失效。
    - API 分页/查询，不一次发送全部 50k chunks。
    - 前端虚拟列表或分页。
    - slider debounce。
    - 选 frame 只更新选中行和播放器，不重建全部列表。
11. `pyproject.toml` package-data 加入 `scripts/advantage_labeler/*.html|*.js|*.css`；建议同时为后续
    UI 使用更通用但受控的 `scripts/*_viz`/labeler patterns。
12. 当 `--host` 非 localhost 时打印无鉴权写接口警告。

### 6.3 验证

- asc/desc 下同一 threshold 的导出 labels 完全相同。
- high advantage 默认 positive；低 advantage 不因 UI asc 排序变 positive。
- tie policy 可复现并写 metadata。
- 非法 override key 报错，summary count 始终等于总 frame 数。
- global/subtask overrides 相互隔离。
- export 后重启 server，stored labels 和 overrides 可恢复。
- headless dry-run 不写文件，headless export 正确写回。
- GT/mock 未加 `--allow_synthetic` 时拒绝 export。
- API 测试覆盖 meta/chunks/preview/export/image 404。
- 用至少 50k synthetic chunk metadata 做性能/结构测试，确认 API 分页且 DOM 不全量渲染。
- 构建 wheel，检查静态资源包含在 wheel 中；从安装目录启动 server，GET `/`、`/app.js`、
  `/style.css` 返回 200。
- 原有 Python 核心测试和 `lerobot-build-dataset --dry_run` 回归通过。

## 7. 推荐执行顺序

1. Milestone 0：atomic IO、metadata merge、fingerprint/stale helper。
2. Milestone 1：严格 subtask/success contract、clip statistics、bin validation。
3. Milestone 1.5：mock prediction 与 synthetic provenance。
4. Milestone 6：新 boundary transition 公式和 paired path contract。
5. Milestone 7 Python core：label semantics、override persistence、headless export、provenance gate。
6. Milestone 7 UI：pagination/cache/virtualization。
7. package-data/wheel smoke。
8. 全量回归和真实样例只读 dry-run。

每一步完成后更新对应原 completion record，增加“2026-07-10 review remediation”小节，写明旧行为、
新行为、迁移影响和重新运行的测试。不要删除原来的完成历史。

## 8. 最终验收命令类别

至少运行：

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

还必须补充并执行：

- atomic failure/rollback tests
- stale provenance tests
- strict subtask contract tests
- synthetic prediction reproducibility tests
- cross-boundary ideal advantage tests
- UI API tests
- 50k pagination/virtualization structural test
- wheel static-asset smoke
- 真实 `strike_match_3` target/mock/advantage dry-run，禁止写入原始样例数据

如果环境没有 ruff，至少运行 `py_compile`、`git diff --check`；completion record 必须如实记录未运行的
工具，不能把源码目录 UI smoke 等同于 wheel 安装验证。
