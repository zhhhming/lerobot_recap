# Value Pipeline Milestone 8 Completion Record

日期：2026-07-13

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 实施顺序中的第五阶段：
Milestone 8，基于 Milestone 7 写入的 `advantage_label_{mode}`，在相似 value/progress group 内对
positive action chunks 做 advantage rank weighting，并提供只读 UI 检查 group 粒度、rank、weight 和
action chunk 图片序列。

本阶段完成 raw dataset 上的 weight 计算、写回、provenance、CLI、UI、package 和 synthetic smoke。
由于 Milestone 2/3/4 的真实 value model 尚未完成，当前真实样例验收使用带明确 provenance 的 mock
prediction；正式实验必须在 model prediction 生成后重新运行 Milestone 6/7/8。

## 修改文件

- 新增 `src/lerobot/value_function/advantage_weights.py`
- 新增 `src/lerobot/scripts/lerobot_compute_advantage_weights.py`
- 新增 `src/lerobot/scripts/advantage_weight_viz/index.html`
- 新增 `src/lerobot/scripts/advantage_weight_viz/app.js`
- 新增 `src/lerobot/scripts/advantage_weight_viz/style.css`
- 新增 `tests/value_function/test_advantage_weights.py`
- 新增 `tests/value_function/test_advantage_weight_viz_api.py`
- 新增 `tests/value_function/test_advantage_weight_viz_package.py`
- 新增 `tests/value_function/test_milestone_8_shadow_smoke.py`
- 新增 `plans/value_pipeline/validate_milestone_8.sh`
- 修改 `src/lerobot/value_function/schema.py`
- 修改 `pyproject.toml`

## 新增 CLI

注册命令：

```bash
lerobot-compute-advantage-weights
```

正式 model prediction 权重示例：

```bash
lerobot-compute-advantage-weights \
  --root /path/to/raw/run \
  --value_mode subtask \
  --group_source auto \
  --group_bin_width 0.1 \
  --q 0.8 \
  --tau 0.08 \
  --w_min 0.1 \
  --w_max 2.0 \
  --positive_group_max_weight 2.0 \
  --min_group_size 4 \
  --negative_weight 1.0
```

synthetic smoke 必须显式允许：

```bash
lerobot-compute-advantage-weights \
  --root /path/to/shadow/run \
  --value_mode global \
  --allow_synthetic
```

`--dry_run` 会完成所有输入、provenance、分组和权重校验并输出 summary，但不修改 parquet 或 metadata。

## 新增字段和 stage

global：

- `advantage_group_id_global: string`
- `advantage_loss_weight_global: float32`

subtask：

- `advantage_group_id_subtask: string`
- `advantage_loss_weight_subtask: float32`

metadata stage：

- `advantage_weights.global`
- `advantage_weights.subtask`

group id 使用稳定且可读的字符串格式：

```text
global:bin:+00005
subtask:0003:bin:+00007
```

负数或超过 `[0, 1]` 的 synthetic/model grouping value 不会静默 clip，而是自然落入 overflow bin；实际
group scalar、bin width 和 source column 会写入 metadata。

## 分组规则

### Global

默认 bin width 为 `0.05`。

- `model_pred`：使用 `value_global_remaining_norm_pred`。
- `gt`：使用 `value_global_remaining_norm_gt`。
- `mock_pred`：使用 `value_global_remaining_frames_mock_pred / global_scale.frames`，只在内存中恢复
  normalized grouping value，不新增模糊的 mock norm 字段。

global 不接受 `group_source=progress`。

### Subtask

默认 bin width 为 `0.1`，group id 始终包含 subtask id。

- `gt_conditioned + group_source=auto`：使用 `value_subtask_id_gt + subtask_progress`。
- `pred_smooth + group_source=auto`：使用
  `value_subtask_id_pred_smooth + value_subtask_remaining_norm_pred_smooth_head`。
- 显式 `group_source=value`：
  - GT 使用 GT subtask norm；
  - mock frame value 按 canonical subtask scale 恢复 norm；
  - model prediction 使用与 inference path 配对的 norm head。
- `pred_smooth + group_source=progress` 会明确报错，禁止把 GT progress 和 predicted boundary 混组。

## Weight 公式和行为

只对同 group 内 label 为 `positive` 的样本排序：

```text
u = rank / max(N - 1, 1)           # best=0, worst=1
w_raw = w_min + (w_max - w_min) * sigmoid((q - u) / tau)
w = w_raw / max(group_w_raw) * positive_group_max_weight
```

默认值：

```text
q=0.8
tau=0.08
w_min=0.1
w_max=2.0
positive_group_max_weight=2.0
min_group_size=4
negative_weight=1.0
ignore_weight=0.0
```

具体行为：

- advantage 从高到低排序，positive weight 单调不增。
- 相同 advantage 使用平均 rank，得到完全相同的权重。
- 达到 `min_group_size` 的 group 内最大 positive weight 严格归一化为 `2.0`。
- 其他 positive 保留 `w_raw` 相对比例，不映射到 `[1, 2]`，因此允许低于 `1.0`。
- 小 group 的 positive 全部回退 `1.0`。
- negative 为 `1.0`。
- ignore 为 `0.0`，不参与 positive rank。
- 参数、label、group scalar、advantage 中的非法值或 NaN 在写回前统一拒绝。

## Atomic write 和 provenance

计算前要求对应的 `advantage_labeling.{mode}` stage 存在且 current，并继续验证其 advantage/value
依赖链。grouping 使用额外 model/GT/mock 字段时，同时校验对应 source stage 和 active output columns。

weight stage fingerprint 覆盖：

- advantage label；
- advantage value；
- 实际 subtask id/value/progress grouping 输入。

写回继续使用 Milestone 0 的全 run 预检、临时 parquet、atomic replace 和 rollback。上游 label、
advantage、value 或 grouping input 改变后，旧 weight stage 会被 stale gate 拒绝。

GT/mock label 默认拒绝 weight export；只有测试可使用 `--allow_synthetic`。metadata/UI 保留：

```text
SYNTHETIC / GT SANITY ONLY — NOT FOR EXPERIMENT
```

## Weight inspection UI

启动命令：

```bash
lerobot-compute-advantage-weights \
  --root /path/to/raw/run \
  --value_mode subtask \
  --serve \
  --port 8002
```

UI 是只读检查工具，不提供手工修改 weight 的入口。能力包括：

- 横向浏览 value/progress bins；
- 显示每个 group 的总数和 positive/negative/ignore 数量；
- 显示 group weight 范围；
- group 内按 advantage 升/降序分页；
- 显示 positive rank、positive group size 和最终 weight；
- action chunk 单帧顺序播放器、camera 切换、键盘左右键；
- 缺图返回包含具体路径的 JSON 404；
- parquet-derived chunk/group cache；
- group 和 chunk 两级分页，50k chunks 不全量返回或创建 DOM。

静态资源已加入 `pyproject.toml` package-data。wheel 安装到独立目录后，`/`、`/app.js`、
`/style.css` 均已验证返回 200。

## 测试覆盖

核心测试包括：

- weight 单调性、最大值 2.0 和 raw weight 比例；
- 非最大 positive 可以低于 1.0；
- negative=1、ignore=0；
- ignore 不进入 rank；
- average tie rank；
- small-group fallback；
- 参数范围和非有限值校验；
- global/subtask 字段、dtype、长度和 metadata；
- `gt_conditioned` progress grouping；
- paired `pred_smooth` value grouping 和错误交叉组合拒绝；
- synthetic gate 和 dry-run；
- stale label 拒绝；
- raw -> LeRobotDataset dry-run 识别 group id/weight；
- visualization API、cache、分页、清晰 image 404；
- 50k chunks 只返回请求页；
- wheel assets 和安装目录 HTTP smoke。

## 一键验收和结果

验收脚本：

```bash
plans/value_pipeline/validate_milestone_8.sh
```

脚本使用 `/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`，依次执行 py_compile、新增测试、既有
value pipeline 回归和真实样例 shadow smoke。本机 HTTP 测试需要允许创建 `127.0.0.1` socket。

2026-07-13 实际结果：

```text
Milestone 8 algorithm/API/wheel: 15 passed
existing value pipeline regression: 117 passed
strike_match_3 shadow pipeline: 1 passed
```

真实样例 shadow smoke 在 pytest `/tmp` 目录中只复制 raw metadata/parquet，执行：

```text
targets -> mock predictions -> advantage -> labels -> weights
```

覆盖 global/subtask 两种模式，共 70 episodes、53,794 frames；确认存在可 rank-weight 的 group，label
和 weight 规则一致。原始样例未被修改，验收后仍然：

```text
value_function_meta.json: absent
extras.parquet columns: [subtask, subtask_progress]
```

## 剩余边界

- 当前只证明 synthetic 数据流和 fake model-pred contract；没有把 mock weight 当成正式训练数据。
- 正式 weight artifact 必须等待 Milestone 2/3/4 产出 model prediction，然后重新运行 Milestone 6/7/8。
- Milestone 9 仍需验证最终 LeRobotDataset/DataLoader batch 中 group/weight 的 shape；本阶段只完成
  builder schema/dry-run 验证。
- Milestone 11 才负责 FM-only weighting、condition dropout fallback 和与旧 `use_rabc` 的互斥。
- UI 不包含浏览器自动化框架；API、前端规模约束、播放器局部更新和 wheel HTTP 行为已有自动测试。
- 当前环境未安装 `ruff`，因此使用 `py_compile`、`git diff --check` 和完整 pytest 验收；没有为此下载
  或新增依赖。
