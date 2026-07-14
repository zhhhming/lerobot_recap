# Value Pipeline Milestone 7 Completion Record

日期：2026-07-09

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 实施顺序中的第四阶段：
Milestone 7，基于 Milestone 6 写入 raw dataset `extras.parquet` 的 advantage 列，
提供 action chunk 排序、positive/negative/ignore 预览、手动覆盖和 label 写回能力。

本阶段只负责生成 label，不计算 group-relative loss weight。

## 修改文件

- 新增 `src/lerobot/value_function/advantage_labeling.py`
- 新增 `src/lerobot/scripts/lerobot_advantage_labeler.py`
- 新增 `src/lerobot/scripts/advantage_labeler/index.html`
- 新增 `src/lerobot/scripts/advantage_labeler/app.js`
- 新增 `src/lerobot/scripts/advantage_labeler/style.css`
- 新增 `tests/value_function/test_advantage_labeling.py`
- 修改 `src/lerobot/value_function/schema.py`
- 修改 `pyproject.toml`

## 新增 CLI

注册了命令：

```bash
lerobot-advantage-labeler
```

主要参数：

- `--root`
- `--value_mode global|subtask`
- `--top_percent`
- `--sort_order desc|asc`
- `--host`
- `--port`
- `--no-browser`

默认端口是 `8001`，避免和 `lerobot-annotate-subtask` 的默认 `8000` 冲突。

## 新增字段

- `advantage_label_global`
- `advantage_label_subtask`

label 类型是 string，取值：

- `positive`
- `negative`
- `ignore`

## 行为定义

核心逻辑在 `src/lerobot/value_function/advantage_labeling.py`：

- `load_advantage_chunks(root, value_mode)` 读取所有 episode 的 advantage chunk。
- `sorted_advantage_chunks(chunks, sort_order)` 排序，invalid chunk 排在末尾。
- `preview_advantage_labels(...)` 根据阈值和手动覆盖生成 label preview。
- `export_advantage_labels(config)` 写回 `extras.parquet`。

global mode 读取：

- `advantage_global_chunk`
- `advantage_global_valid_horizon`
- `advantage_global_is_valid`

subtask mode 读取：

- `advantage_subtask_chunk`
- `advantage_subtask_valid_horizon`
- `advantage_subtask_is_valid`

阈值规则：

- 默认 `top_percent=0.8`。
- 有效 chunk 按排序结果取前 `top_percent` 标为 `positive`。
- 其余有效 chunk 标为 `negative`。
- invalid chunk 默认标为 `ignore`。
- 手动 override 可把任意 chunk 改成 `positive|negative|ignore`，优先级高于阈值规则。
- `top_percent` 支持 `0.8` 或 `80` 两种输入形式。

chunk key 格式：

```text
ep_000012:frame_000345
```

## UI

新增本地 web UI：

```bash
lerobot-advantage-labeler \
  --root /path/to/raw/run \
  --value_mode global
```

UI 能力：

- 左侧显示 action chunk 列表：
  - episode
  - start frame
  - advantage
  - valid horizon
  - preview label
- 支持 global/subtask mode 切换。
- 支持 advantage 高到低/低到高排序。
- 支持 positive 百分比滑条。
- 显示 `positive/negative/ignore` 数量。
- 显示 10%、20%、30%、80% 排序位置 marker。
- 右侧显示当前 chunk 的单帧图片播放器。
- 支持相机选择。
- 支持 Prev/Next、滑条和左右方向键查看 chunk 内 frame。
- 支持单个 chunk 手动设为 `positive`、`negative` 或 `ignore`。
- 支持清除单个手动 override。
- 点击 Export 后写回 label 列。

UI server 沿用 `lerobot_annotate_subtask.py` 的轻量标准库 HTTP 模式：

- `ThreadingHTTPServer`
- 静态 `index.html/app.js/style.css`
- `/api/meta`
- `/api/chunks`
- `/api/preview`
- `/api/export`
- `/api/episode/{episode}/img/{camera}/{frame}`

## Metadata

写回时会更新 `value_function_meta.json`：

- 在 `advantage_labeling.global` 或 `advantage_labeling.subtask` 下记录：
  - `top_percent`
  - `sort_order`
  - `columns_written`
  - `counts`
  - `overrides`
- 同时把 label 字段并入 top-level `columns_written`。

如果 raw run 尚未有 `value_function_meta.json`，会写入一个包含 label 信息的最小 metadata。

## 测试

新增测试覆盖：

- chunk key 生成和解析。
- 读取 raw run 中的 advantage chunks。
- 按 advantage 降序排序，invalid chunk 排末尾。
- `top_percent=0.4` 时 positive/negative/ignore 数量正确。
- `top_percent=40` 百分制输入可正确归一化。
- 手动 override 优先于阈值规则。
- export 后每个 episode `extras.parquet` 增加 `advantage_label_global`。
- label 列长度等于 frame 数。
- 原有 extras 列保留。
- metadata 记录 label summary。
- `--dry_run` 不写回 label 列。
- `lerobot_build_dataset.py --dry_run` 能识别新增 string label 列。

已通过：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m pytest tests/value_function/test_advantage_labeling.py -q
```

结果：`7 passed`

已通过相关回归：

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

结果：`40 passed`

语法检查通过：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python \
  -m py_compile \
  src/lerobot/scripts/lerobot_advantage_labeler.py \
  src/lerobot/value_function/advantage_labeling.py
```

## 使用示例

启动 global advantage labeler：

```bash
lerobot-advantage-labeler \
  --root /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3 \
  --value_mode global \
  --top_percent 0.8
```

启动 subtask advantage labeler：

```bash
lerobot-advantage-labeler \
  --root /path/to/raw/run \
  --value_mode subtask \
  --top_percent 80
```

## 剩余风险

- 当前测试覆盖 Python 核心逻辑和写回行为，没有用浏览器自动化测试 UI 交互。
- UI 只使用现有 PNG 帧路径；如果某个 raw run 缺图，图片 API 会返回 404。
- 本阶段只写 `advantage_label_{mode}`，不计算 `advantage_loss_weight_{mode}`。
- 后续 Milestone 8 需要基于 label 进一步生成 group id 和 loss weight。

## 2026-07-13 review remediation

本节记录对 `completed_milestones_improvement_plan.md` 中 Milestone 7 审阅项的返修。保留上面的
原始完成历史；本节所述行为取代旧行为中与之冲突的部分。

### 旧行为与问题

- `sort_order` 同时决定列表顺序和 positive 方向，切换到升序可能把低 advantage 标成 positive。
- threshold 未定义并列值策略，也未保存 threshold value 和 tie 统计。
- override key 不验证是否属于当前 run，global/subtask override 的恢复和隔离契约不完整。
- UI 没有明确区分 stored、threshold preview、manual override 和最终 preview label。
- 导出直接覆盖旧 label，没有先展示变更摘要并二次确认。
- 只有交互 server，没有可用于自动化的 headless export/dry-run。
- GT/mock advantage 没有正式导出门禁，UI 也没有持续 synthetic 警告。
- `/api/chunks` 会发送全量 chunk，前端全量构建 DOM；slider 和逐帧播放也缺少规模约束。
- package-data 未包含 advantage labeler 静态文件，源码目录能启动不能证明 wheel 可交付。

### 新行为

#### Label 与 threshold 契约

- positive 方向固定为 high advantage；`sort_order=asc|desc` 现在只控制展示顺序。
- 新增 `tie_policy=exact_count|include_all`，默认 `exact_count`：
  - `exact_count` 在 threshold 并列时用 episode/frame 稳定顺序精确选满目标数量；
  - `include_all` 将 threshold value 的全部并列 chunk 选为 positive。
- preview 和 metadata 记录 `positive_direction`、`threshold_value`、`tie_count`、
  `selected_at_tie_count`、目标/实际 threshold positive 数和 valid/invalid 数。
- 每条 preview 明确返回 `stored_label`、`threshold_label`、`manual_override_label`、
  `preview_label`、`label_source`。
- override key 必须是当前 run 中实际存在的 `ep_NNNNNN:frame_NNNNNN`；格式错误、未知 key 和非法
  label 均拒绝，summary count 始终覆盖全部 chunk。

#### 持久化、导出和 provenance

- global/subtask overrides 分开保存在 `advantage_labeling.{mode}.overrides`，server 重启后从 metadata
  恢复；已有 label 从 parquet 恢复为 stored label。
- 新增 `export_change_summary`，报告 unchanged、changed、unset-to-labeled 和逐 transition 统计。
- UI 先调用 `/api/export-preview` 获取 dry-run 变更摘要，再要求用户确认，只有带
  `confirm=true` 的 `/api/export` 才写回。
- 新增 headless CLI：

  ```bash
  lerobot-advantage-labeler \
    --root /path/to/raw/run \
    --value_mode subtask \
    --top_percent 0.8 \
    --tie_policy exact_count \
    --export \
    --dry_run
  ```

- labeler 会验证 advantage stage provenance、input/output fingerprint 和 active columns。
- `model_pred` 默认允许正式 export；`gt`/`mock_pred` 默认拒绝。仅测试和 smoke 可显式加
  `--allow_synthetic`，summary/UI 持续显示 `SYNTHETIC / GT SANITY ONLY — NOT FOR EXPERIMENT`。
- label 写回继续使用 raw IO 的全 run 预检、atomic commit 和 stage dependency metadata。

#### UI/API 规模和交付

- server 增加按 mode 的 parquet-derived chunk cache，并在 extras 文件变化或成功 export 后失效。
- `/api/chunks` 和 `/api/preview` 支持 `page`、`page_size`、episode/label filter，单页上限 500；
  response 只包含当前页，同时保留全局 label counts 和 threshold 统计。
- 前端改为分页，每页 150 条；positive slider 使用 150 ms debounce。
- frame offset、相机切换和左右键只更新当前选择及播放器，不重建 chunk list。
- 模式切换使用各自的 override map；列表直接展示 stored/threshold/override/source/final preview。
- 非 localhost `--host` 会输出“无鉴权写接口暴露”警告。
- `pyproject.toml` 已加入 `scripts/advantage_labeler/*.html|*.js|*.css` package-data。

### 新增和更新测试

`tests/value_function/test_advantage_labeling.py` 现有 12 项核心测试，覆盖：

- asc/desc 显示顺序不同但 label 映射完全相同，high advantage 始终 positive；
- exact-count/include-all tie policy、threshold 元数据和显式 preview 字段；
- 非法/未知 override key 拒绝、总数守恒；
- dry-run 变更摘要、实际写回、stored label 恢复；
- global/subtask override 持久化与重启隔离；
- synthetic export gate 和显式测试逃生口；
- advantage fingerprint stale 拒绝；
- headless dry-run/export；
- `lerobot-build-dataset --dry_run` 同时识别 global/subtask string label。

新增 `tests/value_function/test_advantage_labeler_api.py`，覆盖：

- 静态页面、meta、chunks、preview、export-preview、confirm 后 export；
- page/page-size、cache hit/export invalidation、非法 override 和 page-size 400；
- 未知相机/image 与未知路径 404；
- 50,000 chunk 时 response 只返回 100 条当前页、总页数和全局 count 正确；
- 前端分页、150 ms debounce 和 frame selection 不调用 `renderList` 的结构检查。

新增 `tests/value_function/test_advantage_labeler_package.py`：

- 使用 `pip wheel --no-deps --no-build-isolation` 构建 wheel；
- 检查 wheel 包含 `index.html`、`app.js`、`style.css`；
- `pip install --no-deps --target` 隔离安装；
- 确认 import 来自安装目录，并从该安装目录启动 server；
- GET `/`、`/app.js`、`/style.css` 均返回 200 且内容正确。

### 验证结果

核心专项：

```text
tests/value_function/test_advantage_labeling.py
12 passed
```

API、50k 和前端结构专项：

```text
tests/value_function/test_advantage_labeler_api.py
4 passed
```

wheel 安装 smoke：

```text
tests/value_function/test_advantage_labeler_package.py
1 passed
```

完整 value pipeline 回归命令覆盖 raw IO、targets、mock predictions、advantage、labeler core/API/package
和 subtask data pipeline，结果：

```text
117 passed in 7.34s
```

`py_compile` 与 `git diff --check` 通过。当前环境没有安装 `ruff`，因此未声称运行 ruff。

### 真实样例 shadow-run

继续使用不含图片的 shadow run：

```text
/tmp/lerobot-value-shadow.npiWRg
```

在 shadow 上实际写入 global/subtask mock advantage，然后执行 label dry-run 和带
`allow_synthetic` 的实际 label export：

```text
episodes: 70
frames: 53,794

global mock advantage: valid=53,724, invalid=70
subtask mock advantage: valid=53,724, invalid=70
asc/desc label mapping equal: true

global export: positive=42,978, negative=10,746, ignore=70, overrides=1
subtask export: positive=42,978, negative=10,745, ignore=71, overrides=1
```

global/subtask override 重新加载后各自保持原值；两种 export 均带 synthetic warning，不能被误认为
正式实验数据。原始样例
`/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3` 未被写入：仍无
`value_function_meta.json`，原始 `extras.parquet` 仍只有 `subtask` 和 `subtask_progress`。

### 迁移影响和剩余边界

- 依赖“升序即 low-positive”的旧用法必须迁移；low-positive 不再由排序隐式表达。
- 旧 raw run 若缺少 current advantage stage provenance，labeler 会拒绝加载，需先重跑 Milestone 6。
- GT/mock smoke 必须显式 `--allow_synthetic`；该选项不能用于正式实验产物。
- 已保存 overrides 会在未显式传入新 overrides 时复用；UI 导出前会展示覆盖变化摘要。
- 测试覆盖 HTTP/API、前端结构和 installed-wheel 静态资源，没有引入浏览器自动化框架做像素级或
  真实 DOM 交互测试。
- 真实 model prediction 尚由后续 Milestone 4 提供；本次验证了 model provenance 契约及 mock gate，
  未伪造真实模型实验结果。
- 本阶段仍只生成 label；group id 和 loss weight 属于 Milestone 8。
