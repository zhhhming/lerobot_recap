# Value Pipeline Milestone 5 Completion Record

日期：2026-07-14

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 推荐实施顺序中的第十二阶段：
Milestone 5（value 曲线展示 UI）。

此前 Milestone 2/3/4 已分别作为第九、十、十一阶段完成 value model 架构、训练和整库推理；本阶段按
推荐顺序继续实现 Milestone 5，而不是直接进入编号上的 Milestone 12。下一步才是只使用正式
`model_pred` 重新运行 Milestone 6/7/8，之后重建 LeRobotDataset 并进入端到端训练 smoke。

本阶段实现了一个完全只读的本地 Web UI，用于按 episode/frame 对照 global/subtask、remaining/elapsed、
normalized/frame-unit value，以及 subtask `gt_conditioned` / `pred_smooth` 两条 paired inference path。
UI 不写 `extras.parquet`、`value_function_meta.json` 或任何下游 artifact。

## 修改文件

- 新增 `src/lerobot/scripts/lerobot_value_viz.py`
- 新增 `src/lerobot/scripts/value_viz/index.html`
- 新增 `src/lerobot/scripts/value_viz/app.js`
- 新增 `src/lerobot/scripts/value_viz/style.css`
- 修改 `pyproject.toml`
- 新增 `tests/value_function/test_value_viz_data.py`
- 新增 `tests/value_function/test_value_viz_api.py`
- 新增 `tests/value_function/test_value_viz_package.py`
- 新增 `tests/value_function/test_value_viz_real_sample.py`
- 新增 `plans/value_pipeline/validate_milestone_5.sh`
- 新增本完成记录

没有修改 value model、训练、inference writeback、advantage、label、weight 或 VLA 训练语义。

## 新增 CLI

安装后入口：

```bash
lerobot-value-viz \
  --root /path/to/raw/run \
  --chunk_size 50 \
  --host 127.0.0.1 \
  --port 8003
```

也可直接运行：

```bash
python -m lerobot.scripts.lerobot_value_viz --root /path/to/raw/run
```

默认只绑定 `127.0.0.1`。若显式绑定非本机地址，CLI 会提示服务没有认证并会暴露 raw 图片。

## 后端数据和 API 契约

`ValueRun` 在启动时验证 raw run、episode、相机和跨 episode `extras.parquet` schema；后续只按当前选中的
episode 懒加载 extras。cache 以文件 size/mtime 为签名，文件变化后重新读取，不把整库所有曲线常驻内存。

只读接口：

- `GET /api/meta`
  - root、task、robot、fps、episode、camera；
  - 每条 value series 的 normalized/frame-unit 列可用性；
  - GT / predicted-smoothed boundary 可用性；
  - global/subtask inference provenance 的 current、missing、stale 或 synthetic warning。
- `GET /api/episode/{id}/curves`
  - 支持 `unit=norm|frames`；
  - 支持 `boundary=gt|pred_smooth`；
  - 默认最多返回 2,000 个确定性采样点，上限 10,000，始终保留 episode 首尾 frame；
  - subtask background 使用 run-length interval，不逐 frame 返回重复字符串。
- `GET /api/episode/{id}/frame/{frame}`
  - 返回当前 frame 所有 value 的精确值，不使用降采样值；
  - 返回当前 subtask、时间、action chunk end 和按 boundary 切分后的 chunk segments。
- `GET /api/episode/{id}/img/{camera}/{frame}`
  - 只接受 run metadata 中声明的 camera subdirectory；
  - episode、frame 和 camera 均做白名单/范围校验，不能通过 URL 读取任意路径。

不存在 `extras.parquet` 或某条 value 列时，series 明确返回 `available=false` 和空 points；UI 显示
`unavailable`，不会把缺失数据当 0，也不会因部分模型输出不存在而崩溃。

## UI 行为

- 左侧滚动 episode 列表。
- 当前 frame 图片默认选择 `third_person`，可切换 `left_wrist` / `right_wrist` 等 run 内 camera。
- slider、Prev/Next 和键盘左右键切换 frame；Shift+左右键移动 10 帧。
- 图片、当前 frame 标题、时间、subtask、chart 竖线、高亮点和精确 value card 同步更新。
- normalized 和 frame units 分开绘制，避免把训练归一化尺度和 frame-unit value 混在同一纵轴解释。
- 曲线开关覆盖：
  - global remaining GT / prediction；
  - global elapsed GT；
  - subtask remaining GT；
  - subtask prediction GT-conditioned head；
  - subtask prediction smoothed head；
  - subtask elapsed GT。
- subtask 背景可切换 GT boundary 或 predicted-smoothed boundary，并显示颜色 legend。
- 当前 action chunk 可显示 end marker 以及 boundary-split segments。
- chart 使用 Canvas 绘制限点曲线；当前 frame 的精确值同时参与纵轴范围计算，极值不会因降采样而把高亮点
  画出坐标区域。
- 响应式布局支持窄屏，不依赖外部 CDN 或 JavaScript package。

Milestone 4 没有把 elapsed auxiliary prediction 写回 raw extras，因此本阶段只展示已定义的 elapsed GT；
不会在 Milestone 5 内跨范围新增一套 elapsed inference schema。

## 测试覆盖

`test_value_viz_data.py` 使用两个 episode、三路相机和完整 value 字段的临时 raw run，覆盖：

- 所有 global/subtask、remaining/elapsed、norm/frames series discovery；
- 两条 paired subtask inference path；
- GT 和 predicted-smoothed interval 差异；
- 精确 frame value、chunk end 和跨 boundary segment；
- 50,000-frame -> 2,000-point bounded sampling，保留首尾且无重复；
- 缺失 prediction/全部 value 列时 unavailable 降级；
- 非法 unit、boundary、camera、episode frame 拒绝；
- data helper 调用前后 `extras.parquet` bytes 不变；
- frontend `setFrame()` 同时触发 image、精确 frame data、value card 和 chart 更新，并存在键盘和 chunk
  segment 接线。

`test_value_viz_api.py` 在临时 `127.0.0.1` 服务上覆盖静态资源、meta、curve、frame、图片、HTTP 400/404、
camera/path 安全校验和服务前后 parquet bytes 不变。

`test_value_viz_package.py` 构建 wheel、检查三份静态资源存在、安装到隔离目录，并从安装后的 package 启动
HTTP 服务读取 HTML/JS/CSS，防止 editable checkout 可用但 wheel 漏资源。

`test_value_viz_real_sample.py` 对真实样例：

```text
/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3
```

使用固定随机种子选择 episode，验证三路相机、随机 episode 曲线限点、当前 frame、第三人称图片和
`extras.parquet` byte-for-byte 只读。当前样例只有 `subtask` / `subtask_progress`，所以该 smoke 同时确认
没有 value columns 时 UI 能正常返回 unavailable。

## 一键验收和实际结果

验收脚本：

```bash
plans/value_pipeline/validate_milestone_5.sh
```

包含本地 HTTP/wheel 和真实样例 smoke 的完整命令：

```bash
LEROBOT_RUN_SOCKET_TESTS=1 \
LEROBOT_RUN_REAL_VALUE_VIZ_SMOKE=1 \
plans/value_pipeline/validate_milestone_5.sh
```

2026-07-14 实际结果：

```text
Milestone 5 data + Milestone 4/direct raw dependencies: 36 passed
Value Viz + existing UI API/wheel service tests: 11 passed
Value pipeline + subtask pipeline regression: 174 passed, 7 skipped
real strike_match_3 Value Viz read-only smoke: 1 passed
py_compile: passed
node --check app.js: passed
CLI --help: passed
git diff --check: passed
```

回归 suite 中的 7 个 skip 是需要显式环境变量的真实数据/GPU smoke，不是失败。Socket/wheel suite 已在
允许 `127.0.0.1` 临时端口的环境中完整运行。

## 剩余边界和下一步

- 当前真实 `strike_match_3` 没有 `value_function_meta.json` 或 Milestone 4 model prediction，因此只能在
  该原始样例上验证图片、boundary 和 unavailable 降级；完整两路径曲线由带确定性 model-pred schema 的
  临时 raw run 自动测试覆盖。
- 真实视觉对照需要先训练正式 value checkpoint，再对 writable/formal raw run 运行
  `lerobot-value-infer --subtask_inference_path both`。UI 本身已能直接读取其输出，不需要代码改动。
- 本阶段没有引入 Playwright/Selenium 等浏览器依赖；交互同步由后端 HTTP 集成测试、frontend 接线测试和
  JavaScript syntax check 覆盖。
- 按推荐实施顺序，下一阶段不是 Milestone 12：应先只使用正式 `model_pred` 依次重新运行 Milestone 6、
  7、8，生成 experiment-eligible advantage、label 和 weight；旧 mock-derived artifact 不得复用。
