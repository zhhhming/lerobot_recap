# Value Pipeline Milestone 4 Completion Record

日期：2026-07-14

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 推荐实施顺序中的第十一阶段：
Milestone 4（value model 推理并写回 raw dataset）。

此前 Milestone 0/1/6/7、1.5、8/9/10/11、2/3 已按推荐顺序完成，因此本阶段继续完成
Milestone 2/3/4 组中的最后一项，而不是按 milestone 数字回到 Milestone 11。

本阶段新增训练后 value checkpoint 的完整 raw-run batched inference、checkpoint/raw compatibility、
distributional expectation 输出消费、canonical subtask Viterbi、`gt_conditioned` / `pred_smooth` paired
head writeback、run-level atomic extras merge 和 model-prediction provenance。没有实现 Milestone 5 value
曲线 UI，也没有提前重新运行正式 Milestone 6/7/8 artifact。

## 修改文件

- `src/lerobot/value_function/inference.py`
- `src/lerobot/value_function/__init__.py`
- `src/lerobot/scripts/lerobot_value_infer.py`
- `pyproject.toml`
- `tests/value_function/test_value_infer_writeback.py`
- `plans/value_pipeline/validate_milestone_4.sh`
- 本完成记录

Milestone 0 已定义本阶段需要的 prediction 字段和 `VALUE_INFERENCE_STAGE_PREFIX`，Milestone 2 已实现
distribution expectation 与 paired head selection helper，Milestone 3 已实现 checkpoint strict reload 和
raw value dataset。本阶段直接复用这些契约，没有复制另一套模型、dataset 或 extras 写回实现。

## 新增 CLI

安装后入口：

```bash
lerobot-value-infer \
  --root /path/to/raw/run \
  --checkpoint /path/to/value/checkpoint.pt \
  --mode both \
  --batch_size 8 \
  --device auto \
  --subtask_inference_path both \
  --transition_penalty 0.0 \
  --allow_subtask_skip false
```

参数行为：

- `--mode=global|subtask|both` 只能选择 checkpoint 实际包含的 head；`both` 要求 checkpoint 也是 both。
- `--image_keys` 不提供时使用 checkpoint camera keys；显式提供时必须与 checkpoint 的 key 和顺序完全一致。
- `--subtask_inference_path=gt_conditioned|pred_smooth|both` 控制写回哪些 paired remaining-head value；
  raw classifier ID、confidence 和 Viterbi smooth ID/name 在所有 subtask 推理中都会写回。
- `--allow_subtask_skip=false` 是当前 success-only/full-subtask 数据契约的默认值。此时 Viterbi 强制从
  canonical subtask 0 开始、在最后一个 subtask 结束，每次只允许 stay 或进入下一个 subtask。
- `--allow_subtask_skip=true` 是显式逃生开关，允许向更后的 canonical state 跳转；这类输出仍可能被当前
  正式 advantage 的严格完整路径校验拒绝，不能静默改变第一阶段数据假设。

## Checkpoint/raw compatibility

推理在创建 DataLoader 和写回前验证：

- checkpoint mode 和 requested mode；
- robot type 和 fps；
- image key、camera 顺序与 image feature schema；
- state key、state feature schema 和 state dimension；
- global/subtask bin count；
- global scale；
- canonical subtask order、subtask count 和每个 subtask 的 frame scale；
- target stage current/provenance；
- 如果目标 root 就是 checkpoint 的某个训练 root，当前 target stage fingerprint 必须与训练时一致。

兼容的新 raw root 可以使用同一 checkpoint，不要求路径与训练 root 相同，但上述结构、scale 和 canonical
order 必须一致。本阶段没有实现隐式整数 remap；不兼容时明确失败。

checkpoint 使用 Milestone 3 的 `load_value_function_checkpoint()` strict reload，不会重新读取训练前的
PI0/PI0.5 policy safetensors。推理 metadata 同时记录 checkpoint absolute path、SHA256、step 和 epoch。

## Batched inference 和输出字段

推理使用无 augmentation、非 shuffle 的 `RawValueFrameDataset` / DataLoader，完整消费 checkpoint 内保存的
state normalization。每个 batch 对 model output 做 shape、finite 和 normalized `[0,1]` 校验；所有 episode
和 frame 都成功后才构造 Arrow writeback。

global 输出：

- `value_global_remaining_norm_pred: float32`
- `value_global_remaining_frames_pred: float32`

其中 frame value 使用 checkpoint/raw 兼容性校验后的 canonical global scale 恢复。

subtask classifier 和 smoothing 输出：

- `value_subtask_id_pred: int32`，raw classifier top-1 id；
- `value_subtask_confidence: float32`，raw classifier top-1 probability；
- `value_subtask_id_pred_smooth: int32`；
- `value_subtask_name_pred_smooth: string`。

本仓库 Milestone 0 已将 raw ID canonical 字段固定为 `value_subtask_id_pred`，因此没有另写一个含义重复的
`value_subtask_id_pred_raw` 列。

paired remaining-head 输出：

- `gt_conditioned`：
  - `value_subtask_remaining_norm_pred_gt_head`
  - `value_subtask_remaining_frames_pred_gt_head`
- `pred_smooth`：
  - `value_subtask_remaining_norm_pred_smooth_head`
  - `value_subtask_remaining_frames_pred_smooth_head`

两个 path 都调用 Milestone 2 的 `select_paired_subtask_head()`，head ID 和 boundary ID 不能分别配置。
frame value 按该 path 实际选中的 subtask ID 使用对应 canonical subtask scale 恢复。原模糊的
`value_subtask_remaining_*_pred` 兼容常量仍不作为正式输出或 downstream source。

## Canonical Viterbi

`monotonic_viterbi()` 输入每帧 classifier log probability。默认路径约束为：

1. 第一帧从 canonical ID 0 开始；
2. 每帧只能保持当前 ID 或进入 `ID+1`；
3. 最后一帧必须到达最后一个 canonical subtask；
4. 因而每个 subtask 至少覆盖一帧，压缩 ID sequence 与 canonical order 完全一致；
5. episode frame 数不足以覆盖全部 subtask、概率非有限或不存在合法路径时拒绝写回。

`transition_penalty` 被纳入动态规划分数并写入 stage config。当前默认完整路径的 transition 数固定，默认值
为 `0.0`；显式允许 skip 后，跳跃距离也会计入 penalty。

## Atomic writeback 和 provenance

推理先在内存中完成所有 batch、按 episode 重组并验证 frame index 完整性。model forward、Viterbi、
compatibility 或列构造任一阶段失败时，不调用 extras merge，所有原始 `extras.parquet` bytes 保持不变。

全部成功后调用 Milestone 0 的 `merge_raw_run_extras()`：

- 保留原 subtask、target、mock、advantage 等无关列；
- 对所有 episode 预构造并校验最终 schema；
- 所有 parquet temp staging 完成后才开始 replace；
- commit 失败使用既有 run-level rollback。

stage metadata 分别写为：

- `value_inference.global`
- `value_inference.subtask`

`mode=both` 不创建下游无法识别的单一 `value_inference.both` stage。每个 stage 独立保存 normalized config、
input/output columns 和 fingerprint、checkpoint provenance、`prediction_source=model_pred`、
`synthetic=false`，并依赖 `targets` stage。重跑 inference 会通过既有 dependency graph 把旧
advantage/label/weight 标为 stale。

## 测试覆盖

`tests/value_function/test_value_infer_writeback.py` 使用两 episode fake raw run、确定性 differentiable model
和符合 Milestone 3 格式的 checkpoint，覆盖：

- canonical Viterbi 起止、单调、不可跳过与非法/非有限输入；
- two-episode `both` inference、batch 跨 episode 边界和统一 Arrow schema；
- global normalized/frame scale 恢复；
- raw ID、top-1 confidence、smooth ID/name；
- GT-head 与 smooth-head 的 ID、normalized value 和各自 subtask frame scale 完全 paired；
- `gt_conditioned` / `pred_smooth` 单 path stage manifest 不混入另一条 value column；
- model forward 中途失败前不写任何 episode；
- checkpoint global scale、image schema、state schema、subtask order/scale mismatch 拒绝；
- 同一 training root 的 target stage fingerprint 改变后拒绝旧 checkpoint；
- `value_inference.global` / `.subtask` 的 source、synthetic、dependency、hash 和 output fingerprint；
- `lerobot-build-dataset --dry_run` 识别 prediction dtype/schema；
- 两条 model-pred paired path 都能被现有 Milestone 6 advantage 实际消费；
- inference 重跑使旧 downstream stage stale 后可生成另一条 paired-path advantage。

所有自动测试只修改 pytest 临时目录，没有修改真实 raw dataset。

## 一键验收和实际结果

验收脚本：

```bash
plans/value_pipeline/validate_milestone_4.sh
```

默认结果：

```text
Milestone 4 + direct dependency tests:
87 passed in 3.42s

Value pipeline + subtask pipeline regression（排除受限 sandbox HTTP 文件）:
168 passed, 7 skipped in 5.10s

CLI --help: passed
py_compile: passed
git diff --check: passed
```

当前 sandbox 禁止创建 `127.0.0.1` socket，因此脚本默认排除以下四个与 Milestone 4 无关的既知 HTTP/UI
测试文件：

- `test_advantage_labeler_api.py`
- `test_advantage_labeler_package.py`
- `test_advantage_weight_viz_api.py`
- `test_advantage_weight_viz_package.py`

允许 loopback socket 的环境可设置 `LEROBOT_RUN_SOCKET_TESTS=1` 运行包括它们在内的完整回归。

当前环境没有安装 `ruff`（`No module named ruff`），因此未报告 ruff 结果；已执行 py_compile、专项测试、
相关完整回归、CLI help 和 diff whitespace 检查。

## 可选真实 checkpoint smoke

验收脚本支持：

```bash
LEROBOT_VALUE_CHECKPOINT=/path/to/value/checkpoint.pt \
LEROBOT_VALUE_INFER_SHADOW_RUN=/path/to/writable/shadow/raw_run \
plans/value_pipeline/validate_milestone_4.sh
```

可选变量还包括：

- `LEROBOT_VALUE_INFER_MODE`
- `LEROBOT_VALUE_INFER_PATH`
- `LEROBOT_VALUE_INFER_BATCH_SIZE`
- `LEROBOT_VALUE_INFER_DEVICE`

该路径要求调用者显式提供 writable shadow run，避免验收脚本意外修改正式 raw 数据。

本机当前有真实 `strike_match_3` raw 样例和 PI0 policy checkpoint，但没有 Milestone 3 长时训练产出的正式
value `checkpoint.pt`，因此本次没有伪造“真实全量 53,794 帧 model prediction”结果。正式 value training
完成后，应先在 shadow run 执行上述 smoke，再对正式 raw run 运行 CLI。

## 剩余边界和下一阶段

- 本阶段完成推理能力和可重复验证，不把 synthetic/mock prediction 当作 model prediction。
- 正式 checkpoint 产出后仍需在原始完整 run 上执行本 CLI，才能得到可用于正式实验的 model artifact。
- 推荐顺序的下一阶段是 Milestone 5：实现 global/subtask 曲线和两条 inference path 对照 UI。
- Milestone 5 后必须重新运行 Milestone 6/7/8，只使用本阶段 `model_pred` stage 生成正式 advantage、label
  和 weight；旧 mock-derived artifact 不能复用。
