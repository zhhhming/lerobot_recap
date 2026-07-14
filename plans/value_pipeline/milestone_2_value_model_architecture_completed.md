# Value Pipeline Milestone 2 Completion Record

日期：2026-07-14

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 推荐实施顺序中的第九阶段：
Milestone 2（value model 架构）。

Milestone 0/1、6/7、8/9/10/11 已按推荐顺序完成，其中 Milestone 1 的 review remediation 已包含
Milestone 1.5 synthetic mock prediction。因此 Milestone 11 是第八阶段，本阶段从推荐顺序下一组
Milestone 2/3/4 中先完成模型架构，不按 milestone 数字从 1 顺序重跑。

本阶段实现可训练的 PI0/PI0.5 value model、global/subtask distributional heads、two-hot loss、state
融合、paired subtask head selection 和选择性 policy checkpoint 加载。没有提前实现 Milestone 3 的 raw
dataset/DataLoader/训练 CLI，也没有实现 Milestone 4 的整库推理、Viterbi smoothing 或 extras 写回。

## 修改文件

- 新增 `src/lerobot/value_function/configuration.py`
- 新增 `src/lerobot/value_function/modeling_pi0_value.py`
- 修改 `src/lerobot/value_function/__init__.py`
- 新增 `tests/value_function/test_value_model_shapes.py`
- 新增 `tests/value_function/test_value_model_checkpoint.py`
- 新增 `tests/value_function/test_value_model_real_checkpoint.py`
- 新增 `plans/value_pipeline/validate_milestone_2.sh`
- 新增本完成记录

总计划中建议的 `src/lerobot/value_function/dataset.py` 留到 Milestone 3。episode split、raw image/state
加载、augmentation 和训练 DataLoader 属于训练阶段；本阶段只固定模型接收的 batch 和 target 契约。

没有修改旧的 `src/lerobot/rl/algorithms/recap_value_network_pi0.py`，避免把旧原型的单 global head、
101 bins、`[-1, 0]` support 和无 state 契约混入新 pipeline。

## 模型配置

新增 `ValueFunctionConfig`，支持：

- `mode=global|subtask|both`；默认 `global`，`subtask|both` 必须显式提供动态 `num_subtasks`。
- `backbone_type=pi0|pi05|vision_only`。
- 默认 `paligemma_variant=gemma_2b`，默认截取前 `num_vlm_layers=3` 层。
- 默认三路图片：`left_wrist`、`right_wrist`、`third_person`，允许通过 `image_keys` 改成任意非空组合。
- `num_bins` 以及独立 `global_num_bins` / `subtask_num_bins` override，全部强校验 `>=2`。
- `use_elapsed_aux`、`use_state`、state key/dimension、state/fusion/head hidden dimension。
- freeze vision encoder、freeze backbone、只解冻最后 N 个 VLM layer。
- `remaining_loss_weight=1.0`、`subtask_ce_loss_weight=0.2`、`elapsed_loss_weight=0.0`。
- config `to_dict/from_dict` round-trip，拒绝未知字段、非法 shape/bin/loss/freeze 组合。

默认 pretrained source 按 backbone 自动解析：

- `pi0` / `vision_only`：`lerobot/pi0_base`
- `pi05`：`lerobot/pi05_base`

可使用 `pretrained_path` 指向本地 `model.safetensors` 或包含它的 checkpoint 目录。Milestone 2 默认不
使用 task/subtask 文本，`use_task_text=true` 会明确拒绝，不会静默假设模型看到了语言输入。

## PI0/PI0.5 backbone

新 backbone 只构建 value 需要的模块：

```text
configured camera images
  -> PaliGemma/SigLIP vision tower
  -> multimodal projector
  -> first N PiGemma layers (pi0/pi05)
  -> mean pool
```

`vision_only` 使用相同 vision tower 和 projector，但跳过 PiGemma layers。所有路径都不实例化或加载
action expert。

由于 value model 不输入 text token，PiGemma 直接接收视觉 `inputs_embeds`，不保留 tokenizer、LM head
或 vocabulary embedding。这既固定了 no-task-text 契约，也避免无意义的巨大 vocabulary 参数。

图片输入支持 `[B,C,H,W]` / `[B,H,W,C]` 和最后一个 observation step 的 5D batch，按 PI0 规则 resize
并从 `[0,1]` 转到 `[-1,1]`。configured image key 缺失时明确报错，不静默换用其他相机。

## 选择性 policy checkpoint 加载

旧 RECAP prototype 使用 `safetensors.load_file` 一次加载完整 PI0 policy；当前本地 policy checkpoint
约 16 GB，其中包含与 value model 无关的 action expert 和未使用 VLM layers。

新实现使用 `safe_open` 按 key 读取，只加载当前架构实际拥有的：

- vision tower；
- multimodal projector；
- configured 前 N 个 `language_model.layers`；
- language model final norm。

支持当前 PI0 和 PI0.5 policy checkpoint prefix。action expert、LM head、embedding 和被截掉的 VLM
layers 不读取。required tensor 缺失或 shape 不一致时在 forward 前明确报错。

真实本地 smoke 已分别验证：

- PI0 checkpoint -> global value head；
- PI0.5 checkpoint -> 动态 subtask value heads；
- PI0 checkpoint -> vision-only backbone。

三种路径均使用一个 `third_person` image、一个 batch、8 bins 做离线 forward，输出 shape 正确且全部
finite。默认三路相机和更大 bins 由轻量 shape tests 覆盖，避免真实 checkpoint smoke 重复占用大量
CPU 内存。

## State encoder 和 checkpoint contract

默认使用 `observation.state`：

```text
state
  -> (state - mean) / max(std, eps)
  -> LayerNorm + Linear + GELU
  -> concatenate visual/VLM pooled feature
  -> fusion LayerNorm + Linear + GELU + Dropout
```

state mean/std 是 persistent model buffers，`set_state_normalization_stats()` 强校验维度、finite 和正
std。`checkpoint_payload()` 同时保存：

- 完整 model config；
- model state dict（包括 mean/std buffers）；
- 可审计的 state key、mean、std metadata。

`use_state=false` 完全绕过 state branch，batch 不需要 state key。Milestone 3 负责从训练 roots 计算并
设置真实 normalization stats；在此之前默认 mean=0、std=1。

## Distributional heads 和输出

global 模式输出：

```text
global_remaining_logits: [B, global_num_bins]
global_remaining_value:  [B]
global_elapsed_logits:   [B, global_num_bins]  # optional
global_elapsed_value:    [B]                   # optional
```

subtask 模式输出：

```text
subtask_logits:           [B, num_subtasks]
subtask_remaining_logits: [B, num_subtasks, subtask_num_bins]
subtask_remaining_value:  [B, num_subtasks]
subtask_elapsed_logits:   [B, num_subtasks, subtask_num_bins]  # optional
subtask_elapsed_value:    [B, num_subtasks]                    # optional
```

`both` 同时输出两组 heads。decoded value 使用 softmax distribution 对 `[0,1]` bin support 做 expectation，
argmax 不作为 canonical value。

## Target、loss 和 paired head contract

新增 helper：

- normalized scalar -> adjacent-bin two-hot distribution；
- soft cross entropy；
- distributional expectation decode；
- `[B,num_subtasks,num_bins]` 按 `[B]` subtask id gather；
- paired inference-path head selection。

subtask remaining/elapsed loss 只 gather 当前 `value_subtask_id_gt` head。未被当前样本选择的其他 head
不会收到该样本的 head gradient；subtask classifier 仍在所有 frame 上训练。GT id 的 fractional、负数或
越界值均明确拒绝。

paired helper 只暴露一个 `inference_path`：

```text
gt_conditioned -> value_subtask_id_gt
pred_smooth    -> value_subtask_id_pred_smooth
```

helper 同时返回对应 canonical output field name，调用方不能分别指定 head source 和 boundary/id source。
Milestone 4 将调用该 helper 生成 `_pred_gt_head` / `_pred_smooth_head` 写回列。

loss 总和规则：

```text
remaining_loss_weight * sum(active global/subtask remaining losses)
+ subtask_ce_loss_weight * subtask CE（如启用 subtask）
+ elapsed_loss_weight * sum(active global/subtask elapsed losses)
```

## 测试覆盖

专项单元测试覆盖：

- global bins `64/128/256/512` 输出 shape。
- 动态 `num_subtasks=3/7`，没有写死样例数据的 6 个 subtask。
- global/subtask 独立 bin 数和 `both + elapsed` heads/loss。
- two-hot sum、边界、相邻 bin 插值和 expectation `[0,1]`。
- selected subtask head gradient，以及未选 heads 的零 gradient。
- `gt_conditioned` / `pred_smooth` 正确 paired selection、缺列和非法 path 拒绝。
- `use_state=true|false`、state stats/config/model state round-trip。
- fractional/out-of-range GT subtask id 拒绝。
- config 非法值和未知字段拒绝。
- synthetic safetensors 的 PI0/PI0.5 key mapping、选择性读取、missing/shape mismatch 拒绝。
- freeze backbone/vision 和最后 N 层解冻行为。
- 本地真实 PI0、PI0.5、vision-only checkpoint forward。

## 一键验收和结果

验收脚本：

```bash
plans/value_pipeline/validate_milestone_2.sh
```

脚本支持通过以下环境变量覆盖 Python 和真实 checkpoint：

```text
LEROBOT_PYTHON
LEROBOT_PI0_CHECKPOINT
LEROBOT_PI05_CHECKPOINT
```

最终实际结果：

```text
专项 config/shape/loss/checkpoint tests:
23 passed in 1.54s

现有 value pipeline + subtask data pipeline 回归：
155 passed, 6 skipped in 10.46s

真实 PI0 / PI0.5 / vision-only checkpoint smoke：
3 passed in 11.00s

py_compile: passed
git diff --check: passed
```

完整回归第一次在受限 filesystem/network sandbox 中运行时，7 个已有 UI/API/package tests 因无法绑定
`127.0.0.1` 临时端口而得到 `PermissionError: Operation not permitted`。在允许本机 loopback socket 的
验收环境中原命令重跑后全部通过；这些失败不是代码或测试断言回归。

当前环境没有安装 `ruff`，因此没有报告 ruff 结果；已执行 `py_compile`、专项/完整 pytest、真实模型
smoke 和 `git diff --check`。

## 剩余边界和下一阶段

- 本阶段没有提供训练 CLI。下一阶段 Milestone 3 需要实现 raw dataset adapter、多个 root compatibility、
  episode-level train/val split、augmentation、optimizer/scheduler、metrics 和训练 checkpoint 文件。
- `checkpoint_payload()` 固定了模型 checkpoint 内容，但由 Milestone 3 决定落盘目录、best/last 策略和
  resume 行为。
- 当前只允许 visual + state 输入，不允许 task text；如未来做 text ablation，需要新增 tokenizer/text
  embedding contract 和独立测试，不能静默改变本阶段模型输入。
- subtask Viterbi smoothing 和每 episode predicted id path 属于 Milestone 4；本阶段只固定 paired gather
  API。
- 当前真实 smoke 使用本机已有 fine-tuned PI0/PI0.5 policy checkpoint验证权重兼容性。Hub 默认
  `lerobot/pi0_base` / `lerobot/pi05_base` 未在当前 cache 中；实际训练若使用 Hub source，需要缓存或网络。
- 本阶段没有修改 raw dataset，也没有生成可用于正式 advantage 实验的 model prediction。正式产物仍需
  完成 Milestone 3/4 后重新运行 Milestone 6/7/8。
