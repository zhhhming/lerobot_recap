# Subtask Elapsed-Time Milestone T2 Completion Record

日期：2026-07-17

基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655`

计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

环境：Conda `lerobot-main`，Python 3.12.13，PyTorch 2.10.0+cu128；本阶段测试使用 CPU 和离线模式。

## 完成状态

Milestone T2：训练 noise/dropout 和指标已完成。

本阶段实现训练配置、动态相对噪声、独立 time dropout、训练循环接线、九项窗口累计指标和 PI0/PI0.5
CPU 单步 smoke。没有提前实现 T3 的 time processor、PI0/PI0.5 policy config、converter、token budget 或
checkpoint processor 结构，也没有修改 T4–T8 的 tracker、RTC、deploy、Nero 或 Pico 路径。

T0/T1 已冻结的数据字段和 lookup 语义保持不变；T2 直接消费 T1 wrapper 输出：

```text
subtask_elapsed_seconds
subtask_time_valid
subtask_segment_index  # 仅诊断，T2 helper 不使用
```

## 修改和新增文件

生产代码：

- 新增 `src/lerobot/utils/subtask_time_conditioning.py`；
- 局部修改 `src/lerobot/configs/train.py`；
- 局部修改 `src/lerobot/scripts/lerobot_train.py`。

测试与验收：

- 将 `tests/utils/test_subtask_time_conditioning.py` 的两个 T2 strict xfail 替换为完整可执行测试；
- 新增 `tests/scripts/test_subtask_time_train.py`；
- 新增 `plans/timer_pipeline/validate_subtask_time_milestone_2.sh`；
- 新增本完成记录。

没有修改 `src/lerobot/datasets/subtask_timing.py`、Dataset factory、processor/converter、PI0/PI0.5 config/model、
RTC、deploy、dashboard 或机器人代码。

## Train config 与校验

`TrainPipelineConfig` 新增并可随 train config 序列化：

```python
subtask_time_noise_ratio: float = 0.4
subtask_time_noise_max_seconds: float = 5.0
subtask_time_dropout_prob: float = 0.2
```

校验规则：

- ratio 和 max seconds 必须是 finite、非 bool、非负实数；
- dropout 必须是 finite、非 bool 且位于 `[0, 1]`；
- ratio/max 为 0 是合法的无噪声 ablation，不等于关闭 time；
- time 开启时仅允许 PI0/PI0.5；
- time 开启时要求 `predict_subtask=true`；
- time 开启时拒绝 streaming dataset。

Dataset 必需列仍由 T1 factory 在 metadata/dataset 构造前验证，没有在 T2 重复实现第二套 feature 检查。
T3 尚未增加正式 policy config 字段，因此 T2 保持 `getattr(..., False)` 的前向兼容开关读取。

## Noise/dropout helper

新增：

```python
sample_subtask_time_condition(
    batch,
    noise_ratio,
    noise_max_seconds,
    dropout_prob,
    generator=None,
)
```

实现公式：

```text
amplitude = min(noise_ratio * true_elapsed, noise_max_seconds)
epsilon = (2 * Uniform(0, 1) - 1) * amplitude
noisy = max(0, true_elapsed + epsilon)
kept = valid AND (dropout_uniform >= dropout_prob)
```

输出：

```text
subtask_time_seconds: float32 [B]
subtask_time_condition_kept: bool [B]
```

已锁定的 RNG 契约：

1. 每次 time-enabled helper 调用先 draw 完整 `[B]` noise uniform；
2. 再独立 draw 完整 `[B]` dropout uniform；
3. invalid 样本仍占据两个 draw 的固定位置，但 keep 恒为 false；
4. `p=0`、`p=1` 和 ratio/max 为 0 时仍保持相同 draw shape/order；
5. time disabled 时训练循环完全不调用 helper，因此不消耗 time RNG。

helper 接受 `[B]` 或 `[B,1]` elapsed/valid，严格拒绝缺字段、坏 shape、batch size 不一致、非 bool valid、
bool elapsed、NaN、Inf 和负 elapsed。它浅拷贝 batch 并生成新输出 tensor，不原地修改输入；只做 float32 和
非负 clamp，不 round 到一位小数，文本 round 留给 T3 formatter。

## 训练循环接线

新增 `prepare_training_batch_conditions()`，把训练期条件采样固定为：

```text
raw DataLoader batch
-> advantage condition mask (optional)
-> memory condition mask (optional)
-> subtask-time noise/dropout (optional)
-> preprocessor
-> policy.forward
```

三个 condition 分别调用各自 helper，不共享 keep mask，也不覆盖彼此字段。Current-subtask-to-FM dropout 仍在
PI0/PI0.5 model 内独立采样；T2 测试覆盖 time keep/drop 与 subtask attention keep/drop 的四种边界组合。

Loss 保持现有公式：

```text
loss = FM + subtask_ce_loss_weight * current_subtask_CE
```

Time 不新增 loss、不改变 current CE reduction、不改变 advantage weighting 或 RA-BC，也不删除 current subtask
AR target。

## 指标

新增 `SubtaskTimeTrainingMetrics` 和单 batch helper，按 log window 累计：

```text
subtask_time/valid_fraction
subtask_time/condition_kept_fraction
subtask_time/dropout_fraction_among_valid
subtask_time/true_seconds_mean
subtask_time/true_seconds_max_seen
subtask_time/noisy_seconds_mean
subtask_time/noise_abs_mean
subtask_time/noise_abs_max_seen
subtask_time/clamped_to_zero_fraction
```

统计口径：

- valid fraction 和 condition kept fraction 的分母是 window 内全部样本；
- dropout、true/noisy/noise 和 clamped fraction 的分母是 valid 样本；
- `clamped_to_zero` 只统计 `true > 0` 且 noisy 为 0 的实际 clamp，不把合法的第一帧 `0.0s` 算成 clamp；
- mean 按样本数加权，不对 batch mean 再求平均；
- max 保存整个 window 的真实 extrema；
- 指标只记录到主进程现有 logging/W&B 路径，不进入 loss；
- 每个 log window 后与现有 memory metrics 一样 reset。

## 测试覆盖

T2 聚焦测试共 49 个通过，覆盖：

- `x=0/1/12.5/40.5/95.8`；
- ratio `0/0.4/2.0`、max `0/5/10`；
- `x=0` 始终为 0、1 秒幅度不超过 0.4、40.5/95.8 秒幅度不超过 5；
- 大相对噪声的非负 clamp；
- helper 不 round processor 输入；
- fixed generator 完整复现；
- 两个 full-batch RNG draw 的固定顺序；
- invalid、`p=0`、`p=1` keep 语义；
- 输入不修改、输出 dtype/shape；
- 参数、字段、finite、non-negative、shape 和 dtype 错误；
- 九项 metrics 单 batch 手算、跨 batch 累计、extrema 和 reset；
- config 默认值、序列化、合法 ablation 和 early validation；
- advantage/memory/time 的 8 种边界组合；
- time/current-subtask attention dropout 的 4 种边界组合；
- time off 不调用 helper、不消耗 RNG；
- PI0 和 PI0.5 各完成一次 CPU forward/backward/clip/update，loss、grad norm 和输出 finite。

实际聚焦结果：

```text
49 passed in 1.54s
```

独立 smoke 模式：

```text
2 passed, 23 deselected in 1.43s
```

沙箱内累计聚焦回归：

```text
293 passed, 7 deselected, 4 xfailed, 3 warnings in 4.26s
```

其中：

- 4 个 strict xfail 仅为尚未实施的 T3 processor/config 和 T4 tracker 契约；T2 已无 xfail；
- 7 个 deselected 全是名称含 `workers` 的既有 T1/memory 多进程用例；
- 这些用例在 T1 完成记录的正常进程环境已实际通过，T2 没有修改它们覆盖的 timing/history wrapper；
- 3 个 warning 是 1 个受限环境 NVML warning 和 2 个既有 subtask decode-budget warning。

其他门禁：

```text
py_compile: passed
bash -n: passed
git diff --check: passed
ruff: skipped (not installed in lerobot-main; not recorded as passed)
```

## 一键验收入口

```bash
plans/timer_pipeline/validate_subtask_time_milestone_2.sh [contract|smoke|regression|all]
```

模式：

- `contract`：T2 py_compile/pytest、T0/T1 累计 contract、条件式 Ruff；
- `smoke`：PI0/PI0.5 CPU 单步更新；
- `regression`：memory/advantage/current-subtask 聚焦回归和 T1/M0–M8 累计 regression；
- `all`：依次运行以上三项；默认模式。

脚本固定离线环境，不删除 outputs，最后执行 `bash -n` 和 `git diff --check`。

当前受限沙箱内直接运行包含 `DataLoader(num_workers=2)` 的累计脚本会阻塞，这与 T0/T1 已记录行为一致。
本次申请在沙箱外运行仓库本地脚本被安全审查拒绝，因此没有把正式 `all` 写成 passed；本记录只声明上面实际执行的
49 项 T2 测试、2 项 smoke、293 项沙箱安全累计回归及 T1 已有多 worker 结果。正式正常进程环境可重复执行
`validate_subtask_time_milestone_2.sh all` 获得无排除的累计结果。

## 工作区保护复核

开始和完成时均确认工作区包含尚未提交的 memory M0–M8、计划目录移动和 Nero 用户改动。本阶段没有执行 reset、
checkout、clean、暂存、commit 或无关格式化。

`src/lerobot/configs/train.py` 和 `src/lerobot/scripts/lerobot_train.py` 原先已有 staged memory/advantage 改动；T2
仅在工作树上叠加局部 elapsed-time patch，因此状态显示 `MM` 是保留 staged 内容的预期结果。其他用户文件未被
回退或覆盖。

## 下一阶段输入

T3 可以直接消费 T2 输出：

```text
subtask_time_seconds
subtask_time_valid
subtask_time_condition_kept
```

下一阶段应实现 deterministic formatter/processor、converter route-through、PI0/PI0.5 policy config、pipeline
顺序、token budget、processor registry/serialization 和 checkpoint structural rebuild。T3 不应把 noise/dropout
移入 processor，也不应改变本阶段固定的 RNG、metrics 或 loss 契约。
