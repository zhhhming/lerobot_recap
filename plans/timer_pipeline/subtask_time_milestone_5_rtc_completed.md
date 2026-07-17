# Subtask Elapsed-Time Milestone T5 Completion Record

日期：2026-07-17

基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655`

计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

环境：Conda `lerobot-main`；CPU；`HF_HUB_OFFLINE=1`；`TRANSFORMERS_OFFLINE=1`。

## 完成状态

Milestone T5：RTC transaction 闭环已完成。

本阶段把 T1 的 strict `SubtaskSequenceContract`、T3 的 `SubtaskTimeConditionProcessorStep` 和 T4 的
`SubtaskTimeTracker` 接入异步 `RTCInferenceEngine`。每轮用于 prompt 的 elapsed time 在 observation 和 ActionQueue
资格确认后、build/prepare/preprocess 前只采样一次；模型输出只有在 predict、postprocess、reset-version 检查和
`ActionQueue.merge()` 全部成功后才提交 tracker。History memory 与 time tracker 在同一个 `_state_lock` 临界区提交，
debug reader 不会观察到撕裂状态。

本阶段没有提前实现 T6 的 deploy dataset/config 加载、Space soft pause、Home/full reset 区分、dashboard TIME 行或
Nero README 更新。当前 `reset()` 延续既有 full-reset 语义并清 tracker；普通 pause/resume 的 timer 冻结接线仍属于
T6。

## 修改和新增文件

生产代码：

- 修改 `src/lerobot/inference_engines/rtc.py`；
- 小幅扩展 `src/lerobot/inference_engines/subtask_time_tracker.py`，允许使用调用方已经采样并验证的 monotonic 时间点
  生成 snapshot，避免 prompt 输入阶段重复读取 clock。

测试与验收：

- 新增 `tests/inference_engines/test_rtc_subtask_time.py`；
- 新增 `plans/timer_pipeline/validate_subtask_time_milestone_5.sh`；
- 更新主计划的 T5 完成状态和 completion record 路径；
- 新增本完成记录。

没有修改 Dataset schema、PI0/PI0.5 model、训练循环、processor pipeline、deploy 脚本、terminal dashboard、Nero 或
Pico 文件。

## 测试先行记录

生产代码修改前先新增 11 条 RTC time transaction 测试，首次实际结果：

```text
11 failed in 1.54s
```

11 条失败均为：

```text
TypeError: RTCInferenceEngine.__init__() got an unexpected keyword argument
'subtask_sequence_contract'
```

失败原因精确对应 T5 engine 接口和接线尚未实现，不是 fixture、collection、import、shape 或断言错误。实现后相同
专项测试结果：

```text
11 passed in 1.99s
```

## Engine 构造和关闭路径

`RTCInferenceEngine` 新增可选参数：

```text
subtask_sequence_contract
subtask_time_enabled
subtask_time_clock
```

行为：

- effective time 开启时必须提供 strict sequence contract，并构造一个由 engine state lock 保护的 tracker；
- effective time 关闭时不构造、不维护 tracker，也不注入任何 `subtask_time_*` 字段；
- 参数均有安全默认值，现有 HIL、deploy、memory smoke 和其他 RTC caller 不需要提前接入 T6 配置；
- time 开启但 checkpoint `subtask_generate_at_inference=false` 时发出明确 warning，仍注入 invalid/no-op time，永不
  推进 tracker；
- clock 默认 `time.monotonic`，测试注入 fake clock，不使用真实 sleep 估算 elapsed。

## 推理开始的单次 time snapshot

既有 memory/reset-version snapshot 保持在 observation 读取前，继续保留 Memory Milestone 6 已修正的 reset race
契约。取得 observation 并通过 queue threshold 后，engine 再次在 `_state_lock` 下验证 active/reset version，并为
本次 transaction 采样一次 monotonic time。

该明确时间点传给 tracker snapshot，本轮后续 build、prepare、time field 注入、preprocess 和 predict 都使用同一个
不可变结果：

```text
tracker 尚未 start:
  subtask_time_seconds=[0.0]
  subtask_time_valid=[False]
  subtask_time_condition_kept=[False]

tracker 已 start:
  subtask_time_seconds=[effective_elapsed]
  subtask_time_valid=[True]
  subtask_time_condition_kept=[True]
```

首轮 processor 因 invalid time 保持 task 不变；首个合法 subtask 成功 commit 后，下一轮 processor 生成 canonical：

```text
Subtask elapsed time: 1.2s
```

输入 snapshot 不包含本轮尚未发生的 inference latency；tracker 自身持续使用 monotonic active time，因此本轮 latency
会出现在后续 debug/下一轮输入中。ActionQueue 满时 engine 不暂停 tracker，等待时间同样正常累计。

## Merge 后的原子语义提交

一次成功 transaction 的提交顺序为：

1. predict 返回 batch size 1 action，并读取 batch size 1 subtask candidate；
2. postprocess 成功；
3. 在 `_state_lock` 下复核 shutdown、active 和 reset version；
4. `ActionQueue.merge()` 成功；
5. tracker 按 strict current/next/old/skip/unknown 规则提交 candidate；
6. 在同一 state lock 内提交 history input/output/next/source、last time input 和 inference count。

predict、postprocess 或 merge 抛错时不会触发 tracker commit。推理中发生 reset-version 变化时，candidate 和 action
chunk 都在 merge 前丢弃。`reset()` 在同一个 state 临界区 full-reset tracker、清 last time input、memory、queue 和
observation；之后再 reset policy/preprocessor/postprocessor runtime state。

## Debug snapshot

`debug_snapshot()` 在同一个 `_state_lock` 下取得 tracker frozen snapshot，并新增：

```text
subtask_time_enabled
subtask_time_current_index
subtask_time_current_name
subtask_time_raw_elapsed_seconds
subtask_time_effective_seconds
subtask_time_cap_seconds
subtask_time_valid
subtask_time_running
subtask_time_paused
subtask_time_last_transition
subtask_time_last_rejected_output
subtask_time_last_rejection_reason
subtask_time_last_input_seconds
```

time disabled 时这些 key 仍提供稳定的 disabled/no-time debug 视图，但 engine 内没有 tracker。并发 reader 测试证明
`inference_count`、last output、next memory、time index 和 last time input 只会属于初始状态或某一次完整 commit。

## 专项测试覆盖

11 条新测试覆盖：

- PI0、PI0.5 首轮 invalid time 和成功首项后的次轮有效 time；
- inference-start clock 发生在 build/prepare/preprocess 前，且 prompt snapshot 每轮只采一次；
- inference latency 和 queue-full 等待计入 active elapsed；
- predict、postprocess、merge 三个失败阶段均不推进；
- reset-version race 丢弃 next candidate 并 full-reset tracker；
- history memory 与 time tracker 并发 debug snapshot 不撕裂；
- time disabled 零字段、零 tracker 路径；
- generation disabled warning/no-time ablation；
- multi-sample action/subtask 输出继续 batch-size-1 早失败。

与 T4 tracker 和既有 RTC memory 的组合专项结果：

```text
52 passed in 3.95s
```

## 一键验收入口

```bash
plans/timer_pipeline/validate_subtask_time_milestone_5.sh [contract|data|regression|all]
```

模式：

- `contract`：py_compile、T5 RTC time、T4 tracker、RTC memory 和 T0–T4 累计 contract；
- `data`：复用生产 scanner 离线扫描本地 match/egg 全量轻量列；
- `regression`：T3 checkpoint、memory/subtask/advantage/time 聚焦回归和完整 `tests/policies/rtc`；
- `all`：依次执行以上三项，默认模式。

脚本固定 offline、默认使用 `lerobot-main`，最后执行 `bash -n` 和 `git diff --check`。Ruff 仅在环境中存在时
执行；当前环境未安装 Ruff，因此明确记录 skipped。

## 最终实际验收结果

最终命令：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_5.sh all
```

实际结果：

```text
T5 RTC time + T4 tracker + RTC memory: 52 passed in 3.88s
T4 tracker contract: 30 passed in 1.48s
T1 scanner focus: 23 passed, 21 deselected in 0.02s
T3 processor/converter/time-disabled contract: 54 passed in 1.46s
T0-T3 cumulative time contract: 119 passed, 4 deselected in 1.57s
checkpoint contract: 6 passed in 1.82s
memory/subtask/advantage/RTC/time focused regression:
  263 passed, 6 deselected, 2 warnings in 4.04s
complete RTC policy + memory + time regression:
  223 passed, 3 skipped in 3.76s
script exit code: 0
py_compile: passed
bash -n: passed
git diff --check: passed
ruff: skipped (not installed)
```

两个 warning 是既有测试刻意设置 `subtask_max_decode_tokens > subtask_max_tokens`；3 个 skip 是既有 RTC CUDA
用例。6 个 deselected 是受限环境中排除的既有 DataLoader worker 用例，已在 T1 正常进程环境通过。没有新增
failure、xfail、XPASS 或未说明 warning。

## 真实数据复核

### Strike-match

```text
repo_id: ming326/strike_match_3_subtask
episodes: 70
frames: 53794
fps: 30.0
ordered subtasks: 6
construction: 3.063884s
lookup: 699322 bytes / 0.666925 MiB
largest maximum: 11.533333s
largest deployment cap: 16.533333s
```

### Nero egg

```text
repo_id: ming326/nero_egg_subtask
episodes: 61
frames: 350010
fps: 30.0
ordered subtasks: 12
construction: 19.139473s
lookup: 4550130 bytes / 4.339342 MiB
Stir the beaten eggs.: 43.900000s maximum / 48.900000s cap
Start frying the eggs.: 95.766667s maximum / 100.766667s cap
```

两个数据集均从本地 cache 离线读取，只扫描轻量 parquet 列，没有视频解码、联网或修改 dataset。

## 工作区保护复核

阶段开始和完成时均检查 `git status --short`。工作区在 T5 前已包含 memory M0–M8、timer T0–T4、计划目录移动、
Nero/Pico 及其他用户改动；这些全部保留。本阶段只局部 patch 本记录列出的 RTC、tracker、测试、validation script
和计划文档，没有执行 reset、checkout、删除、暂存、全仓格式化或覆盖无关文件。

## 下一阶段边界

T5 证明 engine 内部 RTC time transaction 已闭环，但生产 deploy 尚未从标注 dataset 构建并传入 sequence contract。
Milestone T6 需要：

- 解析 deploy effective flag 并检查 checkpoint processor presence；
- 加载 dataset metadata、扫描 sequence/cap 并传入 engine；
- 把 Space pause 拆为保留 tracker 的 soft pause，把 Home/显式 reset 保持为 full reset；
- 接入 tracker pause/resume；
- 把本阶段 debug fields 显示到 live/plain TIME 状态行并添加节流事件日志。

因此本记录不能视为 T6 deploy/UI 或 Nero 实机验收已经完成。
