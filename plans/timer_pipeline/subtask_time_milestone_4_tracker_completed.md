# Subtask Elapsed-Time Milestone T4 Completion Record

日期：2026-07-17

基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655`

计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

环境：Conda `lerobot-main`；CPU；`HF_HUB_OFFLINE=1`；`TRANSFORMERS_OFFLINE=1`。

## 完成状态

Milestone T4：序列扫描和纯状态机已完成。

本阶段复用 T1 已实现的轻量 segment scanner 和 strict `SubtaskSequenceContract`，新增独立、可注入 fake clock 的
`SubtaskTimeTracker`，完成严格输出解析、contract 防御性复核、只前进状态机、可暂停 monotonic timer、deployment
cap 和不可变 snapshot。没有提前修改 RTC transaction、deploy 配置、soft pause/home、dashboard 或 Nero/Pico
路径；这些仍属于 T5–T8。

## 修改和新增文件

生产代码：

- 新增 `src/lerobot/inference_engines/subtask_time_tracker.py`。

测试与验收：

- 将 `tests/inference_engines/test_subtask_time_tracker.py` 的 2 条 strict xfail 替换为 30 条完整测试；
- 新增 `plans/timer_pipeline/validate_subtask_time_milestone_4.sh`；
- 更新主计划的 T4 完成状态和 completion record 路径；
- 新增本完成记录。

T4 没有修改 `src/lerobot/inference_engines/rtc.py`、`lerobot_policy_deploy.py`、模型、processor、训练循环、
Dataset schema、Nero 或 Pico 文件。

## Strict failing test 记录

实现前先移除 T4 的 strict xfail 并扩展正式契约，首个实际失败为：

```text
ModuleNotFoundError: No module named 'lerobot.inference_engines.subtask_time_tracker'
1 failed in 1.49s
```

失败原因精确对应 T4 生产模块尚未实现，不是 fixture、collection 或断言错误。

## 输出解析和 normalization

公开纯函数：

```text
parse_subtask_output_name()
```

解析器只接受可完整匹配的：

```text
Subtask: <non-empty name>; Progress: <anything>
```

字段标签大小写不敏感，允许字段周围空白和换行，返回前折叠 name 内连续空白。它拒绝缺字段、空 name、额外前缀和
一次输出中的第二个 `Subtask:` 字段。名称匹配继续复用 T1 的 `normalize_subtask_name()`：trim、折叠空白、
Unicode casefold、去除一个末尾句号类字符；不做编辑距离、词替换或内部标点模糊匹配。

## Contract 防御性复核

`SubtaskTimeTracker` 构造时不盲目信任传入 dataclass，重新验证：

- 类型必须是 `SubtaskSequenceContract`；
- FPS finite 且大于 0；
- sequence 至少有一个 subtask；
- `canonical_name` 经共享 normalizer 后必须等于保存的 `normalized_name`；
- normalized name 全序列唯一，collision 早失败；
- `max_elapsed_seconds` 和 `deployment_cap_seconds` finite 且非负；
- deployment cap 不得小于 dataset maximum。

T1 scanner 已对逐帧 deploy dataset 执行 episode/frame/index、空 label、重复 label 和跨 episode sequence 一致性
验证；T4 真实数据模式直接复用该生产 scanner，没有另写一套 sequence 规则。

## 严格状态机

状态规则实现为：

- `current_index=None`：仅 index 0 可以 `started`；其他已知项为 `rejected_initial_not_first`；
- 输出当前项：`current`，index 和 timer 起点不变；
- 输出紧邻下一项：`advanced`，index 加一且 timer 从零重新开始；
- 输出旧项：`rejected_old`；
- 输出跨级未来项：`rejected_skip`；
- 未知项：`rejected_unknown`；
- 解析失败：`rejected_parse`；
- 到达最后一项后不会 wrap 到 index 0。

old、skip、unknown 和 parse failure 只更新诊断信息，不修改 current index、累计 elapsed 或 timer 起点。成功的 current、
start 和 advance 会清理上一条 rejection 诊断。

## Timer、pause 和 reset

tracker 默认使用 `time.monotonic`，测试注入 fake clock。内部保存：

```text
current_index
accumulated_active_seconds
running_since_monotonic
paused
last_transition_reason
last_rejected_output
last_rejection_reason
```

行为：

- start/advance 的 timer 起点取 commit clock；
- current 重复输出不重启 timer；
- raw elapsed 持续增长，effective elapsed 为 `min(raw, deployment_cap)`；
- pause 把运行段累积进 active elapsed 并冻结；
- resume 从冻结值重新建立 running start，暂停墙钟时间不计入；
- double pause/double resume 幂等；
- pause-before-first-subtask 合法，首项可在 paused 状态确认但 timer 直到 resume 才运行；
- full reset 清空 subtask、elapsed、running 和 rejection 状态，同时保留调用者控制的 paused session 状态，便于 T6
  组合 home/full-reset 语义；
- clock 返回 bool、非数值、NaN、Inf 或负数时早失败；
- fake monotonic clock 倒退时明确抛出 `RuntimeError`，不生成负 elapsed。

## 不可变 snapshot

公开 frozen dataclass：

```text
SubtaskTimeTrackerSnapshot
```

包含：

```text
current_index
current_name
raw_elapsed_seconds
effective_elapsed_seconds
cap_seconds
time_valid
running
paused
last_transition_reason
last_rejected_output
last_rejection_reason
```

尚未确认首项或 full reset 后，snapshot 固定为 no-time：index/name/cap 为 `None`，raw/effective 为 `0.0`，
`time_valid=false`。Snapshot 为 frozen object，调用者不能通过修改 debug view 污染 tracker 状态。

Tracker 故意不自带另一把 lock；T5 会在 RTC semantic state lock 内调用它，使 history memory 和 time tracker 能在
同一个成功 merge transaction 中原子提交，避免额外 lock-order 风险。

## 测试覆盖

30 个 tracker 测试覆盖：

- parser 正常格式、字段大小写、空白/换行、缺字段、空 name、歧义多 subtask 和非 string；
- 大小写、空白、末尾句号匹配，以及内部词差异不做 fuzzy match；
- initial/current/next/old/skip/unknown/parse failure；
- current 不重启、next 归零、末项不 wrap；
- fake clock raw/effective/cap；
- pause 90 秒、resume、pause-before-start 和 double pause/resume；
- full reset 与 paused session 组合；
- clock 倒退和无效 clock value；
- snapshot frozen；
- contract collision、坏 FPS、空 sequence、normalized mismatch、负 maximum 和 cap 小于 maximum。

T1 的 scanner 聚焦测试继续覆盖所有 episode 序列不一致、重复 normalized subtask、坏 frame/index、空 label、第一帧
`0.0s` 和 `max + 5.0s` cap。

## 一键验收入口

```bash
plans/timer_pipeline/validate_subtask_time_milestone_4.sh [contract|data|regression|all]
```

模式：

- `contract`：py_compile、30 个 tracker 测试、23 个 scanner 聚焦测试、T0–T3 累计契约；
- `data`：对本地 match/egg 全量数据运行 T1 生产 sequence/cap 扫描；
- `regression`：T3 checkpoint 契约和 memory/subtask/advantage/RTC/time 聚焦回归；
- `all`：依次执行以上三项，默认模式。

脚本固定 offline、默认使用 `lerobot-main`，最后执行 `bash -n` 和 `git diff --check`。Ruff 仅在环境中存在时
执行；当前环境未安装 Ruff，因此明确记录 skipped，没有写成 passed。

## 最终实际验收结果

最终命令：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_4.sh all
```

实际结果：

```text
tracker contract: 30 passed in 1.50s
scanner focus: 23 passed, 21 deselected in 0.02s
T3 processor/converter/time-disabled contract: 54 passed in 1.50s
T0-T3 cumulative time contract: 119 passed, 4 deselected in 1.57s
checkpoint contract: 6 passed in 1.86s
memory/subtask/advantage/RTC/time focused regression:
  263 passed, 6 deselected, 2 warnings in 4.05s
script exit code: 0
py_compile: passed
bash -n: passed
git diff --check: passed
ruff: skipped (not installed)
```

两个 warning 是既有 `subtask_max_decode_tokens > subtask_max_tokens` 测试 warning。6 个 deselected 是既有受限沙箱
环境下排除的 DataLoader worker 用例，T4 没有修改 Dataset wrapper 或 DataLoader；这些用例已在 T1 正常进程环境
实际通过。T4 原有 2 条 strict xfail 已全部移除，没有 skip 或 xfail 被写成 passed。

## 真实数据结果

### Strike-match

```text
repo_id: ming326/strike_match_3_subtask
episodes: 70
frames: 53794
fps: 30.0
ordered subtasks: 6
construction: 3.159097s
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
construction: 18.899462s
lookup: 4550130 bytes / 4.339342 MiB
Stir the beaten eggs.: 43.900000s maximum / 48.900000s cap
Start frying the eggs.: 95.766667s maximum / 100.766667s cap
```

两个数据集均从本地 cache 离线读取，只扫描轻量列，没有解码视频、联网或修改 dataset。

## 工作区保护复核

阶段开始和结束均检查了 `git status --short`。工作区在 T4 前已包含 memory M0–M8、timer T0–T3、计划目录移动、
Nero/Pico 及其他用户改动；这些全部保留。T4 只局部修改/新增本记录列出的 tracker、tracker tests、T4 validation
script 和 timer plan 文档，没有执行 reset、checkout、删除、全仓格式化或覆盖无关文件。

## 下一阶段边界

T4 只证明纯 tracker 和数据 contract 正确。它尚未接入 `RTCInferenceEngine`，因此当前部署不会注入
`subtask_time_*` 字段，也不会在 `ActionQueue.merge()` 后提交 tracker。成功 merge 后原子提交、reset race、history+time
一致快照和 time-disabled RTC 路径属于 Milestone T5，不能把本记录视为 RTC 闭环已经完成。
