# Subtask Elapsed-Time Milestone T6 Completion Record

日期：2026-07-17

基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655`

计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

环境：Conda `lerobot-main`；CPU；`HF_HUB_OFFLINE=1`；`TRANSFORMERS_OFFLINE=1`。

## 完成状态

Milestone T6：Soft pause、home、deploy 配置和状态面板已完成。

本阶段在 T5 已完成的 RTC transaction/time tracker 闭环之上，完成生产部署配置、标注数据集 sequence/cap 加载、
checkpoint processor 完整性检查、可恢复 soft pause、不可恢复 full reset/home、TIME dashboard、事件/截断日志和 Nero
部署文档。Space pause 现在会停止推理并清 ActionQueue、observation、history memory 及 policy/processor runtime cache，
同时冻结并保留已确认 subtask 的 monotonic elapsed；Right resume 从冻结值继续，暂停墙钟时间不累计。Home 和 fresh
session 使用 full reset，清空 tracker，下一轮必须重新等待序列第一项。

T6 没有修改 PI0/PI0.5 模型结构、训练 loss、数据集 schema、Pico 映射、Nero robot 控制或实机安全参数。真实
checkpoint/GPU/fake BiNero 的完整自动验收属于 T7，真实 Nero/Pico 实机验收属于 T8。

## 修改和新增文件

生产代码和文档：

- 修改 `src/lerobot/inference_engines/rtc.py`；
- 修改 `src/lerobot/scripts/lerobot_policy_deploy.py`；
- 修改 `src/lerobot/utils/terminal_status.py`；
- 修改 `scripts/nero_teleop/README.md`；
- 更新主计划中的 T6 状态及 timer validation/completion 路径。

测试和验收：

- 扩展 `tests/inference_engines/test_rtc_subtask_time.py`；
- 新增 `tests/scripts/test_subtask_time_deploy.py`；
- 扩展 `tests/scripts/test_lerobot_policy_deploy_status.py`；
- 新增 `plans/timer_pipeline/validate_subtask_time_milestone_6.sh`；
- 新增本完成记录。

## 测试先行记录

实现前先添加 T6 strict tests，首次实际命令：

```bash
conda run --no-capture-output -n lerobot-main python -m pytest \
  tests/inference_engines/test_rtc_subtask_time.py \
  tests/scripts/test_subtask_time_deploy.py \
  tests/scripts/test_lerobot_policy_deploy_status.py -q
```

实际失败基线：

```text
23 failed, 26 passed in 2.48s
```

失败均精确对应尚未实现的 `RTCInferenceEngine.soft_pause/full_reset`、deploy flag/margin/processor/dataset helpers、
deploy runtime session controller 和 TIME status line；没有 fixture、collection 或无关 import 错误。实现后相同聚焦集合
加上 cap 节流用例最终为：

```text
56 passed in 2.55s
```

## RTC soft pause、resume 和 full reset

`RTCInferenceEngine` 新增显式 API：

```text
soft_pause()
resume()
full_reset()
```

并保留既有低层 `pause()` 和兼容 `reset()`，避免改变 HIL 等其他 caller 的既有调用契约。

`soft_pause()` 的顺序为：先清 active event；在 engine state lock 内递增 reset version、清 history、冻结 tracker、清
ActionQueue 和 observation；再等待 inference lock 并 reset policy/preprocessor/postprocessor runtime。正在 predict 的
transaction 即使随后返回，也会因 active/reset-version 检查失败而在 merge 前丢弃。Tracker 的 current index 和累计
active elapsed 保留，`last_subtask_time_input_seconds` 作为最近 prompt 诊断保留。

`resume()` 在 state lock 内恢复 tracker monotonic clock 后设置 active event。Double pause/resume 由 tracker 和 event 的
幂等行为保证不会重复累计或产生负时间。

`full_reset()` 先停止 inference，再清 history、tracker、cap-warning 状态、queue、observation 和全部 runtime cache。
Tracker 保留的 paused session bit 会在下一次 `resume()` 时正常解除，但 current index/time 已清空，因此 home 后首轮
仍是 invalid/no-time。

## Deploy 配置、checkpoint 和数据 contract

`PolicyDeployConfig` 新增：

```text
use_subtask_time_conditioning: bool | None = None
subtask_time_deployment_margin_seconds: float = 5.0
```

解析契约：

- `None` 跟随 checkpoint；
- `False` 允许关闭已训练能力做 ablation；
- `True` 只允许 checkpoint config 已启用 time，旧 checkpoint 强制开启会在加载数据和连接机器人前失败；
- margin 必须 finite 且非负；
- checkpoint config 开启时，保存的 preprocessor 必须有且只有一个
  `SubtaskTimeConditionProcessorStep`；duplicate 总是早失败；
- time off 不要求 `dataset.repo_id`，不构造 `LeRobotDataset`，也不扫描 subtask 列；
- time on 要求 `dataset.repo_id`，使用 `download_videos=False`、`select_columns()` 和 T1 的共享
  `scan_subtask_timing()`，不解码视频、不复制状态机规则；
- sequence contract 和 effective flag 显式传入 `RTCInferenceEngine`。

扫描完成只记录一次 dataset/fps/subtask count/margin 摘要，不逐帧或逐推理输出 info。

## Deploy session、home 和动作安全

新增 `_PolicyDeployRuntime`，显式保存 `paused_session_resumable`：

- 初始或 home 后为 false，Right 会先 full reset 再启动 fresh session；
- running/preparing 经 Space soft pause 后为 true；
- resumable session 的 Right 只 resume，不再错误 full reset tracker；
- begin home 立即 full reset，并把 session 标为不可恢复；
- interpolator 和 smoother 在 soft pause/full reset 时都清空；
- engine 同时清 ActionQueue 和 observation，因此旧动作不会在 resume 后泄漏。

测试还锁定 deploy keyboard 的 Right/Space/h/Esc 解析、home waypoint 终点，以及 Nero gripper action 始终 clamp 到配置
上限。

## Dashboard 和事件日志

固定 footer 从五行扩为六行，新增：

```text
[TIME]     disabled
[TIME]     waiting-for-first-subtask
[TIME]     idx=4 running raw=37.2s input=37.2s cap=48.9s subtask=Stir the beaten eggs.
[TIME]     idx=2 paused raw=8.0s input=7.5s cap=12.0s subtask=Third.
```

Live footer 的清除/重画行数随六行更新；plain 模式仍固定最多 1 Hz；普通、多行和 traceback 日志仍会先清 footer 再
恢复，不把 ANSI 写入 file handler。面板同时显示 raw active time、最近一次模型 input、cap、index、状态和 canonical
subtask。

事件日志只在 tracker start/advance、soft pause/resume/full reset、contract 加载和每个 subtask 首次命中 cap 时输出；
old/skip/unknown rejection 使用 debug。专项测试证明同一 subtask 的 cap warning 只出现一次。

## Nero README

`scripts/nero_teleop/README.md` 的 policy deploy 小节新增 time-enabled 示例数据集参数、两个 deploy 配置项和明确语义：

- 初始/home 后 waiting-for-first-subtask；
- Space 清旧动作/runtime 并冻结 time；
- Right 从冻结值继续；
- h/home full reset tracker；
- dashboard raw/input/cap 的含义。

README 原有 `--disable-xet` 用户改动位于 dataset upload 小节，本阶段只局部修改 deploy 小节，没有覆盖该改动。

## 一键验收入口

```bash
plans/timer_pipeline/validate_subtask_time_milestone_6.sh [contract|data|regression|all]
```

模式：

- `contract`：py_compile、T6 RTC/deploy/dashboard、T4 tracker、RTC memory 和 T0–T5 累计 contract；
- `data`：复用生产 scanner 离线扫描本地 match/egg 全量轻量列；
- `regression`：T5 checkpoint/memory/subtask/advantage/time/RTC 累计回归，加 T6 deploy/status 和纯键盘回归；
- `all`：依次执行以上三项，默认模式。

脚本固定 offline、默认使用 `lerobot-main`，最后执行 `bash -n` 和 `git diff --check`。Ruff 仅在环境中存在时运行。

## 最终实际验收结果

最终命令：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_6.sh all
```

实际结果：

```text
T6 RTC/deploy/dashboard + tracker + RTC memory contract:
  97 passed in 4.33s
T6 focused RTC/deploy/dashboard:
  56 passed in 4.21s
T4 tracker contract:
  30 passed in 1.47s
T1 scanner focus:
  23 passed, 21 deselected in 0.02s
T3 processor/converter/time-disabled contract:
  54 passed in 1.52s
T0-T3 cumulative time contract:
  119 passed, 4 deselected in 1.55s
checkpoint contract:
  6 passed in 1.87s
memory/subtask/advantage/RTC/time focused regression:
  266 passed, 6 deselected, 2 warnings in 4.16s
complete RTC policy + memory + time regression:
  227 passed, 3 skipped in 4.30s
T6 deploy/RTC/status regression:
  67 passed in 4.33s
terminal keyboard regression:
  14 passed, 4 deselected in 1.63s
script exit code: 0
py_compile: passed
bash -n: passed
git diff --check: passed
ruff: skipped (not installed)
```

两个 warning 是既有测试刻意设置 `subtask_max_decode_tokens > subtask_max_tokens`。3 个 skip 是既有 RTC CUDA 用例。
6 个 deselected 是既有受限环境中排除的 DataLoader worker 用例；T1 已在正常进程环境通过。4 个 keyboard deselected
是 HIL record 文件中与纯键盘契约无关的录制/删除用例。

Regression 脚本初版曾额外运行整个 `tests/scripts/test_lerobot_hil_record.py`，其中 2 条录制测试在进入断言前因本机
Rerun gRPC `127.0.0.1:9876` 连接超时失败；这不是 T6 生产代码失败。脚本随后收窄为该文件的纯 terminal keyboard
测试，deploy homing/gripper clamp 由新增无硬件单测覆盖。修正后完整 `all` 入口退出码为 0；没有把这次环境失败、
skip 或 deselected 记录成 passed。

## 真实数据结果

### Strike-match

```text
repo_id: ming326/strike_match_3_subtask
episodes: 70
frames: 53794
fps: 30.0
ordered subtasks: 6
construction: 3.091932s
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
construction: 19.034008s
lookup: 4550130 bytes / 4.339342 MiB
Stir the beaten eggs.: 43.900000s maximum / 48.900000s cap
Start frying the eggs.: 95.766667s maximum / 100.766667s cap
```

两个数据集均从本地 cache 离线读取，只扫描轻量 parquet 列，没有解码视频、联网或修改 dataset。

## 工作区保护复核

阶段开始和完成时均检查 `git status --short`。工作区在 T6 前已有 memory M0–M8、timer T0–T5、计划目录移动、
Nero/Pico 和其他用户改动；这些全部保留。本阶段只局部修改本记录列出的 RTC/deploy/dashboard/README/tests/plan
文件，没有执行 reset、checkout、删除、暂存、全仓格式化或覆盖无关文件。最终 `git diff --check` 通过。

## 下一阶段边界

T6 已证明 deploy 配置、真实 sequence contract、soft pause/home/full reset、dashboard 和无硬件动作安全契约正确。
T7 仍需对真实 PI0/PI0.5 checkpoint、真 tokenizer、GPU update、save/reload、production RTC + fake BiNero observation
以及 history/time/advantage/dropout 矩阵做自动验收。T8 仍需真实 Nero + Pico 实机验收；本记录不把 CPU/fake/unit
测试等同于实机通过。
