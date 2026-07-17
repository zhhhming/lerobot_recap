# Memory Pipeline Milestone 7 Completion Record

日期：2026-07-17

基线：`main@8194e710`

Python：`/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`（Conda 环境
`lerobot-main`，CPU 验收显式设置 `CUDA_VISIBLE_DEVICES=''`）

## 完成范围

本次完成 `pi0_pi05_memory_training_deployment_plan.md` 中的 Milestone 7：固定终端状态面板。

`lerobot-policy-deploy` 现在支持 `auto`、`live`、`plain` 三种状态显示模式。交互 TTY 使用固定五行
footer 原地更新 deploy state、实际消费的 keyboard event、RTC latency/timing、最近一次成功 subtask 输出和
该次推理实际使用的 memory。普通日志、warning 和多行异常会临时清除 footer，在其上方完整输出，然后重绘
footer。非 TTY 和 plain 模式不写 ANSI escape。

本阶段没有执行 Milestone 8 的完整 checkpoint、fake robot 全部署、真实数据训练或 Nero 实机验收，也没有
修改 RTC memory 事务、PI0/PI0.5 模型、processor、训练循环或 advantage/RL 逻辑。

## 修改文件

生产代码：

- 新增 `src/lerobot/utils/terminal_status.py`
- 修改 `src/lerobot/scripts/lerobot_policy_deploy.py`

测试、验收和记录：

- 新增 `tests/scripts/test_lerobot_policy_deploy_status.py`
- 新增 `plans/memory_pipeline/validate_milestone_7.sh`
- 新增本完成记录

没有修改 `src/lerobot/utils/utils.py` 的全局 logging 配置。状态组件只在 policy deploy 生命周期内临时包装
`init_logging()` 已安装的 console handler，并在退出时恢复原 handler；现有 file handler 对象和输出路径不受
影响。

## 实现契约

### Live、plain 和 auto

`PolicyDeployConfig` 新增：

```text
status_display: auto | live | plain = auto
status_refresh_hz: float = 4.0
```

- `auto`：console stream 是 TTY 且 `TERM` 非空、非 `dumb` 时使用 live，否则使用 plain；
- `live`：显式启用 ANSI 固定 footer；
- `plain`：固定为每秒一条紧凑状态输出，不受更高 live refresh 配置影响；
- 非法 mode、非有限、零或负 refresh rate 在配置/组件构造时明确报错。

Live footer 固定为五行：

```text
[STATE]    ... [EVENT] ...
[LATENCY]  ...
[TIMING]   ...
[SUBTASK]  ...
[MEMORY]   ...
```

终端宽度不足时每行独立截断并保留起始 label。Plain 模式把相同字段压成一行，适合重定向、CI、systemd
或日志采集。

### 状态来源

- `STATE` 直接读取 deploy 主状态机；
- `EVENT` 只在主循环 `events.pop_latest()` 返回后更新，显示 `right/start`、`space/pause`、`h/home` 或
  `esc/exit` 以及实际消费时刻；
- `LATENCY`、`TIMING` 和 phase 来自一次完整的 `engine.debug_snapshot()`；
- `SUBTASK` 使用 M6 的 `last_subtask_output_text`；
- `MEMORY` 使用 M6 的 `last_memory_input_text`，即最近一次成功 inference 实际使用的输入，而不是 next
  memory 或 policy 临时候选；
- 初次启动和 reset 后空 subtask/memory 显示 `<none>`。

主循环先取得 RTC snapshot，`debug_snapshot()` 返回并释放 engine lock 后才进入 status display update。状态
handler 不调用 engine，从而避免 engine state lock 与 logging/display lock 形成反向依赖。

### Logging 协作与并发

`TerminalStatusDisplay` 查找现有非 file `StreamHandler`，复制其：

- stream；
- formatter；
- level；
- filters；
- terminator。

Live 普通日志流程为：清除现有 footer、写入完整 formatted record、重绘 footer。Logging emit、plain status
write 和 live redraw 共用同一个 `RLock`，但 status update 不反向获取 logging handler lock。

现有 `init_logging()` formatter 只返回 `record.getMessage()`，不会自行附加 `exc_info`。本阶段 wrapper 检测
这一情况，仅在 console 格式结果缺少 traceback 时补入 formatter 生成的完整多行 traceback；已经包含
traceback 的标准 formatter 不会重复输出。

ANSI 只直接写到被包装的 console stream，不生成 ANSI LogRecord，因此 file handler 中不会出现 cursor
control sequence。

### Deploy 日志和终端恢复

原每秒超长 `deploy_loop ...` 从 INFO 降为 DEBUG：详细 loop histogram、obs/fetch/send/sleep 等诊断数据仍
保留，但默认 console 不再滚屏；live/plain 面板负责稳定展示关键 RTC 状态。Slow-loop warning、RTC error、
相机/机器人连接、keyboard、homing 和其他普通日志级别不变。

Normal shutdown、Esc、KeyboardInterrupt 或 fatal engine 分支最终都进入 deploy `finally`。清理 engine、robot
和 listener 后，最外层 `finally` 恢复原 console handler、显示 cursor 并补换行。组件本身 start/stop 幂等，
不会修改 stdin 或 termios，因此不影响 keyboard listener 的 cbreak 设置和恢复。

## 测试先行记录

先新增 M7 契约测试。测试改为直接导入计划新增的独立 terminal component 后，未实现时按预期在 collection
阶段失败：

```text
ModuleNotFoundError: No module named 'lerobot.utils.terminal_status'
```

实现 terminal component 和格式化契约后，首次专项结果：

```text
17 passed in 0.24s
```

随后完成 deploy config、状态机接线、长日志降级、finally 恢复和代码审阅修正，M7 + M6 RTC memory 交接
专项结果：

```text
28 passed in 3.54s
```

专项覆盖：

- fake TTY 初次五行绘制和连续原地更新；
- 普通单行/多行日志清除与重绘 footer；
- `logger.exception()` 完整 traceback；
- auto 的 TTY/TERM 判定；
- plain 1 Hz cadence 和零 ANSI；
- terminal width 截断；
- 2 logging threads + 2 update threads 压测，无交错 write 和 deadlock；
- normal stop 恢复原 handler、cursor 和换行；
- live ANSI 不进入 file handler；
- keyboard event label/消费时间；
- RTC snapshot 到 latency/timing/subtask/memory 五行的精确映射；
- reset/初始空值显示 `<none>`；
- 非法 mode 和 refresh rate。

## 一键验收脚本

脚本：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_7.sh
```

脚本执行：

1. 对 terminal status、policy deploy 和 M7 测试执行 `py_compile`；
2. 复跑 `validate_milestone_6.sh`，累计覆盖 M0–M6；
3. 运行 M7 fake TTY/status 测试和 M6 RTC memory 事务交接测试；
4. Ruff 可用时检查 M7 文件，不可用时明确打印 skipped；
5. 执行 `git diff --check`。

## 最终实际结果

实施前先在受限 sandbox 内运行 M6 基线，在此前 M1–M6 已记录的真实双 worker DataLoader 隔离位置停止
产生进展。终止该受限测试后，在 sandbox 外执行同一 M6 脚本，结果：

```text
216 passed, 6 skipped, 2 warnings in 6.03s
35 passed in 2.07s
59 passed, 2 warnings in 1.76s
60 passed in 1.51s
172 passed, 2 warnings in 6.28s
220 passed, 3 skipped in 3.28s
py_compile: passed
git diff --check: passed
ruff: skipped（未安装）
```

实现完成后在 sandbox 外执行完整 M7 一键脚本：

```text
M0–M3 / memory / subtask / advantage / tokenizer / policy regression:
216 passed, 6 skipped, 2 warnings in 6.03s

Dataset reader/facade core regression:
35 passed in 2.08s

M3 + advantage train/helper integration:
59 passed, 2 warnings in 1.79s

M4 model/attention/reset/logging focused regression:
60 passed in 1.51s

M5 advantage/memory compatibility focused regression:
172 passed, 2 warnings in 6.33s

M6 RTC memory / RTC policy / PI0+PI0.5 inference regression:
220 passed, 3 skipped in 3.53s

M7 terminal status + RTC memory handoff focused regression:
28 passed in 3.63s

py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
```

保存本完成记录后对最终工作区再次执行同一脚本，结果仍全部通过：

```text
216 passed, 6 skipped, 2 warnings in 5.99s
35 passed in 2.09s
59 passed, 2 warnings in 1.76s
60 passed in 1.46s
172 passed, 2 warnings in 6.29s
220 passed, 3 skipped in 3.48s
28 passed in 3.46s
py_compile: passed
git diff --check: passed
ruff: skipped（未安装）
```

6 个累计 skip 是既有 tokenizer CUDA/多 GPU 用例，3 个 RTC skip 是既有 CUDA 用例。UserWarning 是既有
测试刻意设置 `subtask_max_decode_tokens > subtask_max_tokens`；DeprecationWarning 是 Python 3.12 对多线程
进程中 `fork()` 的提示。没有新增失败、xfail 或非预期 warning。

## 工作区保护复核

实施前后均检查工作区。本阶段没有 reset、checkout、暂存、覆盖或格式化无关文件，只增量修改本记录列出
的 M7 文件。

特别保留：

- `scripts/nero_teleop/README.md`
- `src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py`
- `src/lerobot/scripts/lerobot_push_dataset.py`
- `tests/scripts/test_lerobot_push_dataset.py`
- 用户的计划目录迁移和 Milestone 0–6 全部实现、测试与完成记录

## 未运行项和环境风险

- Ruff 未安装，因此没有运行 Ruff lint/format check；
- 未运行既有 CUDA 或多 GPU 测试；
- 未运行真实交互终端、真实 Ctrl-C/keyboard cbreak、相机、robot connect 或 homing；对应行为使用 fake TTY、
  多线程 logging 和 finally 单元契约验证；
- 未运行完整 fake robot `lerobot-policy-deploy`、完整 PI0/PI0.5 checkpoint 或 Nero 实机，这些属于 M8；
- 当前 `lerobot-main` 环境直接 import `lerobot_policy_deploy` 时，在进入 M7 逻辑前由既有 robot import 链的
  `python-can -> sqlite3 -> libicui18n.so.78` 触发动态库错误：系统 `libstdc++.so.6` 缺少
  `CXXABI_1.3.15`。`py_compile` 和独立 status/RTC 测试不受影响，但 M8 fake/真实 deploy 前必须先修复该
  Conda/runtime 动态库环境，不能把当前记录写成完整 deploy 已启动。

以上不阻塞 Milestone 7 的定义：本阶段要求实现、接线并使用 fake TTY/非 TTY、并发 logging 和 RTC snapshot
完成固定面板契约。完整 process import、fake robot 和真实 Nero 属于下一阶段端到端验收，但动态库问题已
明确交接，不能在 M8 中忽略。

## 下一阶段输入

Milestone 8 可以直接使用：

```text
--status_display auto|live|plain
--status_refresh_hz 4.0
```

并按以下顺序继续：

1. 先修复/确认 `lerobot-main` 的 `libstdc++/ICU` 动态库组合，使 deploy 模块和命令可真实启动；
2. 在非 TTY 重定向下确认 plain 输出零 ANSI；
3. 在真实 TTY/fake robot 下观察普通连接日志、keyboard event、Ctrl-C 和 fatal cleanup；
4. 最后连接 Nero，核对第一次 MEMORY=`<none>`、随后 MEMORY 等于上一轮成功 SUBTASK，并验证
   pause/home/restart 清空。
