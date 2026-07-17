# Subtask Elapsed-Time Milestone T0 Completion Record

日期：2026-07-17

基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655`

计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

环境：Conda `lerobot-main`；数据、tokenizer 和 checkpoint 检查均使用离线模式。

## 完成状态

本记录完成 elapsed-time 计划的 Milestone T0：契约基线与 strict failing tests。

本阶段只新增测试、真实数据轻量审计工具、验收脚本和本完成记录，没有修改 Dataset、processor、policy
config、训练循环、模型、RTC、deploy、Nero 或 Pico 生产代码。T1–T8 功能仍未实现，8 个 strict xfail
不能被解释为生产功能已通过。

完成产物按用户要求保存在 `plans/timer_pipeline`；pytest 文件继续位于仓库标准 `tests/` 目录。

## 冻结的核心契约

T0 将以下字段名冻结为后续 milestone 的公共契约：

```text
subtask_elapsed_seconds
subtask_time_valid
subtask_segment_index
subtask_time_seconds
subtask_time_condition_kept
```

同时锁定：

- canonical prompt：`Subtask elapsed time: 1.2s`；
- 真实 elapsed：`(current_frame_index - segment_start_frame) / dataset.fps`；
- 连续段第一帧为 `0.0s`；
- noise amplitude：`min(0.4 * x, 5.0)`，加噪后 clamp 到非负；
- time dropout 默认 `0.2`，与 memory、advantage 和 current-subtask attention dropout 独立；
- deployment cap：该 subtask 最大 end-elapsed `+5.0s`；
- tracker 初始只接受第 0 项，运行后只接受 current 或 immediate next；
- timer 使用 monotonic clock，pause 冻结、resume 延续、full reset/home 清空。

## 新增文件

绿色关闭态基线：

- `tests/processor/test_subtask_time_disabled_baseline.py`

后续 milestone 的 strict failing contracts：

- `tests/datasets/test_subtask_timing.py`
- `tests/utils/test_subtask_time_conditioning.py`
- `tests/processor/test_subtask_time_processor.py`
- `tests/inference_engines/test_subtask_time_tracker.py`

T0 工具与记录：

- `plans/timer_pipeline/subtask_time_m0_data_audit.py`
- `plans/timer_pipeline/validate_subtask_time_milestone_0.sh`
- 本完成记录。

## Time-disabled golden

`test_subtask_time_disabled_baseline.py` 使用离线 deterministic character tokenizer，直接比较最终 prompt、
main token、main attention mask、current-subtask token 和 current-subtask attention mask。

PI0：

```text
main prompt: pick cube\n
main tokenizer length: 48
current target: Subtask: grasp; Progress: 0.4\n
```

PI0.5（`predict_subtask=true`）：

```text
main prompt: Task: pick cube, State: 128 128;\n
main tokenizer length: 200
current target: Subtask: grasp; Progress: 0.4\n
```

两条路径都验证：

- pipeline 中没有 `SubtaskTime` step；
- 输出中没有 `subtask_time_*`；
- tokenizer truncation side 保持当前关闭态；
- current-subtask target 仍是独立 tensor，不进入 main prompt tensor。

## Strict xfail 契约

新增 8 个 `xfail(strict=True, raises=...)`：

| 后续阶段 | 数量 | 冻结内容 |
|---|---:|---|
| T1 Dataset | 2 | scanner elapsed/segment/cap；wrapper O(1) lookup、字段和 raw access |
| T2 Training helper | 2 | noise bounds/RNG/no mutation；valid 与 p=0/p=1 dropout |
| T3 Processor/config | 2 | formatter/no-op canonical prompt；PI0/PI0.5 safe config |
| T4 Tracker | 2 | initial/current/next/old/skip/unknown；timer/cap/pause/reset |

marker 只接受当前缺少模块或字段产生的预期异常。fixture、签名、dtype、shape 或断言错误不会被吞掉；相应
milestone 实现 API 后必须移除 marker，strict XPASS 会使验收失败。

实际结果：

```text
2 passed, 8 xfailed in 1.49s
```

没有 XPASS、普通 fail 或 collection/import fixture 错误。

## 真实数据轻量审计

审计脚本只读取：

```text
index
episode_index
frame_index
subtask
```

没有视频解码、网络访问或 dataset 写入。它验证 FPS、空值、global index、episode-local frame index、非空
label、normalized uniqueness、episode sequence 一致性，并按 `(segment_frames - 1) / fps` 计算统计和 cap。

### Strike-match

```text
repo_id: ming326/strike_match_3_subtask
fps: 30
episodes: 70
frames: 53794
ordered subtasks: 6
all episode sequences equal: true
global/frame indices contiguous: true
largest max end-elapsed: 11.533333s
largest deployment cap: 16.533333s
```

顺序：

1. Pick up the match.
2. move the right arm to ready.
3. Pick up the matchbox.
4. move the left arm to ready.
5. Strike the match and light the candle.
6. Return to the home position.

### Nero egg

```text
repo_id: ming326/nero_egg_subtask
fps: 30
episodes: 61
frames: 350010
ordered subtasks: 12
all episode sequences equal: true
global/frame indices contiguous: true
largest max end-elapsed: 95.766667s
largest deployment cap: 100.766667s
```

关键长段：

```text
Stir the beaten eggs.
  max end-elapsed: 43.900000s
  deployment cap: 48.900000s

Start frying the eggs.
  max end-elapsed: 95.766667s
  deployment cap: 100.766667s
```

完整逐 subtask min/max/cap 可用以下命令重复输出：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_0.sh data
```

## 一键验收入口

```bash
plans/timer_pipeline/validate_subtask_time_milestone_0.sh [contract|data|regression|all]
```

模式：

- `contract`：py_compile、新增契约、聚焦 memory/subtask/advantage/RTC/dashboard 回归、Ruff 检查；
- `data`：两个真实数据集的全量轻量列审计；
- `regression`：调用当前 memory M0–M8 累计 regression；
- `all`：依次执行上述全部门禁；默认模式。

脚本设置：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

数据路径支持通过 `LEROBOT_TIMER_MATCH_ROOT` 和 `LEROBOT_TIMER_EGG_ROOT` 覆盖；Conda 环境支持通过
`LEROBOT_CONDA_ENV` 覆盖。

## 实际回归结果

T0 聚焦回归：

```text
176 passed, 4 warnings in 3.51s
```

4 个 warning 中 2 个是已有 subtask decode-limit warning，2 个是 Python 3.12 多线程进程中 DataLoader
`fork()` 的 DeprecationWarning。

现有 M0–M8 累计 regression：

```text
216 passed, 6 skipped, 2 warnings in 6.07s
35 passed in 2.09s
59 passed, 2 warnings in 1.78s
60 passed in 1.50s
172 passed, 2 warnings in 6.28s
220 passed, 3 skipped in 3.61s
28 passed in 3.63s
```

skip 来自既有 CUDA/多 GPU 条件测试，不是 T0 新增测试。Ruff 未安装在 `lerobot-main`，验收脚本明确输出
`ruff: skipped`，没有记作 passed。

其他门禁：

```text
py_compile: passed
bash -n: passed
real match audit: passed
real egg audit: passed
git diff --check: passed
```

受限沙箱内的 `num_workers=2` DataLoader 测试会阻塞，因此正式 contract 和累计 regression 使用正常进程环境
复验并取得上述完整结果；没有通过删测试、改为 `num_workers=0` 或把 timeout 当成功来绕过门禁。

## 工作区保护

开始与完成时均确认工作区包含 memory M0–M8、计划目录移动和 Nero 用户改动。本阶段没有执行 reset、checkout、
清理、暂存或无关格式化，也没有修改这些既有生产文件。

新增内容与既有改动分离：生产 `src/` 和 Nero/Pico 文件的状态没有被 T0 改写。

## 未完成边界和下一阶段输入

T0 不代表以下功能已经存在：

- timing scanner 或 `SubtaskTimingDataset`；
- train noise/dropout helper 和 metrics；
- time processor、config、token budget 或 checkpoint 保存；
- deploy sequence tracker、RTC transaction、pause/home 或 dashboard；
- checkpoint、fake robot 或 Nero 实机验收。

Milestone T1 的明确入口是 `tests/datasets/test_subtask_timing.py` 中的 2 个 strict xfail。T1 应先实现共享
scanner、sequence contract 和 O(1) wrapper，再移除这两个 marker，并运行 T0+T1 累计门禁。
