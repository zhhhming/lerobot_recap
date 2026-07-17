# Subtask Elapsed-Time Milestone T1 Completion Record

日期：2026-07-17

基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655`

计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

环境：Conda `lerobot-main`，Python 3.12.13，PyTorch 2.10.0；真实数据和回归均使用离线模式。

## 完成状态

Milestone T1：轻量 segment 扫描与训练 lookup 已完成并通过一键验收。

本阶段实现 Dataset 侧共享 segment scanner、严格 sequence contract、O(1) timing lookup、
`SubtaskTimingDataset` 和 factory 的 time/memory 独立组合。没有修改 raw/converted dataset schema，没有实现
训练 noise/dropout、processor、checkpoint、RTC tracker、deploy timer 或 Nero 状态机；这些仍属于 T2–T8。

## 修改和新增文件

生产代码：

- 新增 `src/lerobot/datasets/subtask_timing.py`；
- 局部修改 `src/lerobot/datasets/factory.py`；
- 局部修改 `src/lerobot/datasets/memory_history.py`，只增加 wrapper 组合所需的 raw/select 代理。

测试与验收：

- 扩展 `tests/datasets/test_subtask_timing.py`；
- 新增 `plans/timer_pipeline/subtask_time_m1_real_validation.py`；
- 新增 `plans/timer_pipeline/validate_subtask_time_milestone_1.sh`；
- 新增本完成记录。

没有修改 PI0/PI0.5 config、processor、model、训练循环、RTC、deploy、Nero 或 Pico 代码。

## Scanner 和 sequence contract

共享模块公开：

```text
SubtaskSegmentStats
SubtaskSequenceContract
SubtaskTimingScan
normalize_subtask_name()
validate_subtask_timing_features()
scan_subtask_timing()
SubtaskTimingDataset
```

scanner 只接收以下轻量列：

```text
episode_index
frame_index
index
subtask
```

一次线性扫描生成：

```text
elapsed_seconds: float32 [N]
valid: bool [N]
segment_indices: int64 [N]
sequence_contract
```

真实 elapsed 固定为：

```text
(current_frame_index - segment_start_frame) / dataset.meta.fps
```

因此连续段第一帧和 subtask 切换帧均为 `0.0s`。Segment 的部署统计使用 Python float 按
`(segment_frames - 1) / fps` 计算，不从 float32 lookup 反推，避免 cap 引入 lookup 量化误差。

名称规则与后续 tracker 共用：trim、折叠连续空白、Unicode casefold、去除一个末尾句号类标点。相邻帧按
normalized label 判断是否属于同一段，contract 保留首次出现的 canonical label。

严格验证包括：

- FPS finite 且大于 0；deployment margin finite 且非负；
- 四个必需列存在且长度一致；
- label 为非空 string，normalization 后非空；
- 每个 episode 的 `frame_index` 从 0 连续；
- 每个 episode 内 absolute `index` 逐帧加一；
- selected episode view 可从任意 absolute index 开始，episode 之间允许 index 跳跃；
- episode 只能形成一个连续 row block；
- 所有 episode 的压缩 subtask sequence 必须逐项一致；
- 同一 episode 的压缩序列不得重复 normalized subtask；
- sequence/cap 错误包含 dataset、episode、期望和实际序列诊断。

## Dataset wrapper

`SubtaskTimingDataset` 构造时只调用一次：

```python
dataset.select_columns(["episode_index", "frame_index", "index", "subtask"])
```

之后每次 `__getitem__` 只调用一次底层普通 `__getitem__`，再以当前相对 index O(1) 追加：

```text
subtask_elapsed_seconds: scalar torch.float32
subtask_time_valid: bool
subtask_segment_index: scalar torch.int64
```

返回的 scalar tensor 会 clone，调用方原地修改样本不会破坏共享 lookup。Wrapper 显式代理训练路径使用的
`meta`、`episodes`、`num_frames`、`num_episodes`、`features`、`fps`、`repo_id`、`root`，并代理
`get_raw_item()` 和 `select_columns()`。

`get_raw_item()` 始终返回真正 base dataset 的 raw row，不增加 timing synthetic fields。相应地，
`MemoryHistoryDataset` 也增加了相同的 raw/select 代理，使 wrapper 组合保持 map-style 接口。

## Factory 组合和阶段兼容

factory 使用安全前向兼容读取：

```python
getattr(cfg.policy, "use_subtask_time_conditioning", False)
```

正式 PI0/PI0.5 time config 字段仍由 T3 实现，因此 T1 测试使用轻量 config fixture 启用该路径。Time 开启时
factory 在构建 metadata/dataset 前验证 policy 是 PI0/PI0.5、`predict_subtask=true` 且 dataset 非 streaming，
并在构建 dataset 前验证必需列。

实际组合顺序：

```text
LeRobotDataset
  -> SubtaskTimingDataset   (time enabled)
  -> MemoryHistoryDataset   (memory enabled)
```

已验证四种组合：

- neither：返回原 dataset object，不扫描列；
- time-only：只增加 `subtask_time_*`，不调用 history RNG；
- memory-only：只增加 `memory_*`，不执行 timing scan；
- both：当前帧只解码一次，history 仍走 base raw row。

## Synthetic 和 DataLoader 测试

`tests/datasets/test_subtask_timing.py` 共 44 个通过测试，覆盖：

- 第一帧 `0.0s`；30 FPS 下 frame 33 为 `1.1s`；切换帧归零；
- episode 边界隔离、selected episodes 和不连续 absolute episode 起点；
- normalization、canonical name、最大 end-elapsed 和 `+5.0s` cap；
- 坏 FPS、margin、空 label、缺列、坏 frame/index/episode 和空 dataset；
- 重复 subtask 和 episode sequence 不一致的 early failure；
- 一次 scan、O(1) getitem、lookup storage 防修改、raw/select/属性代理；
- time/memory 四种 factory 组合；
- time-only 不消耗 history RNG，disabled 不扫描；
- timing-only 和 timing+memory wrapper 均通过 `DataLoader(num_workers=0/2)`；
- 10,000 帧 lookup storage 严格为每帧 `4+1+8=13` bytes。

目标 Dataset 与既有 memory wrapper 联合结果：

```text
74 passed in 8.93s
```

双 worker `spawn` 用例在受限 sandbox 内与 T0 记录一样会阻塞，最终在 sandbox 外真实执行通过；没有把 timeout、
skip 或改成 `num_workers=0` 当作通过。

## 真实数据验证

真实验证工具构造生产 `LeRobotDataset -> SubtaskTimingDataset`，不调用帧 `__getitem__`，因此不解码视频。
最终 `all` 验收的实际结果如下。

### Strike-match

```text
repo_id: ming326/strike_match_3_subtask
episodes: 70
frames: 53794
ordered subtasks: 6
construction: 3.092023s
lookup: 699322 bytes / 0.666925 MiB
largest max end-elapsed: 11.533333s
largest deployment cap: 16.533333s
```

### Nero egg

```text
repo_id: ming326/nero_egg_subtask
episodes: 61
frames: 350010
ordered subtasks: 12
construction: 18.955077s
lookup: 4550130 bytes / 4.339342 MiB
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

350010 帧的三个 lookup tensor 小于验收上限 8 MiB，构造完成且规模与 O(N) 设计一致。

## 一键验收入口

```bash
plans/timer_pipeline/validate_subtask_time_milestone_1.sh [contract|data|regression|all]
```

最终执行命令：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_1.sh all
```

脚本设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，默认使用 Conda `lerobot-main`，数据根目录和环境名均可通过 T0 已有环境变量覆盖。

实际结果：

```text
T0/T1 contracts: 46 passed, 6 xfailed
focused memory/subtask/advantage/RTC/dashboard regression: 176 passed, 4 warnings
memory wrapper proxy regression: 30 passed
Dataset reader/facade + memory: 65 passed

M0-M8 cumulative groups:
216 passed, 6 skipped, 2 warnings
35 passed
59 passed, 2 warnings
60 passed
172 passed, 2 warnings
220 passed, 3 skipped
28 passed
```

6 个 strict xfail 全部属于尚未实施的 T2 noise/dropout、T3 processor/config 和 T4 tracker，没有 XPASS。
累计 skip 来自既有 CUDA/多 GPU 条件测试，不是 T1 新增 skip。Warnings 是既有 subtask decode budget warning 和
Python 3.12 多线程进程中的 DataLoader fork deprecation warning。

其他门禁：

```text
py_compile: passed
bash -n: passed
real match lookup: passed
real egg lookup: passed
git diff --check: passed
ruff: skipped (not installed in lerobot-main; not recorded as passed)
```

## 工作区保护复核

开始和完成时都确认工作区包含尚未提交的 memory M0–M8、计划目录迁移和 Nero 用户改动。本阶段没有执行 reset、
checkout、清理、暂存、commit 或无关格式化。

`src/lerobot/datasets/factory.py` 原先已有 staged memory 改动，本阶段只在工作树上叠加 timing 局部 patch；
`src/lerobot/datasets/memory_history.py` 原先是 staged 新文件，本阶段只增加组合代理。因此完成状态中出现 `MM`/`AM`
是保留原 staged 内容并叠加本阶段未暂存修改的预期结果，不代表已有用户内容被覆盖。

## 下一阶段输入

Milestone T2 可以直接使用 wrapper 输出：

```text
subtask_elapsed_seconds
subtask_time_valid
subtask_segment_index
```

下一阶段应实现独立的训练 noise/dropout helper、train config、训练循环接线和 metrics，并移除
`tests/utils/test_subtask_time_conditioning.py` 的两个 strict xfail。T2 不应修改本阶段已冻结的 segment、FPS、
selected episode、sequence contract、factory 组合或 O(1) lookup 语义。
