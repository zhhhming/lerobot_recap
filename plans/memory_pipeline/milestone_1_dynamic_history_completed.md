# Memory Pipeline Milestone 1 Completion Record

日期：2026-07-17

基线：`main@8194e710`

Python：`/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`（Conda 环境 `lerobot-main`）

## 完成范围

本次完成 `pi0_pi05_memory_training_deployment_plan.md` 中的 Milestone 1：动态历史 Dataset
wrapper。

本阶段只实现训练数据侧的同 episode 历史 subtask/progress 动态抽样，以及 `make_dataset()` 的条件包装。
没有实现 Memory prompt/processor、正式 policy/train config 字段、训练 dropout、模型 attention、RTC
闭环或终端状态面板；这些仍属于 Milestone 2–7。

## 修改文件

- 新增 `src/lerobot/datasets/memory_history.py`
- 修改 `src/lerobot/datasets/factory.py`
- 扩展 `tests/datasets/test_memory_history.py`
- 新增 `plans/memory_pipeline/validate_milestone_1.sh`
- 新增本完成记录

没有修改 PI0/PI0.5 config、processor、model、训练循环、RTC 或 deploy 代码。

## 实现契约

### 动态历史读取

`MemoryHistoryDataset` 包装现有 map-style `LeRobotDataset`。每次 `__getitem__`：

1. 当前帧只调用底层普通 `__getitem__` 一次；
2. 使用默认 PyTorch RNG 从闭区间 `[lookback_min_frames, lookback_max_frames]` 抽取 offset，默认
   `[1,12]`；
3. 抽样发生在 episode 起点判断之前；
4. 若 `frame_index-offset<0`，不重抽、不 clamp、不读取历史 row；
5. 否则只调用 `get_raw_item(index-offset)`，不触发历史视频、delta window 或 image transform；
6. 对 episode index、episode-local frame index 和 absolute index 做一致性校验，检测到跨 episode
   或错位立即报错；
7. 历史 subtask 为空、非字符串，或 progress 非标量/非有限数时返回 no-memory。

每个样本固定增加：

```text
memory_subtask: string
memory_subtask_progress: scalar torch.float32
memory_valid: bool
memory_frame_offset: int
```

自然无历史使用空字符串、float32 零值和 `memory_valid=false`，但始终保留实际抽到的 offset。

### 属性代理和 early validation

Wrapper 显式代理训练入口当前依赖的：

```text
meta, episodes, num_frames, num_episodes, features, fps, repo_id, root
```

构造 wrapper 前检查原始 metadata 必须包含：

```text
subtask, subtask_progress, frame_index, episode_index, index
```

缺字段的错误会提示重新构建带逐帧 subtask 标注的 LeRobotDataset。Streaming memory 在创建 metadata
或 dataset 前失败，并提示设置 `--dataset.streaming=false`。

### Factory 接线和阶段兼容

`make_dataset(cfg)` 只在以下条件同时成立时包装：

- `policy.use_memory_conditioning=true`；
- policy type 为 `pi0` 或 `pi05`；
- dataset 非 streaming。

Milestone 2/3 才会正式添加 policy/train memory dataclass 字段，因此本阶段使用安全前向兼容读取：

```python
getattr(cfg.policy, "use_memory_conditioning", False)
getattr(cfg, "memory_lookback_min_frames", 1)
getattr(cfg, "memory_lookback_max_frames", 12)
```

这使 memory-disabled 的当前 PI config 保持原路径，也没有提前使 Milestone 2/3 的严格 xfail 变成
XPASS。后续正式字段加入后，factory 无需重新设计即可启用。

Memory disabled 时 factory 返回原始 dataset 对象，不增加 key、不读取 history、不消耗 history RNG。

## 测试覆盖

`tests/datasets/test_memory_history.py` 当前有 28 个普通通过测试，覆盖：

- `t=0,k=1`、`t=4,k=9` 先抽后判并保留 offset；
- `t=12,k=12` 精确读取 frame 0；
- later/selected episode 起点不读取上一 episode；
- 历史只走 `get_raw_item()`；
- 空/非字符串 subtask 和 NaN、Inf、非标量、非数值 progress；
- 输出 dtype、当前帧字段保留和训练属性代理；
- lookback 参数边界和必需 metadata 字段；
- 相同 worker seed 可重复、不同 worker seed 序列不同、offset 始终在 1–12；
- 真实 `DataLoader(num_workers=2, multiprocessing_context="spawn")` 的固定 seed 可重复和
  两 worker 序列区分；
- PI0/PI0.5 factory 包装、lookback 默认值、memory-disabled identity；
- streaming、缺字段和不支持 policy 的早失败。

Milestone 0 中原来的 3 个 Dataset strict xfail 已移除并扩充为普通绿色测试。其余 7 个 strict
xfail 均属于后续阶段：PI0/PI0.5 memory config 4 个、memory keep-mask 2 个、Memory processor 1 个。

## 一键验收脚本

脚本：

```bash
plans/memory_pipeline/validate_milestone_1.sh
```

脚本行为：

1. 使用 `LEROBOT_PYTHON` 覆盖或默认使用 `lerobot-main` Python；
2. 对新增源码、factory 和测试执行 `py_compile`；
3. 复跑 `validate_milestone_0.sh`，覆盖 memory/subtask/advantage/tokenizer/PI0/PI0.5 专项回归；
4. 运行 `test_dataset_reader.py` 和 `test_lerobot_dataset.py` 核心 Dataset 回归；
5. 若当前 Python 安装了 Ruff，执行 `ruff check` 和 `ruff format --check`；未安装时明确打印
   skipped；
6. 执行 `git diff --check`。

## 实际执行结果

最终命令在沙箱外执行真实双 worker 测试，并显式隐藏 CUDA，使结果与 CPU 基线一致：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_1.sh
```

Milestone 0 + memory 专项回归：

```text
168 passed, 6 skipped, 7 xfailed, 2 warnings in 6.68s
```

Dataset reader/facade 核心回归：

```text
35 passed in 2.24s
```

其他结果：

```text
py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
```

6 个 skip 是已有 tokenizer CUDA/多 GPU 用例；本次显式隐藏 CUDA，没有把 GPU 测试结果写成
通过。2 个 warning 是既有的 `subtask_max_decode_tokens > subtask_max_tokens` 测试 warning。

真实双 worker 用例在受限 sandbox 内会因进程隔离超时；同一用例在 sandbox 外单独执行结果为：

```text
1 passed in 5.56s
```

随后最终一键脚本也在 sandbox 外完整通过，没有残留 pytest 或 multiprocessing 进程。

## 工作区保护复核

本次没有 reset、checkout、暂存、覆盖或格式化用户已有改动。以下既有源码改动保持不变：

- `scripts/nero_teleop/README.md`
- `src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py`
- `src/lerobot/scripts/lerobot_push_dataset.py`
- `tests/scripts/test_lerobot_push_dataset.py`

计划目录迁移/新增内容也未被回退。

## 未运行和未完成项

- Ruff 未安装，因此未运行 Ruff lint/format check；
- 未使用真实包含 subtask/progress 的 converted LeRobotDataset；
- 未下载数据、tokenizer 或 checkpoint；
- 未运行 GPU、完整训练、RTC、fake robot 或 Nero 实机测试；
- 未实现 Milestone 2–8。

以上不阻塞 Milestone 1：本阶段实现和验收均为 Dataset/factory CPU 契约，不依赖模型或实机。

## 下一阶段输入

Milestone 2 可以直接使用本阶段生成的：

```text
memory_subtask
memory_subtask_progress
memory_valid
memory_frame_offset
```

下一阶段应实现共享 subtask/progress formatter、`MemoryConditionProcessorStep`、converter 路由、
PI0/PI0.5 config 字段和 tokenizer budget，并将 `tests/processor/test_memory_processor.py` 中对应的
strict xfail 转为普通绿色测试。Milestone 2 不应改变本阶段的抽样、raw history 或 episode 隔离语义。
