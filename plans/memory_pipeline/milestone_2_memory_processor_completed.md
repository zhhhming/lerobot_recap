# Memory Pipeline Milestone 2 Completion Record

日期：2026-07-17

基线：`main@8194e710`

Python：`/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`（Conda 环境 `lerobot-main`）

## 完成范围

本次完成 `pi0_pi05_memory_training_deployment_plan.md` 中的 Milestone 2：Memory processor、统一
subtask/progress 格式和 main prompt token budget。

本阶段把 Milestone 1 动态 Dataset wrapper 输出的历史 GT 字段，以及后续 RTC 将提供的上一轮完整 prediction，
统一转换为主 prompt 中的 Memory condition。PI0/PI0.5 已具有正式 policy config 字段和 processor pipeline
接线，Memory 开启时使用扩大的有效 token budget 和 left truncation。

本阶段没有实现训练 dropout、Train config、旧 checkpoint processor 结构重建、训练日志、模型 reset、RTC
事务闭环或终端状态面板；这些仍属于 Milestone 3–7。

## 修改文件

生产代码：

- 新增 `src/lerobot/processor/memory_processor.py`
- 修改 `src/lerobot/processor/subtask_processor.py`
- 修改 `src/lerobot/processor/__init__.py`
- 修改 `src/lerobot/processor/converters.py`
- 修改 `src/lerobot/policies/pi0/configuration_pi0.py`
- 修改 `src/lerobot/policies/pi0/processor_pi0.py`
- 修改 `src/lerobot/policies/pi05/configuration_pi05.py`
- 修改 `src/lerobot/policies/pi05/processor_pi05.py`

测试和验收记录：

- 扩展 `tests/processor/test_memory_processor.py`
- 扩展 `tests/processor/test_converters.py`
- 新增 `plans/memory_pipeline/validate_milestone_2.sh`
- 新增本完成记录

没有修改 Milestone 1 的 `MemoryHistoryDataset` 或 factory 行为，也没有修改 train/model/RTC/deploy 源码。

## 实现契约

### 共享 subtask/progress formatter

`format_subtask_output(subtask, progress)` 现在是 current subtask target 和训练 Memory GT 的共享 inner
formatter：

```text
Subtask: {trimmed_subtask}; Progress: {clamped_rounded_progress:.1f}
```

它不添加 outer Memory prefix，也不添加末尾换行。`SubtaskTextProcessorStep` 继续为 current AR target
追加换行，因此既有 target 文本保持：

```text
Subtask: pick; Progress: 0.5\n
```

非空 subtask 的 progress 必须是有限实数；一位小数 round 和 `[0,1]` clamp 规则与原 current target
保持一致。

### MemoryConditionProcessorStep

新增并注册 `memory_condition_processor`，支持两种互斥来源：

训练：

```text
memory_subtask + memory_subtask_progress + memory_valid
```

部署：

```text
memory_text + memory_valid
```

有效且 kept 时追加：

```text
Memory: Subtask: previous action; Progress: 0.7
```

部署 `memory_text` 只做首尾 trim/空白归一化，不解析或重建其内部语义。训练 GT 使用共享 formatter，
因此相同内容的训练和部署来源得到完全相同的 outer Memory block。

Processor 是确定性的，不采随机数。`memory_condition_kept` 缺失时默认请求保留，用于部署和 Milestone 2
直接 processor 调用；Milestone 3 将在 preprocessor 前生成训练 dropout mask。

以下情况都输出 effective `memory_condition_kept=false`，并保证原 task 逐字符不变：

- `memory_valid=false`；
- memory text/subtask 为空；
- requested keep=false；
- inference 缺少所有 memory source 字段。

支持 task/text 的 scalar string 或 batch list，以及 bool/progress 的 scalar、`[B]`、`[B,1]`。batch
长度不匹配、非 bool valid/keep、非法 progress dtype、NaN/Inf、训练/部署来源同时出现都会明确报错。

### Converter 和序列化

`batch_to_transition()` 现在把所有 `memory_*` 字段路由到 complementary data，包括：

```text
memory_subtask
memory_subtask_progress
memory_valid
memory_frame_offset
memory_text
memory_condition_kept
```

Memory step 已从 `lerobot.processor` 导出，并实现完整 `get_config()`；本地 registry save/reload 后配置和
输出一致。

### PI0 / PI0.5 config 和 pipeline

两种 policy config 新增：

```python
use_memory_conditioning: bool = False
memory_tokenizer_max_length: int = 128
```

校验和提示：

- token budget 必须为正；
- `use_memory_conditioning=true` 要求 `predict_subtask=true`；
- 同时关闭 `subtask_generate_at_inference` 会 warning，说明部署 memory 无法更新；
- 默认关闭，旧 config/checkpoint 缺字段时使用 dataclass 安全默认值。

Pipeline 顺序为：

```text
Advantage(optional) -> Memory(optional) -> PI prompt/state -> current subtask -> Tokenizer
```

有效 main prompt 长度：

```text
memory off: PI0=48, PI0.5=200
memory on:  PI0=max(48,128)=128, PI0.5=max(200,128)=200
```

Advantage 或 Memory 任一开启时 tokenizer 使用 left truncation。长 task 的最终 token 测试确认保留
Memory label、完整历史 subtask/progress、Advantage condition，以及 PI0.5 state 尾段。Current subtask
target token 在 memory keep/drop batch 中保持一致。

## 测试覆盖

`tests/processor/test_memory_processor.py` 现有 24 个普通通过测试，覆盖：

- PI0/PI0.5 config 默认值、非法组合和 inference generation warning；
- 共享 formatter、round/clamp 和 current target 换行兼容；
- 训练 GT canonical 文本和部署完整 prediction 空白归一化；
- 训练/部署来源生成相同 Memory block；
- scalar/list、`[B]`、`[B,1]`；
- invalid/empty/dropped/missing source 的 task byte-for-byte no-op；
- batch size、bool dtype、progress dtype、NaN 和混合 source 错误；
- processor registry save/reload；
- Advantage -> Memory -> prompt/state -> current subtask -> Tokenizer 顺序；
- PI0 128 / PI0.5 200 token budget、Memory-only left truncation 和长 prompt suffix 保留；
- current subtask token isolation；
- Memory disabled 不插 step、不改变 tokenizer budget/truncation。

`tests/processor/test_converters.py` 新增所有 memory 字段的 route-through 测试。

Milestone 0 原来的 5 个 Milestone 2 strict xfail 实例已经移除并扩展成普通绿色测试。当前剩余 2 个
strict xfail 都位于 `tests/utils/test_memory_conditioning.py`，对应 Milestone 3 尚未实现的训练 dropout
helper。

## 一键验收脚本

脚本：

```bash
plans/memory_pipeline/validate_milestone_2.sh
```

脚本行为：

1. 使用 `LEROBOT_PYTHON` 覆盖或默认使用 `lerobot-main` Python；
2. 对本阶段生产源码和测试执行 `py_compile`；
3. 复跑 `validate_milestone_1.sh`，因此覆盖 memory history、默认关闭 golden、subtask、advantage、
   tokenizer、converter 和 PI0/PI0.5 train/inference 专项回归；
4. 复跑 Dataset reader/facade 核心回归；
5. Ruff 可用时检查本阶段文件，不可用时明确打印 skipped；
6. 执行 `git diff --check`。

## 实际执行结果

修改前基线在沙箱外执行：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_1.sh
```

结果：

```text
168 passed, 6 skipped, 7 xfailed, 2 warnings in 6.90s
35 passed in 2.25s
```

最终完整验收在沙箱外执行：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_2.sh
```

结果：

```text
193 passed, 6 skipped, 2 xfailed, 2 warnings in 6.44s
35 passed in 2.14s
py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
```

受限 sandbox 内的第一次基线运行停在已知的真实 `DataLoader(..., multiprocessing_context="spawn")`
双 worker 隔离限制；终止后在 sandbox 外完整重跑并通过，没有将受限运行写成测试失败或通过。

6 个 skip 是既有 tokenizer CUDA/多 GPU 用例。2 个 warning 是既有测试刻意设置
`subtask_max_decode_tokens > subtask_max_tokens` 产生的 warning。2 个 strict xfail 是 Milestone 3
预留契约，不属于本阶段遗漏。

## 工作区保护复核

本次没有 reset、checkout、暂存、覆盖或格式化用户已有改动。以下既有用户源码改动保持不变：

- `scripts/nero_teleop/README.md`
- `src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py`
- `src/lerobot/scripts/lerobot_push_dataset.py`
- `tests/scripts/test_lerobot_push_dataset.py`

已有计划目录迁移和 Milestone 0/1 文件也未被回退。Milestone 2 没有重新修改
`src/lerobot/datasets/factory.py` 或 `src/lerobot/datasets/memory_history.py`。

## 未运行和未完成项

- Ruff 未安装，因此未运行 Ruff lint/format check；
- 未运行既有 CUDA 或多 GPU tokenizer 测试；
- 未下载 tokenizer、模型、checkpoint 或数据集；
- 未运行真实 PI0/PI0.5 forward/backward、完整训练 update、RTC、fake robot 或 Nero 实机；
- 未实现 memory dropout、Train config/metrics、旧 checkpoint processor 重建和 resume 语义；
- 未实现 Milestone 4–8。

以上不阻塞 Milestone 2：本阶段 processor/config/tokenization 契约全部由离线 CPU 单元和 pipeline 测试
验证。

## 下一阶段输入

Milestone 3 可直接使用：

```text
MemoryHistoryDataset -> memory_subtask/progress/valid/offset
sample_memory_condition_mask() -> memory_condition_kept
MemoryConditionProcessorStep -> canonical main prompt Memory block
```

下一阶段应实现 `sample_memory_condition_mask()`、Train config/validation、preprocessor 前独立 dropout、旧
checkpoint 非 resume 结构重建、resume 保持、memory metrics，以及 PI0/PI0.5 fake/small 2-step update。
当前 `tests/utils/test_memory_conditioning.py` 中的 2 个 strict xfail 应在 Milestone 3 转为普通绿色，不应
提前改变本阶段 deterministic processor 的职责。
