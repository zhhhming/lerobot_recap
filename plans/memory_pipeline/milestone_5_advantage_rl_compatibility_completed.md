# Memory Pipeline Milestone 5 Completion Record

日期：2026-07-17

基线：`main@8194e710`

Python：`/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`（Conda 环境
`lerobot-main`，CPU 验收显式设置 `CUDA_VISIBLE_DEVICES=''`）

## 完成范围

本次完成 `pi0_pi05_memory_training_deployment_plan.md` 中的 Milestone 5：Advantage/RL
兼容集成。

本阶段用 PI0/PI0.5 的真实 processor pipeline、训练期 advantage/memory keep-mask helper、
`AdvantageWeights` 和 `update_policy()` 联合证明：Memory condition 不读取历史 advantage，不改变当前帧
FM 权重，不改变 current subtask CE 的普通 mean reduction，也不改变 positive/negative/ignore、advantage
dropout fallback 或 RA-BC 互斥语义。

现有生产实现已经满足全部新增契约，因此本阶段没有修改 Dataset、processor、policy config、模型、训练循环或
weight provider 生产代码。RTC memory 事务闭环、终端状态面板和真实 Nero 部署仍属于 Milestone 6–8，
没有提前实施。

## 修改文件

测试：

- 扩展 `tests/scripts/test_advantage_weighted_train.py`
- 扩展 `tests/datasets/test_memory_history.py`

验收和记录：

- 新增 `plans/memory_pipeline/validate_milestone_5.sh`
- 新增本完成记录

## 已验证契约

### PI0/PI0.5 × global/subtask 完整矩阵

新增集成测试覆盖：

```text
PI0 / PI0.5
× advantage global / subtask key
× memory keep / drop
× advantage keep / drop
```

每个组合都使用真实 PI processor factory 和确定性离线 character tokenizer。batch 同时包含 positive、
negative 和 ignore 样本，并依次执行：

```text
raw current-frame batch
-> sample_advantage_condition_mask
-> sample_memory_condition_mask
-> PI0/PI0.5 preprocessor
-> AdvantageWeights
-> update_policy
```

测试同时检查最终 task 中 `Advantage:` / `Memory:` block 是否严格对应各自 keep mask，因此不是只在
weight helper 层构造孤立输入。

### Current-frame FM-only weighting

测试使用当前帧：

```text
label   = [positive, negative, ignore]
weight  = [2, 1, 17]
FM      = [1, 3, 5]
CE      = [2, 4, 6]
CE coef = 0.25
```

Advantage condition 保留时，有效权重严格为 `[2, 1, 0]`：

```text
weighted FM = (2*1 + 1*3) / (2+1) = 5/3
subtask CE  = mean(2, 4, 6) = 4
```

Advantage condition 全部 dropout 时，非 ignore 样本按既有 fallback 得到 `[1, 1, 0]`，weighted FM
为 `(1+3)/2=2`。上述两种权重结果在 memory keep 和 memory drop 下完全一致。

测试直接比较 SGD 后的 FM 参数和 CE 参数，确认 memory/advantage condition 组合只改变预期的 FM 梯度，
CE 始终使用 `0.25 * mean(CE)`，没有按 advantage weight 加权。

### History advantage 隔离

Dataset 测试为历史帧放入与当前帧冲突的 global/subtask label 和 weight：历史帧为 negative/1，当前帧为
positive/2。`MemoryHistoryDataset` 输出只从历史行读取 `memory_subtask` 和
`memory_subtask_progress`；当前 item 的 advantage label/weight 保持 positive/2，且不会产生任何
`memory_advantage_*` 字段。

这锁定了“历史帧只提供文本 memory，loss 始终读取当前帧 advantage”的契约。

### Ignore、dropout 和 RA-BC

- all-ignore batch 在 memory keep/drop 两种状态下，FM 都是计算图安全的 0；current subtask CE 继续更新；
- advantage dropout weight fallback 不读取 `memory_condition_kept`；
- negative 样本仍要求离线 weight=1；
- ignore 始终优先强制 weight=0；
- memory-enabled policy config 下 `use_rabc + use_advantage_weighting` 仍在 config validation 明确拒绝；
- `update_policy()` 的双 provider 防线和既有 RA-BC 路径继续由同一专项文件回归覆盖。

## 一键验收脚本

脚本：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_5.sh
```

脚本执行：

1. 对相关生产源码和本阶段测试执行 `py_compile`；
2. 复跑 `validate_milestone_4.sh`，累计覆盖 M0–M4、Dataset、processor、训练、模型、subtask 和
   advantage 基线；
3. 运行 M5 advantage/memory、PI0/PI0.5、global/subtask、loss、history 和 converter 聚焦回归；
4. Ruff 可用时检查本阶段测试，不可用时明确打印 skipped；
5. 执行 `git diff --check`。

## 实际执行结果

实施前在沙箱外复跑最终 M4 基线：

```text
197 passed, 6 skipped, 2 warnings in 6.01s
35 passed in 2.06s
42 passed, 2 warnings in 1.76s
60 passed in 1.52s
py_compile: passed
git diff --check: passed
ruff: skipped（未安装）
```

新增测试首轮聚焦验证：

```text
tests/scripts/test_advantage_weighted_train.py: 23 passed in 1.51s
new history advantage isolation cases: 2 passed in 0.01s
py_compile: passed
```

最终在沙箱外执行完整 M5 脚本：

```text
M0–M3 / memory / subtask / advantage / tokenizer / policy regression:
216 passed, 6 skipped, 2 warnings in 5.84s

Dataset reader/facade core regression:
35 passed in 2.04s

M3 + advantage train/helper integration（包含新增矩阵）:
59 passed, 2 warnings in 1.73s

M4 model/attention/reset/logging focused regression:
60 passed in 1.52s

M5 advantage/memory compatibility focused regression:
172 passed, 2 warnings in 6.29s

py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
```

6 个 skip 都来自既有 tokenizer CUDA/多 GPU 用例。累计回归的 2 个 warning 是既有测试刻意设置
`subtask_max_decode_tokens > subtask_max_tokens`；训练/M5 专项的 2 个 warning 是 Python 3.12 对
多线程进程中 `fork()` 的 deprecation 提示。没有新增失败、xfail 或非预期 warning。

首次在受限 sandbox 内运行包含双 worker 的累计基线时，在已知的 multiprocessing 隔离位置停止产生进展；
该进程被终止后，按 M1–M4 相同方式在 sandbox 外完整重跑并全部通过。没有把 sandbox 停滞记录为源码失败
或测试通过。

## 工作区保护复核

实施前后均检查 `git status --short`。本阶段没有 reset、checkout、暂存、覆盖或格式化无关文件，只修改
本记录“修改文件”列出的两个测试，并新增验收脚本和本记录。

特别保留：

- `scripts/nero_teleop/README.md`
- `src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py`
- `src/lerobot/scripts/lerobot_push_dataset.py`
- `tests/scripts/test_lerobot_push_dataset.py`
- 用户的计划目录迁移以及 Milestone 0–4 全部现有实现和记录

## 未运行项

- Ruff 未安装，因此没有运行 Ruff lint/format check；
- 未运行既有 CUDA 或多 GPU tokenizer 测试；
- 未加载完整 PI0/PI0.5 checkpoint 做真实 GPU forward/backward 或长训练；
- 未使用重新构建、同时包含 subtask/progress 和 advantage 的真实 LeRobotDataset；
- 未运行 RTC、fake robot、终端 dashboard 或 Nero 实机。

这些项目不阻塞 Milestone 5。本阶段 Definition of Done 是 Advantage/RL 与 Memory 的数学和训练入口兼容
矩阵；完整 checkpoint/data smoke 和实机属于 Milestone 8。

## 下一阶段输入

Milestone 6 可以依赖以下已确认契约：

```text
Memory 是无 loss 的 main prompt condition
历史帧不提供 advantage label/weight
当前帧 AdvantageWeights 只加权 FM
current subtask CE 始终普通 mean
memory keep/drop 不改变 advantage fallback 或 RA-BC 互斥
PI0/PI0.5 × global/subtask 均已通过
```

下一阶段应只实现 `RTCInferenceEngine` 内受锁保护的 memory 事务状态、成功 merge 后提交、reset 清理和
debug snapshot；不要在 RTC 中重新实现训练 weight 或 loss 逻辑。
