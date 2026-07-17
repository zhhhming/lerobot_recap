# Memory Pipeline Milestone 3 Completion Record

日期：2026-07-17

基线：`main@8194e710`

Python：`/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`（Conda 环境 `lerobot-main`，
Python 3.12.13）

## 完成范围

本次完成 `pi0_pi05_memory_training_deployment_plan.md` 中的 Milestone 3：训练 dropout、processor
结构重建和 memory 训练诊断日志。

Milestone 1 的动态历史字段现在会在真实 `lerobot_train` 的 preprocessor 前生成独立
`memory_condition_kept` mask；从旧 checkpoint 非 resume 开启 memory 时，训练入口会按当前 policy
config 和当前 dataset stats 重建 processor，避免 config 已开启但 prompt 中没有 Memory step 的静默失败。
保存后的 checkpoint processor 可直接重新加载和 resume。

本阶段没有修改 PI0/PI0.5 模型 attention、subtask decode/reset、RTC、policy deploy 或终端状态面板；
这些仍属于 Milestone 4–7。

## 修改文件

生产代码：

- 新增 `src/lerobot/utils/memory_conditioning.py`
- 修改 `src/lerobot/configs/train.py`
- 修改 `src/lerobot/scripts/lerobot_train.py`

测试和验收记录：

- 扩展 `tests/utils/test_memory_conditioning.py`
- 新增 `tests/scripts/test_memory_train.py`
- 新增 `plans/memory_pipeline/validate_milestone_3.sh`
- 新增本完成记录

没有修改 Milestone 1/2 的 Dataset wrapper、Memory processor、PI0/PI0.5 processor pipeline 或模型文件。

## 实现契约

### Train config 和早失败

`TrainPipelineConfig` 新增：

```python
memory_lookback_min_frames: int = 1
memory_lookback_max_frames: int = 12
memory_dropout_prob: float = 0.2
```

验证内容：

- lookback 必须是非 bool 的整数，且满足 `1 <= min <= max`；
- dropout 必须在 `[0,1]`；
- memory 只支持 PI0/PI0.5；
- memory 开启时 dataset 必须是 non-streaming。

Dataset metadata 的 `subtask`、`subtask_progress`、frame/episode/index 字段验证继续由 Milestone 1 的
`make_dataset()` / `MemoryHistoryDataset` 在创建训练任务早期执行，没有复制另一套不一致的 schema 校验。

### Memory condition dropout

新增 `sample_memory_condition_mask()`：

```text
eligible = memory_valid AND non_empty(memory_subtask)
kept = eligible AND Bernoulli(1 - memory_dropout_prob)
```

行为：

- 输入 batch 不原地修改；
- 输出 `memory_condition_kept` 为严格 `[B]` bool tensor；
- `p=0` 保留全部 eligible，`p=1` 全部删除；
- natural invalid 和空 memory source 始终为 false；
- 使用 PyTorch RNG，并支持显式 generator 做可重复测试；
- 与 `sample_advantage_condition_mask()` 分开调用、分开采样；
- 不读取或修改 advantage label、weight、keep mask。

真实训练循环顺序现在是：

```text
raw DataLoader batch
-> advantage condition mask（可选）
-> memory condition mask（可选）
-> preprocessor
-> policy.forward
-> 原有 plain / RA-BC / advantage-weighted loss
```

RA-BC 与 advantage weighting 的原有互斥校验保持不变，memory 不进入 loss。

### 旧 checkpoint processor 重建和 resume

训练入口抽出并实际调用 `make_train_pre_post_processors()`：

- 非 resume、`use_memory_conditioning=true` 且存在 `policy.pretrained_path` 时，只把局部 processor 加载
  路径置空；policy 权重来源保持原 checkpoint 不变；
- 随后使用当前 policy config 和 dataset stats 重建 pre/post processor，并打印明确 structural config
  日志；
- resume 时保留 checkpoint processor 加载路径和现有 overrides，加载已保存 Memory step；
- 与既有 `use_relative_actions` structural rebuild 规则兼容。

集成测试创建了一个不含 Memory step 的旧 PI0/PI0.5 processor checkpoint，使用 policy CLI override
加载 `predict_subtask=true/use_memory_conditioning=true`，再经过训练入口 helper，最终 processor 均真实包含
Memory step。测试同时确认 `policy.pretrained_path` 未被改变，因此不会丢失权重加载来源。

checkpoint 测试使用实际 `save_checkpoint()` 保存 processor、train config、optimizer 和 RNG/training step，
并确认：

- `policy_preprocessor.json` 包含 `memory_condition_processor`；
- tokenizer effective max length 为 PI0=128、PI0.5=200；
- resume 重新加载后 Memory step 仍存在；
- training step 恢复为保存的 step 2。

### Memory train metrics

新增 `MemoryTrainingMetrics`，按日志窗口累计：

```text
memory/history_valid_fraction
memory/condition_kept_fraction
memory/dropout_fraction_among_valid
memory/lookback_frames_mean
memory/lookback_frames_min_seen
memory/lookback_frames_max_seen
```

其中 valid/kept/mean 按真实样本数累计，dropout denominator 是 eligible history 数量；lookback 使用所有先抽到
的 offset，包括 episode 开头自然无历史样本，min/max 是窗口真实极值。无 valid history 时 dropout fraction
安全为 0。指标只写 console/WandB 诊断，不进入 policy batch loss 组合。

## 测试覆盖

`tests/utils/test_memory_conditioning.py` 的两个 strict xfail 已移除，并扩展覆盖：

- p=0/1、natural invalid、空 source、输入不原地修改；
- 固定 PyTorch generator 可重复；
- strict bool shape；
- 单 batch 和跨 batch 指标手算、最后小 batch加权及 min/max。

`tests/scripts/test_memory_train.py` 覆盖：

- Train config 默认值、非法 lookback/dropout 和 streaming early failure；
- advantage keep × memory keep 四种组合；
- `batch_size=1`、最后一个小 batch、`num_workers=0/2`；
- PI0/PI0.5 旧 checkpoint + CLI memory override 的真实训练 processor 选择；
- checkpoint processor JSON、save/reload/resume 和 training step；
- PI0/PI0.5 各使用真实 processor pipeline + 轻量可微 policy 完成 2-step optimizer update。

## 一键验收脚本

脚本：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_3.sh
```

脚本行为：

1. 对本阶段生产源码和测试执行 `py_compile`；
2. 复跑 `validate_milestone_2.sh`，覆盖 memory history/processor、默认关闭 golden、subtask、advantage、
   tokenizer、converter、PI0/PI0.5 train/inference 和 Dataset 核心回归；
3. 运行 Milestone 3 memory train、memory helper 和 advantage train/helper 专项；
4. Ruff 可用时执行 lint/format check，不可用时明确打印 skipped；
5. 执行 `git diff --check`。

## 实际执行结果

实施前基线在沙箱外执行：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_2.sh
```

结果：

```text
193 passed, 6 skipped, 2 xfailed, 2 warnings in 6.73s
35 passed in 2.25s
py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
```

2 个 strict xfail 正是 Milestone 3 的 helper 预留契约。

实现后专项首轮：

```text
21 passed, 2 warnings in 1.81s
```

最终完整验收：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_3.sh
```

结果：

```text
Milestone 0–3 / memory / subtask / advantage / tokenizer / policy regression:
197 passed, 6 skipped, 2 warnings in 6.41s

Dataset reader/facade core regression:
35 passed in 2.19s

Milestone 3 + advantage train/helper integration:
42 passed, 2 warnings in 1.82s

py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
remaining xfail: 0
```

6 个 skip 均为既有 tokenizer CUDA/多 GPU 用例。主回归的 2 个 warning 是既有测试刻意构造
`subtask_max_decode_tokens > subtask_max_tokens`。专项的 2 个 warning 是 Python 3.12 提示多线程 pytest
进程中 `fork()` 未来可能弃用；真实 `num_workers=2` 用例已完成通过，没有死锁或残留进程。

受限 sandbox 内两次运行真实 multi-worker 用例都停在进程隔离处；终止对应测试进程后在 sandbox 外完整
重跑通过。没有把受限环境轮询超时写成源码失败或测试通过。

## 工作区保护复核

实施前确认当前工作区包含用户的计划目录迁移、Milestone 0–2 实现以及 Nero/push-dataset 等未提交改动。
本次没有 reset、checkout、暂存、覆盖或格式化无关文件，只增量修改本记录“修改文件”列出的 Milestone 3
文件。所有既有用户改动保持原状。

## 未运行项

- Ruff 未安装，因此没有运行 Ruff lint/format check；
- 未运行既有 CUDA 或多 GPU tokenizer 测试；
- 本地只有 PaliGemma tokenizer/config 缓存，没有完整 PI0/PI0.5 模型权重；
- 未运行完整 PI0/PI0.5 checkpoint 的真实 forward/backward 或 GPU 训练；
- 未使用真实新构建的 subtask/progress LeRobotDataset；
- 未运行 RTC、fake robot、Nero 实机或终端 dashboard。

以上不阻塞 Milestone 3。本阶段验收要求是两个 policy 的 fake/small 2-step update、真实训练 processor
选择和 checkpoint processor/resume 契约；完整模型 attention 属于 Milestone 4，RTC/实机属于后续阶段。

## 下一阶段输入

Milestone 4 可以直接依赖：

```text
MemoryHistoryDataset -> memory_subtask/progress/valid/offset
sample_memory_condition_mask -> memory_condition_kept
MemoryConditionProcessorStep -> main prompt Memory block
lerobot_train -> saved/reloadable memory processor checkpoint
```

下一阶段应只验证 PI0/PI0.5 main prompt memory 进入 prefix、current subtask causal attention和既有 subtask
dropout 保持，并补 policy reset 清理语义状态；不要在模型中新增无必要的 memory tensor参数或独立 loss。
