# Memory Pipeline Milestone 4 Completion Record

日期：2026-07-17

基线：`main@8194e710`

Python：`/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`（Conda 环境 `lerobot-main`，
CPU 验收显式设置 `CUDA_VISIBLE_DEVICES=''`）

## 完成范围

本次完成 `pi0_pi05_memory_training_deployment_plan.md` 中的 Milestone 4：PI0/PI0.5 模型与
attention 回归。

Milestone 2 已把历史 Memory condition 放入 main prompt；本阶段使用两套 policy 的真实
`embed_prefix()`、`make_att_2d_masks()`、`apply_subtask_attention_dropout()` 和训练 `forward()`，配合
轻量可微 fake backbone，证明不需要新增 memory tensor 参数、模型分支、KV-cache 分支或 loss，current
subtask AR 和 action FM 已能按现有 prefix attention 正确看到 Memory。

生产代码仅补齐 reset 语义状态清理，并把重复 subtask change 日志从 INFO 调整为 DEBUG。没有修改
attention、forward、denoise、CE/FM 公式、processor、训练循环、RTC 或 deploy。

Advantage/RL 组合属于 Milestone 5，RTC memory 事务闭环属于 Milestone 6，终端状态面板属于
Milestone 7，均未提前实施。

## 修改文件

生产代码：

- 修改 `src/lerobot/policies/pi0/modeling_pi0.py`
- 修改 `src/lerobot/policies/pi05/modeling_pi05.py`

测试和记录：

- 新增 `tests/policies/pi0_pi05/test_memory_modeling.py`
- 扩展 `tests/policies/pi0_pi05/test_pi0_subtask_inference.py`
- 扩展 `tests/policies/pi0_pi05/test_pi05_subtask_inference.py`
- 新增 `plans/memory_pipeline/validate_milestone_4.sh`
- 新增本完成记录

## 实现契约

### 模型不新增 Memory 分支

Memory keep/drop 已在 processor tokenize 前表现为 main prompt token/mask 的差异。两套模型继续使用：

```text
[image + main prompt（含可选 Memory，双向 prefix）]
[current subtask target（逐 token causal）]
[state/action suffix（PI0）或 action suffix（PI0.5）]
```

测试直接检查两套真实 attention 构造产生的最终 4D mask，并确认：

- Memory keep 时，current subtask 的每个 query 都能看到 Memory prefix token；
- current subtask 内部保持下三角 causal attention；
- suffix 能看到 Memory；
- `subtask_dropout_prob=1` 只移除 suffix 到 current-subtask 列的 attention；
- subtask dropout 不移除 main prompt 中的 Memory；
- Memory drop 时对应 main prompt 位置被 pad mask 完整屏蔽；
- current subtask target token 和 target mask 在所有组合中保持不变。

因此没有为“memory 是新功能”而修改现有 attention 或复制一套 forward/KV-cache 逻辑。

### 四种 dropout 组合和梯度

PI0/PI0.5 均覆盖：

```text
memory keep/drop × current subtask 对 suffix keep/drop
```

四种组合的 FM 与 current subtask CE 均为 finite。组合 loss 可以 backward；Memory keep 时 fake
backbone 的 Memory token embedding 获得来自 AR/FM 可见路径的非零梯度，Memory drop 时该 token 的梯度
严格为零。这同时证明 dropout 是 attention/input 条件差异，没有修改 target 或新增 memory loss。

### Reset 语义状态

PI0Policy 和 PI05Policy 的 `reset()` 现在除原 action queue 外同时清理：

```text
last_subtask_text = ""
_last_logged_subtask_text = None
model._last_subtask_tokens = None
```

这防止 pause/home/restart 后后续 RTC 或 dashboard 读取 stale subtask/token。RTC engine 自身的 memory
状态和 reset-version 事务仍由 Milestone 6 实现。

### Subtask 日志

生成文本变化时仍保留一次 `[subtask]` 日志，但级别从 INFO 降为 DEBUG；相同文本连续生成不会重复记录。
后续 Milestone 7 可从 engine snapshot 固定显示 SUBTASK，而不会与模型 INFO 日志重复刷屏；普通 debug
场景仍能观察生成文本。

### State dict 兼容

测试在 PI0/PI0.5 构造器中替换仅负责减小体积的 backbone，分别创建 memory on/off core model：

- 两者 `state_dict().keys()` 完全相同；
- memory-off state dict 可对 memory-on model 执行 `strict=True` 加载；
- missing keys 和 unexpected keys 均为空。

新增 reset 字段都是非 Parameter Python 状态，因此不引入 checkpoint 权重键。

## 测试先行记录

实施前模型/processor 专项基线：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  /home/zenbot-robot/.conda/envs/lerobot-main/bin/python -m pytest \
  tests/policies/pi0_pi05/test_pi0_subtask_training.py \
  tests/policies/pi0_pi05/test_pi05_subtask_training.py \
  tests/policies/pi0_pi05/test_pi0_subtask_inference.py \
  tests/policies/pi0_pi05/test_pi05_subtask_inference.py \
  tests/processor/test_memory_processor.py \
  tests/processor/test_memory_disabled_baseline.py -q
```

结果：

```text
48 passed in 1.50s
```

新增测试首次运行真实暴露：

- PI0/PI0.5 reset 均未清理 stale subtask 状态；
- PI0/PI0.5 subtask change 仍以 INFO 输出。

attention/四组合断言已通过。首次梯度断言还发现测试自身把“Memory 位置索引 2”误当成“Memory token id
7”读取 embedding gradient；修正测试索引后没有借此改动生产 attention。

完成最小生产修改后，新增 memory model 和两套 inference 测试结果：

```text
20 passed in 1.48s
```

## 一键验收脚本

脚本：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_4.sh
```

脚本执行：

1. 对两套模型和本阶段测试执行 `py_compile`；
2. 复跑 `validate_milestone_3.sh`，累计覆盖 M0–M3、Dataset、processor、训练 dropout、checkpoint
   processor/resume、subtask、advantage 和旧 policy 回归；
3. 运行 M4 memory modeling、PI0/PI0.5 subtask train/inference、Memory processor 和默认关闭 golden；
4. Ruff 可用时检查本阶段文件，不可用时明确打印 skipped；
5. 执行 `git diff --check`。

## 最终实际结果

受限 sandbox 内首次执行 M3 累计基线时，运行到真实
`DataLoader(num_workers=2, multiprocessing_context="spawn")` 后停止产生进展；这是 M1–M3 完成记录已知
的进程隔离限制。终止该次运行后，没有残留 pytest/multiprocessing 进程，也没有把它记录为源码失败。

随后按既有 milestone 相同方式在 sandbox 外执行完整 M4 脚本：

```bash
env CUDA_VISIBLE_DEVICES= plans/memory_pipeline/validate_milestone_4.sh
```

结果：

```text
Milestone 0–3 / memory / subtask / advantage / tokenizer / policy regression:
197 passed, 6 skipped, 2 warnings in 6.06s

Dataset reader/facade core regression:
35 passed in 2.07s

Milestone 3 + advantage train/helper integration:
42 passed, 2 warnings in 1.74s

Milestone 4 model/attention/reset/logging focused regression:
60 passed in 1.49s

py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
```

完成最后测试审阅的小调整（相同 subtask 连续生成只记录一次 DEBUG）后，再次运行 M4 专项与
`git diff --check`：

```text
60 passed in 1.53s
git diff --check: passed
```

完成记录落盘后又执行一次最终完整脚本，确认最终工作区内容：

```text
197 passed, 6 skipped, 2 warnings in 6.05s
35 passed in 2.12s
42 passed, 2 warnings in 1.74s
60 passed in 1.53s
py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
```

6 个 skip 均来自既有 tokenizer CUDA/多 GPU 用例。累计回归的 2 个 warning 是既有测试刻意设置
`subtask_max_decode_tokens > subtask_max_tokens`；训练专项的 2 个 warning 是 Python 3.12 对多线程进程中
`fork()` 的 deprecation 提示。没有新增非预期 warning、xfail 或失败。

## 工作区保护复核

实施前确认工作区已有 M0–M3、计划目录迁移以及 Nero/push-dataset 等用户修改。本次没有 reset、
checkout、暂存、覆盖或格式化无关文件，只修改本记录“修改文件”列出的 Milestone 4 文件。

特别保留：

- `scripts/nero_teleop/README.md`
- `src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py`
- `src/lerobot/scripts/lerobot_push_dataset.py`
- `tests/scripts/test_lerobot_push_dataset.py`
- M0–M3 的所有现有源码、测试和完成记录

## 未运行项

- Ruff 未安装，因此没有运行 Ruff lint/format check；
- 未运行既有 CUDA 或多 GPU tokenizer 测试；
- 本地没有完整 PI0/PI0.5 模型权重，未运行真实 PaliGemma checkpoint GPU forward/backward；
- 未使用真实新构建的 subtask/progress LeRobotDataset；
- 未运行 RTC、fake robot、Nero 实机或终端 dashboard。

这些项目不阻塞 Milestone 4：本阶段计划明确允许 hook/fake backbone 检查 main prompt prefix、attention、
gradient 和 state-dict 契约。完整 checkpoint/GPU/RTC/实机分别属于后续集成阶段。

## 下一阶段输入

Milestone 5 可以直接依赖本阶段确认的模型契约：

```text
Memory 只属于 main prompt condition
current subtask CE 保持普通 per-sample CE/mean
FM 保持现有 per-sample loss
memory/subtask dropout 不修改 current-frame advantage label 或 weight
memory on/off 不改变模型权重结构
```

下一阶段应只扩展 Advantage/RL 兼容矩阵，证明 positive/negative/ignore、global/subtask key、advantage
dropout 和 memory dropout 不污染 FM-only weighting及未加权 current subtask CE；不要在 Milestone 5
重新设计模型 attention 或新增 memory loss。
