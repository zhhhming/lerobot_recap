# Subtask Elapsed-Time Milestone T3 Completion Record

日期：2026-07-17

基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655`

计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

环境：Conda `lerobot-main`；CPU；`HF_HUB_OFFLINE=1`；`TRANSFORMERS_OFFLINE=1`。

## 完成状态

Milestone T3：Processor、PI0/PI0.5 config 和 checkpoint 结构已完成。

本阶段实现确定性的 elapsed-time prompt processor、converter route-through、PI0/PI0.5 policy config、processor
pipeline/token budget、非 resume structural rebuild，以及 checkpoint save/reload/resume/load 契约。没有提前实现 T4–T8
的 deploy sequence scanner、tracker、RTC transaction、soft pause/home、dashboard 或 Nero/Pico 实机路径。

模型 attention 和 loss 公式不需要变化，因此本阶段没有修改 `modeling_pi0.py` 或 `modeling_pi05.py`。

## 修改和新增文件

生产代码：

- 新增 `src/lerobot/processor/subtask_time_processor.py`；
- 修改 `src/lerobot/processor/__init__.py`；
- 修改 `src/lerobot/processor/converters.py`；
- 修改 `src/lerobot/policies/pi0/configuration_pi0.py`；
- 修改 `src/lerobot/policies/pi0/processor_pi0.py`；
- 修改 `src/lerobot/policies/pi05/configuration_pi05.py`；
- 修改 `src/lerobot/policies/pi05/processor_pi05.py`；
- 局部修改 `src/lerobot/scripts/lerobot_train.py` 的 structural processor rebuild 条件和日志。

测试与验收：

- 扩展 `tests/processor/test_subtask_time_processor.py`；
- 扩展 `tests/processor/test_converters.py`；
- 新增 `tests/scripts/test_subtask_time_checkpoint.py`；
- 新增 `plans/timer_pipeline/validate_subtask_time_milestone_3.sh`；
- 新增本完成记录。

T0 的两条 T3 strict xfail 已替换为完整可执行测试。剩余 2 条 strict xfail 只对应尚未实施的 T4 tracker。

## Formatter 和 processor

新增并注册：

```text
format_subtask_elapsed_time()
SubtaskTimeConditionProcessorStep
registry: subtask_time_condition_processor
```

formatter 固定输出 canonical label，不允许通过配置改变：

```text
Subtask elapsed time: 1.2s
```

实现约束：

- 接受 Python real scalar 或单元素 tensor；
- 拒绝 bool、string、多元素 tensor、NaN、Inf 和负数；
- 固定一位小数并使用 fixed-point，不使用科学计数法；
- `-0.0` 规范化为 `0.0`；
- processor 不采噪声、不采 dropout、不读取 dataset 或模型输出；
- 支持 scalar、list/tuple、tensor `[B]` 和 `[B,1]`；
- valid 和 keep 必须是 bool，batch size 必须与 task 一致；
- effective keep 固定为 `valid AND requested_keep`；
- invalid/drop 样本的 task 逐字符不变；
- time 三字段完全缺失时 no-op，并产生全 false effective keep，支持 checkpoint time step 的部署 ablation；
- time source 部分缺字段时早失败，避免静默使用不完整输入；
- 追加 condition 时只规范化 task 尾部空白，然后使用一个换行分隔。

Processor `get_config()` 保存字段键，registry save/reload 后输出一致。

## Converter

`batch_to_transition()` 现在显式 route-through：

```text
subtask_elapsed_seconds
subtask_time_valid
subtask_segment_index
subtask_time_seconds
subtask_time_condition_kept
```

这里没有只依赖 `subtask_time_` 前缀，因为 true elapsed 和 segment index 不属于该前缀。

## PI0 / PI0.5 config

两种 config 均新增：

```python
use_subtask_time_conditioning: bool = False
subtask_time_tokenizer_max_length: int = 128
```

验证行为：

- tokenizer budget 必须为正数；
- time 开启要求 `predict_subtask=true`；
- time 开启但 `subtask_generate_at_inference=false` 时发出明确 warning；
- 旧 config JSON 缺少两个字段时使用 dataclass 安全默认值，保持 time 关闭；
- 默认关闭路径不增加 processor step，不改变 prompt、token 或 tokenizer budget。

## Pipeline 和 token budget

PI0 顺序：

```text
Advantage(optional)
-> Memory(optional)
-> Subtask Time(optional)
-> PI0 Newline
-> Current Subtask target(optional)
-> Tokenizer
```

PI0.5 顺序：

```text
Normalize
-> Advantage(optional)
-> Memory(optional)
-> Subtask Time(optional)
-> PI0.5 State Prompt
-> Current Subtask target(optional)
-> Tokenizer
```

主 prompt tokenizer budget 统一为：

```python
max(
    base tokenizer budget,
    enabled memory budget,
    enabled subtask-time budget,
)
```

Advantage、memory 或 time 任一开启时使用 left truncation。已验证：

- PI0 neither 为 48；time-only/memory-only/both 为 128；
- PI0.5 四种组合均保持 200，不被 128 反向缩短；
- memory + `95.8s` time + PI0.5 state suffix 在真 pipeline character decode 中保留；
- time keep/drop 只改变 main tokens，current-subtask target tensor 逐 tensor 相同。

## Structural rebuild 和 checkpoint

`make_train_pre_post_processors()` 的 structural prompt fields 从 memory 单项扩展为：

```text
use_memory_conditioning
use_subtask_time_conditioning
```

契约：

- 非 resume 使用旧 checkpoint 权重并通过 CLI 开启 time 时，放弃加载旧 processor JSON，按当前 config 和 dataset
  stats 重建 processor；
- 权重 checkpoint 路径本身不被修改；
- 日志列出触发重建的具体 structural 字段；
- memory 原有日志契约保持兼容；
- resume 从保存的 checkpoint processor JSON 加载 time step，不重建；
- checkpoint `config.json` 持久化 time flag 和 budget；
- `policy_preprocessor.json` 恰好包含一个 `subtask_time_condition_processor` 和 effective tokenizer length；
- 普通 `make_pre_post_processors(..., pretrained_path=...)` 可直接加载 time processor，不需要额外手写 override。

## 测试覆盖

T3 新增/扩展测试覆盖：

- canonical formatter、fixed point、负零、大数和错误输入；
- scalar/list/tuple/`[B]`/`[B,1]`；
- invalid/drop/no-source 的 byte-for-byte no-op；
- partial source、batch size、shape、seconds/valid/keep dtype 错误；
- registry、`get_config()`、processor save/reload；
- converter 五字段；
- PI0/PI0.5 config 默认值、验证和 generation warning；
- time-only、memory-only、both、neither 八个 policy/config 组合；
- pipeline 顺序、left truncation、token budget 和长 prompt decode；
- current-subtask target 隔离；
- 旧 config 缺字段加载；
- 非 resume 旧 checkpoint CLI override structural rebuild；
- checkpoint config/processor JSON；
- resume 和普通 checkpoint load。

实现前扩展测试真实失败于：

```text
ModuleNotFoundError: lerobot.processor.subtask_time_processor
ImportError: SubtaskTimeConditionProcessorStep is not exported
```

这证明失败原因是 T3 功能缺失，不是 fixture、collection 参数或断言错误。

## 一键验收入口

```bash
plans/timer_pipeline/validate_subtask_time_milestone_3.sh [contract|checkpoint|regression|all]
```

模式：

- `contract`：py_compile、T3 processor/converter/关闭态，以及 T0–T2 time contract；
- `checkpoint`：旧 config、structural rebuild、save/reload/resume/checkpoint load；
- `regression`：memory、subtask AR、advantage、RTC memory、deploy status 和 T1/T2 time 聚焦回归；
- `all`：依次执行以上三项，默认模式。

脚本固定 offline、默认使用 `lerobot-main`，最后运行 `bash -n` 和 `git diff --check`。Ruff 仅在环境中存在时
执行；当前环境未安装 Ruff，因此明确记录 skipped，没有写成 passed。

## 实际验收结果

最终命令：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_3.sh all
```

实际输出：

```text
T3 processor/converter/time-disabled contract:
54 passed in 1.49s

T0-T2 time cumulative contract:
89 passed, 4 deselected, 2 xfailed in 1.59s

T3 checkpoint contract:
6 passed in 1.90s

memory/subtask/advantage/RTC/time focused regression:
263 passed, 6 deselected, 2 warnings in 4.17s

script exit code: 0
py_compile: passed
bash -n: passed
git diff --check: passed
ruff: skipped (not installed)
```

两个 xfail 均是 T4 `SubtaskTimeTracker` 尚未实现的冻结契约，不属于 T3。两个 warning 是既有
`subtask_max_decode_tokens > subtask_max_tokens` 测试 warning。

`-k "not workers"` 排除了 4/6 个名称含 `workers` 的既有 T1/memory DataLoader 多进程用例。第一次递归调用旧
T2 contract 时，这些用例在当前受限沙箱复现了 T1/T2 完成记录已说明的无输出阻塞，因此 T3 脚本改为显式排除；
没有把 timeout 或 deselected 写成 passed。这些 worker 用例已在 T1 正常进程环境实际通过，且 T3 没有修改 Dataset
wrapper、DataLoader 或 history raw access。

本阶段不需要真实 dataset、GPU、网络、fake robot 或 Nero 实机；这些不属于 T3 验收范围。

## 回归修复记录

首轮累计回归发现 2 个 memory 测试失败，原因是 structural rebuild 日志首字母从既有小写改成了大写。生产逻辑
正确，但这属于已有可观察日志契约回归。实现已恢复原小写子串，同时保留新增的触发字段列表；最终累计回归通过。

## 工作区保护复核

开始和完成时均确认工作区包含尚未提交的 memory M0–M8、timer T0–T2、计划目录移动和 Nero 用户改动。本阶段：

- 没有执行 reset、checkout、clean、commit 或覆盖操作；
- 对已有 staged/modified 文件只做局部 patch；
- 没有格式化无关文件；
- 没有修改 memory processor、PI0/PI0.5 modeling、RTC、deploy、Nero 或 Pico 生产逻辑；
- T4–T8 仍为未完成状态，不能因本记录而标记完成。
