# Value Pipeline Milestone 10 Completion Record

日期：2026-07-13

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 推荐实施顺序中的第七阶段：
Milestone 10（VLA advantage conditioning processor）。

Milestone 9 已经确认 raw label/weight 能进入 LeRobotDataset、DataLoader 和 policy complementary data；
本阶段在该数据入口之上，为 pi0 和 pi0.5 增加可选的主 task prompt conditioning。processor 本身保持
确定性，训练期 classifier-free dropout 由后续训练循环预先生成 `advantage_condition_kept`，不会隐藏
在同时用于部署的 processor 内。

本阶段不实现 Milestone 11 的 Bernoulli mask 生成、dropout weight fallback、ignore loss、FM-only
weighted mean 或 `use_rabc` 互斥逻辑。

## 修改文件

- 新增 `src/lerobot/processor/advantage_processor.py`
- 修改 `src/lerobot/processor/__init__.py`
- 修改 `src/lerobot/processor/tokenizer_processor.py`
- 修改 `src/lerobot/policies/pi0/configuration_pi0.py`
- 修改 `src/lerobot/policies/pi0/processor_pi0.py`
- 修改 `src/lerobot/policies/pi05/configuration_pi05.py`
- 修改 `src/lerobot/policies/pi05/processor_pi05.py`
- 新增 `tests/processor/test_advantage_processor.py`
- 修改 `tests/processor/test_tokenizer_processor.py`
- 修改 `tests/scripts/test_value_extras_build_dataset.py`
- 新增 `plans/value_pipeline/validate_milestone_10.sh`
- 新增本完成记录

`src/lerobot/processor/converters.py` 已在 Milestone 9 完成 `advantage_label_*`、
`advantage_loss_weight_*` 和 `advantage_condition_kept` 的 complementary data 透传及 `[B]` shape
规范化，本阶段经回归确认可直接复用，没有重复修改。

## Advantage processor 契约

新增注册名为 `advantage_condition_processor` 的 `AdvantageConditionProcessorStep`，默认字段为：

```text
label_key=advantage_label_global
condition_format=Advantage: {label}
inference_label=positive
task_key=task
condition_kept_key=advantage_condition_kept
```

确定性行为：

1. `positive|negative` 且 keep mask 为 true 时，在原 task 后追加一行 condition。
2. `ignore`、空 label 或 keep mask 为 false 时不追加，并把 effective keep mask 置 false。
3. label 存在而 mask 不存在时默认保留合法 condition，供 dropout=0 和接口 smoke 使用。
4. label 和 mask 都不存在时视为 eval/deploy，使用 `inference_label=positive|negative|none`。
5. mask 存在但 label 不存在时视为训练数据缺失，不使用部署 positive fallback，避免静默污染训练 prompt。
6. 输出 `advantage_condition_kept` 为严格 `[B]` bool tensor；输入 `[B, 1]` mask 会规范化。
7. 非法 label、非 bool mask、非法高维 shape 或与 task batch size 不匹配均明确报错。
8. condition format 必须包含 `{label}`，processor 支持 registry config 保存和重新加载。

processor 不读取 loss weight，也不自行采样随机数。

## pi0 / pi0.5 配置与接线

两种 policy config 均新增：

```python
use_advantage_conditioning: bool = False
advantage_label_key: str = "advantage_label_global"
advantage_loss_weight_key: str = "advantage_loss_weight_global"
advantage_condition_format: str = "Advantage: {label}"
inference_advantage_label: str = "positive"
```

默认关闭，因此旧 config、旧 checkpoint 和无 advantage 字段的现有数据流保持原行为。开启时的关键顺序：

```text
pi0:
AddBatch -> AdvantageCondition -> Pi0NewLine -> SubtaskText(optional) -> Tokenizer

pi0.5:
Normalize -> AdvantageCondition -> Pi05PrepareStatePrompt -> SubtaskText(optional) -> Tokenizer
```

pi0.5 最终主 prompt 仍保持 `Task -> Advantage -> State -> Action suffix` 的现有 state-token 结构。
`predict_subtask=true` 时，advantage processor 只修改 `task`；`subtask`、`subtask_progress`、subtask AR
token 和 attention mask 保持独立。

## 长 prompt 截断保护

原 tokenizer 默认从右侧截断，而 advantage condition 位于原始 task 之后，长 task 可能把 condition
静默截掉。`TokenizerProcessorStep` 因此新增可序列化的 `truncation_side` override：

- conditioning 关闭：保持 tokenizer 原默认值，不改变旧行为；
- conditioning 开启：pi0/pi0.5 主 prompt 使用 left truncation，优先保留 prompt 尾部的 advantage
  condition 以及 pi0.5 state/action suffix；
- 每次 tokenize 后恢复 tokenizer 原 `truncation_side`，避免共享 tokenizer 状态泄漏。

测试使用 500-word task，并直接检查最终 main token tensor 中仍包含 `Advantage` 和 `positive` token，
不是只检查 tokenization 前的字符串。

## Tokenizer 旧测试契约整理

现有 `TokenizerProcessorStep` 已在之前的 subtask AR 工作中改为
`tokenize_subtask=False` 默认关闭，并要求独立 `subtask_max_length`、显式 EOS 和固定 batch shape。
`tests/processor/test_tokenizer_processor.py` 中仍有一组旧测试假定“只要存在 subtask 就默认 tokenize”，
与当前实现及 `test_subtask_ar_processors.py` 相冲突。本阶段同步把这些测试改为显式开启
`tokenize_subtask=True`，补齐 mock EOS/encode，并按 `[B, subtask_max_length]` 断言。生产默认行为未被
改回旧语义，完整 tokenizer 测试重新通过。

## 新增测试覆盖

`tests/processor/test_advantage_processor.py` 覆盖：

- positive、negative、ignore 和空 label；
- 全 true / 全 false mask，分别模拟 dropout=0 / dropout=1；
- `[B, 1] -> [B]` effective mask；
- global/subtask 自定义 label key 和 condition format；
- eval/deploy positive、negative、none fallback；
- 训练 mask 存在而 label 缺失时不启用 inference fallback；
- 非法 label、mask dtype、shape 和 batch length 拒绝；
- registry、config、pipeline save/reload；
- pi0/pi0.5 processor 顺序、prompt 内容和默认关闭行为；
- `predict_subtask=true` 时相同 subtask 的 AR token 不受 advantage label/mask 影响；
- 长 prompt 的最终 token 保留。

`tests/scripts/test_value_extras_build_dataset.py` 继续使用实际构建的小 LeRobotDataset，并新增确认：

- DataLoader label 和手工训练 mask 经 converter 进入开启 conditioning 的真实 pi0 preprocessor；
- keep=true 的 positive 样本生成 condition；
- keep=false 的 negative 样本不生成 condition；
- tokenizer 实际收到修改后的 task batch。

## 一键验收和结果

验收脚本：

```bash
plans/value_pipeline/validate_milestone_10.sh
```

脚本使用 `/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`，依次执行：

1. 本阶段源码和测试 `py_compile`。
2. advantage/subtask/tokenizer/converter/actual-build dataset 专项回归。
3. Milestone 0/1/1.5/6/7/8/9 核心、UI API 和 wheel package 回归。
4. Milestone 8 全 70 episode shadow pipeline。
5. Milestone 9 两 episode actual-build shadow pipeline。
6. `git diff --check`。

2026-07-13 实际结果：

```text
Milestone 10 processor/tokenizer/dataset integration: 106 passed, 1 skipped
existing value pipeline / API / wheel regression: 132 passed
real sample Milestone 8 + Milestone 9 shadow smoke: 2 passed
total: 240 passed, 1 skipped
py_compile: passed
git diff --check: passed
```

受限 sandbox 内 UI/API/wheel 测试无法绑定 `127.0.0.1`，会得到
`PermissionError: Operation not permitted`。允许本机临时 HTTP socket 后完整脚本通过；这是与前两个
milestone 相同的环境约束，不是源码失败。

主要 warning 为已有 CUDA/NVML、subtask decode limit、bool histogram，以及 Hugging Face
datasets/NumPy scalar conversion deprecation；没有新增测试失败。

## 剩余边界

- 当前 processor 消费 `advantage_condition_kept`，但不会生成随机训练 mask。Milestone 11 必须在调用
  processor 前按 `advantage_condition_dropout_prob` 生成 Bernoulli mask。
- `advantage_loss_weight_key` 已作为 policy config 契约保留，但本阶段不读取或应用 weight。
- Milestone 11 仍需实现 dropped condition -> weight 1、ignore -> weight 0、FM-only weighted mean、
  all-ignore 安全行为以及与旧 `use_rabc` 的互斥。
- 当前 mock/synthetic label 只用于接口 smoke。正式训练仍须等待 Milestone 2/3/4 的 model prediction，
  重新运行 Milestone 6/7/8 并 rebuild dataset。
