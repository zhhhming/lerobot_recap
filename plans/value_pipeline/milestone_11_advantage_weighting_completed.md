# Value Pipeline Milestone 11 Completion Record

日期：2026-07-13

## 完成范围

本次完成 `value_function_subtask_advantage_pipeline_plan.md` 推荐实施顺序中的第八阶段：
Milestone 11（VLA loss weighting provider）。

Milestone 9 已确认离线生成的 `advantage_label_{mode}` 和
`advantage_loss_weight_{mode}` 能进入真实 LeRobotDataset/DataLoader batch；Milestone 10 已为 pi0 和
pi0.5 接入确定性的 advantage prompt processor。本阶段补齐训练期 Bernoulli condition dropout、batch
内预存 weight 读取、dropout/ignore 权重语义、FM-only weighted mean、subtask CE 非加权 mean、全 ignore
安全行为和旧 RA-BC 互斥。

本阶段使用 mock prediction 产物完成数据接口 shadow smoke。它不实现 Milestone 2/3/4 的真实 value
model，也不把 synthetic label/weight 视为正式实验数据；真实模型产物生成后仍需重跑 Milestone 6/7/8
并重新 build dataset。

## 修改文件

- 新增 `src/lerobot/utils/advantage_weights.py`
- 修改 `src/lerobot/configs/train.py`
- 修改 `src/lerobot/scripts/lerobot_train.py`
- 修改 `src/lerobot/policies/pi0/modeling_pi0.py`
- 修改 `src/lerobot/policies/pi05/modeling_pi05.py`
- 新增 `tests/utils/test_advantage_weights.py`
- 新增 `tests/scripts/test_advantage_weighted_train.py`
- 修改 `tests/policies/pi0_pi05/test_pi0_subtask_training.py`
- 修改 `tests/policies/pi0_pi05/test_pi05_subtask_training.py`
- 新增 `tests/value_function/test_milestone_11_shadow_smoke.py`
- 新增 `plans/value_pipeline/validate_milestone_11.sh`
- 新增本完成记录

`src/lerobot/utils/rabc.py` 未修改。Milestone 10 的 advantage processor 和 Milestone 9 的 converter/data
contract 已满足本阶段输入要求，因此没有重复修改。

## 新增训练配置

`TrainPipelineConfig` 新增：

```python
use_advantage_weighting: bool = False
advantage_loss_weight_key: str = "advantage_loss_weight_global"
advantage_label_key: str = "advantage_label_global"
advantage_condition_dropout_prob: float = 0.1
advantage_ignore_label: str = "ignore"
advantage_disable_weight_when_condition_dropped: bool = True
```

配置校验：

- dropout probability 必须在 `[0, 1]`。
- label/weight key 不能为空。
- 第一版只支持 canonical ignore label `ignore`。
- `use_rabc` 和 `use_advantage_weighting` 同时开启会明确报错。
- advantage weighting 当前只允许 pi0/pi05。
- conditioning/weighting 开启时，train-level key 必须与 policy config key 一致，避免 prompt 使用
  subtask label、loss 却误读 global weight 之类的静默交叉组合。
- 所有开关默认关闭或保持既有默认，旧训练配置不启用本路径。

## Train-only classifier-free dropout

新增 `sample_advantage_condition_mask()`，在训练循环调用 preprocessor 之前运行：

1. 只对 `positive|negative` 样本采样 Bernoulli keep。
2. `ignore` 样本始终为 false。
3. `dropout_prob=0` 时所有合法 condition 保留；`dropout_prob=1` 时全部移除。
4. 使用 PyTorch RNG，因此沿用训练 seed/RNG checkpoint 行为。
5. 输入 batch 不原地修改，输出新增严格 `[B]` bool `advantage_condition_kept`。
6. 训练 label 缺失、非法 label、非法概率均立即报错；不会回退到部署时的 positive condition。

只开启 conditioning、不启用 weighting 时也会生成该 mask，因此 classifier-free prompt dropout 可以独立
使用。eval/deploy 不经过离线训练循环，不执行随机 dropout，继续使用 Milestone 10 的固定 inference
condition。

## Effective weight provider

新增 `AdvantageWeights`，从已经过 converter/preprocessor 的 batch 读取：

- `advantage_label_global|subtask`
- `advantage_loss_weight_global|subtask`
- 可选 `advantage_condition_kept`

行为：

- positive：保留 Milestone 8 离线 group-relative weight，包括大于 1 或低于 1 的真实值。
- negative：要求离线 weight 为 `1.0`。
- condition dropped：默认回退 `1.0`；可通过配置关闭该 fallback。
- ignore：优先级最高，强制 `0.0`，不会被 dropout fallback 改回 1。
- 未使用 conditioning/mask 时：完整保留离线 positive/negative weighting，支持 weighting-only ablation。
- `[B]`、`[B, 1]` 和单样本 scalar 会统一成 `[B]`；非法 shape、batch size、NaN/Inf、负权重、非法
  label 或错误 negative weight 均明确拒绝。

provider 同时输出 mean/sum、positive/negative/ignore 数量、dropout 数和 all-ignore 标记供训练日志使用。

## FM-only loss 和 pi0/pi0.5 接口

pi0/pi0.5 原有 `forward(reduction="none")` 返回 per-sample
`FM + subtask_ce_loss_weight * CE`，旧 RA-BC 依赖这个语义。为避免破坏旧路径，本阶段没有改变
`reduction="mean|none"` 的返回值，而是增加：

```python
policy.forward(
    batch,
    reduction="none",
    return_loss_components=True,
)
```

该路径一次 forward 返回：

- `per_sample_fm_loss: [B]`
- `per_sample_subtask_ce_loss: [B]`
- logging-friendly output dict

训练循环按下面公式组合：

```text
fm_loss = sum(weight_i * fm_i) / sum(weight_i)
subtask_ce = mean(subtask_ce_i)
loss = fm_loss + subtask_ce_loss_weight * subtask_ce
```

因此 advantage weight 只改变 FM 梯度；subtask CE 始终普通 mean。`predict_subtask=false` 时 CE component
为零，行为退化为纯 weighted FM。

## 分布式 weighted mean

`distributed_weighted_mean()` 对 weight denominator 做全 rank sum。由于 DDP 会再平均各 rank 梯度，
每个 rank 的 local numerator 乘 `world_size` 后除以 global denominator，使最终平均梯度等价于对 global
batch 直接计算：

```text
sum_all_ranks(weight_i * loss_i) / sum_all_ranks(weight_i)
```

subtask CE 使用同一 helper 和全 1 weight，得到等价的全局普通 mean。

若 global weight sum 为 0（全 ignore），FM 返回与计算图连接的零值，不产生 NaN 且 FM 参数无梯度更新；
若启用了 subtask AR，未加权的 subtask CE 仍可正常更新。

## RA-BC 兼容性

- config 层和 `update_policy()` 层都拒绝同时启用 RA-BC 与 advantage weighting。
- `src/lerobot/utils/rabc.py` 未修改。
- RA-BC 继续调用旧 `reduction="none"`，仍对旧的 per-sample combined loss 使用既有公式。
- 新的 `return_loss_components=True` 只由 advantage weighting 路径使用。

## 使用示例

subtask conditioning + weighting：

```bash
lerobot-train \
  --dataset.repo_id ming326/strike_match_3_value \
  --policy.path /path/to/pi0_or_pi05_checkpoint \
  --policy.use_advantage_conditioning true \
  --policy.advantage_label_key advantage_label_subtask \
  --policy.advantage_loss_weight_key advantage_loss_weight_subtask \
  --use_advantage_weighting true \
  --advantage_label_key advantage_label_subtask \
  --advantage_loss_weight_key advantage_loss_weight_subtask \
  --advantage_condition_dropout_prob 0.1
```

只做 group-relative weighting、不拼 advantage prompt：

```bash
lerobot-train \
  --dataset.repo_id ming326/strike_match_3_value \
  --policy.path /path/to/pi0_or_pi05_checkpoint \
  --policy.use_advantage_conditioning false \
  --use_advantage_weighting true
```

## 测试覆盖

专项单元/集成测试覆盖：

- dropout 0/1、固定 generator 可复现、ignore 永不 keep、输入 batch 不原地修改。
- missing/invalid label、非法 probability、shape 和 batch size 校验。
- positive=2、negative=1 和 positive<1 的实际 weighted mean。
- dropout weight fallback、关闭 fallback、无 conditioning mask 的 weighting-only 行为。
- ignore=0 且优先于 dropout。
- NaN/Inf、负权重和错误 negative weight 拒绝。
- all-ignore FM=0、无 NaN、subtask CE 继续普通 mean。
- pi0/pi0.5 分离 FM/CE component，旧 mean/none 语义不变。
- fake differentiable policy 的参数更新验证 FM 被加权而 CE 梯度不变。
- 模拟两 rank DDP 梯度与 global weighted mean 一致。
- RA-BC 旧路径回归和双开关拒绝。
- train config 合法 global/subtask key 和错误交叉 key。

真实样例 shadow smoke：

```text
2 个真实 episode
  -> targets
  -> mock predictions
  -> global/subtask advantage
  -> global/subtask labels
  -> global/subtask weights
  -> build LeRobotDataset
  -> DataLoader/action chunk
  -> pi0/pi0.5 processor
  -> dropout mask
  -> effective weight provider
```

测试覆盖 pi0/pi0.5 × global/subtask 四种组合，并比较原始样例 `extras.parquet` bytes，确认 shadow 流程
没有修改原始 raw run。

## 一键验收和结果

验收脚本：

```bash
plans/value_pipeline/validate_milestone_11.sh
```

脚本依次执行：

1. 新增/修改源码和测试 `py_compile`。
2. Milestone 11、pi0/pi0.5 loss、processor、converter、actual-build dataset 专项回归。
3. Milestone 0/1/1.5/6/7/8/9/10 核心、UI API 和 wheel package 回归。
4. Milestone 8、9、11 真实样例 shadow smoke。
5. `git diff --check`。

2026-07-13 最终结果：

```text
Milestone 11 / policy / processor / train integration: 141 passed, 1 skipped
existing value pipeline / API / wheel regression: 132 passed
real sample Milestone 8 + 9 + 11 shadow smoke: 3 passed
total: 276 passed, 1 skipped
py_compile: passed
git diff --check: passed
```

UI/API/wheel 回归需要在 `127.0.0.1` 绑定临时测试端口；受限 sandbox 内直接运行会得到
`PermissionError: Operation not permitted`，允许本机 loopback socket 后完整验收通过。warning 仍是已有的
subtask decode limit、bool histogram 以及 Hugging Face datasets/NumPy scalar conversion deprecation，
没有新增测试失败。

## 剩余边界

- 本阶段只验证 synthetic/mock artifact 的训练接口，不允许把它用于正式 VLA 实验。
- DDP 数学和梯度缩放由两 rank 模拟测试覆盖；Milestone 12 的正式 pi0/pi0.5 2-step smoke 可再通过
  `accelerate launch` 覆盖真实多进程运行环境。
- 本阶段没有加载完整 pi0/pi0.5 checkpoint 做优化 step；真实模型 2-step smoke 属于 Milestone 12。
- 第一版只支持 canonical `ignore` label。
- 当前环境没有安装 ruff，因此没有虚构 ruff 结果；`py_compile`、pytest 和 `git diff --check` 均通过。
