# PI0 / PI0.5 历史 Memory 条件、RTC 部署闭环与终端状态面板改造计划

> 面向读者：后续负责实现、测试和真实机器人验收的 agent。
>
> 仓库：`/home/zenbot-robot/repos/lerobot`
>
> 基线：`main`，审视时 HEAD 为 `8194e710`（2026-07-14，`value_pipeline`）。
>
> 日期：2026-07-17。
>
> 本文是实施任务书，不是完成记录。除非某个 milestone 的验收项真实通过，否则不得标记为完成。

---

## 0. 文档用途、实施边界和不可破坏项

### 0.1 本文要解决的问题

当前 PI0 / PI0.5 已能针对每次 observation 自回归生成：

```text
Subtask: {current_subtask}; Progress: {current_progress}
```

本次要在其基础上增加一轮历史记忆：当前推理除了图像、主任务、状态和可选 advantage condition，还要看到上一次模型输出的完整 subtask/progress。训练时用同 episode 的历史 GT subtask/progress 模拟该输入；部署时用上一轮真实模型输出闭环。

同时整理 `lerobot-policy-deploy` 的输出：持续变化的状态、延迟、事件、subtask 和 memory 固定显示，普通连接日志、warning 和异常仍能正常追加，避免每秒长日志刷屏后看不到模型语义输出。

### 0.2 硬性范围

只要求以下组合：

- policy：`pi0`、`pi05`；
- robot：Nero，主要是 `bi_nero_follower`；
- 部署：`lerobot_policy_deploy.py` 的本地 RTC 异步推理；
- 数据：30 FPS LeRobotDataset，带逐帧 `subtask` 和 `subtask_progress`；
- RL：现有 advantage positive/negative/ignore conditioning 和 FM loss weighting；
- 当前 subtask AR：必须继续工作。

不要求：

- 其他 VLA/policy 的 memory 兼容；
- PI0-FAST、SARM、Wall-X、SmolVLA 等模型接入；
- Pico 遥操作程序的固定终端面板；
- 修改 Nero/Pico 采集控制逻辑；
- 本次替用户重建或修正 `nero_egg`；
- 多轮 memory、向量数据库、检索式 memory、跨 episode memory；
- 为 memory 新增一个生成 head 或单独的 memory loss。

其他模型不需要支持新功能，但默认关闭时不得无故破坏 import、配置加载和既有训练。

### 0.3 工作区保护

审视时已有用户改动：

```text
M  scripts/nero_teleop/README.md
M  src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py
M  src/lerobot/scripts/lerobot_push_dataset.py
?? tests/scripts/test_lerobot_push_dataset.py
```

这些改动不属于本计划。实施 agent 必须保留，不得 reset、checkout、覆盖或顺手格式化。

### 0.4 当前执行环境与可选网络代理

当前主要开发和测试环境是 Conda 环境：

```bash
conda activate lerobot-main
```

其现有解释器路径通常为：

```text
/home/zenbot-robot/.conda/envs/lerobot-main/bin/python
```

后续 agent 执行 pytest、训练 smoke、脚本验证或检查已安装依赖时，优先使用该环境；completion record 中应写出实际使用的 Python/Conda 环境，避免把系统 Python 的缺包误判为项目问题。

本机另有 `127.0.0.1:1080` VPN/代理端口。仅当 Hugging Face Hub、模型/tokenizer、数据集或 Python 依赖确实需要联网下载时，可在当前 shell 临时设置：

```bash
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080
```

注意：

- 不需要联网的源码检查、单元测试和本地 checkpoint 部署不要强制依赖代理；
- 不要把代理地址写入 policy checkpoint、processor JSON、dataset metadata 或提交到通用运行配置；
- `lerobot_policy_deploy.py` 当前默认 `hf_hub_offline=true`，本地模型和数据完整时不应访问网络；
- 若下载失败，应在记录中区分离线缓存缺失、代理未启用和真实代码错误；
- 使用代理后只影响当前任务 shell，不要修改用户全局 shell 配置。

---

## 1. 已确认的产品语义

以下决策已经由用户确认，实施时不要重新发明另一套语义。

1. Memory 保存上一轮完整的 `Subtask + Progress` 输出。
2. 训练历史帧偏移先从闭区间 `[1, 12]` 均匀抽样，再判断该帧是否存在。
3. 对 episode 开头，若 `t - k < 0`，该样本无 memory；不得先裁剪成 frame 0，也不得只从可用 offset 中重新抽样。
4. 30 FPS 下 `[1, 12]` 对应约 33–400 ms，是对实测约 70–200 多 ms 延迟的稍宽鲁棒性增强。
5. 自然无历史与 memory dropout 使用相同的模型输入语义：完全不添加 memory block。
6. `memory_dropout_prob=0.2`，与现有 `subtask_dropout_prob=0.2` 独立采样。
7. Advantage/RL 沿用当前语义：当前帧 weight 只加权 FM；当前 subtask CE 不加权；memory 不新增 loss。
8. 部署第一次推理无 memory；每次成功推理后更新；pause/reset/home/重新启动 policy 时清空。
9. `nero_egg` 只作为 raw/LeRobotDataset 格式参考，本任务不负责重建它。
10. 固定状态面板只改 VLA policy deploy，不改 Pico 遥操作程序。
11. 主 task 始终使用当前 LeRobotDataset/部署配置已有的 task；memory 只能追加条件，不能替换、改写或从 subtask 列反推主 task。

---

## 2. 当前代码实际上在做什么

后续 agent 必须以当前源码为准，不要只按旧计划文档猜测。

### 2.1 Subtask 数据和 processor

- `src/lerobot/scripts/lerobot_annotate_subtask.py`
  - 把逐帧 `subtask` 写到 raw episode 的 `extras.parquet`；
  - 按相邻同名 subtask 段计算 `subtask_progress`；
  - progress 为 float32，段内从接近 0 递增到 1。
- `src/lerobot/scripts/lerobot_build_dataset.py`
  - 把 raw `extras.parquet` 列并入 LeRobotDataset；
  - string、float、bool 和 list extras 已有 schema 映射。
- `src/lerobot/processor/converters.py`
  - 已把 `subtask`、`subtask_progress` 和 advantage 字段路由到 complementary data。
- `src/lerobot/processor/subtask_processor.py`
  - `SubtaskTextProcessorStep` 生成：

    ```text
    Subtask: {subtask}; Progress: {progress:.1f}\n
    ```

  - 缺 progress 时回退为 1.0；空 subtask 保持空。
- `src/lerobot/processor/tokenizer_processor.py`
  - 主 prompt 和 subtask target 分开 tokenize；
  - subtask 使用独立定长 token tensor，手动追加 EOS；
  - `tokenize_subtask=False` 默认关闭。

### 2.2 PI0 / PI0.5 当前 prompt

PI0：

```text
{task}\n
```

若启用 advantage conditioning，processor 先把一行 `Advantage: positive|negative` 追加到 task，再由 `Pi0NewLineProcessor` 加最终换行。

PI0.5：

```text
Task: {task}, State: {32个离散state token};\n
```

当 `predict_subtask=False` 时后面还有 `Action: `；开启 subtask AR 后会省略它。PI0.5 会把 task 内换行和下划线整理为空格。

当前默认主 prompt 上限：

- PI0：`tokenizer_max_length=48`；
- PI0.5：`tokenizer_max_length=200`。

Memory 开启时，PI0 的 48 明显偏紧。本计划要求 memory 模式使用至少 128；PI0.5 不得因为统一配置而从 200 反向缩短。

### 2.3 当前模型 attention 和 loss

`modeling_pi0.py` 与 `modeling_pi05.py` 已实现：

```text
[image + main prompt（双向 prefix）]
[current subtask target（逐 token causal）]
[state/action suffix（能看见有效 prefix 和 current subtask）]
```

训练 loss：

```text
loss = FM + subtask_ce_loss_weight * current_subtask_CE
```

现有 `subtask_dropout_prob` 不是删除 subtask CE，而是在 attention mask 上阻止 action suffix 看见 current subtask。Current subtask 自身仍做 CE。

这意味着把 memory 直接加入主 prompt 后：

- current subtask AR 会看到 memory；
- action FM 会看到 memory；
- 不需要改 PaliGemma head、KV cache 结构或 flow matching 公式；
- memory dropout 应在 tokenize 前省略整个文本块，从而同时影响 current subtask AR 和 FM。

### 2.4 当前 advantage/RL

`AdvantageConditionProcessorStep` 把 positive/negative condition 加到主 task；`ignore` 或 dropout 时不加。

`AdvantageWeights` 和训练循环当前执行：

```text
weighted_fm = sum(current_weight_i * fm_i) / sum(current_weight_i)
current_subtask_ce = mean(ce_i)
loss = weighted_fm + subtask_ce_loss_weight * current_subtask_ce
```

Negative 离线 weight 必须为 1；ignore 强制为 0；advantage condition 被 dropout 时默认把 effective FM weight 回退为 1。Memory 不得改变这些规则。

### 2.5 当前 RTC 部署

`RTCInferenceEngine` 在独立线程中完成：

```text
latest observation
  -> build_dataset_frame
  -> prepare_observation_for_inference
  -> preprocessor
  -> policy.predict_action_chunk
  -> postprocessor
  -> ActionQueue.merge
```

PI0 / PI0.5 policy 在 `predict_action_chunk()` 中把生成 token 解码到公开属性 `last_subtask_text`。

因此部署 memory 不能只在 `lerobot_policy_deploy.py` 主控制循环外层拼 prompt。读上一轮输出、注入下一轮输入、处理 reset version 和提交成功条件都必须位于 RTC engine 的推理事务中。

### 2.6 当前 deploy 输出

`lerobot_policy_deploy.py` 每秒输出一条很长的 `deploy_loop ...` 日志，包含 loop、sleep、obs、fetch、send、RTC latency、queue、phase 和 timing。Keyboard、homing、相机连接、subtask 等由其他线程继续输出普通日志，因而屏幕持续滚动。

固定面板必须和 Python logging 协作，而不是简单 `print("\r...")`；否则来自 RTC、相机和机器人线程的日志会破坏显示。

### 2.7 `nero_egg` 只读参考事实

路径：

```text
/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/nero_egg
/home/zenbot-robot/.cache/huggingface/lerobot/ming326/nero_egg
```

审视结果：

- raw：61 episodes、350010 frames、30 FPS；
- raw `extras.parquet`：`subtask: string`、`subtask_progress: float32`；
- 12 个 subtask，当前所有 raw 帧均有标注；
- 现存 converted LeRobotDataset 是标注前版本，不含上述两列；
- raw 中主 task 文本与 egg subtask 内容看起来不一致，但用户明确要求本次只参考格式。

因此所有真实训练前置条件仍是“用户另外生成一个包含 subtask/progress 的 LeRobotDataset”。本计划只能做早失败检查，不能偷偷修改或重建该数据，也不得借 memory 改造纠正其主 task。训练继续读取 LeRobotDataset 中已有 task，部署继续使用明确配置的同一 task。

---

## 3. 目标端到端行为模拟

### 3.1 训练样本

当前样本位于 episode 内 frame `t=100`：

```text
current subtask = "Pour in the beaten eggs and put the bowl back."
current progress = 0.4
```

Dataset wrapper 先抽到 `k=7`，然后读取同 episode frame 93：

```text
memory subtask = "Stir the beaten eggs."
memory progress = 0.9
memory valid = true
```

若 memory dropout 未命中，模型主 prompt 中新增：

```text
Memory: Subtask: Stir the beaten eggs.; Progress: 0.9
```

随后现有 current subtask target 仍为独立 causal 段：

```text
Subtask: Pour in the beaten eggs and put the bowl back.; Progress: 0.4
```

若 memory dropout 命中，主 prompt 完全不出现 `Memory:`，但 current subtask target 和它的 CE 不变。

### 3.2 Episode 开头

当前 `t=4`，先抽到 `k=9`。因为 `4-9<0`：

- 不重抽；
- 不把索引 clamp 到 0；
- 不复制 frame 0；
- 返回 `memory_valid=false`；
- prompt 中没有 `Memory:`。

抽样顺序必须通过测试锁定，避免实现成“只从可用 `[1, min(12,t)]` 中抽”。

### 3.3 Dropout 组合

Memory dropout 与现有 current-subtask-to-FM dropout 独立，action 训练会覆盖四种输入：

| memory keep | current subtask 对 FM keep | Current subtask AR 看见 | Action FM 看见 |
|---|---|---|---|
| true | true | history memory | history memory + current subtask |
| true | false | history memory | history memory |
| false | true | no memory | current subtask |
| false | false | no memory | neither |

注意：第二个 dropout 不影响 current subtask CE；第一个 dropout 会影响 current subtask CE 的条件输入，因为 memory 属于主 prompt。

### 3.4 RTC 部署前两轮

第一轮：

```text
engine memory = empty
prompt has no Memory line
model output = "Subtask: Pick up the fork.; Progress: 0.2"
postprocess + queue merge success
engine commits output as next memory
```

第二轮：

```text
prompt += "Memory: Subtask: Pick up the fork.; Progress: 0.2"
model output = "Subtask: Pick up the fork.; Progress: 0.4"
success -> commit as next memory
```

若第二轮推理、postprocess 或 merge 失败，不得把半完成结果提交为第三轮 memory。

---

## 4. 最终数据契约

### 4.1 原始 LeRobotDataset 必需字段

启用 memory training 时要求：

```text
subtask: string
subtask_progress: float32 scalar/shape(1)
frame_index: int64
episode_index: int64
index: int64
```

subtask 的数量和具体文本完全由数据集决定，代码中不得硬编码 `nero_egg` 的 12 类、类别顺序或名称。

### 4.2 Dataset wrapper 生成字段

建议新增 `MemoryHistoryDataset`，包装现有非 streaming `LeRobotDataset`，每次 `__getitem__` 增加：

```text
memory_subtask: string
memory_subtask_progress: float32
memory_valid: bool
memory_frame_offset: int64       # 仅诊断/测试，表示本次先抽到的 k
```

要求：

- 当前图像/action 仍由原 dataset 正常读取；
- 历史帧只调用 raw row 读取，不重复解码历史视频；
- 历史索引必须留在同 episode；
- selected episodes 下也不能跨边界；
- wrapper 代理 `meta`、`episodes`、`num_frames`、`num_episodes` 等训练脚本依赖属性；
- memory 关闭时不创建 wrapper，保证无额外读取和 RNG 消耗；
- 第一版明确拒绝 `dataset.streaming=true`，不要做一个表面兼容但会跨 shard 出错的实现。

### 4.3 抽样算法

伪代码必须遵守：

```python
k = randint_inclusive(1, 12)  # 第一步始终抽样
if current_frame_index - k < 0:
    return no_memory(offset=k)

history = raw_item_at_current_relative_index_minus_k
assert history.episode_index == current.episode_index
if history.subtask is empty or progress invalid:
    return no_memory(offset=k)
return valid_memory(history, offset=k)
```

不要在 raw 数据或已构建 LeRobotDataset 中永久写 `memory_*` 列；动态抽样应随 epoch/访问变化。

### 4.4 DataLoader RNG

使用 PyTorch worker RNG，使 `num_workers>0` 时每个 worker 不产生完全相同的 offset 序列。测试至少覆盖：

- 固定 seed + 相同 worker 配置可重复；
- 连续访问允许得到不同 offset；
- offset 始终在 1–12；
- 开头样本仍先消耗一次随机抽样。

不要求在改变 worker 数量后保持逐样本完全相同的 offset 序列。

---

## 5. Prompt 和 processor 设计

### 5.1 采用主 prompt 文本块，不新增模型 token 类型

Memory 直接追加到主 task 文本：

```text
Memory: {上一轮完整subtask/progress输出}
```

训练 canonical 示例：

```text
Memory: Subtask: Pick up the fork.; Progress: 0.6
```

部署时：

- 使用 `policy.last_subtask_text` 的完整非空字符串；
- 只做首尾 trim 和空白归一化，不能解析后重新拼出另一个语义；
- 不要求生成结果一定属于某个固定 subtask 列表；
- 空生成结果视为 no memory。

这个设计刻意不增加 `OBS_LANGUAGE_MEMORY_TOKENS`，也不修改 PI 模型的 `embed_prefix()`。原因是 memory 本质是 prompt condition，当前模型已经让 subtask AR 和 action suffix 看到主 prompt；复用它能显著缩小模型侧改动面。

### 5.2 新 processor

建议新增可注册、可序列化的 `MemoryConditionProcessorStep`：

输入支持两种来源：

1. 训练：`memory_subtask + memory_subtask_progress + memory_valid`；
2. 部署：`memory_text + memory_valid`，其中 text 是上一轮完整生成结果。

共同控制字段：

```text
memory_condition_kept
```

行为：

- valid 且 kept 时，向 `task` 末尾追加一行 canonical memory；
- invalid、空 text 或 kept=false 时，task 逐字符保持原值；
- processor 不采随机数；
- processor 不读取 advantage weight；
- progress round/clamp 规则复用 current subtask formatter：一位小数、范围 `[0,1]`；
- batch string/list、scalar、`[B]`、`[B,1]` 都要规范化并校验 batch size；
- 非法 dtype、NaN/Inf、长度不匹配明确报错；
- inference 缺所有 memory 字段时静默走 no-memory，不得用伪默认文本。

优先抽出共享 `format_subtask_output(subtask, progress)`，让 current target 和 training memory 不出现格式漂移。

### 5.3 Processor 顺序

PI0：

```text
AddBatch
-> AdvantageCondition(optional)
-> MemoryCondition(optional)
-> Pi0NewLine
-> Tokenizer
-> Device
-> RelativeAction
-> Normalize
```

PI0.5：

```text
Rename/AddBatch/Relative/Normalize
-> AdvantageCondition(optional)
-> MemoryCondition(optional)
-> Pi05PrepareStatePrompt
-> SubtaskText(optional current target)
-> Tokenizer
-> Device
```

因此 PI0.5 最终 token 顺序仍是：主 task、advantage、memory、state，然后 current subtask AR。`Pi05PrepareStateTokenizerProcessorStep` 会把 task 内换行整理为空格，但 `Memory:` 标签和内容必须保留。

### 5.4 Prompt 长度

PI0 / PI0.5 policy config 都增加：

```python
use_memory_conditioning: bool = False
memory_tokenizer_max_length: int = 128
```

构建 tokenizer step 时：

```text
memory disabled: max_length = existing tokenizer_max_length
memory enabled:  max_length = max(existing tokenizer_max_length, memory_tokenizer_max_length)
```

结果：

- PI0 memory 模式：48 -> 128；
- PI0.5 memory 模式：保持 max(200,128)=200；
- 默认关闭：shape 和数值行为不变。

Advantage 或 memory 任一开启时使用 left truncation，优先保留 prompt 尾部的 condition/state。必须用长 task 测试确认最终 token 中仍有 `Memory:`、历史 subtask/progress，PI0.5 中 state 尾段也未被静默截掉。

### 5.5 Config 约束

- `use_memory_conditioning=true` 仅允许 PI0/PI0.5；
- 要求 `predict_subtask=true`，因为部署 memory 来源就是 subtask AR 输出；
- `memory_tokenizer_max_length>0`；
- `subtask_generate_at_inference=false` 可作为 no-memory 部署 ablation，但应 warning：memory 永远无法更新；
- 所有新增字段默认关闭或安全默认，旧 checkpoint 可加载。

---

## 6. 训练期 history 和 dropout 接线

### 6.1 Train config

`TrainPipelineConfig` 增加：

```python
memory_lookback_min_frames: int = 1
memory_lookback_max_frames: int = 12
memory_dropout_prob: float = 0.2
```

校验：

- `1 <= min <= max`；
- dropout 在 `[0,1]`；
- memory 开启时 dataset 必须非 streaming；
- dataset metadata 必须包含 `subtask` 和 `subtask_progress`；
- policy config 和 train config 组合错误要在创建长训练任务前失败。

### 6.2 Dropout helper

建议新增 `sample_memory_condition_mask()`，在 preprocessor 前执行：

```text
eligible = memory_valid AND non_empty(memory source)
kept = eligible AND Bernoulli(1 - memory_dropout_prob)
batch["memory_condition_kept"] = kept  # strict [B] bool
```

要求：

- 不原地修改输入 batch；
- `p=0` 保留所有 eligible；
- `p=1` 删除所有 memory；
- natural invalid 始终 false；
- 使用 PyTorch RNG；
- 和 advantage condition mask 分别调用、分别采样；
- 不因 memory dropout 改 advantage label、weight 或 condition keep mask。

### 6.3 训练循环顺序

```text
raw DataLoader batch
-> sample_advantage_condition_mask(optional)
-> sample_memory_condition_mask(optional)
-> preprocessor
-> policy.forward
-> existing advantage/RABC/plain loss path
```

Memory mode不得启用 RA-BC 与 advantage weighting 双开；沿用现有互斥校验。

### 6.4 Processor 重建陷阱

当前训练从 `--policy.path` 启动时通常会加载 checkpoint 内保存的旧 preprocessor。只用 CLI 把 `use_memory_conditioning=true` 覆盖到 policy config，并不会自动把新 Memory step 插入旧 pipeline。

必须在 `lerobot_train.py` 中加入结构变更处理：

- 非 resume 且启用 memory 时，从当前 policy config 重新构建 pre/post processors，并使用当前 dataset stats；
- resume 时加载 checkpoint 已保存的 memory processor，保证严格续训；
- 输出明确日志说明是“因 memory structural config 重建 processor”；
- checkpoint 保存后，processor JSON 中必须包含 Memory step 和 effective tokenizer length；
- 部署加载该 checkpoint 时不得依赖用户再次手写 processor override。

这是本任务的必测项，不可只在直接调用 `make_pi*_pre_post_processors()` 的单测中通过。

### 6.5 训练日志

至少记录：

```text
memory/history_valid_fraction
memory/condition_kept_fraction
memory/dropout_fraction_among_valid
memory/lookback_frames_mean
memory/lookback_frames_min_seen
memory/lookback_frames_max_seen
```

日志只用于诊断，不进入 loss。

---

## 7. Model、subtask 和 loss 的兼容策略

### 7.1 预期模型源码改动很小

由于 memory 已进入主 prompt，`PI0Pytorch.forward/sample_actions` 和 `PI05Pytorch.forward/sample_actions` 不应新增 memory tensor 参数，也不应复制一套 KV-cache 生成逻辑。

模型文件主要需要：

- 确认扩大后的 main prompt mask/position ids 正常；
- 保留 current subtask AR causal attention；
- 保留现有 current subtask attention dropout；
- `reset()` 时清空 `last_subtask_text`、last logged text 和临时生成 token，避免 dashboard 显示 stale 输出；
- 将高频/重复 subtask info 输出交给部署状态面板；普通非 dashboard 场景至少保留 debug 可观测性。

若实现 agent 发现必须修改 attention，需先用具体 failing test 证明；不要仅为了“memory 是新功能”而增加模型分支。

### 7.2 Loss 不变

Plain training：

```text
loss = mean(FM) + subtask_ce_loss_weight * mean(current_subtask_CE)
```

Advantage weighting：

```text
loss = weighted_mean(current_frame_weight, FM)
     + subtask_ce_loss_weight * mean(current_subtask_CE)
```

Memory：

- 无 target；
- 无 CE；
- 无独立 weight；
- 不读取历史帧 advantage label/weight；
- 历史帧只提供 subtask/progress 文本。

### 7.3 Positive / negative / ignore

- positive：使用该当前帧原 weight；history 可来自同一轨迹的历史帧；
- negative：仍要求当前帧离线 weight=1；不因为 memory 存在而变 positive；
- ignore：FM weight=0；若 current subtask AR 开启，CE 仍可训练，并按 memory keep 状态决定是否看到 history；
- advantage condition dropped：按现有规则把 effective FM weight回退到 1；memory keep/drop 不参与该判断；
- memory dropout：不得把 current sample weight 改成 1 或 0。

### 7.4 默认关闭回归

`use_memory_conditioning=false` 时必须满足：

- dataset 未包装；
- 不抽 history RNG；
- batch 无 `memory_*`；
- pipeline 无 Memory step；
- main prompt max length仍为 PI0=48、PI0.5=200；
- prompt token/mask 与当前基线相同；
- FM/CE/advantage loss 相同；
- 旧 checkpoint 和旧 processor 正常加载。

---

## 8. RTC 部署 memory 状态机

### 8.1 状态归属

Memory 真值放在 `RTCInferenceEngine`，而不是：

- deploy 主控制循环的局部字符串；
- processor 的隐藏 mutable state；
- policy model 的 KV cache；
- 全局变量。

建议 engine 字段：

```text
_memory_text_for_next_inference
_last_memory_input_text
_last_subtask_output_text
_memory_source_inference_id
```

这些字段通过现有 state/inference lock 保护，并纳入 `debug_snapshot()`。

### 8.2 一次推理事务

在 `_rtc_loop` 中按以下顺序：

1. 记录当前 `reset_version`。
2. 在 lock 下 snapshot `memory_text_for_next_inference`。
3. build/prepare 当前 observation。
4. 若 policy memory 开启且 snapshot 非空，加入：

   ```text
   memory_text=[snapshot]
   memory_valid=[true]
   memory_condition_kept=[true]
   ```

   第一次或 reset 后则显式 false/空值，processor 不添加 block。
5. 执行 preprocessor。
6. 执行 `policy.predict_action_chunk()`。
7. 从 `policy.last_subtask_text` 取得 candidate；只 trim/normalize whitespace。
8. 执行 postprocessor。
9. 再检查 shutdown、active 和 `reset_version`。
10. `ActionQueue.merge()` 成功。
11. 最后原子提交：
    - `last_memory_input_text = 本轮snapshot`；
    - `last_subtask_output_text = candidate`；
    - `memory_text_for_next_inference = candidate或empty`；
    - inference count/source id 更新。

任何 6–10 步异常都不得更新 next memory。

### 8.3 Reset 语义

以下路径都已调用或必须调用 engine reset：

- 初次 prepare/start；
- pause；
- home；
- 从 home 回到 policy 前再次 prepare；
- 显式 engine reset；
- policy/environment episode reset。

Reset 必须同时清：

- action queue、RTC processor state；
- pre/post processor state；
- engine memory；
- policy `last_subtask_text`；
- dashboard subtask/memory 显示。

### 8.4 推理关闭和异常输出

- `use_memory_conditioning=false`：engine 不注入字段、不维护 memory；
- `subtask_generate_at_inference=false`：输出 warning，运行成永远 no-memory 的 ablation；
- candidate 空：下一轮 no-memory；
- candidate 非 canonical 但非空：完整作为文本使用，不按固定类别解析；
- deployment batch 不是 1 时明确拒绝或逐样本设计，第一版不得悄悄只取第 0 个。

---

## 9. `lerobot-policy-deploy` 固定状态面板

### 9.1 目标显示

交互式 TTY 固定显示类似：

```text
[STATE]    running       [EVENT] right/start @ 14:32:10
[LATENCY]  total=104.2ms delay=4f queue=31 phase=between_inferences
[TIMING]   build=0.3 prep=1.2 preprocess=2.4 predict=98.7 post=0.6 merge=0.2ms
[SUBTASK]  Subtask: Pick up the fork.; Progress: 0.4
[MEMORY]   Subtask: Pick up the fork.; Progress: 0.2
```

字段值原地更新，label 和行位置稳定。

相机连接、机器人连接、配置、warning、slow-loop、异常等普通日志：

- 先临时清除 footer；
- 在 footer 上方输出完整日志；
- 重新绘制 footer；
- 不吞日志，不把 exception traceback压成一行。

### 9.2 实现建议

不新增 Rich 依赖。新增一个线程安全、TTY-aware 的 console handler/display：

- 复用 `init_logging()` 已安装 console handler 的 formatter 和 level；
- 文件 handler（若存在）不受影响；
- logging write 和 footer redraw 共用 lock；
- 支持多行 log 和终端宽度截断；
- shutdown/finally 恢复 cursor、换行和 terminal 状态；
- keyboard listener 的 cbreak 模式不得被破坏。

Deploy config 建议：

```python
status_display: Literal["auto", "live", "plain"] = "auto"
status_refresh_hz: float = 4.0
```

- `auto`：stderr 是 TTY 且 TERM 可用时 live，否则 plain；
- `live`：显式启用 ANSI；
- `plain`：无 ANSI，每秒一条紧凑带 label 状态日志，适合重定向和日志采集。

不要在全仓库全局替换 logging；该 handler 只在 `lerobot_policy_deploy` 生命周期内安装和卸载。

### 9.3 状态更新来源

- `STATE`：deploy state machine；
- `EVENT`：主循环实际消费的 keyboard event，而不是 listener 收到但后来被覆盖的旧事件；
- `LATENCY/TIMING`：`engine.debug_snapshot()`；
- `SUBTASK`：最近一次成功提交的 current output；
- `MEMORY`：最近一次成功推理实际使用的 memory input；第一次显示 `<none>`。

现有每秒超长 `logger.info("deploy_loop ...")` 在 live 模式改为 status update；slow-loop warning、fatal error 仍是普通日志。PI0/PI0.5 内部的 subtask change info 不应再与固定行重复刷屏，可降为 debug，真实值由 snapshot 显示。

### 9.4 非目标

- 不做 curses 全屏 UI；
- 不接管 Pico teleop 输出；
- 不隐藏相机连接或机器人初始化日志；
- 不把所有 debug 指标删掉；
- 不让 ANSI escape 写进非 TTY 日志文件。

---

## 10. 必读代码与预计修改文件

行号会变化，agent 应搜索类/函数名。

### 10.1 开始实现前必须完整阅读

| 类别 | 文件 | 重点符号 |
|---|---|---|
| 既有设计 | `plans/pi0_pi05_subtask_generation_plan.md` | subtask AR prompt、attention、dropout |
| 既有 RL | `plans/value_pipeline/milestone_10_advantage_conditioning_completed.md` | advantage prompt |
| 既有 RL | `plans/value_pipeline/milestone_11_advantage_weighting_completed.md` | FM-only weighting |
| policy config | `src/lerobot/policies/pi0/configuration_pi0.py` | tokenizer、subtask、advantage config |
| policy config | `src/lerobot/policies/pi05/configuration_pi05.py` | 同上 |
| prompt pipeline | `src/lerobot/policies/pi0/processor_pi0.py` | `make_pi0_pre_post_processors` |
| prompt pipeline | `src/lerobot/policies/pi05/processor_pi05.py` | state prompt 和 pipeline 顺序 |
| subtask format | `src/lerobot/processor/subtask_processor.py` | current target canonical text |
| tokenization | `src/lerobot/processor/tokenizer_processor.py` | main prompt max/truncation、subtask target |
| advantage prompt | `src/lerobot/processor/advantage_processor.py` | deterministic condition pattern |
| batch routing | `src/lerobot/processor/converters.py` | complementary data |
| dataset creation | `src/lerobot/datasets/factory.py` | `make_dataset` |
| dataset access | `src/lerobot/datasets/lerobot_dataset.py` | `__getitem__`、`get_raw_item` |
| episode bounds | `src/lerobot/datasets/dataset_reader.py` | relative/absolute index、padding |
| train config | `src/lerobot/configs/train.py` | validation |
| train loop | `src/lerobot/scripts/lerobot_train.py` | processor load/rebuild、dropout、loss |
| RL helper | `src/lerobot/utils/advantage_weights.py` | dropout和effective weight语义 |
| model | `src/lerobot/policies/pi0/modeling_pi0.py` | subtask decode、reset、loss components |
| model | `src/lerobot/policies/pi05/modeling_pi05.py` | 同上 |
| RTC | `src/lerobot/inference_engines/rtc.py` | inference事务、reset、snapshot |
| deploy | `src/lerobot/scripts/lerobot_policy_deploy.py` | state machine、keyboard、每秒 timing |
| logging | `src/lerobot/utils/utils.py` | `init_logging` handler/formatter |

### 10.2 预计新增文件

命名可小幅调整，但职责不能混乱：

```text
src/lerobot/datasets/memory_history.py
src/lerobot/processor/memory_processor.py
src/lerobot/utils/memory_conditioning.py
src/lerobot/utils/terminal_status.py

tests/datasets/test_memory_history.py
tests/processor/test_memory_processor.py
tests/utils/test_memory_conditioning.py
tests/inference_engines/test_rtc_memory.py
tests/scripts/test_lerobot_policy_deploy_status.py
```

### 10.3 预计修改文件

```text
src/lerobot/configs/train.py
src/lerobot/datasets/factory.py
src/lerobot/processor/__init__.py
src/lerobot/processor/converters.py
src/lerobot/policies/pi0/configuration_pi0.py
src/lerobot/policies/pi0/processor_pi0.py
src/lerobot/policies/pi0/modeling_pi0.py          # 仅 reset/输出状态，避免大改 forward
src/lerobot/policies/pi05/configuration_pi05.py
src/lerobot/policies/pi05/processor_pi05.py
src/lerobot/policies/pi05/modeling_pi05.py        # 同上
src/lerobot/scripts/lerobot_train.py
src/lerobot/inference_engines/rtc.py
src/lerobot/scripts/lerobot_policy_deploy.py
```

若实际实现需要改 `lerobot_build_dataset.py`，应先证明通用 subtask/progress 数据集无法满足契约；不要为动态 memory 向离线数据永久增加派生列。

---

## 11. 分阶段任务、顺序和验收标准

必须按依赖顺序推进。每个 milestone 完成后写对应 completion record，记录实际改动、命令、pass 数和未运行项。

### Milestone 0：契约测试和默认关闭基线

目标：先把本文关键语义写成失败测试，防止实现中悄悄改变。

任务：

- 固化 PI0/PI0.5 memory disabled prompt/token golden；
- 固化现有 subtask dropout 和 advantage loss component 行为；
- 为 config 默认值和非法组合写测试；
- 记录当前专项测试结果，不改生产逻辑。

验收：

- memory disabled 的 PI0/PI0.5 processor tensor shape 和 prompt 与当前 HEAD 一致；
- 现有 subtask/advantage专项测试全过；
- 新测试明确表达 `[1,12]`、先抽后判、无 memory 与 dropout 同语义；
- `git diff --check` 通过。

### Milestone 1：动态历史 Dataset wrapper

目标：每次读取当前样本时，低成本返回同 episode 的历史 subtask/progress。

任务：

- 实现 `MemoryHistoryDataset`；
- 在 `make_dataset(cfg)` 中仅当 PI0/PI0.5 memory 开启时包装；
- early validation 必需字段和 non-streaming；
- 保持训练脚本依赖属性代理；
- 增加 offset/valid 诊断字段。

验收测试：

- `t=0,k=1`、`t=4,k=9` 均无 memory，offset 保留抽样值；
- `t=12,k=12` 正确取 frame 0；
- episode B 的前几帧绝不读 episode A；
- 随机 offset 只在 1–12；
- 历史读取不触发第二次视频 decode；
- memory disabled 时底层 dataset访问次数和输出 key 不变；
- 缺 subtask/progress 或 streaming 给出可执行的错误提示。

### Milestone 2：Memory processor、格式和 token budget

目标：训练历史 GT 与部署历史 prediction 生成相同主 prompt block。

任务：

- 抽共享 subtask/progress formatter；
- 实现、注册、导出 `MemoryConditionProcessorStep`；
- converters 路由全部 memory 字段；
- PI0/PI0.5 config 增加开关和 128 token budget；
- 按 §5.3 接入两个 processor pipeline；
- memory 模式设置 effective main prompt length 和 left truncation。

验收测试：

- 训练字段生成精确 canonical 文本；
- inference `memory_text` 保留完整 subtask/progress；
- invalid/empty/dropped 时原 task 不变；
- PI0 effective length=128；PI0.5 不低于200；
- 长 task 中最终 token 仍包含 Memory、历史 subtask、progress 和 PI0.5 state 尾部；
- advantage + memory 顺序正确；
- current subtask target token 不受 formatter 串线影响；
- processor registry save/reload 后输出相同；
- memory disabled golden 继续通过。

### Milestone 3：训练 dropout、processor 重建和日志

目标：把动态 history 稳定接入真实 `lerobot_train`，并保证从旧 checkpoint 开启 memory 时 pipeline 真正更新。

任务：

- 实现 `sample_memory_condition_mask()`；
- 加 Train config 和 validation；
- 在 preprocessor 前接入独立 dropout；
- 非 resume 的结构变更强制按当前 policy config 重建 processor；
- resume 继续加载保存的 memory processor；
- 增加 memory train metrics。

验收测试：

- dropout 0/1、固定 RNG、natural invalid、输入不原地修改；
- advantage keep 与 memory keep 的四种组合均可构造；
- 从不含 Memory step 的旧 checkpoint + CLI开启 memory，最终 pipeline 确实含 Memory step；
- checkpoint save/reload、resume 后 step 和 processor config 保持；
- batch=1、最后一个小 batch、num_workers=0/2 均通过；
- 两个 policy 各完成 fake/small 2-step update；
- memory metrics 数值与人工 batch 对得上。

### Milestone 4：PI0/PI0.5 模型与 attention 回归

目标：证明不新增 memory 模型分支也能让 AR/FM 正确看到 prompt memory。

任务：

- 用 hook 或 fake backbone 检查 main prompt token 确实进入 prefix；
- 验证 current subtask causal mask不变；
- 验证 existing subtask dropout不变；
- 补 policy reset 清理语义状态；
- 整理 subtask change logging，避免与 dashboard 重复。

验收测试：

- PI0/PI0.5 current subtask CE 在 memory keep/drop 两种输入都可反传；
- FM 在四种 dropout 组合都 finite；
- memory dropout 不改变 target/CE mask；
- subtask dropout 不删除 memory；
- reset 后 `last_subtask_text` 和临时 token为空；
- `predict_subtask=false/use_memory=false` 旧 forward/inference 回归通过；
- 不新增无必要的 model state dict keys，旧 checkpoint 权重加载无额外 missing keys。

### Milestone 5：Advantage/RL 兼容集成

目标：明确证明 memory 没有污染 current-frame weight 和现有 loss 分解。

任务：

- 扩展 advantage weighted train tests；
- 覆盖 global/subtask label key；
- 覆盖 positive/negative/ignore、advantage dropout、memory dropout；
- 保持 RA-BC 互斥。

验收测试：

- positive weight=2、negative=1、ignore=0 的 weighted FM 与手算一致；
- 历史帧即使有别的 label/weight也不被读取；
- memory keep/drop 下 current subtask CE 数学上都是未加权 mean；
- all-ignore batch：FM安全为0，current subtask CE仍训练；
- advantage condition dropped 的 weight fallback不受 memory keep影响；
- RA-BC + advantage weighting仍明确拒绝；
- PI0/PI0.5 × global/subtask 四种组合专项通过。

### Milestone 6：RTC memory 事务闭环

目标：上一轮成功输出成为下一轮输入，且不存在跨 reset 或失败提交。

任务：

- engine 增加受锁保护的 memory state；
- preprocessor 前注入 memory；
- queue merge 成功后提交 next memory；
- reset 清理；
- snapshot 暴露 last input/current output；
- PI0/PI0.5 共用，不复制两套 engine。

验收测试：

- 第一次 inference 无 memory；
- 输出 A 后第二次 prompt含完整 A；输出 B 后第三次含 B；
- predict、postprocess、merge 三种失败分别不提交 candidate；
- reset version 在推理中变化时不提交；
- pause/home/restart 后下一轮无 memory；
- empty output 使下一轮无 memory；
- memory disabled 时 engine输入与基线相同；
- debug snapshot 在并发读取下无 torn state；
- PI0/PI0.5 fake RTC tests均通过。

### Milestone 7：固定终端状态面板

目标：交互部署不刷延迟长日志，同时保留所有普通日志。

任务：

- 实现 TTY-aware status handler；
- 将 deploy state/event/RTC/subtask/memory接入；
- live/plain/auto config；
- 替换每秒超长 info；
- finally 可靠恢复终端。

验收测试：

- fake TTY 中 label 行固定，连续 update 不增长行数；
- 普通日志出现时 footer清除、日志输出、footer重绘；
- exception多行 traceback不破坏 footer；
- 非TTY输出零 ANSI escape；
- plain模式每秒紧凑日志仍含 state/latency/subtask/memory；
- 多线程 log/update 压测无交错半行和 deadlock；
- keyboard event展示消费到的最新事件；
- shutdown、Ctrl-C、fatal engine error 后 cursor/换行正常；
- 相机连接和 homing complete 等日志仍可见。

### Milestone 8：端到端 smoke 和真实 Nero 部署

目标：从真实格式数据到 checkpoint 再到 Nero RTC 完整验收。

前置：用户提供已经重新构建、实际包含 `subtask/subtask_progress` 的 LeRobotDataset。不能把当前 stale `nero_egg` converted dataset当作已满足前置。

任务：

- 从数据集中抽样打印 current/history/prompt，人工核对；
- PI0、PI0.5 各做小数据 overfit/smoke；
- 运行 advantage on/off 小矩阵；
- 加载保存 checkpoint 本地 RTC fake robot；
- 最后 Nero 实机短时运行。

验收：

- DataLoader 多轮能看到 1–12 多种 offset 和预期开头 no-memory；
- 训练 loss finite，FM/CE/memory metrics均有记录；
- checkpoint 单独目录可加载，不依赖源码内临时对象；
- deploy 第一次 MEMORY=`<none>`，随后 MEMORY严格等于上一轮 SUBTASK；
- pause/home/restart 清空；
- 70–200+ ms 延迟下 RTC queue/action正常，无新增卡顿或死锁；
- fixed dashboard 中状态可读，相机/机器人日志未丢；
- memory off checkpoint作为 A/B 基线可以同脚本运行。

---

## 12. 推荐命令形态

具体 checkpoint/dataset 路径由执行环境替换。

### 12.1 PI0 memory training

```bash
lerobot-train \
  --dataset.repo_id <repo_with_subtask_progress> \
  --policy.path <pi0_subtask_checkpoint> \
  --policy.predict_subtask true \
  --policy.use_memory_conditioning true \
  --policy.memory_tokenizer_max_length 128 \
  --policy.subtask_dropout_prob 0.2 \
  --memory_lookback_min_frames 1 \
  --memory_lookback_max_frames 12 \
  --memory_dropout_prob 0.2
```

### 12.2 PI0.5 memory + advantage

```bash
lerobot-train \
  --dataset.repo_id <repo_with_subtask_progress_and_advantage> \
  --policy.path <pi05_subtask_checkpoint> \
  --policy.predict_subtask true \
  --policy.use_memory_conditioning true \
  --policy.use_advantage_conditioning true \
  --policy.advantage_label_key advantage_label_subtask \
  --policy.advantage_loss_weight_key advantage_loss_weight_subtask \
  --use_advantage_weighting true \
  --advantage_label_key advantage_label_subtask \
  --advantage_loss_weight_key advantage_loss_weight_subtask \
  --advantage_condition_dropout_prob 0.1 \
  --memory_lookback_min_frames 1 \
  --memory_lookback_max_frames 12 \
  --memory_dropout_prob 0.2
```

### 12.3 RTC deploy

Checkpoint 已保存 policy config 和 memory processor 后，正常只需：

```bash
lerobot-policy-deploy \
  --robot.type bi_nero_follower \
  --policy.path <memory_checkpoint>/pretrained_model \
  --dataset.repo_id <matching_dataset> \
  --status_display auto
```

No-memory ablation：

```bash
lerobot-policy-deploy \
  ... \
  --policy.subtask_generate_at_inference false
```

该 ablation 应 warning，并保持每轮 MEMORY=`<none>`。

---

## 13. 最小实验矩阵

在正式长训前至少跑：

| Policy | Memory | Advantage condition/weight | 目的 |
|---|---:|---:|---|
| PI0 | off | off | 默认关闭回归 |
| PI0 | on | off | memory 主路径 |
| PI0 | on | subtask | RL兼容 |
| PI0.5 | off | off | 默认关闭回归 |
| PI0.5 | on | off | memory 主路径 |
| PI0.5 | on | subtask | RL兼容 |

每个 memory-on smoke 至少检查 dropout `(0,0)`、`(1,0)`、`(0,1)`、默认 `(0.2,0.2)`；其中 tuple 顺序为 `(memory_dropout, subtask_dropout)`。

---

## 14. 风险和必须主动检查的陷阱

### 14.1 Prompt 截断

PI0 从48增到128仍可能遇到超长 task/subtask。不能只检查 tokenize 前字符串，必须检查最终 token 解码或 token子序列。PI0.5 要同时保护 state 尾部。

### 14.2 训练/部署格式漂移

训练来自 GT 字段，部署来自完整生成文本。若两边一个写 `Previous Subtask`、另一个写 `Subtask`，会制造无意义 domain gap。必须共享 outer Memory formatter，inner deployment text不重新解释。

### 14.3 Exposure bias

训练 memory 是 GT，部署是模型 prediction，天然存在噪声差异。第一版用 0.2 memory dropout和更宽 1–12 延迟增强处理，不在本计划中增加离线模型预测回灌。若真实实验仍不稳定，再单独设计 memory corruption/self-generated memory，不要暗中塞入本 milestone。

### 14.4 跨 episode 泄漏

不能只用 global `index-k`；必须先用 episode-local `frame_index` 判定，并断言历史 episode一致。

### 14.5 Processor 从旧 checkpoint加载

这是最容易出现“config显示已开启但 prompt根本没有memory”的静默失败。必须有真实 train entrypoint 测试，不接受只测 factory。

### 14.6 RTC 提交时机

不能在 `predict_action_chunk` 返回后立即更新 memory；postprocess、reset-version check和 queue merge尚可能失败。

### 14.7 Dashboard 与 logging 死锁

RTC engine持有 state lock 时不要调用可能回到 engine snapshot的 status handler。先复制 snapshot，释放 engine lock，再更新 dashboard。Logging handler自身锁也不得反向获取 inference lock。

### 14.8 非 TTY 污染

CI、重定向文件、systemd日志中不能出现 cursor up/clear ANSI。`auto` 判定和 plain fallback必须测试。

### 14.9 当前参考数据不是训练就绪证明

raw `nero_egg` 有新标注，但现存 converted dataset没有对应列。任何“真实数据 smoke通过”都必须打印 loaded metadata/row证明实际使用的是新构建数据，不能只凭目录名判断。

---

## 15. 完成定义（Definition of Done）

只有同时满足以下条件，整个任务才算完成：

1. PI0、PI0.5 都能训练和部署 memory，且共享相同数据/processor/RTC契约。
2. 历史 offset严格先从1–12抽样，再做 episode起点判定。
3. Memory dropout和现有 subtask dropout默认0.2、相互独立，四种组合测试覆盖。
4. Memory使用上一轮完整 subtask/progress；训练和部署 prompt格式一致。
5. PI0 memory prompt有效上限至少128；PI0.5不低于当前200。
6. Advantage positive/negative/ignore和FM-only weighting数学回归通过。
7. Memory不新增loss，不读取历史 advantage weight，不改变 current subtask CE reduction。
8. RTC只在成功 merge后提交 memory，reset/pause/home后清空。
9. Live dashboard固定显示 state/event/latency/timing/subtask/memory，普通日志仍正常。
10. 非TTY无ANSI污染，plain模式可读。
11. Memory disabled时数据、prompt、token shape、loss和旧checkpoint行为保持当前基线。
12. 专项单测、集成测试、两种policy小训练smoke、fake RTC和Nero实机smoke均有真实记录。
13. completion records列出未运行项；不得把 mock/synthetic结果写成实机通过。
14. 用户已有工作区改动未被覆盖。

---

## 16. Agent 交接要求

每个实施 agent 开始前：

1. 读完本文和 §10.1 指定文件；
2. 执行 `git status --short`，保护用户改动；
3. 先跑对应基线测试；
4. 只实施当前 milestone，不顺手扩展其他模型；
5. 先写/更新测试，再宣称契约成立。

每个 milestone 结束时在 `plans/memory_pipeline/` 写 completion record，至少包含：

- 日期和基线 commit；
- 修改文件；
- 关键契约；
- 实际执行命令；
- pass/fail/skip数量；
- 未运行的 GPU、完整 checkpoint 或实机项；
- 对下一 milestone 的明确输入和剩余风险。

最终 agent 必须运行所有 memory、subtask、advantage、RTC、deploy专项回归及 `git diff --check`，并把真实结果写入最终完成记录。
