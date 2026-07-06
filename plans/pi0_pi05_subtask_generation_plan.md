# PI0 / PI0.5 Subtask 自回归生成改造计划

> 目标读者：负责实现的智能体。本文档给出背景、设计、逐文件改动点、验收标准与实施顺序。
> 环境：conda 环境 `lerobot-main`；如需访问 HuggingFace Hub，先 `export https_proxy=http://127.0.0.1:1080 http_proxy=http://127.0.0.1:1080`（paligemma tokenizer 大概率已有本地缓存）。
> 仓库：`/home/zenbot-robot/repos/lerobot`，所有路径相对 `src/lerobot/`。

---

## 0. 背景与硬性约束

本仓库当前的实际用法收敛为：**pi0 / pi05 两个模型 + bi_nero 双臂机器人 + pico 遥操作**。采数用 `scripts/lerobot_raw_record.py`（原始格式），标注用 `scripts/lerobot_annotate_subtask.py`，构建数据集用 `scripts/lerobot_build_dataset.py`，训练用 `scripts/lerobot_train.py`，部署用 `scripts/lerobot_policy_deploy.py`（RTC 异步推理）。**其他模型/机器人兼容性不需要考虑**，但不要无故破坏它们的 import。

要实现的功能：在 pi0/pi05 的 VLM backbone（PaliGemma）上增加 **subtask 自回归（AR）生成**：

1. **训练**：prompt 后面拼接 ground-truth subtask 文本段（含 progress），该段内部 causal attention，对其计算 next-token CE loss；action expert 的 noisy action tokens 通过联合前向 attend 到（复用）这些文本特征；flow matching loss 不变，两个 loss 加权求和。
2. **鲁棒性**：训练时以一定概率让 action expert **看不到** subtask 段（attention mask 层面 dropout），使得部署时可以完全跳过 AR、按训练中"无 subtask 段"的分布推理，性能仍有保障。
3. **部署**：每次推理 action chunk 前，先用 VLM 做一次 KV-cache 的贪心 AR 生成 subtask 文本（含 progress），生成的 token 的 KV 直接留在 cache 里给 denoise 步骤 cross-attend；生成文本可取出记录。留 config 开关可关闭 AR（退化为原始推理）。
4. **兼容性（硬约束）**：所有新行为由 config 开关控制，默认关闭。关闭时训练/推理数值行为与现状**完全一致**（旧 checkpoint、无 subtask 数据集的小任务微调不受任何影响）。

已确认的决策（用户已拍板）：
- pi05 与 pi0 **同时改**（先实现并测试 pi05，再镜像到 pi0）。
- **原地修改 + config 开关**，不新建 policy 类型。
- 部署时**每个 chunk 都做 AR**，另留整体关闭开关。
- progress 作为 **AR 文本的一部分**，1 位小数（0.0–1.0，10 档）。

---

## 1. 现状梳理（实现前必读的代码）

### 1.1 模型侧

| 文件 | 关键内容 |
|---|---|
| `policies/pi05/modeling_pi05.py` | `PI05Pytorch.embed_prefix()`(L640) 图像+文本嵌入，att_masks 全 0（双向）；`embed_suffix()`(L683) noisy action + adaRMS time cond，att_masks `[1]+[0]*(chunk-1)`；`forward()`(L730) 训练联合前向，`(_, suffix_out)` 丢弃了 prefix 输出；`sample_actions()`(L785) 先 prefix 前向建 KV cache，再 denoise 循环；`denoise_step()`(L865) suffix attend 全部 valid prefix（`prefix_pad_2d_masks`）；`make_att_2d_masks()`(L108) **cumsum 块状 attention 机制**（见下）；`PI05Policy.forward()`(L1248) 组 batch、调模型、聚合 loss。 |
| `policies/pi0/modeling_pi0.py` | 结构与 pi05 平行。差异：state 是连续向量经 `state_proj` 放在 **suffix 开头**（att_mask=1，L796），prompt 里没有 state；time 通过 `action_time_mlp_*` 融合而非 adaRMS；`forward`/`sample_actions`/`denoise_step` 多一个 `state` 参数。 |
| `policies/pi0_fast/modeling_pi0_fast.py` | **重要参考实现**：`_create_custom_attention_mask_fast()`(L452) 手工构造"图文双向 + 尾段 causal"的 2D mask；`forward()`(L493) 用 `paligemma.lm_head` 对尾段算 shift 后的 next-token CE（含 pad mask 处理）；`sample_actions_fast()`(L593) 无 cache AR 解码；`sample_actions_fast_kv_cache()`(L689) **KV-cache 增量 AR 解码**（每步只喂新 token、position_ids 续接、mask 增长）。AR 相关代码基本可以照此模式写。 |
| `policies/pi_gemma.py` | `PaliGemmaForConditionalGenerationWithPiGemma` 继承 HF `PaliGemmaForConditionalGeneration`，**带 `lm_head`**；pi0/pi05 的 `_fix_pytorch_state_dict_keys` 里已经把 checkpoint 的 `lm_head.weight` 复制进 `embed_tokens`（两者 tied），pi0_fast 已验证 `paligemma.lm_head` 可直接用于 logits。 |

**`make_att_2d_masks` 的 cumsum 机制（本次改动的核心杠杆）**：`att_masks` 是每 token 一个 0/1 标记，1 表示"开启新块"。对每个 query，它能 attend 到所有 `cumsum(att_masks)` 小于等于自己的 key。因此：

- prefix（图像+prompt）全 0 → cumsum=0，块内双向，且**天然看不到后面 cumsum≥1 的 token**；
- 给 subtask 段每个 token 标 1 → cumsum 逐 token 递增 → **段内严格 causal**，且能看到全部 prefix；
- action 段 `[1]+[0]*(chunk-1)` → 能看到前面一切（含 subtask 段）。

也就是说训练时的 attention 结构**不需要新的 mask 构造函数**，只需要给 subtask 段 append `[1]*S` 的 att_masks。padding（右 pad）通过 `pad_masks` 自动屏蔽，`position_ids = cumsum(pad_masks)-1` 也自动跳过 pad 位。

### 1.2 Processor / 数据流

- `policies/pi05/processor_pi05.py`：`Pi05PrepareStateTokenizerProcessorStep` 把 state 离散化 256 桶拼进 prompt：`"Task: {task}, State: {d0 d1 ...};\nAction: "`；随后 `TokenizerProcessorStep`（paligemma tokenizer，`max_length=200`，右 pad 到定长）。
- `policies/pi0/processor_pi0.py`：`Pi0NewLineProcessor` 只给 task 加 `\n`，prompt 即 `"{task}\n"`。
- `processor/tokenizer_processor.py`：`TokenizerProcessorStep` **已经支持 subtask**——`get_subtask()`(L144) 从 complementary_data 读 `"subtask"`，tokenize 后写入 `OBS_LANGUAGE_SUBTASK_TOKENS` / `OBS_LANGUAGE_SUBTASK_ATTENTION_MASK`（常量在 `utils/constants.py` L29-31 已定义）。但目前 subtask 与 task 用同一套 max_length/special-token 设置，需扩展（见 §4.2）。
- `processor/converters.py` `_extract_complementary_data()`(L156)：**已经**把 batch 里的 `"subtask"` 列路由进 complementary_data（L170）。`"subtask_progress"` 尚未路由，需加一行。
- 训练 batch：`LeRobotDataset.__getitem__` 返回的 dict 含数据集所有列（string 列按原样返回），`lerobot_train.py` L426 `batch = preprocessor(batch)` 走上述管线。**训练脚本本身预计零改动**（loss_dict 里的新键会经 `output_dict` 自动进 wandb 日志）。

### 1.3 数据采集/标注链路

- `lerobot_raw_record.py` 产出 raw run：`run_meta.json` + `ep_XXXXXX/{info.json, frames.parquet, 相机目录/*.png}`。
- `lerobot_annotate_subtask.py`：网页标注器，把每帧的 subtask 标签导出为每 episode 的 `extras.parquet`（单 string 列，默认列名 `subtask`，未标注帧为 `default_value`，默认 `""`）。
- `lerobot_build_dataset.py`：`_load_extras_schema()`(L175) 读 extras 列并入 feature schema（string → `{"dtype":"string","shape":(1,)}`，float/int 走 `str(pa_type)`），逐帧 merge 进 LeRobotDataset。

---

## 2. 总体设计

### 2.1 序列布局与 prompt 格式

**prefix prompt 按开关取两种形态**：

- `predict_subtask=False`（默认）：与现状**完全一致**（pi05：`"Task: {task}, State: {state};\nAction: "`；pi0：`"{task}\n"`），保证旧 checkpoint / 无 subtask 微调逐 bit 兼容。
- `predict_subtask=True`：pi05 prefix 改为 `"Task: {task}, State: {state};\n"`（**去掉 `Action: ` 尾巴**，否则训练文本会变成语义混乱的 `"Action: Subtask: ..."`）；pi0 的 prompt 本来就是 `"{task}\n"`，不变。该形态下，无论样本有无 subtask、部署是否做 AR，prefix 一律不含 `Action: `——**subtask-enabled checkpoint 的训练与部署分布严格一致**。

训练时（`predict_subtask=True` 且样本有 subtask 标签），完整序列：

```
┌────────────────────── PaliGemma (VLM) ──────────────────────┐┌── action expert ──┐
[img×N (各256 tok)][prompt tok (右pad至200)][subtask AR 段 (右pad至S_max)][ (pi0: state)  noisy action ×chunk]
 att=0 双向         att=0 双向              att=1 逐token causal          att=[1]([1])+[0]*...
```

**subtask AR 段文本**（在 processor 中构造）：

```
"Subtask: {subtask}; Progress: {progress:.1f}\n"   →  tokenize(add_special_tokens=False) + [EOS]
```

- 推理 seed：`tokenizer("Subtask:", add_special_tokens=False)`（**不带尾空格**——sentencepiece 会把空格并进下一词 `▁pick`，这样 seed token 序列是训练文本的严格前缀）。
- progress ∈ {0.0, 0.1, …, 1.0}，由每帧的 `subtask_progress` float 四舍五入到 1 位小数。
- 样本 subtask 为空字符串或缺失 → 该段**全 padding**（tokens 全 0、mask 全 False）：不产生 CE loss，action expert 因 pad mask 也 attend 不到，行为自动退化。这是"混合有/无标注数据共训"的机制。

### 2.2 训练 loss

```
total = fm_loss.mean() + subtask_ce_loss_weight * ce_loss
```

- CE 按 pi0_fast `forward()` 的写法：取 prefix 输出的最后 S 个位置过 `paligemma.lm_head`，logits 左移/targets 右移做 next-token 预测，按（右移后的）subtask pad mask 求 masked mean。段内第一个 token（"Sub"）没有前驱预测它，天然不进 loss——**不要**尝试用 prompt 段最后一个有效 token 去预测它（prompt 右 pad 导致该位置逐样本可变，收益可忽略）。
- **模型返回 per-sample CE**：`ce_loss_per_sample: [B] = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)`，subtask 全空的样本严格为 0。policy 层聚合：`reduction="mean"` 记 `ce_loss_per_sample.mean()`；`reduction="none"`（RA-BC 路径，当前不使用但保持语义正确）逐样本相加。
- **dtype 处理**：lm_head 的输入 hidden 按 `lm_head.weight.dtype` 对齐（不要把 hidden 强转 float32 再过 bf16 权重的 linear，会 dtype 不匹配）；**logits 转 float32** 后再算 CE，保证数值稳定。
- `loss_dict` 中分别记录 `fm_loss`、`ce_loss`（wandb 自动可见），便于调 `subtask_ce_loss_weight`。

### 2.3 Subtask 条件 dropout（鲁棒性）

训练 forward 中，对 batch 每个样本以 `subtask_dropout_prob` 概率独立采样"丢弃"：把 **suffix 全部行**（pi05：action 行；pi0：state+action 行）对 **subtask 段所有列** 的 attention 置 False（在 `make_att_2d_masks` 之后、转 4D 之前直接改 `att_2d_masks`）。注意：

- CE loss **照常计算**（subtask 段自身行不动）；
- 被丢弃样本的 action expert 所见与"无 subtask 样本"完全一致——这正对应部署时不做 AR 的情形。

### 2.4 部署推理两种模式

`predict_subtask=True` 的 checkpoint，`sample_actions` 内：

- **模式 A（`subtask_generate_at_inference=True`，默认）**：
  1. prefix 前向 `use_cache=True` 建 KV（现有代码不动）；
  2. 新增 `_generate_subtask()`：先一次前向喂入 seed tokens（`"Subtask:"` 约 3 个 token），**seed 段内必须 causal**——构造"每个 seed token attend 全部 valid prefix + seed 段内下三角"的 mask（多层网络里 seed token 的 KV 依赖下层 attention 输出，全可见会污染 cache，且与训练分布不一致）；随后贪心逐 token 解码（照抄 pi0_fast `sample_actions_fast_kv_cache` 的 KV-cache 增量模式：每步只喂新 token embedding，position_ids 从 `prefix_pad_masks.sum()` 续接，attention 行 = 全部 valid prefix + 已生成 token），直到 EOS 或 `subtask_max_decode_tokens`；
  3. 生成 token 的 KV 已留在 `past_key_values` 中；把 `prefix_pad_masks` 右侧 append 相应个数的 True（seed + 生成 + EOS）；
  4. denoise 循环**零改动**——`denoise_step` 里 `prefix_pad_2d_masks` 本来就让 suffix attend 所有 valid prefix 位置，KV 变长后自动生效（position_ids 用 `prefix_pad_masks.sum()` 偏移，同样自动正确）；
  5. 解码文本存到 `self._last_subtask_text`（PI05Policy/PI0Policy 属性），文本变化时 `logging.info` 一条。
- **模式 B（`subtask_generate_at_inference=False`）**：跳过 2–4。注意：由于 §2.1 中 prefix 已去掉 `Action: `，模式 B **不是**"与原始 pi05 逐 bit 一致"，而是**与训练时 dropout / 无 subtask 样本分支分布一致**（这正是 dropout 训练所保障的部署形态）。只有 `predict_subtask=False` 才逐 bit 等于原始行为。

部署脚本通过 `--policy.subtask_generate_at_inference=false` 即可切换，`lerobot_policy_deploy.py` 的 `__post_init__` 已支持 policy CLI overrides，无需改脚本参数解析。

---

## 3. Config 改动

`policies/pi05/configuration_pi05.py` 与 `policies/pi0/configuration_pi0.py` 各加（字段名两边保持一致）：

```python
# --- Subtask AR generation ---
predict_subtask: bool = False          # 总开关：训练构造 AR 段 + 推理允许 AR
subtask_max_tokens: int = 48           # 训练时 AR 段右 pad 定长（含 EOS）
subtask_ce_loss_weight: float = 0.25   # CE loss 权重（需实验调参，见 §8）
subtask_dropout_prob: float = 0.2      # 训练时 action expert 看不到 subtask 段的概率
subtask_generate_at_inference: bool = True   # 部署是否做 AR（False=模式 B）
subtask_max_decode_tokens: int = 48    # 推理 AR 最大解码步数
subtask_decode_temperature: float = 0.0      # 0=greedy
```

`__post_init__` 校验：`predict_subtask=True` 时 `train_expert_only` 必须为 False（VLM 需要梯度）；`subtask_max_decode_tokens <= subtask_max_tokens` 不强制但给 warning。

---

## 4. 数据与 Processor 改动

### 4.1 标注器导出 progress（`scripts/lerobot_annotate_subtask.py`）

`export_extras()`(L183)：在写 `extras.parquet` 时额外写一列 `subtask_progress`（`pa.float32()`）：

- 对每个 episode 的 labels 序列，找出**相邻且标签相同的非空段**；段内第 i 帧（0-based，段长 L）progress = `(i + 1) / L`；
- 未标注/空标签帧 progress = `0.0`；
- 保留原有"合并已有 extras 列"的逻辑（注意 `feature_name` 和 `subtask_progress` 两列都要在合并时视为自己拥有的列，避免自我合并残留旧值）。

同时修改 `scripts/lerobot_build_dataset.py` `_load_extras_schema()` L219：float 列的 dtype 目前取 `str(pa_type)`（`pa.float32()` 得到 `"float"`，不是合法的 LeRobot feature dtype）。**直接把 `pa.types.is_floating` 分支显式映射为 `"float32"`**（int/bool 分支同理检查一遍），不要依赖 `str(pa_type)` 的偶然结果。验收：build 出的数据集 `__getitem__` 能取到 float 的 `subtask_progress`（测试 7 实测整条链路）。

### 4.2 Tokenizer 步骤扩展（`processor/tokenizer_processor.py`）

`TokenizerProcessorStep` 加字段（进 `get_config()` 以便随 checkpoint 序列化）：

```python
subtask_max_length: int = 48
tokenize_subtask: bool = False   # 必须显式开启且默认 False：当前实现只要 complementary data
                                 # 里有 "subtask" 就会 token 化，不加开关会波及其他策略和现有测试
```

行为改动（仅 `tokenize_subtask=True` 时）：

- subtask 文本用 `add_special_tokens=False` tokenize（不要 BOS），**手动 append `eos_token_id`**，右 pad/截断到 `subtask_max_length`；截断时 `logging.warning`（标注文本过长应在数据侧发现）；
- 空字符串 `""` → tokens 全 0、attention mask 全 False（不是"只有 EOS"）；
- `transform_features()` 补上 subtask 两个 key 的 PolicyFeature 声明。

### 4.3 Subtask 文本组装步骤（新增，`processor/` 下共享）

新建注册步骤 `SubtaskTextProcessorStep`（pi0/pi05 共用），放在各自 pipeline 中 `TokenizerProcessorStep` **之前**：

- 从 complementary_data 读 `subtask`（string 或 list[string]）与 `subtask_progress`（float 或 tensor）；
- 非空 subtask → 覆写 complementary_data `"subtask"` 为 `f"Subtask: {s.strip()}; Progress: {round(p,1):.1f}\n"`；空/缺失 → 置 `""`;
- `subtask_progress` 缺失但 subtask 存在时 progress 按 1.0 处理并 warning（容忍旧标注数据）；
- 在 `processor/__init__.py` 中导出该步骤（对照现有步骤的导出方式），否则 pi0/pi05 的 processor 文件 import 不到。

### 4.4 converters（`processor/converters.py`）

`_extract_complementary_data()` L170 旁边加一行路由 `"subtask_progress"`。

### 4.5 pipeline 接线（`policies/pi05/processor_pi05.py`、`policies/pi0/processor_pi0.py`）

`make_pi05_pre_post_processors` / `make_pi0_pre_post_processors`：当 `config.predict_subtask` 时：

- 在 input_steps 中插入 `SubtaskTextProcessorStep`，并给 `TokenizerProcessorStep` 传 `tokenize_subtask=True, subtask_max_length=config.subtask_max_tokens`；
- **pi05 专有**：给 `Pi05PrepareStateTokenizerProcessorStep` 加布尔字段（如 `omit_action_suffix: bool = False`，进 `get_config()` 序列化），为 True 时 prompt 模板改为 `f"Task: {cleaned_text}, State: {state_str};\n"`（去掉 `Action: `，见 §2.1）；pipeline 接线处按 `config.predict_subtask` 传值。

关闭时 pipeline 与现状完全一致。

**推理时注意**：部署时 batch 没有 `subtask` 列（AR 是模型内部做的），`SubtaskTextProcessorStep` 和 tokenizer 的 subtask 分支都要对"缺失"静默跳过（现有 `get_subtask()` 已返回 None，保持该行为）。

---

## 5. Modeling 改动 —— pi05（先做这个）

文件：`policies/pi05/modeling_pi05.py`。所有新逻辑用 `if subtask_tokens is not None:` / `if self.config.predict_subtask:` 分支包裹，参数默认 None 保证旧调用路径不变。

### 5.1 `PI05Pytorch.embed_prefix`（L640）

加可选参数 `subtask_tokens=None, subtask_masks=None`。有值时：用 `embed_language_tokens` 嵌入（同样乘 `sqrt(dim)`），append 到 embs/pad_masks，`att_masks += [1] * S`。（现有 att_masks 是 python list 再转 tensor，直接续接即可。）

### 5.2 `PI05Pytorch.forward`（L730）

签名加 `subtask_tokens=None, subtask_masks=None`。改动点：

1. `embed_prefix` 传入 subtask 参数；
2. dropout：`make_att_2d_masks` 之后，若训练且 `subtask_dropout_prob>0` 且有 subtask：采样 `drop = torch.rand(bsize, device=...) < p`，对 drop 样本把 `att_2d_masks[b, -suffix_len:, sub_start:sub_end] = False`（`sub_start = prefix_len_without_subtask`，`suffix_len = chunk_size`）；
3. 联合前向把 `(_, suffix_out)` 改为 `(prefix_out, suffix_out)`；
4. CE：`sub_hidden = prefix_out[:, sub_start:sub_end]`，dtype 对齐 `lm_head.weight.dtype` 后过 `paligemma.lm_head`，**logits 转 float32**，shift 后按 §2.2 计算 **per-sample** `ce_loss_per_sample: [B]`（参考 pi0_fast L558-585 的 shift/mask 写法，但按样本维聚合且注意 dtype 顺序）；
5. 返回值从单 Tensor 改为 `(fm_losses, ce_loss_per_sample)`（无 subtask 时后者为 None），同步更新 `PI05Policy.forward` 的解包。注意 gradient checkpointing 包裹的 `forward_func` 需要把 prefix_out 一起返回。

### 5.3 `PI05Pytorch.sample_actions`（L785）+ 新方法 `_generate_subtask`

- prefix KV 建好后（L819-825 之后），若 `config.predict_subtask and config.subtask_generate_at_inference`：调 `_generate_subtask(past_key_values, prefix_pad_masks, seed_tokens)` → 返回 `(generated_token_ids, extended_prefix_pad_masks, past_key_values)`；后续 denoise 用扩展后的 pad_masks。
- `_generate_subtask` 实现要点（模板：pi0_fast `sample_actions_fast_kv_cache` L689 起）：
  - seed tokens 由 policy 层传入（tokenizer 不放在 `PI05Pytorch` 里，见 5.4）；seed 段一次前向喂入，但 **mask 必须是"attend 全部 valid prefix + seed 段内下三角"**（不能全可见，见 §2.4 模式 A 第 2 步的说明），之后再逐 token 贪心；
  - 每步 position_ids = 当前 valid 长度 - 1 续接；attention mask 行向量 = 已有全部 valid 位置；
  - 遇 `eos_token_id` 停止（部署 batch=1；实现时仍写成 batch-safe：维护 finished 掩码，finished 样本后续位置 pad）；
  - `temperature>0` 时按 softmax 采样，否则 argmax；
  - 返回的 token ids 供 policy 层 decode 成文本。
- `sample_actions` 返回值保持 Tensor 不变；生成的 token ids 通过返回值扩展（`return_subtask_tokens=True` 关键字）或存到 `self._last_subtask_tokens`，二选一，推荐后者（不动 RTC 调用链）。

### 5.4 `PI05Policy`

- `__init__`：`predict_subtask=True` 时懒加载 `AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")` 存为 `self._paligemma_tokenizer`（pi0_fast 同款），预计算 seed token ids 与 eos id 传给模型；
- `forward()`(L1248)：从 batch 取 `OBS_LANGUAGE_SUBTASK_TOKENS`/`OBS_LANGUAGE_SUBTASK_ATTENTION_MASK`（缺失则 None）传入模型；loss 聚合 `loss = fm.mean() + w * ce_per_sample.mean()`；`reduction="none"` 分支为 per-sample `fm_i + w * ce_i`（模型已返回 per-sample CE，此分支自然正确；RA-BC 场景当前不使用，无需额外处理）；loss_dict 记 `fm_loss`、`ce_loss`；
- `predict_action_chunk()`(L1231)：推理后 decode `self.model._last_subtask_tokens`（`skip_special_tokens=True`）存 `self.last_subtask_text`，与上次不同则 `logging.info("[subtask] %s", text)`；
- `from_pretrained`：确认 `lm_head` 权重可用（现有 remap 已复制到 embed_tokens 且两者 tied；加载后加一次 assert/log）。

---

## 6. Modeling 改动 —— pi0 移植

文件：`policies/pi0/modeling_pi0.py`、`processor_pi0.py`、`configuration_pi0.py`。与 pi05 的差异清单（其余逐条镜像 §3–§5）：

1. `forward`/`sample_actions`/`denoise_step` 多一个 `state` 参数，位置保持不变；
2. suffix 序列 = `[state(1 tok), action(chunk)]`，**dropout 时 suffix 全部行（含 state 行）都屏蔽 subtask 列**，`suffix_len = 1 + chunk_size`；
3. prompt 构造在 `Pi0NewLineProcessor`，无 state 文本；`SubtaskTextProcessorStep` 同样适用；
4. pi0 用 `action_time_mlp_*` 而非 adaRMS——与本次改动无交互，不动；
5. cumsum att_masks 布局：`[0]*img + [0]*lang + [1]*S ‖ [1](state) + [1]+[0]*(chunk-1)`，语义与 pi05 相同（state/action 都能看到 subtask 段，除非被 drop）。

---

## 7. 训练 / 部署脚本

- `scripts/lerobot_train.py`：**预计零改动**。验证两点：(a) batch 中 string 列 `subtask` 与 float 列 `subtask_progress` 能经默认 collate 到达 preprocessor；(b) `ce_loss` 出现在 wandb 日志。DDP 的 `find_unused_parameters=True` 已开（L179），整 batch 无 subtask 时 lm_head 无梯度不会报错。
- `scripts/lerobot_policy_deploy.py`：**预计零改动**（AR 开关走 `--policy.subtask_generate_at_inference=false`；subtask 文本由 policy 内部 logging 输出）。可选增强：在 1 Hz 的 `deploy_loop` 日志里加 `subtask=%s`（读 `policy.last_subtask_text`），顺带用现有 `fetch_avg_ms` 观察 AR 增加的延迟。

---

## 8. 验收标准与测试计划

新建 `tests/policies/test_pi05_subtask.py`（pi0 对应镜像一份或参数化）。除单测外，每个 milestone 有明确验收方式。**回归基线做法**：改动前先在当前 HEAD 上跑一个固定 seed 的 forward/sample_actions，把输出张量存成 `.pt` 作为 golden（脚本放 scratch 即可，不入库），改动后 `predict_subtask=False` 必须与 golden `torch.allclose`。

1. **mask 结构单测**：用小维度手工构造 att_masks（`[0]*P + [1]*S + [1]+[0]*(A-1)`）过 `make_att_2d_masks`，断言：prefix 行看不到 subtask/action 列；subtask 段内严格下三角；action 行全可见；subtask pad 位行列全 False。
2. **默认关闭回归**：`predict_subtask=False` 时 `forward` loss 与 `sample_actions` 输出和 golden 逐位一致；preprocessor 输出**不包含**任何 subtask token keys，pi05 prompt 仍以 `"\nAction: "` 结尾（即使 batch 里带有 `subtask` 列——`tokenize_subtask=False` 必须挡住它）。
3. **空 subtask 等价性 + CE=0**：同一 `predict_subtask=True` 配置下（固定 noise/time 注入，`forward(noise=..., time=...)` 已支持），subtask 段全 pad 时的 fm loss == 不传 subtask（`subtask_tokens=None`）时的 fm loss，且 `ce_loss_per_sample` 逐样本严格为 0。（注意：不再与 `predict_subtask=False` 比较——两者 prompt 不同，fm loss 本就不同。）
4. **dropout 等价性**：`predict_subtask=True` 下强制 `subtask_dropout_prob=1.0` 时，任意非空 subtask 下的 fm loss == 空 subtask 时的 fm loss（同 seed）。
5. **AR 一致性**：随机初始化小模型（`paligemma_variant` 无小档，可直接用 gemma_300m 双塔或缩小 config 构造）上，teacher-forcing 前向的逐位 argmax 与 `_generate_subtask` 贪心解码结果一致（喂相同前缀）。这是 KV-cache/position_ids/mask 增长逻辑（含 seed 段 causal mask）的关键测试。
6. **processor 集成**：构造含 `subtask`/`subtask_progress` 的假 batch，`predict_subtask=True` → preprocessor 输出含正确 shape 的 subtask token keys；`"Subtask: pick; Progress: 0.5\n"` 文本正确；**pi05 prompt 不含 `Action: `**；空 subtask 输出全 pad；无 subtask 列的 batch 不报错。
7. **数据链路**：对一个真实 raw run 跑 annotator 的 `export_extras()`（可直接调函数构造 annotations），检查 `subtask_progress` 段内线性且段尾为 1.0；`lerobot_build_dataset` 构建后 `dataset[i]` 同时取到 `subtask`(str) 与 `subtask_progress`(float)。
8. **过拟合冒烟**（GPU，人工执行）：取一个已标注小数据集，`predict_subtask=True` 训 ~500 步：`ce_loss` 显著下降；训毕对训练集样本做 AR，生成文本与标签基本一致（含 progress 档位大致正确）。
9. **部署冒烟**（真机，人工执行）：模式 A 下 subtask 文本随任务阶段推进变化、`fetch_avg_ms` 增幅在可接受范围（预期 +100~500ms/chunk 量级）；模式 B 下行为与普通 pi05 部署无差异。

单测运行方式：`conda run -n lerobot-main pytest tests/policies/test_pi05_subtask.py -x -q`（tokenizer 需要网络时先设 §0 的代理，或依赖本地缓存）。

---

## 9. 实施顺序（建议的 milestone 切分）

| # | 内容 | 验收 |
|---|---|---|
| M1 | 数据侧：annotator 导出 `subtask_progress`；build_dataset dtype 验证/修正；converters 路由 | 测试 7 |
| M2 | Config 字段 + `SubtaskTextProcessorStep` + `TokenizerProcessorStep` 扩展 + 两个 pipeline 接线 | 测试 6 |
| M3 | pi05 训练路径：embed_prefix/forward（mask、CE、dropout、loss 聚合） | 测试 1–4 + golden 回归 |
| M4 | pi05 推理路径：`_generate_subtask` + sample_actions 接入 + policy 文本输出 | 测试 5 + 2（sample_actions 回归） |
| M5 | pi0 镜像移植（§6 差异清单） | pi0 版测试 1–5 |
| M6 | 集成：小数据过拟合训练 + 真机部署冒烟 | 测试 8、9 |

M1/M2 与 M3/M4 可并行；M5 必须在 M4 验收后开始。

---

## 10. 风险与开放问题

- **`subtask_ce_loss_weight` 需要实验确定**：CE（初期 ~几个 nat）与 flow-matching MSE（<1）量级不同。默认 0.25 只是起点；分开记录两条 loss 曲线，若 fm loss 退化则下调。
- **AR 延迟**：gemma-2B 每 token 一次前向，48 token 上限时最坏几百 ms。RTC 是异步推理，理论上可吸收，但要盯 `rtc_last_latency_ms` 是否触发 delay 报警；不可接受时用模式 B 或减小 `subtask_max_decode_tokens`。
- **标注文本长度**：`subtask_max_tokens=48` 覆盖不了很长的中文/英文标注时会截断（tokenizer 步骤有 warning）；标注时建议 subtask 名保持简短英文短语。
- **进度档位**：1 位小数 = 10 档；若后续想要 2 位，只改 `SubtaskTextProcessorStep` 的格式串与文档，模型侧无改动（可作为 config 化的后续项）。
- **torch.compile**：`compile_model=True` 与新分支（动态 AR 循环）不兼容的风险——`predict_subtask=True` 时直接禁用 compile 或只 compile denoise 路径；在 config 校验里给出明确报错/警告。
- **PEFT/train_expert_only**：subtask 训练需要 VLM 梯度；若显存紧张，可建议 `freeze_vision_encoder=True`（SigLIP 冻结不影响 CE 路径学习），但 language model 必须可训。
- **混合数据配比**：全空 subtask 的 batch 对 lm_head 无梯度是合法的；但若有/无标注数据混训，建议按数据集层面混合而非 episode 内混合，避免同一任务内监督信号不一致。
