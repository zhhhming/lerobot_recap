# Value Function, Subtask Value, Advantage Labeling, and RA-BC Training Plan

日期：2026-07-09

最近审阅：2026-07-10

目标仓库：`/home/zenbot-robot/repos/lerobot`

运行环境：优先使用 `/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`，或 `conda activate lerobot-main` 后运行。当前系统默认 `python` 不可用，`python3` 环境缺少 `pyarrow`，不要用它跑本计划里的数据脚本。

网络环境：如需下载 Hugging Face 模型或 tokenizer，使用本机 VPN 端口 `1080`；优先设置 `HF_HUB_OFFLINE=1` 或 `local_files_only=true` 使用本地缓存，只有缓存缺失时再走网络。

## 1. 背景和目标

当前想验证的是一个以 pi0/pi0.5 为主的离线 value function 和 advantage 标注流程，最终服务于类似 pi*0.6 / RECAP 的 advantage-conditioned policy training。核心不是重写 VLA 主干，而是把每帧 state 的 value 先离线估计出来，再基于 action chunk 的起止 state 计算 advantage，生成 `positive/negative` 条件和 loss weight，最后在 pi0/pi0.5 训练时可选地启用。

整体目标分四层：

1. 训练独立的 value function。
   支持全局 remaining-frame baseline，也支持 subtask 内 remaining-frame value；可选增加 elapsed-from-start 辅助 head。

2. 把 value 预测写回 raw dataset。
   raw 数据每个 episode 已有 `frames.parquet`、图片目录和 `extras.parquet`。value、subtask value、advantage、positive/negative、loss weight 都应优先作为 `extras.parquet` 的附加列保存。

3. 计算 action chunk advantage 和人工/半自动划分。
   对每个 action chunk 计算 advantage，提供可视化排序和比例阈值选择，把 `advantage_label` 写回 raw dataset。

4. 在 VLA 训练里可选启用 advantage condition 和 loss weighting。
   训练时根据 `advantage_label` 把 prompt 增加 `Advantage: positive` 或 `Advantage: negative`；按概率 classifier-free dropout；当 condition 被 dropout 时同步禁用 loss weight。

约束：

- 仓库现阶段只需要关注 `pi0` / `pi0.5`。
- 机械臂只关注 `bi_nero`。
- 遥操作设备只关注 `pico`。
- 不需要保持 gr00t、ACT、diffusion 等模型的完整兼容性，但公共 dataset builder 不应被无谓破坏。
- 当前第一阶段数据只包含成功 episode，不接收失败、超时或人工中止 episode。运行 pipeline 时应把
  `all_episodes_successful=true` 作为显式数据假设写入 metadata；如果以后引入失败 episode，必须先重新
  设计 terminal/outcome target，不能继续直接把 episode 结束当成成功终点。
- 每个 episode 必须包含同一套 subtask，按同一固定顺序各出现一次，并且每个 subtask 只能形成一个
  连续 segment。Milestone 1 必须强校验这个约束，不能只校验 subtask id 不回退。

## 2. 当前代码和数据现状

需要优先阅读这些文件：

- `src/lerobot/scripts/lerobot_raw_record.py`
  raw 采集脚本。格式是 run root 下每个 episode 一个目录，包含 `frames.parquet`、`events.jsonl`、图片目录和 `info.json`。

- `src/lerobot/scripts/lerobot_annotate_subtask.py`
  subtask 标注和导出脚本。导出时写每个 episode 的 `extras.parquet`，当前支持 `subtask` 和 `subtask_progress`。

- `src/lerobot/scripts/lerobot_build_dataset.py`
  raw -> LeRobotDataset builder。它已经会读取 `extras.parquet`，把 string/float/int/bool/list 类型列自动并入最终 LeRobotDataset。

- `src/lerobot/scripts/lerobot_train.py`
  离线训练入口。已经支持 `use_rabc`，会让 policy 返回 per-sample loss 并乘权重。

- `src/lerobot/utils/rabc.py`
  当前 RA-BC 权重提供器。它从 parquet 读 progress，再按 chunk delta 算权重。后续可改造成直接读 dataset 中的 `advantage_loss_weight`，或新增一个并行的 advantage weights provider。

- `src/lerobot/policies/pi0/modeling_pi0.py`
  pi0 forward 已支持 `reduction="none"`，返回 batch 级 per-sample loss，这正好满足 loss weighting。

- `src/lerobot/policies/pi0/configuration_pi0.py`
  pi0 配置入口。需要新增 advantage conditioning/dropout 相关配置时放这里。

- `src/lerobot/policies/pi0/processor_pi0.py`
  pi0 preprocessor pipeline。prompt 或 subtask text 的修改应在 processor 层实现，而不是在 dataset 内硬改 `task` 文本。

- `src/lerobot/processor/subtask_processor.py`
  当前 subtask 文本格式为 `Subtask: ...; Progress: x.x`，可参考新增 advantage text processor。

- `src/lerobot/processor/converters.py`
  当前只把 `task`、`subtask`、`subtask_progress`、index 等作为 complementary data 透传。后续要透传 `advantage_label` 和 `advantage_loss_weight`。

- `src/lerobot/rl/algorithms/recap_value_network_pi0.py`
  旧的 pi0/PaliGemma value network 原型，可参考 PaliGemma 权重加载和 layer 截断方式。

- `src/lerobot/rl/algorithms/recap_train_value_network_pi0.py`
  旧的 value training 原型，可参考 two-hot / soft CE target 生成方式，但新实现建议围绕 raw dataset + subtask value 重新组织。

样例 raw dataset：

- `/home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3`
- 这个数据集只作为测试样例和格式参考，后续实现绝不能写死 repo id、任务文本、episode 数、subtask 数量或 subtask 名称。所有 subtask 数量、顺序、名称都必须从当前 raw run 的 `annotation_config.json` / `extras.parquet` / 用户提供配置中解析。
- `run_meta.json` 中 features 为：
  - `action`
  - `observation.state`
  - `observation.images.left_wrist`
  - `observation.images.third_person`
  - `observation.images.right_wrist`
- `robot_type` 是 `bi_nero_follower`
- `fps` 是 30
- 当前有 70 个 episode
- 每个 episode 的 `extras.parquet` 当前已有：
  - `subtask: string`
  - `subtask_progress: float`
- 样例 subtask 顺序为 6 段：
  - `Pick up the match.`
  - `move the right arm to ready.`
  - `Pick up the matchbox.`
  - `move the left arm to ready.`
  - `Strike the match and light the candle.`
  - `Return to the home position.`
- 之后真正做强化学习的任务可能不是该任务，但 raw dataset 结构、subtask 标注方式和三路图片结构预计类似。

## 3. 对当前想法的审视和建议默认值

### 3.0 GT label、model prediction 和重新标注的关系

这里必须区分两件事：

- `*_gt`：从某条成功 episode 当前帧往后数得到的监督标签，用来训练 value model。它仍有
  episode-specific 偏差，例如不同示范速度、停顿或恢复动作会改变 remaining frames；当前范围不包含
  失败 episode。
- `*_pred`：value model 对所有 frame 重新预测出来的 value。这个才是后续 advantage 和 VLA 训练默认应该使用的字段。

也就是说，即使 raw dataset 已经人工标注了 subtask，仍然要用训练好的 value model 对整个 dataset 的每一帧重新标注 value。subtask 标注只用于：

- 生成 subtask 内 remaining/elapsed 的训练标签；
- 给 subtask classifier 提供监督；
- 在离线计算 advantage 时提供 GT subtask 边界；
- 展示和 debug。

后续默认数据流应该是：

```text
人工 subtask 标注 -> 生成 noisy GT value target -> 训练 value model -> value model 重新预测所有 frame -> 用 predicted value 计算 advantage
```

第一版可以支持 `--value_source gt|mock_pred` 做 sanity check，但实验和 VLA 训练必须使用
`--value_source model_pred`。

这里的 `GT` 不是模型预测，而是由人工 subtask 边界和 episode/frame 位置直接计算出的监督标签。
由于 global/subtask remaining GT 在同一 segment 内每帧严格减少 1，直接用 GT 计算出的 centered
advantage 理论上恒为 0（跨边界按下文定义后仍应为 0）。因此：

- GT advantage 只能验证公式、字段写回、UI、build dataset 和训练 batch 数据流。
- positive/negative 的真实排序只允许使用 value model prediction。
- value model 尚未完成时，可以从 GT frame-unit value 生成带固定 seed 的 Gaussian-noise mock
  prediction 做端到端 smoke；必须使用独立的 `*_mock_pred` 列，并在 metadata 标记
  `prediction_source=synthetic_gt_gaussian_noise`。
- label/weight export 默认拒绝 GT 或 mock prediction；只有显式 `--allow_synthetic true` 才能用于测试，
  且 synthetic 结果不得作为正式 VLA 实验数据。

### 3.1 全局 value 和 subtask value 都应该实现

不要等选定全局或 subtask 后再做后续流程。建议从第一版就让所有脚本接受 `--value_mode global|subtask`，列名带 prefix，避免互相覆盖。

推荐列名：

- `value_global_remaining_frames_gt`
- `value_global_remaining_norm_gt`
- `value_global_remaining_norm_pred`
- `value_global_remaining_frames_pred`
- `value_global_elapsed_frames_gt`，可选
- `value_global_elapsed_norm_gt`，可选
- `value_subtask_id_gt`
- `value_subtask_name_gt`
- `value_subtask_remaining_frames_gt`
- `value_subtask_remaining_norm_gt`
- `value_subtask_remaining_norm_pred`
- `value_subtask_remaining_frames_pred`
- `value_subtask_elapsed_frames_gt`，可选
- `value_subtask_elapsed_norm_gt`，可选
- `value_subtask_id_pred_raw`
- `value_subtask_confidence`
- `value_subtask_id_pred_smooth`
- `value_subtask_name_pred_smooth`
- `value_subtask_remaining_norm_pred_gt_head`
- `value_subtask_remaining_frames_pred_gt_head`
- `value_subtask_remaining_norm_pred_smooth_head`
- `value_subtask_remaining_frames_pred_smooth_head`

训练阶段可以用 normalized target，但存储阶段必须同时保存 frame-unit value。后续 advantage 以 frame-unit canonical value 为准，避免不同 subtask 归一化尺度混在一起。

subtask model 是多 head 模型，所以“某帧的 remaining prediction”必须同时说明选择了哪个 head。
第一版只允许下面两个成对一致的 inference path，禁止把一个 path 的 value 和另一个 path 的边界混用：

- `gt_conditioned`：使用人工 `value_subtask_id_gt` 选择对应 remaining head，同时使用人工 GT subtask
  边界切分 chunk。它是已完整人工标注 raw dataset 上的默认离线 advantage 路径；虽然选择 head 时使用
  GT id，但 remaining 数值本身仍然是 value model prediction。
- `pred_smooth`：先对模型的 subtask classifier 做单调 Viterbi smoothing，再用 smoothed predicted id
  选择对应 remaining head，并使用同一条 smoothed predicted path 切分 chunk。它用于部署语义验证、
  未标注数据和与 `gt_conditioned` 的对照实验。

CLI 使用单一参数 `--subtask_inference_path gt_conditioned|pred_smooth` 同时决定 head selection 和边界，
不再暴露可以任意组合的独立 `--subtask_source`。`value_subtask_remaining_*_pred` 这种不带 head source
的模糊列名不作为 canonical 字段。

### 3.2 归一化策略

全局 value：

- ground truth raw value：`remaining_frames = episode_end_frame - current_frame`
- 推荐训练归一化：`remaining_norm = remaining_frames / global_value_scale_frames`
- 默认 `global_value_scale_frames = p95(max_episode_remaining)`，并提供 `--global_scale max|p95|manual`
- 如果使用 `p95`，超过 1 的训练 target 默认 clip 到 `[0, 1]`，但 metadata 里必须记录实际 scale，并额外写 `value_global_remaining_norm_gt_is_clipped` 方便检查 clip 比例。

subtask value：

- ground truth raw value：`subtask_remaining_frames = subtask_end_frame - current_frame`
- 每个 subtask 单独 scale：`scale_k = p95(length_of_subtask_k)` 或 `max(length_of_subtask_k)`
- 训练 target：`remaining_norm = subtask_remaining_frames / scale_k`
- 存储预测时必须用 `frames_pred = norm_pred * scale_k` 复原。
- 超过 1 的 subtask target 同样默认 clip，并写 `value_subtask_remaining_norm_gt_is_clipped`。如果某个 subtask clip 比例过高，例如超过 5%-10%，说明该 subtask 的 p95 scale 不适合当前任务，应改用 `max` 或 `manual` scale。

原因：value function 训练需要数值稳定，但 action chunk advantage 的物理单位是“帧/动作单位”，所以最终计算必须回到 frame units。

注意：clip 会损失长尾长度区分能力。它是为了避免少数异常长 episode/subtask 拉坏大多数样本的 bin resolution。后续实验如果很关心长尾，应该改用 `max`、手动 scale，或把 bin 支持范围改成 `[0, v_max]`，而不是继续默认 `[0, 1]`。

### 3.3 value bin 数和输出解码

训练 label 用 two-hot，不用 hard one-hot。

`256` 只是默认值，不能写死。所有相关脚本和模型配置都必须支持：

- `--num_bins`：同时作用于全局和 subtask。
- `--global_num_bins` / `--subtask_num_bins`：如果需要分开设置。
- checkpoint metadata 保存实际 bin 数，推理脚本从 checkpoint 读取，CLI 只能在明确 override 时覆盖。

经验默认：

- 中等长度全局 episode：`256` 起步。
- 很长程任务：可以提高到 `512` 或更高。
- subtask 很短时：可以降低到 `64` 或 `128`，避免过细 bin 造成空 bin 和过拟合。

推理时不要“最大 bin + 相邻第二大”这种启发式。推荐 distributional expectation：

```text
pred_norm = sum_i softmax(logits)[i] * bin_center[i]
```

如果要更稳，可同时保存：

- `argmax_bin`
- `expected_value`
- `entropy`
- `top1_prob`

实际 value 用 expected value，argmax 只用于 debug。

### 3.4 subtask 多头输出

推荐结构：

```text
shared trunk
  -> subtask classifier: num_subtasks
  -> remaining heads: num_subtasks * num_value_bins
  -> optional elapsed heads: num_subtasks * num_value_bins
```

训练 loss：

- subtask CE：所有帧都训。
- remaining CE：只训练当前 GT subtask 对应的 head。
- elapsed CE：如果开启，只训练当前 GT subtask 对应的 head。

不要让其他 subtask head 对同一帧也吃 loss；边界 soft overlap 可以作为第二阶段增强，不作为第一版硬需求。

### 3.5 subtask 边界稳定化

模型预测的 subtask 序列可能出现 `1,2,2,1,1,2` 抖动。后处理必须利用“单个任务内 subtask 顺序固定且不可回退”的先验。

当前数据范围进一步保证：每个 episode 包含全部 subtask，严格按 canonical order 各出现一次，每个
subtask 只有一个连续 segment。target preparation 应从显式配置或第一条完整 episode 得到 canonical
order，并要求所有 episode 的压缩 label sequence 与它完全一致。

推荐第一版后处理：

1. 得到每帧 `p(subtask=k)`。
2. 限制路径只能单调不降。
3. 用动态规划/Viterbi 找最大概率路径：
   - 状态 `k` 可保持 `k` 或转移到 `k+1`。
   - 转移惩罚可配，默认 `transition_penalty=0.0`。
   - 允许跳过非常短的 subtask 需显式开关，默认不允许跳过。
4. 输出 `value_subtask_id_pred_smooth` 和 `value_subtask_name_pred_smooth`。

第一版离线 advantage 默认使用上文定义的 `gt_conditioned` path；模型预测 subtask 仍需对每帧输出并
做 smoothing，主要用于 `pred_smooth` 对照、部署/未标注 frame 推理和 debug。两个 path 的 boundary
source 与 remaining head source 必须保持成对一致。

### 3.6 action chunk advantage 公式

全局 remaining value 的推荐公式：

```text
progress = V_remaining(s_t) - V_remaining(s_t+h)
expected = valid_h
advantage = progress - expected
```

其中 `valid_h` 是 chunk 内真实有效 frame 数。靠近 episode 末尾时 LeRobot 会 pad 最后一帧，不能把 pad 的 49 帧当成真实推进；必须用 episode 边界截断。

如果最终只想排序，也可以存：

- `advantage_raw = V_start - V_end`
- `advantage_centered = V_start - V_end - valid_h`

建议后续 positive/negative 默认用 `advantage_centered` 排序，但 UI 里提供切换。

subtask value 不使用 `completed_subtask_scale_sum` 这类绝对累计尺度。原因是手工 subtask 通常没有可靠的绝对长度，subtask value 更适合表示“同一 subtask 内的相对差值”。把不同 subtask 的 p95/max scale 强行相加，会把 scale 误差引入 advantage。

推荐第一版使用“按边界手动切分 chunk”的相对差分：

```text
chunk: [t, t+h]
根据 GT subtask 或 smoothed predicted subtask，把 chunk 切成若干同-subtask segment
每个 segment j: [a_j, b_j], subtask = k
segment_progress_j = V_k_remaining(a_j) - V_k_remaining(b_j)
within_subtask_progress = sum_j(segment_progress_j)
boundary_progress = boundary_transition_value * num_crossings
total_progress = within_subtask_progress + boundary_progress
advantage = total_progress - valid_horizon
```

默认 `boundary_transition_value=1.0`。一个 chunk 有 `valid_horizon` 个真实 transition；同 subtask
remaining 差分覆盖其中的 `valid_horizon - num_crossings` 个 transition，而每个跨边界的相邻帧
transition 直接计 1 单位 progress。这样理想轨迹无论是否跨边界，`total_progress=valid_horizon`，
centered advantage 都为 0。

这不是在 centered advantage 计算完成后额外加 `+1 bonus`。旧公式
`sum(segment_progress - segment_expected) + boundary_bonus * crossings` 会让理想的跨边界 chunk
得到正 advantage，不符合本计划现在的定义。CLI 应将参数命名为 `--boundary_transition_value`；
如需保留旧 `--boundary_bonus`，只能作为 deprecated alias 并明确迁移语义。

边界帧定义要写清楚：

- 如果 segment 结束点正好是旧 subtask 最后一帧，用旧 subtask head 估值。
- 如果下一帧已经进入新 subtask，后一个 segment 从新 subtask 第一帧开始。
- 靠近边界的 GT/pred subtask 抖动先用单调 Viterbi smoothing 处理，再切分。
- 同时写出 `advantage_{mode}_num_crossings`、`advantage_{mode}_within_subtask_horizon` 和
  `advantage_{mode}_boundary_progress`，便于验证完整 horizon 是否被覆盖。

这样做只依赖同一个 subtask head 内的 value 差值，不需要跨 subtask 比较绝对 value。

### 3.7 positive/negative 和 loss weight

positive/negative 标注应该是独立字段，不要只隐含在权重里。

推荐列名：

- `advantage_global_chunk`
- `advantage_subtask_chunk`
- `advantage_valid_horizon`
- `advantage_label_global`
- `advantage_label_subtask`
- `advantage_loss_weight_global`
- `advantage_loss_weight_subtask`
- `advantage_group_id_global`
- `advantage_group_id_subtask`

loss weight 公式推荐：

```text
u = rank / max(N - 1, 1)     # best = 0, worst = 1
w_raw = w_min + (w_max - w_min) * sigmoid((q - u) / tau)
w_positive = w_raw / max(group_w_raw) * positive_group_max_weight
```

默认值：

- `q = 0.8`
- `tau = 0.08`
- `w_min = 0.1`
- `w_max = 2.0`
- `positive_group_max_weight = 2.0`
- 每个 group 只把 raw weight 按同一比例缩放，使组内最大 positive 权重等于 2.0；其他 positive
  权重保留原 sigmoid/rank 相对比例，不做 `[1,2]` 区间映射，因此可以低于 1.0。
- group 小于 `min_group_size` 时没有可靠 rank，默认所有样本回退 1.0；只有达到最小 group size 的
  positive 样本才执行 group rank weighting。可以显式配置 small-group policy 做后续 ablation。

只对 positive 样本做 group rank weighting。negative 样本和 condition dropout 样本默认 weight=1.0，
ignore 样本 weight=0.0。组内最优 positive 的 FM loss 相对 weight=1 的 negative/dropout 有 2 倍权重；
其他 positive 按其原始 rank weight 决定，可能高于或低于 1。不要再增加一个统一乘所有样本权重的
`advantage_weight_scale`：在 weighted mean 中统一缩放全部权重会被分母抵消。若要调整 positive 的
整体相对强度，应修改 `positive_group_max_weight`，且只作用于 positive group normalization。

### 3.8 classifier-free advantage conditioning

不要把 `positive/negative` 直接拼进原始 dataset 的 `task` 字段。processor 只做确定性文本拼接；
随机 dropout 必须由训练循环在调用 processor 之前生成，不能藏在同时用于部署的 processor 内：

```text
training loop: label + Bernoulli mask -> advantage_condition_kept
deterministic processor: kept label -> append "\nAdvantage: positive\n"
deploy/eval: no random dropout; missing dataset label 时使用 inference_advantage_label=positive
```

当 advantage text 被 dropout：

- 该样本 loss weight 必须回退到 1.0。
- 需要在 batch 里保留一个布尔 mask，例如 `advantage_condition_kept`，让训练权重 provider 知道哪些样本真的用了 condition。

训练和部署规则：

- train：只有 `positive|negative` label 才参与 Bernoulli keep；dropout 后不追加文本且 FM weight 回退 1.0。
- train ignore：不追加文本，weight=0.0；ignore 优先于 dropout fallback，不能被改回 1.0。
- eval/deploy：不执行随机 dropout，默认固定追加 `Advantage: positive`；提供
  `inference_advantage_label=positive|negative|none` 做 ablation。
- pi0 和 pi0.5 都必须接入同一个 deterministic advantage processor，但分别在各自 prompt/state-token
  processor 之前的正确位置拼接文本。

当前 pi0 的 subtask AR 训练是另一条文本/token 路径：`SubtaskTextProcessorStep` 会把 `subtask` 和 `subtask_progress` 构造成 `Subtask: ...; Progress: ...`，再由 tokenizer 生成 subtask tokens。advantage condition 只追加到主 task prompt，不应覆盖或复用 `subtask` 字段。两者可以同时存在：

```text
main task prompt: original task + optional "Advantage: positive"
subtask AR segment: "Subtask: ...; Progress: ..."
```

测试里必须覆盖 `predict_subtask=true` 且 `use_advantage_conditioning=true` 的组合。

## 4. 需要用户最终确认的问题

这些问题不阻塞第一版实现，因为都可以配置化。当前已确认的默认值如下：

1. value model 使用 `pi0_base` 还是 `pi0.5_base` 作为默认预训练权重？
   已确认第一版默认 `lerobot/pi0_base`；pi0.5 后续作为配置项补充。

2. value model 默认输入三路图片还是只用第三人称？
   已确认第一版默认三路：`left_wrist`、`right_wrist`、`third_person`；提供 `--image_keys` 可改成只用 `observation.images.third_person`。

3. subtask value 训练时是否使用文字 task/subtask 输入？
   已确认第一版不输入 task 文本，减少可记忆信息；subtask id 只作为监督 head，不作为输入条件。

4. elapsed auxiliary 的默认权重是多少？
   elapsed 指“从当前 episode/subtask 起点到当前帧经过了多少帧”。建议默认关闭，即 `elapsed_loss_weight=0.0`；做 ablation 时用 `0.25` 起步。

5. global/subtask scale 用 `max` 还是 `p95`？
   已确认训练默认 `p95`，advantage 计算永远用复原后的 frame units。超过 1 的 target 默认 clip，并记录 clip mask/clip rate。

6. positive 切分默认比例。
   这里的比例指 positive 比例。建议脚本默认 `top_percent=0.8`，即排序前 80% 标 positive，其余标 negative；必须支持 UI 手动覆盖后写入 label。

7. subtask 标注流程。
   继续使用现有 `lerobot_annotate_subtask.py`。标注完成并 export 出 `extras.parquet` 后，再运行本计划中的 value target/value train/value infer/advantage 处理脚本。

8. episode outcome 范围。
   已确认第一阶段所有 episode 都是成功 episode，不输入失败、超时或人工中止 episode。metadata 必须记录
   此数据假设；以后范围变化时重新设计 target。

9. subtask 结构约束。
   已确认每个 episode 包含全部 subtask，按固定顺序各出现一次，每个 subtask 只有一个连续 segment。
   target preparation 默认严格校验，不满足直接报错。

10. 跨 subtask 边界 transition。
    已确认每个相邻 subtask 边界 transition 计 `1.0` 单位 progress，然后统一减完整
    `valid_horizon`，不是在 centered advantage 之后额外加 bonus。

11. VLA 权重和部署条件。
    已确认 positive FM weight 保留 group 内 raw sigmoid/rank 比例，只按同一比例缩放到组内最大值 2；
    其他 positive 不设下限且可以低于 1。negative/dropout 为 1，ignore 为 0；weight 只作用于 FM loss。
    部署默认固定使用 `Advantage: positive`，不执行随机 dropout。

## 5. 总体任务顺序

### Milestone 0: 定义字段、metadata 和 raw extras 合并工具

目标：先把所有离线产物的字段、metadata、读写行为固定下来。

新增/修改文件：

- 新增 `src/lerobot/value_function/__init__.py`
- 新增 `src/lerobot/value_function/raw_io.py`
- 新增 `src/lerobot/value_function/schema.py`
- 新增测试 `tests/value_function/test_raw_value_io.py`

功能要求：

- 能读取 raw run root，枚举 episode、frame、图片路径、`frames.parquet`、`extras.parquet`。
- 能把新列 merge 到 episode 的 `extras.parquet`，保留已有 `subtask`、`subtask_progress` 和未来其他 extras 列。
- 能写 run-level metadata，例如 `value_function_meta.json`：
  - `pipeline_schema_version`
  - `all_episodes_successful`
  - value mode
  - bin count
  - global scale
  - subtask names/order
  - subtask scale dict
  - checkpoint path
  - image keys
  - created_at
- 每个 stage 的 metadata 记录独立 `created_at`、完整 config、输入列、输入 fingerprint、输出列和
  prediction source；重跑上游 stage 后应检测并拒绝继续使用 stale 下游 label/weight，或显式将其标为 invalid。
- `extras.parquet` 和 metadata 使用同目录临时文件写入、fsync/close 后 atomic replace；所有 episode
  预校验成功后才进入 commit 阶段，失败时不能留下部分 episode 新、部分 episode 旧的状态。
- 所有新增列长度必须等于 `frames.parquet` 行数。
- `lerobot_build_dataset.py` 无需大改；如果发现 list/float 类型 schema 不支持，再局部扩展。

测试标准：

- 构造临时 raw run，已有 `extras.parquet` 两列，写入 value 新列后旧列不丢。
- extras schema 在所有 episode 一致。
- `lerobot-build-dataset --dry_run` 能识别新增列。
- 模拟写入失败时原 `extras.parquet` 仍可读，且不会出现部分 episode 已更新。
- 重算 target/advantage 后旧下游 artifact 会被 provenance 检查识别为 stale。

### Milestone 1: 生成 value ground truth

目标：从 raw dataset 和 subtask 标注生成训练 value 所需的 GT 标签，不依赖 value model。

新增文件：

- `src/lerobot/scripts/lerobot_value_prepare_targets.py`
- `src/lerobot/value_function/targets.py`
- 测试 `tests/value_function/test_value_targets.py`

功能要求：

- 支持 `--root /path/to/raw/run`。
- 支持 `--mode global|subtask|both`。
- 支持 `--num_bins 256`，并支持 `--global_num_bins` / `--subtask_num_bins` 覆盖；bin 数写入 metadata，不能在模型或推理脚本里写死。
- 支持 `--global_scale max|p95|manual`。
- 支持 `--subtask_scale max|p95|manual`。
- 支持 `--elapsed_aux true|false`。
- 生成并写入 GT 列：
  - global remaining/elapsed frames + norm
  - subtask id/name/order
  - subtask remaining/elapsed frames + norm
- 对 p95 scale 下超过 1 的 norm target 做 clip，写 clip mask，并在 summary 里报告 clip rate。
- 对未标注 subtask 的 frame，默认报错；可选 `--allow_unlabeled skip|default_subtask|error`。
- 第一阶段默认 `--require_all_subtasks true`、`--require_single_segment_per_subtask true`：从显式
  canonical order 或第一条完整 episode 得到顺序，并要求每个 episode 的压缩 subtask sequence 完全
  相同，每个 subtask 恰好一个连续 segment。不一致时输出 episode、缺失/重复 subtask 和边界位置。
- metadata 写入 `all_episodes_successful=true` 和 canonical `subtask_order`。
- clip rate 使用实际 frame 计数做加权聚合；某 episode 不含某 subtask 时不得用 0 稀释该 subtask 统计。
- 校验 `num_bins/global_num_bins/subtask_num_bins >= 2`。
- `strike_match_3` 只能作为 smoke 数据和格式样例；测试不能假设 subtask 数固定为 6。

测试标准：

- 对长度为 10 的 episode，全局 remaining 应为 `[9,8,...,0]` 或按明确定义一致。
- 对 subtask segment `[0..3]`，remaining 应为 `[3,2,1,0]`。
- 归一化能被 scale 正确复原。
- 每个 episode 缺 subtask、subtask 重复出现、顺序不同都应报错。
- 不同长度 episode 的 aggregate clip rate 等于全体 clipped frames / 全体 eligible frames。
- 样例 raw dataset 上 dry run 能打印当前数据的 episode 数、subtask 数和每段长度统计；测试只能检查统计来自数据本身，不能 assert 固定为 70 个 episode 或 6 个 subtask。

### Milestone 1.5: synthetic value prediction smoke（仅测试）

目标：在真实 value model 完成前，用可复现的 GT+Gaussian noise 验证 predicted-value 字段、advantage、
label、weight 和 VLA batch 接口，避免把恒为 0 的 GT advantage 当成排序输入。

功能要求：

- 新增独立 CLI 或 `prepare-targets` 子命令，支持 `--seed`、`--noise_std_frames`，可选 temporal smoothing。
- 写独立的 `*_mock_pred` 列，绝不覆盖 model prediction canonical 列。
- metadata 标记 `prediction_source=synthetic_gt_gaussian_noise`、seed、sigma 和 source GT fingerprint。
- downstream label/weight export 默认拒绝 synthetic；测试时必须显式 `--allow_synthetic true`。
- completion 文档和 UI 显著显示 `SYNTHETIC / NOT FOR EXPERIMENT`。

测试标准：

- 相同 seed 输出完全一致，不同 seed 输出不同。
- sigma=0 时恢复 GT，centered advantage 接近 0。
- synthetic provenance 不能被误识别为 model prediction。

### Milestone 2: value model 架构

目标：实现可训练的 value model，支持全局 binned value 和 subtask 多头 binned value，bin 数可配置。

建议新增文件：

- `src/lerobot/value_function/configuration.py`
- `src/lerobot/value_function/modeling_pi0_value.py`
- `src/lerobot/value_function/dataset.py`
- `tests/value_function/test_value_model_shapes.py`

模型要求：

- 支持 `backbone_type=pi0|pi05|vision_only`。
- 第一版优先 `pi0`。
- 可从 `lerobot/pi0_base` 或本地 checkpoint 加载 PaliGemma/VLM 权重。
- 支持裁剪前 N 层 VLM，默认 `num_vlm_layers=3` 或 `8` 由显存决定。
- 支持 freeze vision encoder、freeze backbone、只训练最后 N 层和 heads。
- 支持 image keys 配置，默认三路图片。
- 明确支持 `observation.state`：默认启用可配置的 state encoder/projection，并与视觉/VLM pooled feature
  融合；提供 `use_state=false` 做 vision-only ablation。checkpoint 保存 state key、维度和 normalization stats。
- 不默认输入 task 文本；如复用 PaliGemma text path，可喂固定空 prompt 或固定 task prompt，但配置里要能关闭。
- 默认 `pretrained_path=lerobot/pi0_base`。

head 要求：

- global：
  - `remaining_logits: [B, num_bins]`
  - optional `elapsed_logits: [B, num_bins]`
- subtask：
  - `subtask_logits: [B, num_subtasks]`
  - `remaining_logits: [B, num_subtasks, num_bins]`
  - optional `elapsed_logits: [B, num_subtasks, num_bins]`
- `num_subtasks` 从当前 raw run 的标注配置/数据中解析；不能按 `strike_match_3` 的 6 个 subtask 写死。
- 提供统一 helper 根据一条 subtask id path gather `[B, num_subtasks, num_bins]` 中对应 head；
  `gt_conditioned` 和 `pred_smooth` 必须调用同一 helper，且输出字段带 head source。

loss 要求：

- two-hot CE / KL loss。
- global remaining loss。
- optional elapsed auxiliary loss。
- subtask CE loss。
- subtask remaining head 只对 GT subtask index gather 后算 loss。
- 所有 loss weight 可配置：
  - `remaining_loss_weight=1.0`
  - `subtask_ce_loss_weight=0.2`
  - `elapsed_loss_weight=0.0` 默认关闭

测试标准：

- 随机 fake batch 能 forward。
- global 输出 shape 正确。
- subtask 输出 shape 随 `num_subtasks` 变化正确。
- `num_bins=64/128/256/512` 都能构建模型并 forward。
- two-hot target sum 为 1。
- expectation decode 输出在 `[0, 1]`。
- subtask gather loss 不更新非当前 head 的 loss target。
- `gt_conditioned` gather 与 GT id 一致，`pred_smooth` gather 与 smoothed id 一致，禁止交叉组合。
- `use_state=true|false` 都能 forward，state normalization 能随 checkpoint round-trip。

### Milestone 3: value 训练脚本

目标：可独立训练 value model，与 VLA 训练解耦。

新增文件：

- `src/lerobot/scripts/lerobot_train_value_function.py`
- `tests/value_function/test_train_value_smoke.py`

功能要求：

- 输入 raw run root 或多个 raw run root。
- 自动读取 images、state、GT value columns、subtask columns。
- 多 root 默认要求 robot type、fps、image/state schema、canonical subtask order 和 target scale metadata
  兼容；不兼容时明确报错。若以后允许 remap，必须提供显式 name mapping，不能按整数 id 静默合并。
- 启动训练前验证每个 root 都声明 `all_episodes_successful=true`。
- train/val split 支持：
  - 按 episode split，默认 90/10。
  - 指定 `--val_episodes`。
- 支持 augmentation：
  - 默认轻量 image augmentation，包括 random resized crop / small rotation / brightness-contrast-saturation jitter。
  - 可选 gaussian blur、random grayscale、轻微 noise。
  - augmentation 参数必须可配置，并且默认不要强到破坏机器人操作语义。
  - 不应只依赖 blur 防记忆；episode-level validation、限制可训练层数、subtask 局部目标同样重要。
- 支持 checkpoint 保存：
  - `checkpoint.pt`
  - `config.json`
  - `value_function_meta.json`
  - `train_metrics.jsonl`
- 支持 quick smoke：
  - `--max_steps 2`
  - `--batch_size 2`
  - CPU 可跑 shape，不要求真实收敛。

监控指标：

- train/val remaining CE
- decoded MAE in normalized units
- decoded MAE in frame units
- subtask classification accuracy
- elapsed MAE，如果开启
- 同一 subtask/progress bin 内预测方差
- episode 内 value 单调性违规比例
- p95 target clip rate，全局和每个 subtask 分开统计

测试标准：

- 在一个小 fake raw run 上 2 step smoke 通过。
- 在真实样例 raw run 上能成功构建 dataset 并跑一个 forward batch。
- checkpoint 能重新 load 并输出一致 shape。

### Milestone 4: value 推理并写回 raw dataset

目标：用训练好的 value model 跑完整 raw run，把每帧预测写进 `extras.parquet`。

新增文件：

- `src/lerobot/scripts/lerobot_value_infer.py`
- `src/lerobot/value_function/inference.py`
- `tests/value_function/test_value_infer_writeback.py`

功能要求：

- 输入：
  - `--root`
  - `--checkpoint`
  - `--mode global|subtask|both`
  - `--batch_size`
  - `--image_keys`
  - `--subtask_inference_path gt_conditioned|pred_smooth|both`
- 输出：
  - global norm pred + frame pred
  - subtask raw id pred + confidence
  - subtask smoothed id/name
  - GT-head-selected subtask norm/frame prediction，字段带 `_pred_gt_head`
  - smoothed-head-selected subtask norm/frame prediction，字段带 `_pred_smooth_head`
  - optional entropy/top1 debug columns
- subtask smoothing 使用单调 Viterbi。
- 即使 dataset 已经有 GT subtask，也必须对所有 frame 写入 model prediction；GT 只作为训练监督和 debug 对照。
- 如果有 GT subtask，默认同时写两个 paired inference path 便于对照。advantage 通过单一
  `--subtask_inference_path` 选择 path，不能独立混选 boundary source 和 head source；value 数值默认始终
  使用 model prediction。
- checkpoint 与 raw run 的 subtask order、scale、image/state schema 不兼容时拒绝推理，除非有显式 remap。

测试标准：

- 写回后所有 episode extras schema 一致。
- 预测列长度等于 frames 行数。
- smoothed subtask id 单调不降。
- 每个 paired path 的 head id 与 boundary id 完全一致。
- `lerobot-build-dataset --dry_run` 可以识别预测列。

### Milestone 5: value 曲线展示 UI

目标：提供一个 web UI 看 episode 图片和 value 曲线，辅助比较全局/subtask/elapsed。

新增文件：

- `src/lerobot/scripts/lerobot_value_viz.py`
- `src/lerobot/scripts/value_viz/index.html`
- `src/lerobot/scripts/value_viz/app.js`
- `src/lerobot/scripts/value_viz/style.css`

UI 要求：

- 选择 raw root。
- 左侧 episode 列表。
- 主区域显示当前 frame 图片，默认第三人称，可切换三路图。
- 下方/右侧坐标轴：
  - 横轴 frame
  - 纵轴 value
  - 当前 frame 竖线
  - 当前点高亮
- 支持曲线：
  - global GT remaining norm
  - global pred remaining norm
  - subtask norm value
  - subtask frame-unit remaining value
  - 可选展示按边界切分后的 chunk segment
- subtask 显示：
  - 背景区间按 subtask 着色
  - 可切换 GT subtask / predicted smoothed subtask
- 支持键盘左右移动 frame。

测试标准：

- 对样例 raw run 启动本地服务。
- 随机 episode 能加载图片和曲线。
- 当前 frame 切换时图片、竖线和数值同步。
- 缺少某条 value 列时 UI 不崩，显示 unavailable。

### Milestone 6: action chunk advantage 计算

目标：根据 value 预测或 GT value，为每个 frame/action chunk 写入 advantage。

新增文件：

- `src/lerobot/scripts/lerobot_compute_advantage.py`
- `src/lerobot/value_function/advantage.py`
- `tests/value_function/test_advantage.py`

功能要求：

- 支持 `--value_mode global|subtask`。
- 支持 `--value_source gt|mock_pred|model_pred`；正式 label export 只接受 `model_pred`，其余仅用于 smoke。
- 支持 `--chunk_size 50`，默认从 pi0 config 或 CLI 来。
- 支持末尾 padding 处理：
  - `valid_horizon = min(chunk_size, episode_len - 1 - frame_index)`
  - 如果 `valid_horizon <= 0`，advantage 可设为 0，并标记 `advantage_is_valid=false`。
- global:
  - `advantage = V_remaining_start - V_remaining_end - valid_horizon`
- subtask:
  - 默认按 subtask 边界切分 chunk，只在同一 subtask head 内计算 relative remaining 差值。
  - 不使用 `completed_subtask_scale_sum` 或其他跨 subtask 绝对累计尺度。
  - 支持 `--subtask_inference_path gt_conditioned|pred_smooth`，同时选择 paired boundary/head source。
  - 支持 `--boundary_transition_value`，默认 `1.0`；每个 boundary transition 计入 total progress，
    centered advantage 最后统一减完整 valid horizon。
- 写入：
  - `advantage_{mode}_chunk`
  - `advantage_{mode}_valid_horizon`
  - `advantage_{mode}_is_valid`
  - `advantage_{mode}_start_value`
  - `advantage_{mode}_end_value`
  - `advantage_{mode}_num_crossings`
  - `advantage_{mode}_within_subtask_horizon`
  - `advantage_{mode}_boundary_progress`

测试标准：

- 构造线性理想 episode：如果每帧推进 1，advantage 应接近 0。
- 构造卡住片段：end value 未下降足够时 advantage 为负。
- 跨 subtask chunk 被拆成多个同-subtask segment，每段只使用自己的 subtask head。
- 理想线性 episode 跨一个或多个边界时 advantage 仍为 0。
- `gt_conditioned` 只能读取 GT-head prediction + GT boundary；`pred_smooth` 只能读取 smooth-head
  prediction + smooth boundary，交叉组合必须报错。
- 改变 subtask scale 不应影响同一 subtask segment 内的 relative advantage，除非预测 frame-unit value 本身改变。
- 末尾 padding 不把重复最后一帧算作 50 帧真实推进。

### Milestone 7: advantage 排序、positive/negative 标注 UI

目标：可视化 action chunks，按 advantage 排序，选择切分比例并写入 label。

新增文件：

- `src/lerobot/scripts/lerobot_advantage_labeler.py`
- `src/lerobot/scripts/advantage_labeler/index.html`
- `src/lerobot/scripts/advantage_labeler/app.js`
- `src/lerobot/scripts/advantage_labeler/style.css`

UI 要求：

- 左侧 action chunk 列表，项名：
  - `ep_000012 frame_000345 adv=-3.21`
- 支持按 advantage 从高到低/低到高排序。
- 展示排序方向不得改变 positive 的语义：positive 默认永远表示高 advantage。若要实验性选择低
  advantage，使用单独且醒目的 `positive_direction=high|low`，不能复用 UI sort order。
- 列表显示 10%、20%、30%、80% 等比例位置 marker。
- 选中 chunk 后，右侧是一个 action chunk 顺序播放器，默认展示第三人称单帧大图；按左右键或点击 step 控件逐帧查看 chunk 内图片。
- 可选显示缩略图时间条，但不要把所有 frame 拼成一个大图作为主展示。
- 显示：
  - episode
  - start frame
  - 当前 chunk 内 offset
  - valid horizon
  - advantage
  - current label
- 支持设置阈值：
  - 例如 top 80% -> positive，其余 negative。
- 支持手动覆盖单个 chunk。
- 手动 override 必须持久化并可在重启 UI 后恢复；UI 同时显示 stored label、threshold preview 和
  override source，不能静默覆盖旧人工结果。
- 对大 run 使用分页/虚拟列表、服务端 chunk cache 和 slider debounce；禁止每次选帧重建全部 DOM。
- 提供无浏览器 export 模式，例如 `--export --top_percent 0.8 --dry_run`，供端到端 smoke 使用。
- 检查 advantage provenance；GT/mock 默认拒绝 export，只有显式 `--allow_synthetic true` 可测试。
- 写回：
  - `advantage_label_{mode}`，string: `positive|negative|ignore`

测试标准：

- 样例 raw run 可启动并列出 chunks。
- 调整阈值后 preview label 数量正确。
- export 后每个 episode `extras.parquet` 增加 label 列。
- label 列长度等于 frame 数；靠近 episode 尾部无效 chunk 可标 `ignore`。
- wheel/package 安装后 HTML/JS/CSS 资源存在并能返回 200。
- 50k chunks 规模下列表使用分页/虚拟化，不创建 50k DOM 节点；切换 frame 不重新加载全部 parquet。
- 重启 server 后已有 overrides 和 stored labels 可恢复。
- `sort_order=asc|desc` 只改变展示顺序，不改变同一 threshold 下导出的 label。

### Milestone 8: group-relative loss weight 计算和展示

目标：按相似 value/progress group 内的 rank 生成 loss weight，并提供可视化检查 group 粒度。

新增文件：

- `src/lerobot/scripts/lerobot_compute_advantage_weights.py`
- `src/lerobot/scripts/advantage_weight_viz/index.html`
- `src/lerobot/scripts/advantage_weight_viz/app.js`
- `src/lerobot/scripts/advantage_weight_viz/style.css`
- `tests/value_function/test_advantage_weights.py`

分组要求：

- global:
  - 默认按 `value_global_remaining_norm_pred` 或 GT norm 的 bin 分组。
  - `--group_bin_width` 默认 `0.05`。
- subtask:
  - `group = subtask_id + floor(subtask_progress_or_value / bin_width)`。
  - 默认 `--group_bin_width 0.1`。
- 只对 `advantage_label=positive` 的样本在 group 内排序。
- group 太小默认跳过 rank weighting：
  - `--min_group_size 4`
  - 小 group weight 设为 1.0。

权重要求：

- 使用 sigmoid rank 公式生成 `w_raw`，再按
  `w_raw / group_max(w_raw) * positive_group_max_weight` 缩放；默认
  `w_min=0.1`、`w_max=2.0`、`positive_group_max_weight=2.0`。
- 达到 `min_group_size` 的 group 内只保证最大 positive 权重为 2.0，其他 positive 保留原始相对
  比例并允许低于 1.0；小 group 默认全部回退 1.0。
- 最终列：
  - `advantage_group_id_{mode}`
  - `advantage_loss_weight_{mode}`
- negative 样本默认 weight `1.0`；condition dropout 在训练时回退 `1.0`。
- ignore 样本 weight `0.0` 或训练时过滤；第一版建议 `0.0` 并保留 label。

展示 UI 要求：

- 展示横轴 value/progress bins。
- 每个 bin 下列出 action chunks。
- 支持点选 chunk 后用顺序播放器逐帧查看第三人称图片序列。
- 显示 group 内 advantage rank 和 weight。

测试标准：

- 构造 group 中 advantage 降序，positive 权重应单调不增，且组内最大值为 2.0。
- 除最大值外的 positive 权重与 `w_raw` 比例一致，不强制大于等于 1；negative 为 1，ignore 为 0。
- `ignore` 样本不进入 positive rank。
- UI 能显示 group 分布，缺图时返回明确错误。

### Milestone 9: raw -> LeRobotDataset 集成验证

目标：确认新增字段能进入最终 LeRobotDataset，训练 batch 能拿到。

涉及文件：

- `src/lerobot/scripts/lerobot_build_dataset.py`
- `src/lerobot/datasets/factory.py`
- `tests/scripts/test_value_extras_build_dataset.py`

功能要求：

- 尽量不改 builder；利用现有 extras 合并。
- 如果新增 bool/list 字段有 schema 问题，再补齐 `_load_extras_schema` 类型映射。
- 确认最终 dataset sample 中包含：
  - `advantage_label_global` 或 `advantage_label_subtask`
  - `advantage_loss_weight_global` 或 `advantage_loss_weight_subtask`
  - value debug columns，可选 include/exclude

测试标准：

- 用小 raw run build 一个本地 LeRobotDataset。
- `dataset[0]` 可以读到 advantage label/weight。
- 真实 DataLoader + policy preprocessor batch 能保留 label、weight 和 condition mask；数值 extras 从
  `[B, 1]` 规范化为严格 `[B]`，禁止与 per-sample loss `[B]` 广播成 `[B, B]`。
- `policy.action_delta_indices` 仍会让 `action` 取 chunk。
- 不要求 value 列也取 chunk；advantage 是以 start frame 为单位存储，训练时读 start frame 的权重即可。

### Milestone 10: VLA advantage conditioning processor

目标：训练时可选地把 positive/negative 作为 prompt 条件，并支持 classifier-free dropout。

新增/修改文件：

- 新增 `src/lerobot/processor/advantage_processor.py`
- 修改 `src/lerobot/processor/__init__.py`
- 修改 `src/lerobot/processor/converters.py`
- 修改 `src/lerobot/policies/pi0/processor_pi0.py`
- 修改 `src/lerobot/policies/pi0/configuration_pi0.py`
- 修改 `src/lerobot/policies/pi05/processor_pi05.py`
- 修改 `src/lerobot/policies/pi05/configuration_pi05.py`
- 测试 `tests/processor/test_advantage_processor.py`

配置建议：

```python
use_advantage_conditioning: bool = False
advantage_label_key: str = "advantage_label_global"
advantage_loss_weight_key: str = "advantage_loss_weight_global"
advantage_condition_format: str = "Advantage: {label}"
inference_advantage_label: str = "positive"
```

processor 行为：

- 如果 `use_advantage_conditioning=false`，完全保持原训练行为。
- 如果 label 是 `positive|negative`，把文本追加到 task prompt。
- 如果 label 是 `ignore` 或空，默认不追加。
- processor 不自行随机 dropout；训练循环预先生成 `advantage_condition_kept`，processor 只按 mask
  确定性拼接。
- 输出/保留 `advantage_condition_kept`，供 loss weighting 使用。
- 如果 `predict_subtask=true`，advantage processor 仍只处理 `task`，不能改写 `subtask` 或 `subtask_progress`。

测试标准：

- positive label 会生成包含 `Advantage: positive` 的 prompt。
- dropout=1.0 时 prompt 不包含 advantage。
- dropout=0.0 时全部保留。
- 原始 subtask processor 和 tokenizer 仍能正常工作。
- `predict_subtask=true` 时，subtask tokens 仍由 `subtask` / `subtask_progress` 生成，不被 advantage label 覆盖。
- pi0 和 pi05 processor 都覆盖，并验证各自 tokenizer/state prompt 顺序。
- eval/deploy 缺少 dataset label 时固定生成 `Advantage: positive`；`inference_advantage_label=none`
  时生成无条件 prompt。
- 长 task 接近 tokenizer 上限时，测试必须确认 advantage condition token 没有被静默截断；必要时预留
  token budget 或把 condition 放到不会被截断的位置。

### Milestone 11: VLA loss weighting provider

目标：训练时读取 batch 内预存的 loss weight，而不是只支持旧 SARM progress parquet。

新增/修改文件：

- 新增 `src/lerobot/utils/advantage_weights.py`
- 修改 `src/lerobot/scripts/lerobot_train.py`
- 修改 `src/lerobot/configs/train.py`
- 可选保留 `src/lerobot/utils/rabc.py` 不动，避免破坏旧路径。
- 测试 `tests/utils/test_advantage_weights.py`

配置建议：

```python
use_advantage_weighting: bool = False
advantage_loss_weight_key: str = "advantage_loss_weight_global"
advantage_label_key: str = "advantage_label_global"
advantage_condition_dropout_prob: float = 0.1
advantage_ignore_label: str = "ignore"
advantage_disable_weight_when_condition_dropped: bool = True
```

训练行为：

- 当 `use_advantage_weighting=false`，原训练逻辑不变。
- 当开启时：
  - 训练循环在 processor 前按 dropout prob 生成 `advantage_condition_kept`，eval/deploy 不走此逻辑。
  - policy 暴露 per-sample FM loss；subtask CE 继续普通 mean，不乘 advantage weight。
  - 从 batch 读 `advantage_loss_weight_key`。
  - 如果 `advantage_condition_kept=false` 且配置要求禁用，weight 设为 1.0。
  - label 为 `ignore` 时 weight 设为 0.0，或由配置选择过滤。
  - `fm_loss = sum(weight * per_sample_fm_loss) / (sum(weight) + eps)`，这里的 sum 和分母都是当前
    batch（以及分布式训练所需的等价全局归约）上的 weighted mean。
  - `loss = fm_loss + subtask_ce_loss_weight * mean(per_sample_subtask_ce)`。
  - positive 使用离线生成的 group-relative 权重（每组最大值 2，其他值可低于 1），negative/dropout
    为 1；各样本按实际权重影响混合 batch。只有把所有样本权重统一乘同一常数时才会被分母抵消。
  - batch 全部为 ignore 时不得产生 NaN；应过滤 ignore sample、跳过 FM update 或使用明确 fallback。
- 现有 `use_rabc` 和新 `use_advantage_weighting` 不能同时开启，除非明确实现组合逻辑。第一版应互斥并报错。

测试标准：

- fake policy 返回 per-sample loss，weight 后 scalar loss 符合预期。
- 某个 positive=2、negative=1 的混合 batch 中，该 positive FM 对结果的贡献是 negative 的两倍；
  另一个权重低于 1 的 positive 必须按其实际权重贡献。
- subtask CE 不随 advantage weight 改变。
- dropout mask 生效时 weight 回退。
- `use_rabc` 原测试不受影响。

### Milestone 12: 端到端 smoke 和实验矩阵

目标：确认从 raw 标注到 VLA batch 的完整路径可跑。

建议新增脚本：

- `scripts/value_pipeline_smoke.sh` 或文档化命令集合。

端到端流程：

1. `lerobot-value-prepare-targets`
2. `lerobot-train-value-function --max_steps 2`
3. `lerobot-value-infer`
4. `lerobot-compute-advantage`
5. `lerobot-advantage-labeler` 手动或 CLI 阈值 export
6. `lerobot-compute-advantage-weights`
7. `lerobot-build-dataset --dry_run`
8. `lerobot-build-dataset`
9. `lerobot-train --steps 2 --policy.type pi0 --use_advantage_conditioning true --use_advantage_weighting true`

在 value model 完成前，步骤 4-9 只能读取带明确 synthetic provenance 的 mock prediction，并仅作为
接口 smoke。正式实验必须重新从步骤 3 的 model prediction 开始生成 advantage、label、weight 和
LeRobotDataset；不得复用 synthetic label/weight。

第一批实验矩阵：

- global value baseline
- global value + elapsed auxiliary
- subtask value
- subtask value + elapsed auxiliary
- subtask value + group-relative weighting
- subtask value + group-relative weighting + classifier-free advantage conditioning

核心指标：

- value val MAE in frames
- episode 内 value 单调性
- 同类 subtask/progress bin 内 value 方差
- advantage ranking 与真实推进效率的相关性
- positive/negative 分布是否覆盖所有 subtask
- VLA 训练 loss 是否稳定
- rollout 成功率、耗时、卡住率

## 6. 实施顺序建议

推荐按下面顺序做：

1. 返修已完成 Milestone 0/1/6/7：atomic/provenance、严格 subtask 约束、新边界公式、UI/package。
2. Milestone 1.5：生成带 provenance 的 GT+Gaussian-noise mock prediction。
3. Milestone 8/9/10/11：用 mock prediction 跑通 weight、真实 DataLoader batch、pi0/pi05 prompt、
   train-only dropout 和 FM-only weighting；此阶段只算接口 smoke。
4. Milestone 2/3/4：实现、训练并推理真实 value model，产出 paired `gt_conditioned` / `pred_smooth`
   model prediction。
5. Milestone 5：value/subtask 曲线和两条 inference path 对照 UI。
6. 重新运行 Milestone 6/7/8：只用 model prediction 生成正式 advantage、label 和 weight。
7. Milestone 12：重新 build dataset，分别对 pi0、pi05 做 2-step smoke，再开始正式实验矩阵。

这个顺序仍然允许在 value model 完成前隔离验证 VLA 数据流，但 synthetic artifact 与正式 artifact 有
强 provenance 隔离，不会把恒为零的 GT advantage 或 mock label 误用于实验。

## 7. 推荐命令形态

以下是后续实现完成后的目标命令形态，不要求现在可运行。

准备 target：

```bash
conda activate lerobot-main
lerobot-value-prepare-targets \
  --root /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3 \
  --mode both \
  --num_bins 256 \
  --global_scale p95 \
  --subtask_scale p95
```

训练 value：

```bash
lerobot-train-value-function \
  --root /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3 \
  --mode subtask \
  --image_keys observation.images.left_wrist observation.images.right_wrist observation.images.third_person \
  --pretrained_path lerobot/pi0_base \
  --num_bins 256 \
  --elapsed_loss_weight 0.0 \
  --output_dir outputs/value/strike_match_3_subtask_pi0
```

推理写回：

```bash
lerobot-value-infer \
  --root /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3 \
  --checkpoint outputs/value/strike_match_3_subtask_pi0/checkpoint.pt \
  --mode subtask \
  --subtask_inference_path both
```

计算 advantage：

```bash
lerobot-compute-advantage \
  --root /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3 \
  --value_mode subtask \
  --value_source model_pred \
  --subtask_inference_path gt_conditioned \
  --boundary_transition_value 1.0 \
  --chunk_size 50
```

计算权重：

```bash
lerobot-compute-advantage-weights \
  --root /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3 \
  --value_mode subtask \
  --group_bin_width 0.1 \
  --q 0.8 \
  --tau 0.08 \
  --w_min 0.1 \
  --w_max 2.0 \
  --positive_group_max_weight 2.0 \
  --negative_weight 1.0
```

构建 dataset：

```bash
lerobot-build-dataset \
  --runs /home/zenbot-robot/.cache/huggingface/lerobot/raw/ming326/strike_match_3 \
  --output_repo_id ming326/strike_match_3_value \
  --video true \
  --push_to_hub false \
  --force true
```

VLA 训练：

```bash
lerobot-train \
  --dataset.repo_id ming326/strike_match_3_value \
  --policy.path /path/to/pi0_or_pi05_checkpoint \
  --policy.use_advantage_conditioning true \
  --policy.advantage_label_key advantage_label_subtask \
  --use_advantage_weighting true \
  --advantage_condition_dropout_prob 0.1 \
  --advantage_loss_weight_key advantage_loss_weight_subtask \
  --advantage_label_key advantage_label_subtask
```

## 8. 风险和注意事项

1. 不要把 normalized value 当成 advantage 单位。
   normalized 只服务训练；advantage 必须用 frame units 或连续 potential。

2. 不要忽略 episode 尾部 padding。
   action chunk 会 pad 最后一帧，但 advantage 计算要用真实 valid horizon。

3. 不要只按 value 分组。
   value 接近不一定状态相似。第一版至少加 subtask id；后续可加视觉 embedding 近邻。

4. 不要在 dataset 里硬改 task prompt。
   advantage 条件应在 processor 层动态拼接，方便 dropout 和 ablation。

5. 不要让 `use_rabc` 和新 advantage weighting 同时静默生效。
   第一版互斥，后续再考虑组合。

6. subtask 边界处 value 不连续是正常风险。
   第一版不直接比较边界两侧不连续的 head value，而是同 subtask 内做 relative difference，并将每个
   boundary transition 单独记为 1 单位 progress。UI 必须分别展示 within-subtask progress、crossing
   数和 boundary progress，不能把不同 head 的 start/end value 当成一条连续绝对曲线解释。

7. value model 太强会记忆 episode。
   真正的缓解手段是目标定义、subtask 局部化、episode-level validation、augmentation、冻结/限制 backbone，而不是只加 blur。

8. raw `extras.parquet` 是中心集成点。
   所有离线产物尽量写 extras，不要改 `frames.parquet`，减少对采集脚本和 build 脚本的影响。

9. 上游重算会让下游 artifact 过期。
   每个 stage 必须记录 input fingerprint；发现 value/advantage 已变化时，旧 label/weight/dataset build
   不得静默继续使用。

10. 所有本地 web UI 都必须随 wheel 打包静态资源，并对 50k+ frame/chunk 使用分页或虚拟化。

## 9. 文档完成标准

后续 agent 在开始实现前，应先确认：

- 已读本计划中第 2 节列出的关键文件。
- 已用 `lerobot-main` 环境检查样例 raw dataset。
- 已决定本次 milestone 的输入/输出字段名。
- 已为本 milestone 添加最小测试。
- 不跨 milestone 做大范围重构。

每完成一个 milestone，需要在 `plans/` 下追加一个简短完成记录，写明：

- 改了哪些文件。
- 新增了哪些 CLI。
- 新增了哪些字段。
- 跑了哪些测试。
- 哪些风险仍然存在。
