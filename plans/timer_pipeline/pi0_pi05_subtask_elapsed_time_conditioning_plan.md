# PI0 / PI0.5 Subtask Elapsed-Time 条件训练与 Nero RTC 部署计划

> 面向读者：后续负责实现、测试、checkpoint 验证和 Nero 实机验收的 agent。
>
> 仓库：`/home/zenbot-robot/repos/lerobot`
>
> 基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655`（`value_pipeline`）。
>
> 现有增量基线：`plans/memory_pipeline/pi0_pi05_memory_training_deployment_plan.md` 及其 Milestone 0–8 已实现的工作区改动。
>
> 日期：2026-07-17。
>
> 本文是新的增量实施任务书，不是完成记录。任何阶段只有在对应验收真实通过后才能标记完成。

---

## 0. 文档用途和实施原则

### 0.1 本文要解决的问题

当前 PI0 / PI0.5 memory 只把上一轮模型生成的完整 subtask/progress 文本作为下一轮输入：

```text
Memory: Subtask: {previous_subtask}; Progress: {previous_progress}
```

这个条件能告诉模型“上一轮说了什么”，但不能告诉模型“当前 subtask 已执行多久”。尤其在以下情况中，
progress 文本和一轮历史无法表达真实时间：

- 相同 subtask 在不同 episode 中持续时间差异很大；
- RTC 推理间隔、动作队列和模型延迟并非严格固定；
- 长动作（例如煎蛋、搅拌）需要持续数十秒；
- 模型可能在 subtask 边界附近交替生成前后两个 subtask；
- 部署时需要一个独立于推理次数、按真实经过时间前进的条件。

本任务新增一个可选、文本形式的当前 subtask elapsed-time condition。训练时从逐帧 subtask 连续段和数据集
FPS 计算真实秒数，加入相对随机噪声和独立 dropout；部署时从标注数据集提取固定 subtask 顺序，使用只前进、
不后退、不跳级的状态机驱动可暂停的 monotonic timer。

### 0.2 最终目标

启用功能后，模型主 prompt 可以同时或分别包含：

```text
Memory: Subtask: Pick up the fork.; Progress: 0.8
Subtask elapsed time: 1.2s
```

两个条件彼此独立：

- history memory 描述上一轮完整生成输出；
- subtask elapsed time 描述部署状态机当前已确认 subtask 的有效运行时间；
- 任一功能关闭时，另一功能仍可工作；
- 两者都关闭时维持当前默认路径，不改变已有 checkpoint 行为。

### 0.3 硬性范围

只要求以下组合：

- policy：`pi0`、`pi05`；
- subtask：必须启用当前已有的 subtask AR；
- robot：`bi_nero_follower`，以及其 Nero 单臂实现所需兼容路径；
- 数据采集背景：Pico 双臂遥操作 `bi_pico_nero_teleop`；
- 部署：`lerobot_policy_deploy.py` + `RTCInferenceEngine`；
- 数据：带逐帧 `subtask`、`subtask_progress`、episode/frame/index 的非 streaming LeRobotDataset；
- 采集数据当前为 30 FPS，但实现必须读取 `dataset.meta.fps`，不能硬编码 30；
- 已有 history memory、subtask AR、advantage conditioning/weighting、relative actions 和 RTC 行为必须兼容。

不要求：

- PI0-FAST、SmolVLA、SARM、Wall-X、XVLA、ACT 等其他模型接入；
- 新增数值 embedding、时间编码 head、额外 loss 或时间预测 target；
- 修改 Pico 手柄映射、IK、遥操作状态机或 Nero 采集频率；
- 把 elapsed time 永久写回 raw `extras.parquet` 或已生成 LeRobotDataset；
- 支持 streaming dataset 或多数据集训练；
- 模糊字符串相似度、LLM 分类器或向量检索式 subtask 匹配；
- 允许部署 subtask 回退、跳级或任意重排；
- 跨 episode、跨任务或跨进程保存 timer。

其他 policy 不需要支持新字段，但时间功能默认关闭时不得破坏公共 import、配置加载、processor registry、
Dataset factory 或通用训练路径。

### 0.4 工作区保护

本文建立时工作区包含尚未提交的 history-memory 全套实现、计划目录移动以及 Nero 相关用户改动。实施 agent 必须：

- 把当前工作区视为新功能的真实基线；
- 不执行 `git reset --hard`、`git checkout --` 或其他覆盖用户改动的命令；
- 不回退 `plans/memory_pipeline` 的 Milestone 0–8 文件；
- 不顺手格式化或改写无关的 Nero/Pico 文件；
- 每个 milestone 开始和完成时记录 `git status --short`；
- 若同一文件存在无关用户修改，只做局部 patch 并在完成记录中说明。

### 0.5 开发环境

优先环境：

```bash
conda activate lerobot-main
```

推荐用以下形式运行会同时 import PyTorch、sqlite/ICU 或 CUDA 的命令：

```bash
conda run --no-capture-output -n lerobot-main <command>
```

本机代理端口为 `127.0.0.1:1080`。只有本地 cache 缺失且确实要联网时，才在当前 shell 临时设置：

```bash
export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080
```

不得把代理写入 checkpoint、dataset metadata、通用配置或用户全局 shell 文件。现有数据、tokenizer 和 checkpoint
可离线使用时应设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。

---

## 1. 已确认的产品语义

以下决策已经由用户确认，实施 agent 不得自行替换成另一套含义。

1. 时间使用文本 prompt，不新增数值 embedding。
2. 时间行与 history `Memory:` 行分开，canonical 格式为：

   ```text
   Subtask elapsed time: 1.2s
   ```

3. 时间保留一位小数。
4. 当前帧真实时间为：

   ```text
   x = (current_frame_index - current_subtask_segment_start_frame) / dataset.fps
   ```

5. 连续 subtask 段第一帧为 `0.0s`，不是 `1 / fps`。
6. 训练噪声为相对均匀噪声，比例 `0.4`，最大幅度 `5.0s`：

   ```text
   a = min(0.4 * x, 5.0)
   epsilon ~ Uniform(-a, +a)
   noisy = max(0.0, x + epsilon)
   ```

7. 时间 condition 的训练 dropout 默认 `0.2`，并与 history memory、current-subtask attention、advantage
   condition dropout 独立采样。
8. elapsed-time condition 与 history memory 独立开关，但 elapsed time 必须依赖 `predict_subtask=true`；部署时还
   必须能生成 subtask。
9. checkpoint 保存时间能力；deploy 默认跟随 checkpoint，只允许关闭已训练的时间 condition 做 ablation，不能给
   没有对应 processor 的旧 checkpoint 强行开启。
10. 部署从指定标注 LeRobotDataset 扫描逐帧 `subtask` 列提取顺序，不在代码中硬编码 egg/match 文本。
11. 部署状态机只接受当前 subtask 或紧邻的下一个 subtask；永不后退、不允许跳级。
12. 紧邻下一个 subtask 在一次成功 RTC transaction 中出现一次即推进，不做多轮投票或 debounce。
13. 未知输出、解析失败、旧 subtask 或跨级 subtask 不改变状态和 timer。
14. 部署输入按每个 subtask 在训练数据中的最大真实持续时间加 `5.0s` 截断。
15. timer 基于 monotonic clock，与推理次数无关；模型每轮推理开始时读取一次时间快照。
16. 普通 pause 冻结并保留已确认 subtask 和累计 elapsed time，暂停期间不累计；resume 从冻结值继续。
17. Home、显式 full reset、新任务开始和进程重启彻底清空 subtask/time 状态。
18. 本任务不修改 raw 或 converted dataset schema；训练期通过轻量列扫描预计算 lookup。

### 1.1 对“噪声 0.4”的精确定义

本文把用户确认的 `0.4` 固定解释为相对噪声比例，而不是固定 `0.4s`：

```python
amplitude = min(0.4 * true_elapsed_seconds, 5.0)
noise = uniform(-amplitude, amplitude)
```

例子：

| 真实时间 x | 噪声幅度 a | 加噪后理论区间 | clamp 后区间 |
|---:|---:|---:|---:|
| 0.0s | 0.0s | [0.0, 0.0] | [0.0, 0.0] |
| 1.0s | 0.4s | [0.6, 1.4] | [0.6, 1.4] |
| 5.0s | 2.0s | [3.0, 7.0] | [3.0, 7.0] |
| 12.5s | 5.0s | [7.5, 17.5] | [7.5, 17.5] |
| 40.5s | 5.0s | [35.5, 45.5] | [35.5, 45.5] |

只有训练加噪；部署 timer 本身不加随机噪声。

### 1.2 Pause 的最终语义

普通 Space pause 与 full reset 必须分开：

- Space pause：停止推理、清动作队列并冻结 timer；保留 `current_subtask_index` 和累计 active elapsed；
- Right resume：恢复 timer，暂停时长不计入 elapsed；
- Home：full reset，清空时间状态；home 完成后仍是一个无时间状态的新 paused session；
- 初次启动或 home 后重新启动：必须先等模型成功输出序列第一个 subtask 才启动 timer；
- engine fatal error、进程退出：状态不持久化；
- 显式 full reset：清空 timer；
- history memory 继续遵守原计划的 reset 语义。普通 pause 后首轮允许 history memory 为空、time 仍保留，二者
  不要求同生同灭。

原因：普通 pause 通常不改变机器人所在物理 subtask，若把暂停墙钟时间计入 elapsed 会产生错误大值；若每次 pause
都丢失 subtask time，又会让恢复后的模型突然回到 no-time 输入。Home 则实际改变物理状态，必须清空。

---

## 2. 当前代码实际在做什么

实施前必须阅读源码，不得只按本文伪代码猜测。

### 2.1 Subtask 标注和数据集

- `src/lerobot/scripts/lerobot_annotate_subtask.py`
  - 写逐帧 `subtask` 和 `subtask_progress`；
  - progress 按同 episode 内相邻同名的连续段计算；
  - 长度为 `N` 的段 progress 是 `(offset + 1) / N`；
  - 现有代码没有 elapsed-time 列。
- `src/lerobot/scripts/lerobot_build_dataset.py`
  - 合并 raw `extras.parquet`；
  - 生成 converted LeRobotDataset；
  - 本任务不要求修改它或重建现有数据。
- `src/lerobot/datasets/lerobot_dataset.py`
  - `select_columns()` 可只读取轻量 parquet 列；
  - `get_raw_item()` 不解码视频；
  - `__getitem__()` 才走正常视频、delta window 和 transform 路径。
- `src/lerobot/datasets/memory_history.py`
  - `MemoryHistoryDataset` 动态抽取上一历史帧；
  - 它解决的是 history memory，不知道当前连续段起点；
  - 新时间功能不能把 `memory_frame_offset` 当 elapsed time。
- `src/lerobot/datasets/factory.py`
  - 仅 `use_memory_conditioning=true` 时创建 history wrapper；
  - 新时间功能需要能在 history memory 关闭时独立创建时间 wrapper；
  - 当前 streaming memory 会早失败。

### 2.2 Current subtask 和 history memory processor

- `src/lerobot/processor/subtask_processor.py`
  - current target canonical 文本为 `Subtask: ...; Progress: ...\n`。
- `src/lerobot/processor/memory_processor.py`
  - history memory 是一个 deterministic main-prompt processor；
  - processor 自己不采 dropout、不做随机数；
  - 训练 history GT 和部署上一轮 prediction 共用一个步骤。
- `src/lerobot/processor/converters.py`
  - 把 `subtask_*`、`memory_*` 路由到 complementary data；
  - 新的 time 字段也必须显式 route-through。
- `src/lerobot/policies/pi0/processor_pi0.py`
  - 当前顺序：Advantage → Memory → PI0 newline → Current subtask → Tokenizer。
- `src/lerobot/policies/pi05/processor_pi05.py`
  - 当前顺序：Normalize → Advantage → Memory → State prompt → Current subtask → Tokenizer。

### 2.3 PI0 / PI0.5 model

当前模型 attention 已允许：

```text
[image + main prompt]
    -> current subtask causal AR
    -> state/action flow matching suffix
```

因此 elapsed time 进入 main prompt 后自然同时影响 current subtask CE 的条件输入和 action FM，不需要：

- 新增模型 tensor 参数；
- 修改 PaliGemma embedding；
- 新增 token type；
- 修改 flow matching 公式；
- 新增 elapsed-time loss。

`src/lerobot/policies/pi0/modeling_pi0.py` 与 `pi05/modeling_pi05.py` 主要作为回归阅读和 reset/生成输出来源，
原则上不应为时间条件新增 attention 分支。

### 2.4 训练入口

- `src/lerobot/configs/train.py`
  - 已有 `memory_dropout_prob=0.2` 和 history lookback；
  - 尚无 elapsed noise/dropout 配置。
- `src/lerobot/scripts/lerobot_train.py`
  - DataLoader batch 先采 advantage mask，再采 memory mask，再进 preprocessor；
  - history memory 开启且非 resume 时会重建结构性 processor；
  - resume 加载 checkpoint 内保存的 processor；
  - memory metrics 已有独立 accumulator。

时间条件应复用这一结构，但不能复用 `memory_dropout_prob`，因为二者必须独立。

### 2.5 RTC 和 deploy

- `src/lerobot/inference_engines/rtc.py`
  - 独立线程运行 build → prepare → preprocess → predict → postprocess → queue merge；
  - history memory 只有在 merge 成功后才提交；
  - `debug_snapshot()` 在 state lock 下返回完整状态；
  - `pause()` 当前只清 active event；
  - `reset()` 会清 history memory、queue、observation 和 processor/policy 状态。
- `src/lerobot/scripts/lerobot_policy_deploy.py`
  - 当前 `pause_policy()` 调用 `clear_policy_state()`；
  - `clear_policy_state()` 执行 `engine.pause()` 后紧接 `engine.reset()`；
  - `prepare_policy()` 同样先全量清理；
  - 因此若不拆分 soft pause/full reset，新 timer 会在每次 Space pause 时被错误清空。
- `src/lerobot/utils/terminal_status.py`
  - 当前固定显示 STATE、LATENCY、TIMING、SUBTASK、MEMORY；
  - 新时间 tracker 状态应加入面板，便于真机确认。

### 2.6 Nero 与 Pico 的关系

需要阅读：

- `src/lerobot/robots/bi_nero_follower/`；
- `src/lerobot/teleoperators/bi_pico_nero_teleop/`；
- `src/lerobot/teleoperators/pico_nero_teleop/`；
- `scripts/nero_teleop/README.md`；
- `scripts/nero_teleop/run_bi_teleop.py`。

它们用于确认 30 FPS 数据来源、Nero action/observation schema、home 和 pause 的物理语义。本任务预计不改 Pico
映射或 Nero 控制代码；若 implementation 发现必须修改，必须先用具体失败证明，不能为了接入 prompt 顺手侵入
遥操作链路。

---

## 3. 已有真实数据审视结果

### 3.1 Strike-match 数据

```text
repo_id: ming326/strike_match_3_subtask
root: /home/zenbot-robot/.cache/huggingface/lerobot/ming326/strike_match_3_subtask
fps: 30
episodes: 70
frames: 53794
```

所有 70 个 episode 的压缩序列完全相同：

```text
1. Pick up the match.
2. move the right arm to ready.
3. Pick up the matchbox.
4. move the left arm to ready.
5. Strike the match and light the candle.
6. Return to the home position.
```

不存在空标注、重复 subtask 或 timestamp/frame 不一致。单段真实 end-elapsed 范围约 1.2–11.53 秒。

### 3.2 Nero egg 数据

```text
repo_id: ming326/nero_egg_subtask
root: /home/zenbot-robot/.cache/huggingface/lerobot/ming326/nero_egg_subtask
fps: 30
episodes: 61
frames: 350010
```

所有 61 个 episode 的 12 个 subtask 顺序完全相同，不存在空标注或重复 subtask。单段真实 end-elapsed 范围约
3.53–95.77 秒，其中：

```text
1. Pick up the oil bottle and pour in the oil.
2. Pick up the salt shaker and add some salt.
3. Bring the bowl of beaten eggs to the pan.
4. Pick up the fork.
5. Stir the beaten eggs.
6. Pour in the beaten eggs and put the bowl back.
7. Place the serving bowl in front of the pan.
8. Pick up the pan and the spatula.
9. Start frying the eggs.
10. Pour the eggs into the bowl.
11. Put down the bowl and the spatula.
12. Place the bowl of eggs on the left, then return to the starting position.
```

- `Stir the beaten eggs.` 中位约 31.83 秒、最大约 43.90 秒；
- `Start frying the eggs.` 中位约 53.73 秒、最大约 95.77 秒。

这证明时间实现必须支持 40.5s、95.8s 一类值，不能按短任务只设计成个位数秒。

### 3.3 数据结论

当前两个目标数据集满足第一版严格部署状态机的前置条件：

- 同一数据集所有 episode 顺序一致；
- 每个 subtask 每 episode 只出现一次；
- 相邻段边界明确；
- FPS 和 timestamp 一致；
- converted dataset 已包含 `subtask` 和 `subtask_progress`。

任务实现仍必须做通用 early validation，不能因为这两个数据刚好干净而省略错误处理。

---

## 4. 最终端到端行为模拟

### 4.1 训练样本

假设当前样本：

```text
episode = 3
frame_index = 420
current subtask segment starts at frame 300
fps = 30
```

真实 elapsed：

```text
x = (420 - 300) / 30 = 4.0s
```

训练 helper 计算：

```text
a = min(0.4 * 4.0, 5.0) = 1.6s
epsilon ~ Uniform(-1.6, 1.6)
```

假设抽到 `epsilon=-0.47`：

```text
noisy = max(0, 4.0 - 0.47) = 3.53s
formatted = 3.5s
```

若 time dropout 未命中，主 prompt 追加：

```text
Subtask elapsed time: 3.5s
```

若 history memory 同时保留：

```text
Memory: Subtask: Pick up the fork.; Progress: 0.8
Subtask elapsed time: 3.5s
```

current subtask target 仍是独立 causal target，不被改写：

```text
Subtask: Stir the beaten eggs.; Progress: 0.2
```

### 4.2 Subtask 第一帧

```text
current frame = segment start frame
x = 0.0s
a = 0.0s
noisy = 0.0s
```

保留 condition 时输入 `0.0s`。不得写成 `0.033s`，也不得为了制造噪声引入负数。

### 4.3 Dropout 组合

History、time 和 current-subtask-to-FM dropout 全部独立，因此至少覆盖：

| history keep | time keep | current subtask 对 FM keep | Current subtask AR 条件 | Action FM 条件 |
|---|---|---|---|---|
| true | true | true | history + time | history + time + current subtask |
| true | false | true | history | history + current subtask |
| false | true | true | time | time + current subtask |
| false | false | true | neither | current subtask |
| true | true | false | history + time | history + time |
| false | false | false | neither | neither |

Time dropout 不删除 current subtask CE；它只改变 CE 和 FM 所看到的 main-prompt condition。

### 4.4 部署首轮

启动时：

```text
tracker index = none
timer = not started
time condition = absent
```

模型第一次成功提交：

```text
Subtask: Pick up the match.; Progress: 0.1
```

状态机识别为序列第 0 项，在 RTC merge 成功后的 commit 点：

```text
current index = 0
timer start = monotonic commit time
```

同一次推理没有 time condition；下一次推理才读到约 `0.xs`。

### 4.5 正常前进和边界抖动

当前 index=0。连续输出当前 subtask 时 timer 不重启：

```text
Pick up the match. -> keep index 0, elapsed continues
```

第一次成功提交紧邻 subtask：

```text
move the right arm to ready. -> index 1, elapsed reset to 0
```

之后模型短暂又输出旧 subtask：

```text
Pick up the match. -> ignored, index remains 1
```

旧输出不会让 timer 回退或重启。

### 4.6 跳级和未知输出

当前 index=1，若输出 index=3：

```text
move the left arm to ready. -> ignored as skipped future subtask
```

若输出无法解析、未在数据集序列中，或 normalization 后有歧义：

```text
tracker unchanged
timer continues
diagnostic rejection reason updated
```

### 4.7 Pause / resume

Pause 前：

```text
current index = 4
raw active elapsed = 37.2s
```

暂停 90 秒：

```text
time condition remains logically 37.2s
no inference runs
wall-clock pause duration is not accumulated
```

Resume 后 0.8 秒开始下一轮推理：

```text
time condition = 38.0s
```

Action queue、interpolator、smoother 和 policy runtime cache 仍需按安全要求清理；只保留 tracker 的
`current_subtask_index + accumulated_active_elapsed`。

### 4.8 部署截断

若数据中某 subtask 最大真实 end-elapsed 为 `43.9s`：

```text
deployment cap = 43.9 + 5.0 = 48.9s
```

即使 tracker 原始 active elapsed 已达 70 秒，prompt 仍使用：

```text
Subtask elapsed time: 48.9s
```

状态面板应同时显示 raw elapsed 与 effective/clamped elapsed，便于发现模型长期未推进，而不是把截断伪装成
真实 timer 停止。

---

## 5. 数据扫描、lookup 和序列契约

### 5.1 不修改数据集 schema

第一版不在 annotate/build 阶段生成 `subtask_elapsed_time` 列，原因是：

- 现有两个 converted dataset 可以直接使用，无需重建约 11 GB 视频数据；
- elapsed 是由 `subtask + episode boundary + frame_index + fps` 唯一派生的轻量信息；
- 加噪和 dropout 必须动态发生，本来也不能永久写死；
- 避免 raw annotation、converted schema 和旧数据版本分叉；
- 350010 帧的 float32 lookup 约 1.4 MB，内存成本很小。

### 5.2 新增共享扫描模块

新增：

```text
src/lerobot/datasets/subtask_timing.py
```

它应提供两个层次：

1. 纯函数扫描给定轻量列，生成 segment contract；
2. `SubtaskTimingDataset` wrapper 给训练样本追加 true elapsed 字段。

共享 contract 至少包含：

```python
@dataclass(frozen=True)
class SubtaskSegmentStats:
    canonical_name: str
    normalized_name: str
    max_elapsed_seconds: float
    deployment_cap_seconds: float

@dataclass(frozen=True)
class SubtaskSequenceContract:
    fps: float
    ordered_subtasks: tuple[SubtaskSegmentStats, ...]
```

统计口径固定为：

```text
segment end elapsed = (segment_frame_count - 1) / fps
max_elapsed_seconds = 该 subtask 在所有扫描 episode 中最大的 segment end elapsed
deployment_cap_seconds = max_elapsed_seconds + deployment_margin_seconds
```

这里不能使用 `segment_frame_count / fps`，否则会与“第一帧为 0.0s”的训练定义错开一帧。

部署 sequence extraction 和训练 lookup 必须复用同一 normalization/segment 规则，不能各写一套。

### 5.3 训练 wrapper 输出

`SubtaskTimingDataset.__getitem__()` 在不修改原样本的前提下追加：

```text
subtask_elapsed_seconds: scalar torch.float32   # true x，尚未加噪
subtask_time_valid: bool
subtask_segment_index: int64                    # episode 内压缩序号，仅诊断/测试
```

训练 helper 后再增加：

```text
subtask_time_seconds: [B] float32               # 加噪、clamp 后的 processor 输入
subtask_time_condition_kept: [B] bool
```

不要复用 `memory_valid`、`memory_condition_kept` 或 `memory_subtask_progress`。

### 5.4 预计算算法

wrapper 构造时通过 `dataset.select_columns()` 一次读取：

```text
episode_index
frame_index
index
subtask
```

逐 episode 单次线性扫描：

```python
for each episode:
    require frame_index == 0..N-1
    require absolute index increments by one within the episode
    for each contiguous non-empty subtask segment [start, end):
        for relative row r in [start, end):
            elapsed[r] = (frame_index[r] - frame_index[start]) / fps
            valid[r] = True
            segment_index[r] = local_segment_index
```

复杂度：

```text
startup: O(number_of_frames)
getitem: O(1)
memory: O(number_of_frames)
```

严禁在每个 `__getitem__` 中向前回溯直到 subtask 改变，这会让长段和随机 DataLoader 访问退化。

### 5.5 Dataset wrapper 组合

时间和 history memory 独立，factory 必须支持：

```text
base LeRobotDataset
  -> SubtaskTimingDataset       (if time enabled)
  -> MemoryHistoryDataset      (if history enabled)
```

或相反顺序，但必须通过测试证明：

- 属性代理完整；
- `get_raw_item()` 仍指向真正 base dataset，history 不读取 wrapper 合成字段；
- 两个 wrapper 同开时不重复视频解码；
- time-only 不消耗 history offset RNG；
- history-only 不执行 timing scan；
- 两者关闭返回原 dataset object。

优先让 wrapper 显式代理 `get_raw_item()`、`select_columns()` 和训练依赖属性，避免 wrapper 叠加时接口丢失。

### 5.6 序列提取验证

部署扫描每个 episode，把相邻相同标签压缩成序列：

```text
[A, A, A, B, B, C] -> [A, B, C]
```

第一版必须要求：

- 所有使用的 episode 序列逐项一致；
- label 非空 string；
- normalization 后名称唯一；
- 同一 episode 的压缩序列中不得重复同一 normalized subtask；
- FPS finite 且大于 0；
- frame index 从 0 连续；
- 每个 subtask 至少一帧；
- 每个 subtask 都能计算最大 end-elapsed 和 cap。

错误时列出 dataset、episode、期望序列和实际差异并早失败。不要静默采用“出现次数最多”的序列。

### 5.7 选择 episodes

训练 wrapper 只扫描 `DatasetConfig.episodes` 选中的 view；lookup 下标必须与该 view 相对下标一致。部署第一版扫描
`PolicyDeployDatasetConfig` 指定数据集的全部 episode。若后续需要 deployment episode filter，应显式新增配置，
不能复用训练进程内对象。

---

## 6. 训练噪声、dropout 和指标

### 6.1 Train config

`TrainPipelineConfig` 新增：

```python
subtask_time_noise_ratio: float = 0.4
subtask_time_noise_max_seconds: float = 5.0
subtask_time_dropout_prob: float = 0.2
```

校验：

- ratio finite 且 `>=0`；
- max seconds finite 且 `>=0`；
- dropout finite 且位于 `[0,1]`；
- 时间开启时 dataset 非 streaming；
- dataset features 包含 `subtask`、`episode_index`、`frame_index`、`index`；
- policy 仅 PI0/PI0.5；
- `predict_subtask=true`。

`ratio=0` 或 `max_seconds=0` 是合法无噪声 ablation，不等于关闭 time condition。

### 6.2 新增 helper

新增：

```text
src/lerobot/utils/subtask_time_conditioning.py
```

核心函数：

```python
sample_subtask_time_condition(
    batch,
    noise_ratio,
    noise_max_seconds,
    dropout_prob,
    generator=None,
) -> dict
```

要求：

- 不原地修改输入 batch；
- 输入 true elapsed 为 finite、非负 `[B]` 或 `[B,1]`；
- valid 必须是 bool `[B]`；
- 使用 PyTorch RNG；
- 输出 float32 `[B]` 和 bool `[B]`；
- 对 eligible batch 固定 RNG 调用顺序；
- 噪声和 dropout 使用独立随机 draw；
- invalid 样本 kept 恒为 false；
- `p=0` 保留所有 valid；
- `p=1` 全部删除；
- clamp 到 0 发生在加噪后；
- 不在 helper 中 round 到一位小数，round 只在文本 formatter 中完成；
- time disabled 时不调用 helper，不消耗 time RNG。

固定每个 batch 的顺序：先为整个 batch draw noise，再 draw dropout。即使某样本 invalid，也保持 shape 固定；
测试锁定该行为，避免后续重构改变复现序列。

### 6.3 与其他 condition 的调用顺序

训练循环：

```text
raw DataLoader batch
-> sample_advantage_condition_mask(optional)
-> sample_memory_condition_mask(optional)
-> sample_subtask_time_condition(optional)
-> preprocessor
-> policy.forward
-> existing loss
```

三者必须分别采样。不得用一个 shared keep mask，也不得因 time dropout 改 advantage weight 或 history validity。

### 6.4 指标

新增窗口累计指标：

```text
subtask_time/valid_fraction
subtask_time/condition_kept_fraction
subtask_time/dropout_fraction_among_valid
subtask_time/true_seconds_mean
subtask_time/true_seconds_max_seen
subtask_time/noisy_seconds_mean
subtask_time/noise_abs_mean
subtask_time/noise_abs_max_seen
subtask_time/clamped_to_zero_fraction
```

这些指标不进入 loss。分布式训练时沿用现有 accelerator/主进程记录约定，不为每个 worker 重复刷日志。

### 6.5 Loss 不变

继续：

```text
loss = FM + subtask_ce_loss_weight * current_subtask_CE
```

Advantage weighting 仍只作用于 FM；time condition 不新增 loss、不改变 current CE reduction、不参与 RA-BC 或
advantage label 语义。

---

## 7. Prompt、processor 和 policy config

### 7.1 新 processor

新增：

```text
src/lerobot/processor/subtask_time_processor.py
```

并注册：

```text
subtask_time_condition_processor
```

`SubtaskTimeConditionProcessorStep` 是 deterministic processor，只接受已经决定好的：

```text
subtask_time_seconds
subtask_time_valid
subtask_time_condition_kept
```

有效且 kept 时追加：

```text
Subtask elapsed time: {seconds:.1f}s
```

invalid、empty source 或 kept=false 时必须让 task 逐字符保持原样。

### 7.2 Formatter 约束

新增纯函数：

```python
format_subtask_elapsed_time(seconds: float) -> str
```

要求：

- 拒绝 bool、string、NaN、Inf、负数；
- 接受 Python real scalar 或单元素 tensor；
- 输出固定一位小数；
- `-0.0` 必须规范化成 `0.0`；
- 不使用科学计数法；
- 不添加外层换行；
- canonical label 大小写固定。

### 7.3 Processor batch 契约

支持：

- scalar；
- list/tuple；
- tensor `[B]`；
- tensor `[B,1]`。

严格校验 task batch size、dtype、valid/keep bool 类型。Processor 不采噪声、不读取 dataset、不读取 subtask
output，也不自行推断 validity。

### 7.4 Pipeline 顺序

PI0：

```text
AddBatch
-> Advantage(optional)
-> History Memory(optional)
-> Subtask Time(optional)
-> Pi0NewLine
-> Current Subtask target(optional)
-> Tokenizer
-> Device/Relative/Normalize
```

PI0.5：

```text
Rename/AddBatch/Relative/Normalize
-> Advantage(optional)
-> History Memory(optional)
-> Subtask Time(optional)
-> Pi05 state prompt
-> Current Subtask target(optional)
-> Tokenizer
-> Device
```

时间行放在 state prompt 之前，以便 PI0.5 最终仍保留 state suffix；它与 history memory 是两个独立 processor
step，而不是把 time 拼进 `Memory:` 内部。

### 7.5 Policy config

PI0 和 PI0.5 config 新增：

```python
use_subtask_time_conditioning: bool = False
subtask_time_tokenizer_max_length: int = 128
```

约束：

- budget 必须正数；
- time 开启要求 `predict_subtask=true`；
- time 开启而 `subtask_generate_at_inference=false` 时 warning，说明只能训练或部署 no-time ablation；
- 默认关闭，旧 checkpoint 缺字段时安全加载；
- 不要求 `use_memory_conditioning=true`。

### 7.6 Token budget

effective main tokenizer length：

```python
budgets = [tokenizer_max_length]
if use_memory_conditioning:
    budgets.append(memory_tokenizer_max_length)
if use_subtask_time_conditioning:
    budgets.append(subtask_time_tokenizer_max_length)
effective = max(budgets)
```

Advantage、history memory 或 time 任一开启时使用 left truncation。至少测试：

- PI0 time-only 从 48 扩到 128；
- PI0 memory+time 保持 128；
- PI0.5 保持至少 200，不被反向缩短；
- 长 egg task + long history subtask + 95.8s time 时仍保留 `Memory:`、`Subtask elapsed time:` 和 PI0.5
  state suffix；
- current subtask target tensor 与 time keep/drop 独立，不被拼进 main token tensor。

### 7.7 Converter 和序列化

`batch_to_transition()` route-through：

```text
subtask_elapsed_seconds
subtask_time_valid
subtask_segment_index
subtask_time_seconds
subtask_time_condition_kept
```

新 processor 必须：

- 从 `lerobot.processor` 导出；
- 注册可反序列化；
- `get_config()` 完整；
- save/reload 后输出一致；
- checkpoint 的 processor JSON 中包含该 step 和 effective tokenizer length。

### 7.8 Processor 结构重建

`make_train_pre_post_processors()` 当前只把 history memory 视为结构性 override。条件应扩为：

```text
history memory enabled OR subtask time enabled
```

非 resume 从旧 checkpoint 开启 time 时必须重建 processor；resume 必须加载保存的 time processor。日志明确说明
触发重建的结构性字段，不能只写泛化的 “Memory conditioning”。

---

## 8. 部署数据加载和开关

### 8.1 Deploy config

`PolicyDeployConfig` 新增：

```python
use_subtask_time_conditioning: bool | None = None
subtask_time_deployment_margin_seconds: float = 5.0
```

解析：

- `None`：跟随 checkpoint 的 `policy.use_subtask_time_conditioning`；
- `False`：即使 checkpoint 支持也关闭，作为 ablation；
- `True`：只有 checkpoint config 已启用且加载的 preprocessor 含 time step 时才允许；否则早失败；
- margin 第一版默认/正式值为 `5.0`，finite 且非负。

不要让 deploy override 偷偷重建一个旧 checkpoint 的 processor。

### 8.2 数据集要求

effective deploy time 开启时必须提供：

```text
--dataset.repo_id
--dataset.root (本地数据需要时)
```

启动顺序：

```text
load policy config
-> resolve effective time flag
-> load dataset metadata
-> validate robot/fps/features
-> scan lightweight subtask columns
-> build sequence contract and caps
-> build RTC engine
```

若 time 关闭，部署不扫描 subtask 列，也不因 dataset 缺 subtask 而失败。

### 8.3 Processor presence 检查

部署加载 checkpoint processor 后验证：

- effective time on：恰好一个 `SubtaskTimeConditionProcessorStep`；
- effective time off：允许 step 存在，但 engine 不注入 time 字段，processor 应走 no-op；
- checkpoint config on 但 processor step 缺失：早失败并提示 checkpoint processor 不完整；
- duplicate step：早失败。

---

## 9. Subtask 输出解析与严格状态机

### 9.1 新增独立 tracker

新增可单测类，不把逻辑全部堆进 RTC loop：

```text
src/lerobot/inference_engines/subtask_time_tracker.py
```

职责：

- 保存有序 subtask contract；
- 解析/匹配模型输出；
- 维护 current index；
- 维护 active elapsed、pause/resume；
- 应用 deployment cap；
- 输出不可变 snapshot；
- 不读取 robot、不调用 model、不操作 ActionQueue。

### 9.2 输出解析

模型当前公开 `last_subtask_text`，典型值：

```text
Subtask: Pick up the match.; Progress: 0.4
```

解析器只接受可明确提取 subtask 部分的格式。字段标签使用大小写不敏感识别：

```text
Subtask: <name>; Progress: <anything>
```

name normalization：

1. trim；
2. 连续空白折叠为一个空格；
3. Unicode casefold；
4. 去除末尾一个句号类标点差异；
5. 不删除内部标点、不换词、不做编辑距离。

数据集 sequence 在启动时使用相同 normalization，并验证无 collision。

### 9.3 严格推进规则

状态 `current_index=None`：

- 匹配 index 0：start；
- 任何其他 index：ignore；
- unknown：ignore。

状态 `current_index=i`：

- 匹配 i：keep，timer 不重启；
- 匹配 i+1：advance，timer 从 commit clock 重启；
- 匹配 `<i`：reject old；
- 匹配 `>i+1`：reject skip；
- unknown/parse failure：reject unknown。

末尾 index：

- 重复末尾：继续计时并 cap；
- 任何旧项：忽略；
- 不自动结束 task，也不 wrap 到 index 0。

### 9.4 Timer 数据结构

不要只保存一个 wall-clock start，因为 pause/resume 需要排除暂停时间。状态至少包含：

```text
current_index: int | None
accumulated_active_seconds: float
running_since_monotonic: float | None
paused: bool
last_transition_reason: str
last_rejected_output: str
```

读取 raw elapsed：

```python
raw = accumulated_active_seconds
if running_since is not None:
    raw += now - running_since
effective = min(raw, current_subtask.deployment_cap_seconds)
```

所有 clock 注入使用 callable，默认 `time.monotonic`，测试使用 fake clock，不用真实 sleep。

### 9.5 RTC transaction 接线

每轮：

1. 在 state lock 下快照 reset version、history memory 和 time tracker；
2. 使用当前 monotonic time 计算 effective elapsed；
3. time 有效且 deploy effective flag on 时注入：

   ```text
   subtask_time_seconds=[effective]
   subtask_time_valid=[True]
   subtask_time_condition_kept=[True]
   ```

4. tracker 尚未 start 时显式注入：

   ```text
   subtask_time_seconds=[0.0]
   subtask_time_valid=[False]
   subtask_time_condition_kept=[False]
   ```

   processor 必须 no-op；time deploy 整体关闭时才完全不注入 `subtask_time_*` 字段；
5. preprocess/predict/postprocess；
6. 读取 subtask candidate；
7. reset/version/active 检查；
8. `ActionQueue.merge()`；
9. merge 成功后，在同一个 state commit 中更新 history memory 和 tracker；
10. predict、postprocess、merge、reset race 失败时 tracker 不推进。

状态机绝不能在 `predict_action_chunk()` 返回后、queue merge 前就推进，否则失败推理会永久改变 timer。

### 9.6 推理开始时取时间

“每次模型开始推理就获取时间”固定为：在 build/prepare/preprocess 前、该 RTC transaction 已拿到 observation 和
queue 资格后，创建一次 monotonic snapshot。本轮后续所有 time 字段和 debug 记录都使用这个 snapshot，不能在
tokenizer 前后多次读取造成一个 batch 内不一致。

---

## 10. Soft pause、full reset 与部署状态机改造

### 10.1 当前问题

当前：

```python
pause_policy()
  -> clear_policy_state()
      -> engine.pause()
      -> engine.reset()
```

这会把 pause 和 full reset 混成一个操作。新功能必须拆开，但仍要清理动作队列保证安全。

### 10.2 Engine API

避免用含义模糊的 bool 到处穿透，采用显式 API：

```python
engine.soft_pause()
engine.resume()
engine.full_reset()
```

若实现时为保持公共兼容而保留现有 `pause/reset` 名称，可以新增
`reset_runtime_preserving_subtask_time()`；无论内部命名如何，下表契约不得改变：

| 操作 | 推理线程 | ActionQueue | obs | policy/processor cache | history memory | time tracker |
|---|---|---|---|---|---|---|
| soft pause | stop | clear | clear | reset | clear | freeze + preserve |
| resume | start | empty warmup | new obs | clean | empty | resume clock |
| full reset | stop/controlled | clear | clear | reset | clear | clear |
| home | stop | clear | clear | reset | clear | clear |

不要为了保留 timer 而保留旧 ActionQueue 或旧 observation。

### 10.3 Deploy session 标志

`lerobot_policy_deploy.py` 需要区分：

- 初始 paused：不可 resume semantic state；
- Space 导致的 soft paused：可 resume；
- home 后 paused：不可 resume旧 semantic state；
- fatal error：不可 resume；
- 正常 preparing/running。

增加 `paused_session_resumable` 或等价显式状态，而不是仅从字符串 `state="paused"` 猜测。

### 10.4 Race 条件

至少覆盖：

- 推理进行中按 pause：候选 output 不提交，tracker 冻结在 pause 时刻；
- pause 后 RTC thread 不再读取 timer；
- resume 的首轮 time 等于冻结 elapsed + resume 后 active 时间；
- 推理进行中 full reset：候选丢弃，tracker 清空；
- pause 后按 home：旧 tracker 立即清空；
- home 完成后 right start：首轮无 time；
- 连续 pause/resume 不重复累计同一段时间；
- double pause/double resume 幂等或明确报错，不能产生负 elapsed。

---

## 11. Dashboard、日志和可观测性

### 11.1 RTC debug snapshot

新增：

```text
subtask_time_enabled
subtask_time_current_index
subtask_time_current_name
subtask_time_raw_elapsed_seconds
subtask_time_effective_seconds
subtask_time_cap_seconds
subtask_time_running
subtask_time_paused
subtask_time_last_transition
subtask_time_last_rejection_reason
subtask_time_last_input_seconds
```

snapshot 必须在同一个 state lock 下复制，不能看到 index 已推进但 timer 仍属于旧 subtask 的 torn state。

### 11.2 状态面板

在现有 footer 增加一行，例如：

```text
[TIME]     idx=4 running raw=37.2s input=37.2s cap=48.9s subtask=Stir the beaten eggs.
```

无 time：

```text
[TIME]     disabled
```

尚未识别第一项：

```text
[TIME]     waiting-for-first-subtask
```

paused：

```text
[TIME]     idx=4 paused elapsed=37.2s ...
```

Plain/non-TTY 输出仍遵守现有刷新节流，不能恢复成每控制 tick 刷屏。

### 11.3 日志

只在事件发生时用 info/warning：

- sequence contract 加载摘要；
- tracker start/advance；
- old/skip/unknown rejection 使用 debug 或节流 warning，避免边界抖动刷屏；
- pause freeze、resume、full reset；
- time cap 首次命中，可节流 warning；
- config ablation。

不得每轮 info 打印 time。

---

## 12. 需要阅读和预计修改的文件

### 12.1 实施前必读

核心计划与完成记录：

- `plans/memory_pipeline/pi0_pi05_memory_training_deployment_plan.md`
- `plans/memory_pipeline/milestone_1_dynamic_history_completed.md`
- `plans/memory_pipeline/milestone_2_memory_processor_completed.md`
- `plans/memory_pipeline/milestone_3_memory_training_completed.md`
- `plans/memory_pipeline/milestone_4_memory_modeling_completed.md`
- `plans/memory_pipeline/milestone_6_rtc_memory_completed.md`
- `plans/memory_pipeline/milestone_7_terminal_status_completed.md`
- `plans/memory_pipeline/milestone_8_automated_validation_completed.md`

数据链路：

- `src/lerobot/scripts/lerobot_annotate_subtask.py`
- `src/lerobot/scripts/lerobot_build_dataset.py`
- `src/lerobot/datasets/lerobot_dataset.py`
- `src/lerobot/datasets/dataset_reader.py`
- `src/lerobot/datasets/factory.py`
- `src/lerobot/datasets/memory_history.py`
- `src/lerobot/configs/default.py`

Processor/policy：

- `src/lerobot/processor/memory_processor.py`
- `src/lerobot/processor/subtask_processor.py`
- `src/lerobot/processor/converters.py`
- `src/lerobot/processor/__init__.py`
- `src/lerobot/policies/pi0/configuration_pi0.py`
- `src/lerobot/policies/pi0/processor_pi0.py`
- `src/lerobot/policies/pi0/modeling_pi0.py`
- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/pi05/processor_pi05.py`
- `src/lerobot/policies/pi05/modeling_pi05.py`

训练/部署：

- `src/lerobot/configs/train.py`
- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/inference_engines/rtc.py`
- `src/lerobot/scripts/lerobot_policy_deploy.py`
- `src/lerobot/utils/terminal_status.py`

Nero/Pico：

- `src/lerobot/robots/bi_nero_follower/bi_nero_follower.py`
- `src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py`
- `src/lerobot/teleoperators/bi_pico_nero_teleop/bi_pico_nero_teleop.py`
- `src/lerobot/teleoperators/pico_nero_teleop/teleop_state_machine.py`
- `scripts/nero_teleop/README.md`

现有测试模板：

- `tests/datasets/test_memory_history.py`
- `tests/datasets/test_memory_m8_real_dataset.py`
- `tests/processor/test_memory_processor.py`
- `tests/processor/test_memory_disabled_baseline.py`
- `tests/utils/test_memory_conditioning.py`
- `tests/scripts/test_memory_train.py`
- `tests/inference_engines/test_rtc_memory.py`
- `tests/scripts/test_lerobot_policy_deploy_status.py`
- `tests/policies/pi0_pi05/test_memory_modeling.py`
- `plans/memory_pipeline/m8_checkpoint_rtc_smoke.py`

### 12.2 预计新增

- `src/lerobot/datasets/subtask_timing.py`
- `src/lerobot/processor/subtask_time_processor.py`
- `src/lerobot/utils/subtask_time_conditioning.py`
- `src/lerobot/inference_engines/subtask_time_tracker.py`
- `tests/datasets/test_subtask_timing.py`
- `tests/processor/test_subtask_time_processor.py`
- `tests/utils/test_subtask_time_conditioning.py`
- `tests/inference_engines/test_subtask_time_tracker.py`
- `tests/inference_engines/test_rtc_subtask_time.py`
- milestone validation scripts 和 completion records。

### 12.3 预计修改

- `src/lerobot/datasets/factory.py`
- `src/lerobot/processor/__init__.py`
- `src/lerobot/processor/converters.py`
- `src/lerobot/policies/pi0/configuration_pi0.py`
- `src/lerobot/policies/pi0/processor_pi0.py`
- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/pi05/processor_pi05.py`
- `src/lerobot/configs/train.py`
- `src/lerobot/scripts/lerobot_train.py`
- `src/lerobot/inference_engines/rtc.py`
- `src/lerobot/scripts/lerobot_policy_deploy.py`
- `src/lerobot/utils/terminal_status.py`
- 对应现有 tests 和 Nero deploy README。

模型文件原则上只读回归；除非 failing test 证明 reset/生成接口存在缺口，否则不要修改。

---

## 13. 实施顺序与每阶段验收

每个 milestone 都必须：先写失败测试、实现最小生产改动、跑本阶段和累计回归、写 completion record。不得先做
RTC 再补数据契约，因为部署 cap 和 sequence tracker 依赖同一扫描结果。

### Milestone T0：契约基线与 strict failing tests

任务：

1. 冻结本计划中的字段名、公式、prompt 和状态机规则；
2. 添加 time-disabled PI0/PI0.5 prompt/token golden；
3. 添加 strict xfail 或明确失败测试，覆盖后续 dataset、helper、processor、tracker 接口；
4. 记录当前 memory M0–M8 回归基线；
5. 记录真实两个数据集的 sequence/duration 审计脚本输出。

测试标准：

- time disabled golden 与当前输出逐 tensor 相同；
- 新测试失败原因只能是功能尚未实现，不得是 fixture/import 错误；
- 当前 history memory、subtask、advantage、RTC、dashboard 测试保持原通过数；
- `git diff --check` 通过。

完成记录：

```text
plans/memory_pipeline/subtask_time_milestone_0_contract_completed.md
```

### Milestone T1：轻量 segment 扫描与训练 lookup

任务：

1. 实现共享 segment scanner 和 sequence contract；
2. 实现 `SubtaskTimingDataset`；
3. factory 支持 time-only、memory-only、both、neither；
4. 支持 selected episodes；
5. 对 streaming、缺列、空 label、不一致 index 早失败；
6. 不修改 raw/build dataset schema。

测试标准：

- 第一帧 0.0s；frame 33 在 30 FPS 且段起点 0 时为 1.1s；
- subtask 切换帧重新为 0.0s；
- episode 边界不串段；
- 同名但非连续段被视为两个 segment，并因第一版部署重复约束在 sequence validation 中失败；
- `__getitem__` 不向后扫描、不额外解码视频；
- 350010 帧 egg lookup 构造可完成，内存规模合理；
- time-only 不产生 `memory_*`，memory-only 不产生 `subtask_time_*`；
- both wrapper 属性、raw access、DataLoader `num_workers=0/2` 正常；
- disabled 返回原对象且不扫描列。

完成记录：

```text
plans/memory_pipeline/subtask_time_milestone_1_dataset_completed.md
```

### Milestone T2：训练 noise/dropout 和指标

任务：

1. 新增 train config 与校验；
2. 实现加噪/dropout helper；
3. 接入训练循环；
4. 实现 time metrics；
5. 验证与 advantage/history/subtask dropout 独立。

测试标准：

- x=0 始终 noisy=0；
- x=1 幅度不超过 0.4；x=40.5 幅度不超过 5；
- 所有输出 finite 且非负；
- 固定 generator 可复现；
- p=0/p=1 边界正确；
- helper 不修改输入；
- 三种 condition RNG 分离，mask 不互相覆盖；
- metrics 与手算一致；
- time off 不调用 helper、不消耗 RNG；
- CPU 单步 train smoke 通过。

完成记录：

```text
plans/memory_pipeline/subtask_time_milestone_2_training_completed.md
```

### Milestone T3：Processor、PI0/PI0.5 config 和 checkpoint 结构

任务：

1. 实现 formatter 和 processor；
2. 注册/export/converter 接线；
3. PI0/PI0.5 config、pipeline、token budget；
4. structural processor rebuild；
5. save/reload/resume 契约。

测试标准：

- exact canonical `Subtask elapsed time: 1.2s`；
- invalid/drop task byte-for-byte 不变；
- scalar/batch shape 和错误 dtype 覆盖；
- PI0/PI0.5 pipeline 顺序正确；
- time-only、memory-only、both、neither 四种 config；
- 旧 config 安全默认关闭；
- PI0 effective 128、PI0.5 不低于 200；
- 长 prompt 保留 condition/state；
- current subtask target 与 time keep/drop 隔离；
- 非 resume 旧 checkpoint + CLI time override 确实重建 processor；
- resume 加载已保存 time step；
- deploy load 不需要再次手写 override。

完成记录：

```text
plans/memory_pipeline/subtask_time_milestone_3_processor_completed.md
```

### Milestone T4：序列扫描和纯状态机

> 完成状态（2026-07-17）：已完成并通过 `contract`、真实数据和累计回归验收。实现、测试范围与实际结果见
> `plans/timer_pipeline/subtask_time_milestone_4_tracker_completed.md`。

任务：

1. 从 deploy dataset 构建 strict sequence contract；
2. 实现 parser/normalizer/collision 检查；
3. 实现 start/current/next/old/skip/unknown 状态；
4. 实现 fake-clock timer、cap、pause/resume/full reset；
5. 提供不可变 debug snapshot。

测试标准：

- 两个真实 dataset 分别提取唯一 6/12 项顺序；
- 所有 episode 不一致时早失败；
- 大小写、空白、末尾句号差异可匹配；
- 内部词差异不做 fuzzy match；
- initial 只能接受 index0；
- next 一次推进；old/skip 永不改变 index；
- current 重复不重启 timer；
- pause 90 秒不计时，resume 正确继续；
- full reset 后无有效 time；
- cap=dataset max +5，raw/effective 同时可见；
- fake clock 出现倒退时明确报错或 clamp，不能产生负值；
- normalization collision 早失败。

完成记录：

```text
plans/timer_pipeline/subtask_time_milestone_4_tracker_completed.md
```

### Milestone T5：RTC transaction 闭环

> 完成状态（2026-07-17）：已完成并通过 RTC transaction、真实数据和累计回归验收。实现、测试范围与实际结果见
> `plans/timer_pipeline/subtask_time_milestone_5_rtc_completed.md`。

任务：

1. engine 接收 sequence contract 和 effective flag；
2. 推理开始快照 time 并注入 processor；
3. merge 后原子提交 tracker；
4. 处理 predict/postprocess/merge/reset race；
5. debug snapshot 增加 time 状态；
6. time 和 history memory transaction 共存。

测试标准：

- 首轮无 time，首个 subtask commit 后次轮有 time；
- time 取样发生在 inference transaction 开始且一轮只取一次；
- inference latency 计入 active elapsed；
- queue 满等待期间 timer 继续；
- predict/postprocess/merge 任一失败都不推进；
- reset-version 变化丢弃候选；
- history commit 和 time commit snapshot 不撕裂；
- time disabled 不注入任何 time 字段、不维护 tracker；
- checkpoint generation disabled 时 time 永远无效并有 warning；
- batch size 非 1 继续早失败。

完成记录：

```text
plans/timer_pipeline/subtask_time_milestone_5_rtc_completed.md
```

### Milestone T6：Soft pause、home、deploy 配置和状态面板

> 完成状态（2026-07-17）：已完成并通过 soft pause/home、deploy 配置、真实数据、dashboard 和累计回归验收。
> 实现、测试范围与实际结果见
> `plans/timer_pipeline/subtask_time_milestone_6_deploy_completed.md`。

任务：

1. 拆分 deploy soft pause/full reset；
2. 清 queue/runtime cache 但保留 frozen tracker；
3. home/full reset 清 tracker；
4. deploy flag 和 dataset early validation；
5. dashboard TIME 行和节流日志；
6. README 更新 Nero 部署命令和语义。

测试标准：

- 初始 pause/right start 首轮无 time；
- running→Space→Right 保留并冻结 time；
- pause 时 queue/obs/runtime cache 已清；
- pause 中 h/home 清 tracker；
- home 后 right 首轮无 time；
- repeated pause/resume 幂等；
- `deploy flag=None/False/True` 解析正确；
- 旧 checkpoint 强制 True 早失败；
- time off 不要求 dataset subtask；
- live/plain dashboard 都显示正确且日志不破版；
- keyboard、homing、Nero action clamp 既有测试通过。

完成记录：

```text
plans/timer_pipeline/subtask_time_milestone_6_deploy_completed.md
```

### Milestone T7：真实数据、checkpoint 和 fake-robot 自动验收

> 完成状态（2026-07-17）：已完成全部非真机自动验收。真实数据、真 tokenizer、PI0/PI0.5 GPU update、
> checkpoint 严格 reload、condition matrix、production RTC + fake BiNero、pause/resume/home 和累计回归结果见
> `plans/timer_pipeline/subtask_time_milestone_7_automated_completed.md`。

任务：

1. 对 egg/match 全量数据跑 timing contract；
2. PI0/PI0.5 真 tokenizer prompt decode；
3. 各至少一个真实 checkpoint 单步/短步 GPU update；
4. checkpoint save/reload；
5. production RTC + fake BiNero observation 闭环；
6. pause/resume/home fake integration；
7. time/history/advantage/dropout matrix。

最低矩阵：

```text
history off / time off
history on  / time off
history off / time on
history on  / time on
time dropout 0 / 1 / 0.2
time noise ratio 0 / 0.4
advantage off / on
PI0 / PI0.5
```

测试标准：

- loss、FM、CE、grad norm finite；
- time dropout 不删除 CE；
- processor JSON 和 config 持久化；
- 离线 reload 无 missing/unexpected key；
- 真 tokenizer decode 含 canonical time；
- fake RTC 完成 initial→next→old rejection→pause/resume→home reset；
- time off checkpoint 与 T0 golden 一致；
- M0–M8 和 T0–T6 累计回归通过；
- 记录 CUDA、显存、Python、dataset/checkpoint 路径和实际命令。

完成记录：

```text
plans/timer_pipeline/subtask_time_milestone_7_automated_completed.md
```

### Milestone T8：Nero + Pico 实机验收

本阶段不能用 fake robot 代替。

前置：

- 机器人周围清空；
- 低速、夹爪限幅、急停和人工接管可用；
- 先用 Pico/teleop 确认左右臂、相机、home 正常；
- 使用与 checkpoint 对应的标注 dataset sequence；
- 首次运行关闭 compile，缩短 execution horizon；
- status dashboard 可见。

验收步骤：

1. time off 基线短跑；
2. time on，确认首轮 waiting、首 subtask 后启动；
3. 观察 current 重复输出 timer 连续；
4. 观察 next 一次输出后立即归零且不回退；
5. Space pause 5–10 秒，确认 time 冻结；
6. Right resume，确认从冻结值继续；
7. 再 pause 后 home，确认 tracker 清空；
8. 重新启动，确认必须再次从序列第一项开始；
9. 若安全可控，观察至少一次边界旧输出被忽略；
10. egg 长 subtask 观察时间超过 40 秒仍正确格式化；
11. 记录 cap 是否命中、模型是否长期卡住和物理行为；
12. time on/off 做同条件短时对比。

通过标准：

- 没有旧动作在 pause/resume 后泄漏；
- timer 与 dashboard 语义一致；
- pause 不累计、home 清空；
- subtask 不回退、不跳级；
- RTC 异常不导致 tracker 错进；
- Nero 动作、相机、夹爪和 home 无回归；
- Pico 人工接管/采集路径无回归；
- completion record 写明真实机器人、人员、时长、配置、结果和未覆盖风险。

完成记录：

```text
plans/memory_pipeline/subtask_time_milestone_8_nero_completed.md
```

---

## 14. 累计测试矩阵

### 14.1 Dataset

- synthetic segment boundaries；
- selected episodes；
- empty/invalid labels；
- repeated labels；
- inconsistent sequences；
- bad FPS/frame/index；
- real match/egg；
- DataLoader workers；
- wrapper composition；
- disabled zero-overhead path。

### 14.2 Processor/config

- PI0/PI0.5；
- time-only/history-only/both/neither；
- valid/invalid/drop；
- batch/scalar/dtype errors；
- token budget/truncation；
- registry/save/reload；
- old config；
- structural rebuild/resume。

### 14.3 Training

- ratio 0/0.4；
- max 0/5；
- dropout 0/0.2/1；
- x=0/1/12.5/40.5/95.8；
- independent masks；
- metrics；
- advantage/RABC compatibility；
- PI0/PI0.5 CPU/GPU smoke。

### 14.4 Tracker/RTC

- initial/current/next/old/skip/unknown；
- parser normalization/collision；
- cap；
- pause/resume/full reset/home；
- predict/postprocess/merge failure；
- reset race；
- history+time atomic snapshot；
- queue wait and inference latency；
- time deploy ablation。

### 14.5 Deploy/UI/Nero

- config resolution；
- missing dataset/checkpoint processor；
- TTY/non-TTY；
- keyboard state transitions；
- action queue safety；
- fake BiNero；
- true BiNero；
- Pico preflight/manual override。

---

## 15. Definition of Done

只有同时满足以下条件，整个任务才算完成：

1. PI0 和 PI0.5 可独立开启 elapsed-time text condition；
2. history memory 和 time 支持四种独立组合；
3. 训练 elapsed 来自连续 subtask 段起点和 dataset FPS，第一帧 0.0s；
4. 无需修改或重建现有 raw/converted 数据集；
5. 训练噪声严格为 ratio 0.4、最大 ±5s、非负 clamp；
6. time dropout 默认 0.2 且与其他 dropout 独立；
7. processor canonical 文本固定并保存进 checkpoint；
8. time enabled 的旧 checkpoint override 能正确重建，resume 正确加载；
9. deploy 从标注 dataset 严格提取唯一 sequence 和 per-subtask cap；
10. tracker 只接受 current/next，永不回退、不跳级；
11. tracker 只在成功 RTC merge 后提交；
12. 每轮推理开始读取 monotonic time，不按推理次数估算；
13. deploy effective time 按 dataset maximum +5s 截断；
14. Space pause 冻结并保留，resume 继续，home/full reset 清空；
15. dashboard 能看见 current subtask、raw/effective/cap、pause 和 rejection；
16. time disabled 的 prompt/tensor/checkpoint/deploy 基线无回归；
17. history memory、subtask、advantage、relative RTC 既有测试无回归；
18. real match/egg、真 tokenizer、真 PI0/PI0.5 checkpoint 和 fake RTC 自动验收通过；
19. Nero + Pico 实机安全验收完成并有记录；
20. 每个 milestone 有可重复 validation script、实际结果和 completion record。

---

## 16. 实施禁区和常见错误

- 不把 `subtask_progress` 直接乘一个固定时长当 elapsed；progress 不包含真实段长。
- 不把 history `memory_frame_offset` 当 elapsed；它是随机历史偏移。
- 不在每次 `__getitem__` 向前扫描 label。
- 不硬编码 30、egg 12 类或 match 6 类。
- 不把 time 拼进 history `Memory:` 文本后再共用一个 dropout。
- 不在 processor 内采 noise/dropout。
- 不让模型自己输出 time，也不新增 time loss。
- 不在 queue merge 前推进 tracker。
- 不用 fuzzy matching 驱动不可逆状态。
- 不把 skip 当 next。
- 不因 current subtask 重复输出重启 timer。
- 不用 `time.time()`，避免系统时钟调整；使用 monotonic clock。
- 不让普通 pause 墙钟时间进入 elapsed。
- 不为保留 timer 而保留旧 ActionQueue。
- 不让 home 后恢复旧 tracker。
- 不在 time off 时强制扫描 dataset 或要求 subtask 列。
- 不声称 fake robot 通过等于 Nero 实机通过。

---

## 17. 建议的一键验收入口

每个阶段新增：

```text
plans/timer_pipeline/validate_subtask_time_milestone_0.sh
...
plans/timer_pipeline/validate_subtask_time_milestone_8.sh
```

最终自动化入口建议支持：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_8.sh data
plans/timer_pipeline/validate_subtask_time_milestone_8.sh regression
plans/timer_pipeline/validate_subtask_time_milestone_8.sh gpu
plans/timer_pipeline/validate_subtask_time_milestone_8.sh checkpoints
plans/timer_pipeline/validate_subtask_time_milestone_8.sh automated
```

脚本必须：

- 默认使用 `lerobot-main`；
- 明确 offline/online；
- 不删除用户 outputs；
- 临时 fixture 使用 `mktemp -d` 并在记录中说明；
- GPU smoke 和真机分开；
- Ruff 不存在时明确 skipped；
- 执行 `py_compile`、目标 pytest、累计 regression、`git diff --check`；
- 不把 skip、xfail 或未跑真机写成 passed。
