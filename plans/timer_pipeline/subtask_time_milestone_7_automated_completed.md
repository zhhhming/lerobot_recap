# Subtask Elapsed-Time Milestone T7 Automated Validation Completion Record

日期：2026-07-17

基线：`main@8194e71096d58ee2d82bbe8b47b35d3ad5f2d655` 加当前 memory M0–M8、timer T0–T6 工作区改动。

计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

## 完成状态

Milestone T7：真实数据、checkpoint 和 fake-robot 自动验收已完成。

本阶段完成了两个真实数据集全量 timing contract、PI0/PI0.5 真 PaliGemma tokenizer、真实 checkpoint GPU
update、time/history/advantage/dropout/noise matrix、checkpoint 保存和离线严格 reload、真实数据三路图像与
BiNero observation 驱动的 production `RTCInferenceEngine`、soft pause/resume/home，以及 memory M0–M8 和 timer
T0–T6 累计回归。

本阶段没有连接 Nero/Pico、没有发送真实机器人动作，也没有把 fake RTC 解释为实机通过。Nero + Pico 实机安全验收
仍属于 T8。

## 环境和前置资源

实际环境：

```text
Conda: lerobot-main
Python: 3.12.13
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 4090
GPU memory: 24564 MiB
Driver: 580.159.03
offline: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
```

执行前可用磁盘约 `97 GB`。T7 输出最终约 `17 GB`，没有删除或覆盖已有 outputs。

真实数据：

```text
match: /home/zenbot-robot/.cache/huggingface/lerobot/ming326/strike_match_3_subtask
egg:   /home/zenbot-robot/.cache/huggingface/lerobot/ming326/nero_egg_subtask
```

原始 checkpoint：

```text
PI0:
/home/zenbot-robot/models/lerobot/
pi0_nero_egg_relative_bs256_20260715_155327/checkpoints/019000/pretrained_model

PI0.5:
/home/zenbot-robot/models/lerobot/
pi05_strike_match_3_subtask_relative_bs192_20260706_200321/checkpoints/010000/pretrained_model
```

time+memory 训练从已完成的 memory M8 checkpoint 启动：

```text
outputs/memory_m8_automated/pi0_memory/checkpoints/last/pretrained_model
outputs/memory_m8_automated/pi05_memory/checkpoints/last/pretrained_model
```

## 新增验收文件

- `tests/datasets/test_subtask_time_t7_real_data.py`
  - 环境变量显式启用，不让普通单测依赖本地大数据；
  - 使用真实 egg 样本和真实 PI0/PI0.5 tokenizer；
  - 覆盖 time-only、memory+time 和 time-disabled prompt；
  - 检查 canonical elapsed 文本、PI0/PI0.5 tokenizer budget、left truncation 和 no-time 基线。
- `plans/timer_pipeline/subtask_time_t7_train_log_check.py`
  - 解析真实训练日志；
  - 强制要求预期 update 数量以及 finite loss/grad norm；
  - 为每个训练 case 保存机器可读 JSON。
- `plans/timer_pipeline/subtask_time_t7_checkpoint_rtc_smoke.py`
  - 加载真实完整权重与保存的 processor；
  - 检查 clean strict reload、真实模型 FM/CE/total loss；
  - 覆盖 time/history/dropout/noise matrix；
  - 用 dataset state/三路图像构造 fake BiNero observation；
  - 驱动 production RTC 完成 initial/next/old/pause/resume/home；
  - 输出完整 JSON 报告。
- `plans/timer_pipeline/validate_subtask_time_milestone_7.sh`
  - 提供 `data`、`regression`、`gpu`、`checkpoints`、`automated` 模式；
  - 复用 T6 和 memory M8 验收入口；
  - 默认离线，拒绝覆盖已有 T7 output。
- 本完成记录。

主计划的 T7 completion record 路径从错误的 `plans/memory_pipeline` 更正为与 T0–T6 一致的
`plans/timer_pipeline`。

本阶段没有修改 Dataset、processor、policy、model、training、RTC、deploy、dashboard、Nero 或 Pico 生产代码。

## 真实数据与真 tokenizer

全量 scanner 复验：

| Dataset | Episodes | Frames | FPS | Subtasks | Lookup | 关键最大值 / cap |
|---|---:|---:|---:|---:|---:|---|
| strike-match | 70 | 53794 | 30 | 6 | 0.667 MiB | 11.533333s / 16.533333s |
| Nero egg | 61 | 350010 | 30 | 12 | 4.339 MiB | 95.766667s / 100.766667s |

Egg 长段再次确认：

```text
Stir the beaten eggs.: 43.900000s maximum / 48.900000s cap
Start frying the eggs.: 95.766667s maximum / 100.766667s cap
```

真 tokenizer 用例使用 episode 0、frame 20 的真实 elapsed `0.6666667s`，PI0 和 PI0.5 在 time-only、
memory+time 下均解码出：

```text
Subtask elapsed time: 0.7s
```

PI0 effective main tokenizer length 为 `128`，PI0.5 为 `200`；time-enabled 路径均为 left truncation。
time-disabled 路径没有 time step/field/text，PI0 保持 `48`，PI0.5 保持 `200`，并与 T0 deterministic golden
联合通过。

实际结果：

```text
6 passed in 10.79s
```

## 真实 GPU update

共同配置：

```text
device=cuda
dtype=bfloat16
batch_size=1
num_workers=0
compile_model=false
gradient_checkpointing=true
freeze_vision_encoder=true
optimizer=SGD(lr=1e-5, momentum=0, weight_decay=0)
subtask_time_noise_ratio=0.4
subtask_time_noise_max_seconds=5.0
subtask_time_dropout_prob=0.2
```

SGD 无动量仅用于在 24 GB GPU 上执行完整模型 smoke 并避免保存大型 Adam optimizer state，不是正式长训练推荐配置。

实际 update：

| Policy/case | Steps | Loss | Grad norm | 保存 checkpoint |
|---|---:|---|---|---|
| PI0 memory+time | 2 | 10.185, 11.153 | 224.935, 147.561 | yes |
| PI0.5 memory+time | 2 | 6.693, 1.776 | 78.582, 18.768 | yes |
| PI0 time-only | 1 | 12.432 | 202.424 | no |
| PI0.5 time-only | 1 | 2.950 | 25.289 | no |
| PI0 memory+time+advantage | 1 | 12.397 | 168.771 | no |
| PI0.5 memory+time+advantage | 1 | 1.224 | 21.283 | no |

所有 loss 和 grad norm 均 finite。time-only 证明 history 结构关闭时 time 可独立训练；advantage 两个 case 使用
临时只读派生数据视图：

```text
/tmp/lerobot-t7-advantage.6WBGUq/nero_egg_subtask_advantage
positive=123215, negative=114484, ignore=112311
videos -> source dataset read-only symlink
```

原 egg dataset 没有被修改。

## 保存、processor 与严格 reload

保存结果：

| Policy | Checkpoint | model.safetensors | Time step | Memory step | Main tokenizer |
|---|---|---:|---:|---:|---:|
| PI0 | `outputs/subtask_time_t7_automated/pi0_both/checkpoints/000002/pretrained_model` | 8,892,502,944 bytes | 1 | 1 | 128 |
| PI0.5 | `outputs/subtask_time_t7_automated/pi05_both/checkpoints/000002/pretrained_model` | 9,354,050,752 bytes | 1 | 1 | 200 |

两个 `config.json` 均持久化：

```text
predict_subtask=true
subtask_generate_at_inference=true
use_memory_conditioning=true
use_subtask_time_conditioning=true
memory_tokenizer_max_length=128
subtask_time_tokenizer_max_length=128
dtype=bfloat16
```

两个保存的 `policy_preprocessor.json` 均恰好包含一个 `memory_condition_processor` 和一个
`subtask_time_condition_processor`，顺序符合 PI0/PI0.5 pipeline 契约。两种 policy 在 offline `strict=True`
reload 时均输出 `All keys loaded successfully!`，没有 missing/unexpected key。

## 真实模型 condition matrix

每个保存后的完整模型只加载一次，对真实 egg batch 运行以下五组 forward。所有 total/FM/current-subtask CE
均 finite；即使 time dropout=1，CE 仍存在且大于 0，没有删除 current-subtask target。

PI0：

| Case | history keep | time keep | ratio/dropout | Total | FM | CE |
|---|---|---|---|---:|---:|---:|
| both clean | yes | yes | 0 / 0 | 10.8075 | 2.8073 | 32.0009 |
| history only | yes | no | 0 / 1 | 12.6630 | 4.6332 | 32.1191 |
| time only noisy | no | yes | 0.4 / 0 | 12.4243 | 4.3558 | 32.2740 |
| neither noisy | no | no | 0.4 / 1 | 12.5420 | 4.5326 | 32.0377 |
| default | yes | yes | 0.4 / 0.2 | 10.6131 | 2.6057 | 32.0297 |

PI0.5：

| Case | history keep | time keep | ratio/dropout | Total | FM | CE |
|---|---|---|---|---:|---:|---:|
| both clean | yes | yes | 0 / 0 | 2.3029 | 1.4416 | 3.4451 |
| history only | yes | no | 0 / 1 | 4.9884 | 4.1136 | 3.4993 |
| time only noisy | no | yes | 0.4 / 0 | 3.1826 | 2.3616 | 3.2840 |
| neither noisy | no | no | 0.4 / 1 | 3.0695 | 2.0447 | 4.0989 |
| default | yes | yes | 0.4 / 0.2 | 2.1679 | 1.3037 | 3.4567 |

该矩阵覆盖：

```text
history off/on
time off/on
time dropout 0/1/0.2
time noise ratio 0/0.4
advantage off/on（真实 update）
PI0/PI0.5
```

## Production RTC + fake BiNero

fake RTC 使用真实 checkpoint、真实 tokenizer、真实 egg 数据集 state 和三路 `(3,480,640)` 图像，将其还原为
`bi_nero_follower` raw observation，再调用 production `RTCInferenceEngine`。没有使用 stub policy，action 是真实模型
计算结果且逐次验证 finite。

为了确定性验证不可逆 tracker，本测试适配层在每次真实 action/subtask generation 后：

1. 先记录模型自然生成的 subtask 文本；
2. 再把 candidate 替换为数据 contract 中预定的 first/next/old 文本；
3. 让 production RTC 按正常 merge transaction 提交。

这样证明的是 checkpoint/processor/model/action/RTC/tracker 的完整接口和事务语义，不把两步 smoke 后的自然生成
质量误报为可部署质量。

PI0 和 PI0.5 均完成四种 effective mode：

| Mode | Inferences | History contract | Time contract |
|---|---:|---|---|
| history off / time off | 2 | 无 memory input/update | 无 tracker/time fields |
| history on / time off | 2 | 首轮空、次轮使用完整前轮输出 | 无 tracker/time fields |
| history off / time on | 5 | 无 memory input/update | full sequence below |
| history on / time on | 5 | history transaction 同时提交 | full sequence below |

time-on 两种 mode 均验证：

```text
initial index0: first inference has no time input
next: one successful output advances to index1
old: index0 output is rejected_old and never moves backwards
soft pause: raw elapsed freezes at 5.0s
pause clock: advancing fake clock by 90s keeps 5.0s
resume: first input is 5.8s after 0.8s active time
home/full reset: current index and elapsed are cleared
fresh start: first inference again has no time and must accept index0
```

PI0 peak allocated CUDA memory为 `8648.9 MiB`，PI0.5 为 `9099.5 MiB`。

自然生成文本作为诊断保存在 JSON 报告中：两步 PI0 checkpoint 仍生成非 canonical 乱码；PI0.5 重复生成
`Return to the home position.`。因此这两个 T7 smoke checkpoint 不能直接用于 T8 实机，它们只用于自动化结构和事务
验收。

## Time-off checkpoint A/B

原始 PI0 和 PI0.5 checkpoint 分别以 memory/time off 驱动真实模型 RTC：

```text
PI0: tokenizer=48, finite action, no memory/time state, latency=328.3–506.7ms
PI0.5: tokenizer=200, finite action, no memory/time state, latency=507.0–677.4ms
```

两者 strict load 均为 clean，reset 清空 runtime；同时 T0 两条 byte/tensor golden 继续通过：

```text
2 passed in 1.45s
```

## 累计回归与静态门禁

实际分组结果：

```text
T6 RTC/deploy/dashboard + tracker + RTC memory contract:
  97 passed in 4.37s
T6 focused RTC/deploy/dashboard:
  56 passed in 4.25s
T4 tracker contract:
  30 passed in 1.48s
T1 scanner focus:
  23 passed, 21 deselected in 0.02s
T3 processor/converter/time-disabled contract:
  54 passed in 1.48s
T0-T3 cumulative time contract:
  119 passed, 4 deselected in 1.58s
checkpoint contract:
  6 passed in 1.87s
memory/subtask/advantage/RTC/time focused regression:
  266 passed, 6 deselected, 2 warnings in 3.90s
complete RTC policy + memory + time regression:
  227 passed, 3 skipped in 4.24s
T6 deploy/RTC/status regression:
  67 passed in 4.28s
terminal keyboard regression:
  14 passed, 4 deselected in 1.61s
T0 time-disabled golden:
  2 passed in 1.45s
T7 true tokenizer:
  6 passed in 10.79s
```

2 个 warning 是既有测试刻意设置 decode token 上限；3 个 skip 是既有 RTC CUDA 条件用例；deselected 是 T0–T6
已记录的 worker/非目标筛选，不是 T7 新失败。T7 的完整 GPU smoke 另行真实通过。

其他门禁：

```text
py_compile: passed
bash -n: passed
git diff --check: passed
ruff: skipped (lerobot-main 未安装)
```

## 一键验收入口

```bash
plans/timer_pipeline/validate_subtask_time_milestone_7.sh data
plans/timer_pipeline/validate_subtask_time_milestone_7.sh regression
plans/timer_pipeline/validate_subtask_time_milestone_7.sh gpu
plans/timer_pipeline/validate_subtask_time_milestone_7.sh checkpoints
plans/timer_pipeline/validate_subtask_time_milestone_7.sh automated
```

语义：

- `data`：两个全量数据 contract 和真 tokenizer；
- `regression`：timer T0–T6、memory M0–M8 和关闭态 golden；
- `gpu`：要求全新 output root，运行 update、保存、matrix、reload、fake RTC；
- `checkpoints`：不重新训练，复验已有 T7 checkpoint；
- `automated`：依次运行全部非真机验收。

默认输出：

```text
outputs/subtask_time_t7_automated/
  logs/
  reports/
  pi0_both/checkpoints/000002/pretrained_model/
  pi05_both/checkpoints/000002/pretrained_model/
```

`reports/pi0_checkpoint_rtc.json` 和 `reports/pi05_checkpoint_rtc.json` 保存完整真实模型 matrix、自然/注入输出、
RTC transition、环境和显存信息。

## 工作区保护

阶段开始和完成时均检查 `git status --short`。工作区原有 memory M0–M8、timer T0–T6、计划目录移动、Nero/Pico
和其他用户改动全部保留。

本阶段只新增本记录列出的 T7 tests/scripts、更新主计划 T7 状态，并写入独立 T7 outputs。没有执行 reset、checkout、
删除、暂存、全仓格式化或覆盖无关文件；没有修改原 dataset、原 checkpoint 或 memory M8 checkpoint。

## T8 边界

T7 完成不代表整个 elapsed-time 项目 Definition of Done 已完成。仍未验收：

- 真实 `bi_nero_follower` connect/disconnect；
- 真实三路相机、动作发送、夹爪和 home；
- Pico 人工接管与采集路径；
- 真实 TTY keyboard 和人在回路急停；
- 正式长训练 checkpoint 的 subtask 生成质量；
- time on/off 的真实物理行为对比。

这些只能在机器人周围清空、低速/限幅/急停准备完成且用户现场同意后进入 T8。
