# Memory Pipeline Milestone 8 Automated Validation Completion Record

日期：2026-07-17

基线：`main@8194e71096d5`

环境：

- Conda：`lerobot-main`，Python 3.12.13；
- PyTorch：2.10.0+cu128；
- GPU：NVIDIA GeForce RTX 4090，24564 MiB；
- 数据和模型均使用 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；
- 所有实际 deploy import、训练和 checkpoint smoke 使用
  `conda run --no-capture-output -n lerobot-main`。

## 完成范围和状态

本记录完成 `pi0_pi05_memory_training_deployment_plan.md` Milestone 8 中所有不连接真实机械臂的自动化验收：

- 真实 `nero_egg_subtask` LeRobotDataset 全量契约检查；
- PI0、PI0.5 真实 PaliGemma checkpoint 的 GPU 训练 update；
- memory on/off、advantage on/off 和四种 memory/subtask dropout 组合；
- 独立 checkpoint 保存、processor 序列化、离线重新加载；
- 使用真实数据集 state/三路相机 observation 的 production RTC fake-robot 闭环；
- M0–M7 memory/subtask/advantage/model/RTC/dashboard 累计回归；
- runtime ABI/import、非 TTY/live dashboard 和 logging 契约。

本记录**不是整个 Milestone 8 的最终完成记录**。没有连接真实 `bi_nero_follower`，没有发送真实机器人动作，
也没有进行真实相机、homing、pause/restart 和物理安全验收。按照总计划 Definition of Done，M8 仍需最后的
Nero 实机短时运行；不得把本记录引用为“真机已通过”。

## 新增自动验收文件

- `tests/datasets/test_memory_m8_real_dataset.py`
  - 环境变量显式启用的真实数据测试，普通单测环境中安全 skip；
  - 检查全量 schema、episode/index 连续性、实际视频解码、多 worker history 和真实 tokenizer prompt。
- `plans/memory_pipeline/m8_make_advantage_fixture.py`
  - 从 source dataset 创建临时 advantage 测试视图；
  - 只重写小型 parquet/meta，11 GB 视频使用只读符号链接；
  - source dataset 永不修改。
- `plans/memory_pipeline/m8_checkpoint_rtc_smoke.py`
  - 加载完整 PI0/PI0.5 权重与保存的 processor；
  - 运行四种真模型 dropout forward；
  - 把 dataset observation 转换成 BiNero raw state/camera 形式并驱动 production
    `RTCInferenceEngine`；
  - 同脚本支持 `--memory-mode=off` A/B 基线。
- `plans/memory_pipeline/validate_milestone_8.sh`
  - `data`：runtime/data/prompt；
  - `regression`：M0–M7 累计回归；
  - `gpu`：首次训练、checkpoint、fake RTC、advantage matrix；
  - `checkpoints`：不重新训练，复验已有 checkpoint；
  - `automated`：按顺序运行全部非真机验收。
- 本完成记录。

没有为 M8 修改 model、processor、dataset、RTC 或 deploy 生产代码。M8 测试没有发现需要新增生产分支的
缺陷。

## M7 runtime 遗留问题处理

M7 记录中的 `CXXABI_1.3.15` 问题已定位为启动环境的动态库加载顺序：直接执行 Conda Python 绝对路径时，
`torch` 会先加载系统 `libstdc++.so.6`，随后 `sqlite3 -> ICU 78` 需要较新的 CXXABI 而失败。

以下门禁已实际通过：

```text
conda run --no-capture-output -n lerobot-main python -c \
  'import torch; import sqlite3; import lerobot.scripts.lerobot_policy_deploy'

deploy import: passed
cuda: True 1
```

因此 M8 脚本统一使用 `conda run`，没有把机器特定的 `LD_PRELOAD` 或 import hack 写进生产源码。

## 真实数据集验收

数据集：

```text
repo_id: ming326/nero_egg_subtask
root: /home/zenbot-robot/.cache/huggingface/lerobot/ming326/nero_egg_subtask
```

全量 parquet 和实际 LeRobotDataset 加载结果：

```text
codebase_version: v3.0
robot_type: bi_nero_follower
fps: 30
episodes: 61
frames: 350010
subtasks: 12
null subtask/progress: 0
progress finite and in [0,1]: true
global index contiguous: true
episode-local frame_index contiguous: true
```

实际解码了 `left_wrist`、`right_wrist`、`third_person` 三路 `(3,480,640)` 图像。PI0/PI0.5 均通过：

- episode 0 和 episode 1 的 frame 0 先抽 offset，再返回 no-memory；
- 非边界样本 history 与 current 保持同 episode，offset 始终在 1–12；
- 连续访问 24 次至少出现 4 种 offset；
- DataLoader `num_workers=0/2` 均成功；
- prompt 的实际 PaliGemma decode 包含 `Memory:`、`Subtask:` 和 `Progress:`；
- PI0 effective tokenizer length=128；PI0.5=200 且保留 `State:`。

专项结果：

```text
3 passed, 4 warnings in 5.34s
```

4 个 warning 是 Python 3.12 对多线程进程中 `fork()` 的既有 DeprecationWarning；两个真实 worker 配置均
完成，没有死锁或残留进程。

## 真实 checkpoint 训练 smoke

共同配置：

```text
GPU: RTX 4090 single process
dtype: bfloat16
batch_size: 1
episodes: [0]
steps: 2
compile_model: false
gradient_checkpointing: true
freeze_vision_encoder: true
optimizer: SGD(lr=1e-5, momentum=0)
memory lookback: [1,12]
memory_dropout_prob: 0.2
subtask_dropout_prob: 0.2
```

SGD 无动量仅用于在单张 24 GB GPU 上完成真实 full-model smoke，并避免保存数十 GB Adam state；它不是正式
长训练推荐优化器。

### PI0

输入 checkpoint：

```text
/home/zenbot-robot/models/lerobot/
pi0_nero_egg_relative_bs256_20260715_155327/checkpoints/019000/pretrained_model
```

该 checkpoint 原本 `predict_subtask=false`。非 resume CLI override 开启 subtask+memory 后，训练入口明确打印
structural processor rebuild，并用当前 egg dataset stats 重建 processor。

```text
step 1: loss=10.405, grad_norm=222.436, offset=8, memory kept
step 2: loss=10.808, grad_norm=146.277, offset=6, memory dropped
```

保存目录：

```text
/home/zenbot-robot/repos/lerobot/outputs/memory_m8_automated/
pi0_memory/checkpoints/000002/pretrained_model
```

### PI0.5

输入 checkpoint：

```text
/home/zenbot-robot/models/lerobot/
pi05_strike_match_3_subtask_relative_bs192_20260706_200321/checkpoints/010000/pretrained_model
```

```text
step 1: loss=6.622, grad_norm=96.052, offset=1, memory kept
step 2: loss=1.963, grad_norm=17.989, offset=3, memory dropped
```

保存目录：

```text
/home/zenbot-robot/repos/lerobot/outputs/memory_m8_automated/
pi05_memory/checkpoints/000002/pretrained_model
```

两个 checkpoint 均完整保存并离线重新加载，state dict 无 missing/unexpected key。保存结果：

| Policy | model.safetensors | Memory step | main tokenizer |
|---|---:|---:|---:|
| PI0 | 8,892,502,944 bytes | 1 | 128 |
| PI0.5 | 9,354,050,752 bytes | 1 | 200 |

合计自动化输出约 17 GB。两者 config 均持久化：

```text
predict_subtask=true
use_memory_conditioning=true
memory_tokenizer_max_length=128
```

## 真模型 dropout 矩阵

保存后的完整模型各只加载一次，对同一个真实 egg dataset batch 运行：

```text
(memory_dropout, subtask_dropout)
(0,0), (1,0), (0,1), (0.2,0.2)
```

PI0：

| 组合 | total | FM | current subtask CE |
|---|---:|---:|---:|
| 0 / 0 | 12.5598 | 4.5538 | 32.0239 |
| 1 / 0 | 12.8198 | 4.7955 | 32.0971 |
| 0 / 1 | 12.4484 | 4.4424 | 32.0239 |
| 0.2 / 0.2 | 12.6184 | 4.6125 | 32.0239 |

PI0.5：

| 组合 | total | FM | current subtask CE |
|---|---:|---:|---:|
| 0 / 0 | 2.2722 | 1.4016 | 3.4823 |
| 1 / 0 | 5.9510 | 4.9217 | 4.1172 |
| 0 / 1 | 5.1683 | 4.2978 | 3.4823 |
| 0.2 / 0.2 | 2.2120 | 1.3414 | 3.4823 |

所有数值 finite。相同 memory 输入下，把 subtask dropout 从 0 改成 1 不改变 current subtask CE；memory drop
会改变 CE 的 prompt condition，符合计划语义。

## Advantage on/off 自动矩阵

`nero_egg_subtask` 本身没有 advantage 字段。自动脚本创建临时只读派生视图：

```text
/tmp/lerobot-m8-advantage.vFsUjS/nero_egg_subtask_advantage
```

它包含：

```text
advantage_label_subtask: positive=123215, negative=114484, ignore=112311
advantage_loss_weight_subtask: positive=2, negative=1, ignore=0
videos: symlink to source dataset
```

原数据集未修改。PI0、PI0.5 均从刚保存的 memory checkpoint 重新启动真实训练入口，启用 advantage prompt、
FM-only weighting 和 memory，并完成一个 GPU update：

```text
PI0:   loss=13.083, grad_norm=243.576
PI0.5: loss=1.286,  grad_norm=21.059
```

两次 update 均 finite，processor 因 memory structural config 使用当前 advantage fixture stats 重建。精确
positive/negative/ignore 数学和 all-ignore 安全路径继续由 M5 的 172 个聚焦测试覆盖。

## Dataset-backed fake RTC 和 A/B

fake RTC 不使用 stub policy。脚本加载保存后的完整 policy、保存的 processor 和 dataset 三路真实图像，把
state/image 转回 BiNero raw observation，再驱动 production `RTCInferenceEngine`。

### Memory on

PI0：

```text
first memory input: <none>
second memory input == complete first output: true
latency: 493.3 ms / 475.9 ms（复验值）
reset cleared memory/subtask/source id: true
```

PI0 起点 checkpoint 没有 subtask 训练，仅做两步 smoke 后生成的是非 canonical 乱码。RTC 按产品契约保留并
回灌完整非空输出，因此事务测试通过；这不代表该两步 checkpoint 有可用的 subtask 语义质量，也不应直接拿去
控制真实机器人。

PI0.5：

```text
first output: Subtask: Return to the home position.; Progress: 0.8
second memory input: Subtask: Return to the home position.; Progress: 0.8
latency: 278.0 ms / 244.3 ms（复验值）
reset cleared memory/subtask/source id: true
```

PI0.5 输入 checkpoint 来自 strike-match，fake RTC 只证明跨 task 数据下 engine/checkpoint 契约成立，不代表
egg 动作质量。

### Memory off A/B

使用原始 checkpoint 和同一脚本的 `--memory-mode=off`：

```text
PI0:   tokenizer=48,  action finite, subtask empty, memory/source id始终为空, latency=330.5 ms
PI0.5: tokenizer=200, action finite, subtask="Pick up the match...", memory/source id始终为空,
       latency=655.7 ms
```

这证明 memory disabled 时不注入字段、不扩大 PI0 prompt、不维护 next memory；PI0.5 原有 subtask 推理仍可
独立工作。

## 累计回归

实际分组结果：

```text
M0–M3 / memory / subtask / tokenizer / policy:
216 passed, 6 skipped, 2 warnings in 6.35s

Dataset reader/facade:
35 passed in 2.17s

M3 train/helper integration:
59 passed, 2 warnings in 1.86s

M4 model/attention/reset:
60 passed in 1.52s

M5 advantage/memory compatibility:
172 passed, 2 warnings in 6.29s

M6 RTC/policy inference:
220 passed, 3 skipped in 3.43s

M7 dashboard/RTC handoff:
28 passed in 3.49s

M8 real dataset/prompt:
3 passed, 4 warnings in 5.34s
```

6 个累计 skip 和 3 个 RTC skip 均为既有 CUDA/multi-GPU 用例；本阶段另行运行的完整模型 GPU smoke 已通过。
warning 是既有 tokenizer 配置 warning、Python 3.12 `fork()` DeprecationWarning，以及加载 PI checkpoint 时已有的
vision embedding diagnostic warning。state dict 全部成功加载。

`py_compile`、`bash -n` 和 `git diff --check` 均通过。`lerobot-main` 没有安装 Ruff，因此 Ruff 仍明确 skip。

## 工作区保护

实施前后均保留用户现有 dirty worktree，没有 reset、checkout、暂存、覆盖或格式化无关文件。新增内容仅限本
记录列出的 M8 测试/脚本/文档和独立 `outputs/memory_m8_automated` 运行产物。

原 dataset、原 PI0/PI0.5 checkpoint 和 Nero/Pico 配置均未修改。临时 advantage fixture 仅在 `/tmp` 创建，
且视频为 source 的只读符号链接。

## 未运行项和下一步

未运行：

- 真实 `bi_nero_follower` connect/disconnect；
- 真实三路相机初始化日志；
- 真实 action send、pause、home、restart；
- 真实 TTY keyboard cbreak 和人在回路的 Ctrl-C；
- 真实 70–200+ ms 控制下机械臂动作连续性和物理安全；
- 正式长训练或 subtask 语义 overfit。

下一步只应在用户准备好机械臂、清空工作区并现场确认后进行。真机候选不能直接使用本记录中的两步 PI0 smoke
checkpoint；应使用经过足够 subtask+memory 训练且人工检查生成质量的 egg checkpoint。完成真机验收后，再写
最终 `milestone_8_end_to_end_completed.md`。

