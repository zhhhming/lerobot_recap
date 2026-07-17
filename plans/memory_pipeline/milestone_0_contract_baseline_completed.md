# Memory Pipeline Milestone 0 Completion Record

日期：2026-07-17

基线：`main@8194e710`

Python：`/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`（Conda 环境 `lerobot-main`）

## 完成范围

本次完成 `pi0_pi05_memory_training_deployment_plan.md` 中的 Milestone 0：契约测试和默认关闭基线。

本阶段只新增测试、验收脚本和本完成记录，没有修改 Dataset、processor、policy config、模型、训练循环、
RTC 或 deploy 生产逻辑。动态 history、memory prompt、训练 dropout 和部署闭环仍属于后续 milestone，
不得把本记录视为这些功能已经实现。

## 新增文件

- `tests/processor/test_memory_disabled_baseline.py`
- `tests/datasets/test_memory_history.py`
- `tests/utils/test_memory_conditioning.py`
- `tests/processor/test_memory_processor.py`
- `plans/memory_pipeline/validate_milestone_0.sh`
- 本完成记录

## 已固化的绿色基线

`test_memory_disabled_baseline.py` 使用离线 deterministic character tokenizer，直接锁定最终 prompt、token
tensor 和 attention mask，而不是只检查 tokenize 前的输入字符串：

- PI0：prompt 为 `pick cube\n`，main token shape 为 `[1, 48]`；
- PI0.5：prompt 为 `Task: pick cube, State: 128 128;\nAction: `，main token shape 为
  `[1, 200]`；
- 两种 policy 的 tokenizer 均保持原 truncation side；
- pipeline 中没有 Memory step；
- 处理结果没有 `memory_*` 字段。

验收脚本同时运行现有 subtask、advantage 和 loss 专项测试，继续锁定：

- current subtask 的 causal attention 布局；
- subtask dropout 只遮断 suffix 到 current subtask 的 attention，不删除 current subtask CE；
- advantage weight 只作用于 FM；
- current subtask CE 使用普通 mean；
- all-ignore batch 的 FM 为安全零值，CE 仍能训练；
- PI0/PI0.5 无 subtask 时保持 FM-only 行为。

## 后续功能的红灯契约

尚未存在的 API 使用 `pytest.mark.xfail(strict=True, raises=...)` 登记为严格预期失败。marker 只接受当前
缺少模块、字段或构造参数导致的异常；AssertionError、错误的部分实现或其他异常不会被吞掉。对应 API
实现后，正确通过会成为严格 XPASS 并令验收失败，实施 milestone 必须同步移除 marker，使测试转为普通
绿色测试。

当前 10 个预期 xfail 分布：

- 3 个 Dataset history 契约：先从 `[1,12]` 抽样、`t=4/k=9` 无历史、`t=12/k=12`
  读取 frame 0、episode B 起点不读取 episode A，并要求历史读取只走 `get_raw_item()`；
- 2 个 train keep-mask 契约：natural invalid 和 dropout 都输出 false、`p=0/1` 边界、输入不原地修改、
  固定 PyTorch generator 可重复；
- 4 个 PI0/PI0.5 config 契约：默认关闭、token budget 默认 128、开启 memory 要求
  `predict_subtask=true`、token budget 必须为正；
- 1 个 Memory processor 契约：invalid、empty 和 dropped 三种情况均逐字符保留原 task，并统一输出
  false keep mask。

## 一键验收脚本

脚本：

```bash
plans/memory_pipeline/validate_milestone_0.sh
```

脚本行为：

1. 使用 `LEROBOT_PYTHON` 覆盖或默认使用 `lerobot-main` Python；
2. 设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避免 ROS 的外部 pytest entrypoint 污染项目测试；
3. 对新增测试执行 `py_compile`；
4. 运行 memory baseline/contract 以及现有 subtask、tokenizer、converter、advantage、PI0/PI0.5
   训练和推理专项回归；
5. 执行 `git diff --check`。

## 实际执行结果

实施前专项基线：

```text
138 passed, 6 skipped, 3 warnings in 1.63s
```

新增测试后的完整 Milestone 0 验收：

```text
140 passed, 6 skipped, 10 xfailed, 3 warnings in 1.72s
py_compile: passed
git diff --check: passed
```

6 个 skip 均来自现有 `tests/processor/test_tokenizer_processor.py`：5 个要求 CUDA，1 个要求至少 2 张
GPU。它们不是 memory 测试跳过项。

warning 为已有的 CUDA/NVML 初始化提示，以及测试刻意构造 `subtask_max_decode_tokens` 大于训练 token
上限时的已有 warning；没有新增非预期 warning 或失败。

## 工作区保护复核

执行前确认工作区除计划文档记录的改动外，还包含用户正在进行的计划文件目录迁移/新增。此次没有
reset、checkout、覆盖、暂存或格式化这些内容，也没有修改以下既有用户源码改动：

- `scripts/nero_teleop/README.md`
- `src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py`
- `src/lerobot/scripts/lerobot_push_dataset.py`
- `tests/scripts/test_lerobot_push_dataset.py`

## 未运行和未完成项

- 未运行需要 CUDA 或多 GPU 的既有 tokenizer 测试；
- 未下载模型、tokenizer 或数据集；
- 未运行完整 PI0/PI0.5 checkpoint、训练 update、RTC、机器人或 Nero 实机测试；
- 未实现 Milestone 1–8 的任何生产功能；
- 10 个严格 xfail 必须由后续对应 milestone 实现并转绿，不能作为最终完成状态保留。

Milestone 1 的明确输入是 `tests/datasets/test_memory_history.py` 中的 3 个 Dataset 契约；下一阶段应先
实现 `MemoryHistoryDataset` 和 factory 接线，并把这 3 个测试从严格 xfail 转为普通通过测试。
