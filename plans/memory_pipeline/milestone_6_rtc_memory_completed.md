# Memory Pipeline Milestone 6 Completion Record

日期：2026-07-17

基线：`main@8194e710`

Python：`/home/zenbot-robot/.conda/envs/lerobot-main/bin/python`（Conda 环境
`lerobot-main`，CPU 验收显式设置 `CUDA_VISIBLE_DEVICES=''`）

## 完成范围

本次完成 `pi0_pi05_memory_training_deployment_plan.md` 中的 Milestone 6：RTC memory 事务闭环。

PI0/PI0.5 在异步 `RTCInferenceEngine` 中共享同一套 memory 状态机：上一轮只有在 policy predict、
postprocess、reset-version 检查和 `ActionQueue.merge()` 全部成功后，才会成为下一轮 prompt 的 Memory。
失败、空输出或 reset 不会泄漏半完成候选状态。

本阶段没有实现 Milestone 7 的固定终端状态面板，也没有进行 Milestone 8 的完整 checkpoint、真实数据或
Nero 实机部署。

## 修改文件

生产代码：

- 修改 `src/lerobot/inference_engines/rtc.py`

测试：

- 新增 `tests/inference_engines/test_rtc_memory.py`
- 更新 `tests/policies/rtc/test_action_queue.py`
- 更新 `tests/policies/rtc/test_latency_tracker.py`

验收和记录：

- 新增 `plans/memory_pipeline/validate_milestone_6.sh`
- 新增本完成记录

没有修改 PI0/PI0.5 model、processor、训练循环、advantage/RL、ActionQueue 或 LatencyTracker 生产代码，
也没有修改 `lerobot_policy_deploy.py` 的状态机。Deploy 现有 `clear_policy_state()` 已统一用于初次 prepare、
pause、home 和重新 prepare，并执行 `engine.pause()`、`engine.reset()`，因此可直接获得本阶段新增的 memory
清理语义。

## RTC memory 事务契约

### 受锁状态和 snapshot

Engine 新增由 `_state_lock` 统一保护的状态：

```text
_memory_text_for_next_inference
_last_memory_input_text
_last_subtask_output_text
_memory_source_inference_id
```

`debug_snapshot()` 在同一个临界区内复制这些字段，并公开同名无下划线 key。Milestone 7 可直接使用：

- `last_subtask_output_text` 显示最近一次成功提交的 SUBTASK；
- `last_memory_input_text` 显示该次推理实际使用的 MEMORY；
- `memory_text_for_next_inference` 用于诊断下一轮待用条件；
- `memory_source_inference_id` 标识非空 next memory 的来源提交。

并发 reader 测试持续读取 snapshot，同时完成 20 次 RTC commit；观察结果只能属于初始状态或某一个完整
commit，不会看到 inference count、input、output、next memory 和 source id 相互撕裂的组合。

### 一次推理事务

每轮推理现在执行：

1. 在 `_state_lock` 下原子快照 `reset_version` 和 next memory；
2. 构建并准备 observation；
3. memory conditioning 开启时，在 preprocessor 前注入 batch size 1 字段：

   ```text
   memory_text=[snapshot]
   memory_valid=[bool(snapshot)]
   memory_condition_kept=[bool(snapshot)]
   ```

4. 执行 preprocessor 和 `predict_action_chunk()`；
5. 要求 action/subtask 输出来自 deployment batch size 1；
6. 对 `last_subtask_text` 只做首尾和空白规范化，不解析或重建内部 subtask/progress；
7. 执行 postprocessor；
8. 检查 shutdown、active 和 `reset_version`；
9. 执行 `ActionQueue.merge()`；
10. merge 返回后，在 `_state_lock` 内一次性提交 input/output/next/source id 和 inference count。

predict、postprocess 或 merge 抛出异常时，候选 subtask 可能已经存在于 policy 临时属性中，但 engine 的最近
成功 output 和 next memory 均保持不变。推理中途 reset-version 变化时同样不提交。

### 首轮、空输出、关闭和 ablation

- 第一次推理和 reset 后显式注入 empty/invalid/false，Memory processor 不改变 task；
- 成功输出 A 后第二轮 prompt 使用完整 A，成功输出 B 后第三轮使用完整 B；
- 成功生成空白文本时，当前 output 规范化为空字符串，下一轮恢复 no-memory；
- `use_memory_conditioning=false` 时不注入任何 `memory_*` 字段，也不维护 next memory；
- 即使 memory 关闭，最近一次成功 subtask output 仍写入 snapshot，供后续 dashboard 显示；
- `subtask_generate_at_inference=false` 时 next memory 永远为空，形成配置 warning 所描述的 no-memory
  deployment ablation；
- 非 batch size 1 的 action 或 list subtask 输出明确报错，不会静默只取第 0 个样本。

### Reset 竞态修正

原实现先增加 `reset_version`，释放 state lock 后再清 observation。显式 reset 与 active inference 并发时，
理论上存在“推理读到 reset 前 observation，却快照到 reset 后 version”的窄窗口。

本阶段把 memory、queue 和 observation 清理放进同一个 state 临界区，并把 transaction 的 version/memory
快照提前到 observation 读取之前：

- reset 发生在快照后：最终 version check 丢弃候选；
- reset 发生在快照前：新事务只能在 observation 已清空后继续；
- reset 同时清理 policy、preprocessor 和 postprocessor 状态。

这使显式 reset、pause、home 和重新 prepare 都不会跨边界提交旧 observation 或旧 memory。

## 测试先行记录

新增测试首次在未修改 RTC engine 时执行：

```text
11 failed in 5.00s
```

失败点均为预期缺失契约：无 memory 字段注入、无 memory snapshot key、无事务提交/清理状态，以及未拒绝
multi-sample output。实现后同一专项结果：

```text
11 passed in 3.50s
```

后续 reset 原子性调整后复跑：

```text
11 passed in 3.55s
```

专项覆盖：

- PI0/PI0.5 第一轮、A→第二轮、B→第三轮；
- predict/postprocess/merge 三个失败提交点；
- inference 中 reset-version 变化；
- reset 后重新 resume 的第一轮 no-memory；
- 空输出清空 next memory；
- memory disabled 和 generation disabled ablation；
- deployment batch size 1 防线；
- 20 次 commit 与并发 debug snapshot reader。

## 既有 RTC 测试契约同步

第一次把此前未纳入 M5 脚本的整个 `tests/policies/rtc` 加入验收时，结果为：

```text
6 failed, 214 passed, 3 skipped in 3.97s
```

6 个失败均与本次 `rtc.py` 修改无关，而是 `de9b01ec`（2026-06-02）生产代码更新后测试未同步：

- `LatencyTracker` 生产代码已从永久 `max_latency` 属性改为随 deque 滚动的 `max()`，4 个测试仍读取旧属性；
- `ActionQueue` 已改为以 merge lock 内的 action index delta 为真值，2 个测试仍断言旧 warning 或没有模拟
  inference 期间实际消费 action。

本阶段只同步测试到当前生产契约：改为断言 `max()` 和新 warning，并在 typical RTC workflow 中真实消费
5 个 inference-delay action。没有为了让旧测试通过而回退当前 ActionQueue/LatencyTracker 生产逻辑。

同步后的完整 RTC 聚焦回归：

```text
220 passed, 3 skipped in 3.24s
```

## 一键验收脚本

脚本：

```bash
CUDA_VISIBLE_DEVICES='' plans/memory_pipeline/validate_milestone_6.sh
```

脚本执行：

1. 对 RTC engine、本阶段测试和同步的 RTC 测试执行 `py_compile`；
2. 复跑 `validate_milestone_5.sh`，累计覆盖 M0–M5；
3. 运行新增 RTC memory、完整 `tests/policies/rtc` 和 PI0/PI0.5 subtask inference 回归；
4. Ruff 可用时检查本阶段文件，不可用时明确打印 skipped；
5. 执行 `git diff --check`。

## 最终实际结果

实施前在受限 sandbox 内运行 M5 基线时，再次停在 M1–M5 已记录的真实双 worker DataLoader 隔离位置。
终止该受限运行后，在 sandbox 外执行相同 M5 命令，结果：

```text
216 passed, 6 skipped, 2 warnings in 6.02s
35 passed in 2.11s
59 passed, 2 warnings in 1.77s
60 passed in 1.50s
172 passed, 2 warnings in 6.26s
py_compile: passed
git diff --check: passed
ruff: skipped（未安装）
```

最终在 sandbox 外执行完整 M6 一键脚本：

```text
M0–M3 / memory / subtask / advantage / tokenizer / policy regression:
216 passed, 6 skipped, 2 warnings in 6.01s

Dataset reader/facade core regression:
35 passed in 2.07s

M3 + advantage train/helper integration:
59 passed, 2 warnings in 1.75s

M4 model/attention/reset/logging focused regression:
60 passed in 1.52s

M5 advantage/memory compatibility focused regression:
172 passed, 2 warnings in 6.27s

M6 RTC memory / RTC policy / PI0+PI0.5 inference regression:
220 passed, 3 skipped in 3.55s

py_compile: passed
git diff --check: passed
ruff: skipped（lerobot-main 环境未安装 Ruff）
```

M0–M5 的 6 个 skip 是既有 tokenizer CUDA/多 GPU 用例。M6 的 3 个 skip 是既有 RTC CUDA 用例。
两个 UserWarning 是既有测试刻意设置 `subtask_max_decode_tokens > subtask_max_tokens`；两个
DeprecationWarning 是 Python 3.12 对多线程进程中 `fork()` 的提示。没有新增失败、xfail 或非预期 warning。

## 工作区保护复核

实施前后均检查 `git status --short`。本阶段没有 reset、checkout、暂存、覆盖或格式化无关文件，只修改本记录
“修改文件”中列出的 M6 文件。

特别保留：

- `scripts/nero_teleop/README.md`
- `src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py`
- `src/lerobot/scripts/lerobot_push_dataset.py`
- `tests/scripts/test_lerobot_push_dataset.py`
- 用户的计划目录迁移以及 Milestone 0–5 全部现有实现和记录

## 未运行项

- Ruff 未安装，因此没有运行 Ruff lint/format check；
- 未运行既有 CUDA 或多 GPU 测试；
- 未加载完整 PI0/PI0.5 checkpoint 或真实 PaliGemma 权重；
- 未运行完整 fake robot `lerobot-policy-deploy`；
- 未运行真实 Nero、相机、home/pause 键盘交互或长时间 RTC soak；
- 未实现或测试 Milestone 7 的 live/plain/auto 固定终端状态面板。

以上不阻塞 Milestone 6：本阶段定义是 engine 内 memory transaction、失败/并发/reset 契约和 PI0/PI0.5
共享 fake RTC 验收。完整 checkpoint、fake robot 和 Nero 实机属于 Milestone 8。

## 下一阶段输入

Milestone 7 可以直接读取 `engine.debug_snapshot()` 的：

```text
last_subtask_output_text
last_memory_input_text
memory_text_for_next_inference
memory_source_inference_id
```

Dashboard 更新时应先取得 snapshot、释放 engine lock，再获取 status/logging handler lock，避免形成
engine lock 与 logging lock 的反向依赖。Milestone 7 不应在 deploy 主循环另建一套 memory 真值。
