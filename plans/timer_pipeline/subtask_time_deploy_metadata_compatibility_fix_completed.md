# Subtask-Time Deploy Metadata Compatibility Fix Completion Record

日期：2026-07-20

基线：`main@b94eebeca71397ede447373c07052287fb489b7b`

关联计划：`plans/timer_pipeline/pi0_pi05_subtask_elapsed_time_conditioning_plan.md`

## 完成状态

timer 部署数据集 metadata 兼容性问题已修复并通过聚焦单测、两个真实本地数据集、T6 contract、T6 regression、
T7 regression 和静态检查。

本次修复只修改 deploy metadata 检查与对应无硬件测试；没有修改 Dataset schema、timer scanner、processor、
PI0/PI0.5、RTC transaction、checkpoint、机器人控制、动作限幅或真实数据。验证过程没有连接 Nero/Pico，也没有发送
机器人动作。

## 原问题与复现

`src/lerobot/scripts/lerobot_policy_deploy.py::_check_metadata_compatibility()` 原先直接比较：

```text
dataset_meta.features
vs.
runtime robot features + DEFAULT_FEATURES
```

真实 `ming326/strike_match_3_subtask` 和 `ming326/nero_egg_subtask` metadata 除 action、observation 和默认索引字段外，
还包含逐帧 dataset-only 标注：

```text
subtask
subtask_progress
```

BiNero 运行时只产生 action、observation state 和三路相机 features，不产生这两个离线标注字段。因此旧实现的严格
`DeepDiff` 会在硬件 features 实际完全匹配时仍抛出：

```text
Dataset metadata compatibility check failed
```

timer 开启时又必须提供 `dataset.repo_id` 扫描 subtask sequence/cap，所以该错误会阻断正常 timer 部署。

原调用顺序还有一个安全性问题：`robot.connect()` 在 metadata 检查之前执行。也就是说不兼容配置会先连接硬件，
再因 metadata 失败。

## 测试先行记录

修改生产代码前，先在 `tests/scripts/test_subtask_time_deploy.py` 增加 annotation 和连接顺序回归测试。首次执行：

```bash
conda run --no-capture-output -n lerobot-main \
  python -m pytest tests/scripts/test_subtask_time_deploy.py -q
```

旧实现结果：

```text
2 failed, 27 passed in 1.60s
```

两个失败分别对应：

1. `subtask` / `subtask_progress` 被误判为缺失的运行时硬件 features；
2. metadata mismatch 发生前 fake robot 的 `connect()` 已被调用。

其余新增严格性负例在旧实现上已通过，证明测试本身没有把 robot type、FPS 或硬件 feature mismatch 错当成合法。

## 实现内容

### 1. 只忽略明确的 dataset 标注字段

`lerobot_policy_deploy.py` 新增不可变 allowlist：

```python
_DEPLOY_DATASET_ANNOTATION_FEATURES = frozenset({"subtask", "subtask_progress"})
```

`_check_metadata_compatibility()` 构造新的过滤字典，只从 metadata 比较侧排除上述两个字段；不会原地修改
`dataset_meta.features`。

以下检查保持严格：

- robot type；
- FPS；
- action dtype/shape/names；
- observation state dtype/shape/names；
- camera key/dtype/shape/names；
- 所有不在 allowlist 中的未知额外字段。

没有采用“忽略所有 dataset 额外字段”的宽松策略，因此未知 schema 差异仍会 early fail。

timer 标注 contract 也没有被放宽。effective timer on 时，既有
`validate_subtask_timing_features()` 和 `scan_subtask_timing()` 仍负责要求并扫描 `subtask`、episode/frame/index、固定
序列和 deployment cap。

### 2. 兼容性检查前移到连接之前

现在 deploy 在构造 robot 和 runtime features 后立即执行：

```text
dataset provided -> metadata compatibility check
dataset omitted  -> policy/runtime compatibility check
```

只有检查通过后才创建 status display、加载完整 policy/processor 并调用 `robot.connect()`。metadata/policy mismatch
不会连接机器人。

## 新增测试覆盖

`tests/scripts/test_subtask_time_deploy.py` 新增：

- annotation dataset 可通过 metadata compatibility；
- 调用前后 `dataset_meta.features` 完全一致；
- robot type mismatch 仍失败；
- FPS mismatch 仍失败；
- action shape mismatch 仍失败；
- action dtype mismatch 仍失败；
- observation names mismatch 仍失败；
- camera feature 缺失仍失败；
- 未知额外 camera feature 仍失败；
- 未知 `operator_note` 额外字段仍失败；
- compatibility failure 时 fake robot `connect()`、status display 和 policy load 均未启动。

实现后相同聚焦命令结果：

```text
31 passed in 1.60s
```

## 真实数据无硬件验证

使用真实 `BiNeroFollowerConfig`、默认 action/observation processors 和生产 `_build_dataset_features()` 构造运行时
features，然后直接调用生产 `_check_metadata_compatibility()`。没有调用 `robot.connect()`。

结果：

```text
PASS ming326/strike_match_3_subtask:
  annotation_only=['subtask', 'subtask_progress'], robot_connected=False

PASS ming326/nero_egg_subtask:
  annotation_only=['subtask', 'subtask_progress'], robot_connected=False
```

两个 dataset 的 metadata 在检查前后均保持不变。

## 累计回归结果

### T6 contract

命令：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_6.sh contract
```

最终退出码：`0`。

```text
107 passed
56 passed
30 passed
23 passed, 21 deselected
54 passed
119 passed, 4 deselected
```

### T6 regression

命令：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_6.sh regression
```

最终退出码：`0`。

```text
6 passed
266 passed, 6 deselected, 2 warnings
227 passed, 3 skipped
77 passed
14 passed, 4 deselected
```

两个 warning 是既有 subtask decode budget warning；三个 skip 是既有 RTC CUDA 条件用例；deselected 是验收脚本的
既有筛选，不是本次新增失败。

### T7 regression

命令：

```bash
plans/timer_pipeline/validate_subtask_time_milestone_7.sh regression
```

最终退出码：`0`。该入口重新执行 T6 contract/regression，并额外通过 time-disabled golden：

```text
2 passed
```

preflight 输出一次受限环境 NVML warning，CUDA 为 false；该 regression 模式为 CPU 验收，不要求 GPU。

## 静态门禁

以下检查通过：

```text
python -m py_compile: passed
git diff --check: passed
T6/T7 bash -n: passed（由验收入口执行）
ruff: skipped（lerobot-main 环境未安装）
```

## 修改文件

- `src/lerobot/scripts/lerobot_policy_deploy.py`
- `tests/scripts/test_subtask_time_deploy.py`
- `plans/timer_pipeline/subtask_time_deploy_metadata_compatibility_fix_completed.md`

## 剩余边界

本记录证明 metadata 过滤、严格硬件 mismatch 防护、连接前 early failure 和既有 timer 自动回归正确，不等同于 T8
实机验收。正式连接 Nero/Pico、真实三路相机、动作发送、home、急停、Pico 人工接管和物理 time on/off 对比仍需按
T8 安全流程单独执行。
