# Nero 双臂遥操作 / 数据采集 / 训练 / 部署 指南

本文档介绍在 LeRobot 上为 **Nero 双臂机器人 + Pico XR 遥操作器** 新增的一整套工具链，
覆盖从遥操作、数据采集、数据集构建、上传、训练到策略部署的完整流程。

> 运行环境：本仓库所有命令默认在虚拟环境 **`lerobot-main`** 中执行。
> 上手前需要按自己的硬件配置两处：相机序列号（见 [相机配置](#2-相机配置orbbec-序列号)）和
> USB→CAN 端口映射（见 [配置 USB 口](#配置-usb-口-️)）。

---

## 目录

- [整体流程](#整体流程)
- [一次性准备](#一次性准备)
  - [1. 激活 CAN（`activate_can.sh`）](#1-激活-canactivate_cansh)
  - [2. 相机配置（Orbbec 序列号）](#2-相机配置orbbec-序列号)
  - [3. 机器人与遥操作器](#3-机器人与遥操作器)
- [脚本说明](#脚本说明)
  - [run_bi_teleop.py — 快速遥操作自检](#run_bi_teleoppy--快速遥操作自检)
  - [lerobot-hil-record — 直接生成 LeRobotDataset](#lerobot-hil-record--直接生成-lerobotdataset)
  - [原始数据工作流：raw-record / annotate-subtask / build-dataset](#原始数据工作流raw-record--annotate-subtask--build-dataset)
  - [lerobot-push-dataset — 上传到 HuggingFace](#lerobot-push-dataset--上传到-huggingface)
  - [nero_candle_pi0_relative.sh — 下载 / 统计 / 训练](#nero_candle_pi0_relativesh--下载--统计--训练)
  - [lerobot-policy-deploy — 部署（含 RTC）](#lerobot-policy-deploy--部署含-rtc)

---

## 整体流程

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│ 硬件准备     │ →  │ activate_can.sh（CAN）+ 相机序列号 + 机器人/遥操作配置 │
└─────────────┘     └──────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                                 ▼
   ┌────────────────────┐                          ┌──────────────────────┐
   │ run_bi_teleop.py   │ 遥操作自检（不存数据）       │  采数据              │
   └────────────────────┘                          └──────────────────────┘
                                                              │
                       ┌──────────────────────────────────────┴───────────────┐
                       ▼                                                        ▼
        ┌──────────────────────────────┐               ┌──────────────────────────────────────────┐
        │ A) lerobot-hil-record         │               │ B) lerobot-raw-record（存原始数据）          │
        │    直接生成 LeRobotDataset     │               │    → lerobot-annotate-subtask（可选标注）   │
        │                               │               │    → lerobot-build-dataset（构建数据集）     │
        └──────────────────────────────┘               └──────────────────────────────────────────┘
                       │                                                        │
                       └───────────────────────┬────────────────────────────────┘
                                               ▼
                                  ┌──────────────────────────┐
                                  │ lerobot-push-dataset      │ 上传到 HuggingFace Hub
                                  └──────────────────────────┘
                                               │
                                               ▼
                      ┌────────────────────────────────────────────────┐
                      │ nero_candle_pi0_relative.sh                     │
                      │   download → (relative 时) stats → verify-stats │
                      │   → train                                        │
                      └────────────────────────────────────────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │ lerobot-policy-deploy     │ 上机部署（RTC 实时分块推理）
                                  └──────────────────────────┘
```

采集数据有两条路线：

- **A（直采）**：`lerobot-hil-record` 实时录制时直接落盘成 `LeRobotDataset`，最快。
- **B（原始数据 → 后处理）**：`lerobot-raw-record` 先保存原始数据（逐帧 PNG + parquet），
  可以单独删/查/标注 episode，再用 `lerobot-build-dataset` 构建成 `LeRobotDataset`。
  适合需要做 subtask 标注、特征筛选、或想保留原始数据二次处理的场景。

---

## 一次性准备

### 1. 激活 CAN（`activate_can.sh`）

双臂通过两路 USB-to-CAN 适配器与上位机通信，使用前必须先激活并把网卡重命名为 `left` / `right`。

```bash
./scripts/nero_teleop/activate_can.sh
# 若 CAN 数量与预期不符想跳过校验，可加 --ignore：
./scripts/nero_teleop/activate_can.sh --ignore
```

`activate_can.sh` 是一个薄封装，默认调用**同目录下随仓库附带的** `can_muti_activate.sh`
（从 `nero_pyagxarm` vendored 进来，便于直接在本仓库里改 USB 口映射）：

```bash
# scripts/nero_teleop/activate_can.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAN_SCRIPT="${NERO_CAN_SCRIPT:-${SCRIPT_DIR}/can_muti_activate.sh}"
exec bash "$CAN_SCRIPT" "$@"
```

- 如需改用别处的脚本，可设环境变量 `NERO_CAN_SCRIPT=/path/to/can_muti_activate.sh`。
- 脚本会 `modprobe gs_usb`，枚举系统 CAN 接口，用 `ethtool` 读出每个接口的 USB `bus-info`，
  再按预定义映射把接口重命名为 `left` / `right` 并设置 1 Mbps 比特率。

#### 配置 USB 口 ⚠️

USB 口到左右臂的映射在 **`scripts/nero_teleop/can_muti_activate.sh`** 顶部：

```bash
declare -A USB_PORTS
USB_PORTS["1-5.2:1.0"]="right:1000000"   # 右臂：USB 口 1-5.2:1.0，1 Mbps
USB_PORTS["1-5.1:1.0"]="left:1000000"    # 左臂：USB 口 1-5.1:1.0，1 Mbps
```

换主机、换 USB 插口或线序变化时，需要更新这里的 `1-5.x:1.0` 端口号。
查询当前 CAN 接口对应的 USB 端口：

```bash
ip link show type can                 # 列出 canX 接口
ethtool -i can0 | grep bus-info       # 读出该接口的 USB bus-info（即 1-5.x:1.0）
```

把读到的 `bus-info` 填到 `USB_PORTS` 里对应 `left` / `right` 即可。

---

### 2. 相机配置（Orbbec 序列号）

当前使用 **Orbbec** 相机，通过 `pyorbbecsdk v2` 接入。相机在机器人配置里以字典形式声明，
**每台相机靠 `serial_number_or_name` 字段定位**。

默认相机配置在：

```
src/lerobot/robots/bi_nero_follower/config_bi_nero_follower.py
```

```python
def nero_left_cameras_config() -> dict[str, CameraConfig]:
    return {
        "left_wrist": OrbbecCameraConfig(
            serial_number_or_name="CP2AB530007Z",   # ← 改成你左腕相机的序列号
            width=640, height=480, fps=30,
        ),
        "third_person": OrbbecCameraConfig(
            serial_number_or_name="CP2R553000EP",    # ← 第三人称相机序列号
            width=640, height=480, fps=30,
        ),
    }

def nero_right_cameras_config() -> dict[str, CameraConfig]:
    return {
        "right_wrist": OrbbecCameraConfig(
            serial_number_or_name="CP2R553000NZ",    # ← 右腕相机序列号
            width=640, height=480, fps=30,
        ),
    }
```

`OrbbecCameraConfig`（`src/lerobot/cameras/orbbec/configuration_orbbec.py`）的可配置字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `serial_number_or_name` | `None` | **相机序列号或名称**，用于唯一定位设备 |
| `color_mode` | `RGB` | 颜色模式 |
| `rotation` | `NO_ROTATION` | 图像旋转 |
| `warmup_s` | `1` | 预热秒数 |
| `auto_exposure` | `True` | 自动曝光 |
| `exposure` | `300` | 关闭自动曝光时的手动曝光值 |

**如何拿到序列号**：把 Orbbec 相机接上后，可用 pyorbbecsdk 自带的设备枚举示例列出在线设备的
serial number，或参考厂商工具。拿到后替换上面三处 `serial_number_or_name` 即可。

---

### 3. 机器人与遥操作器

#### 机器人：`bi_nero_follower`

- 实现：`src/lerobot/robots/bi_nero_follower/`
- 注册类型名：`bi_nero_follower`
- 结构：内部封装两个单臂 `NeroFollower`（`src/lerobot/robots/nero_follower/`），分别带 `left_` / `right_` 前缀。
- 每条臂：**7 个关节**（`joint_1.pos` … `joint_7.pos`）+ **AGX 夹爪**（`gripper.pos`，行程 0–0.1 m），
  并会输出末端位姿（`ee_x/y/z`、`ee_roll/pitch/yaw`）。
- 控制模式（`control_mode`）：`j`（move_j 关节位控）/ `js`（直通）/ `mit`（MIT 阻抗 + 重力补偿，默认）。

单臂配置 `NeroFollowerConfigBase` 关键字段（`config_nero_follower.py`）：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `can_channel` | — | `"left"` / `"right"`，对应 [步骤 1](#1-激活-canactivate_cansh) 重命名的接口 |
| `firmware_version` | `"auto"` | 固件版本：`auto` / `default` / `v111` |
| `control_mode` | `"mit"` | 控制模式 |
| `speed_percent` | `60` | 电机速度百分比 0–100 |
| `control_hz` | `90` | 执行线程频率 |
| `gripper_force_n` | `1.0` | 夹爪力（N，0–3） |
| `home_joints_rad` | `[0,0.35,0,1.75,0,0,-0.6]` | 回零关节角 |
| `mit_kp` / `mit_kd` | 见源码 | MIT 模式各关节增益 |
| `mit_gravity_urdf_path` | 见源码 | 重力补偿用 URDF（按仓库相对路径自动解析，一般无需改） |

> 双臂顶层配置 `BiNeroFollowerConfig` 里有一个顶层 `cameras` 字段：当它非空时会被赋给左臂，
> 此时各臂自带的相机配置会被忽略。默认留空，使用上面 `nero_left/right_cameras_config()` 的分臂相机。

#### 遥操作器：`bi_pico_nero_teleop`

- 实现：`src/lerobot/teleoperators/bi_pico_nero_teleop/`，单臂版在 `pico_nero_teleop/`。
- 注册类型名：`bi_pico_nero_teleop`
- 输入设备：**Pico / Meta Quest XR 手柄**，通过 `xrobotoolkit_sdk` 本地读取手柄位姿与按键
  （**不是网络遥操作**，命令读到后经 IK 直接走 CAN 发给机械臂）。
- IK：使用 Placo 求解器，默认 90 Hz。
- 按键约定：左臂 home 键 `Y`，右臂 home 键 `B`；grip 扳机作为"死人开关"激活，trigger 控制夹爪开合。

单臂配置 `PicoNeroTeleopConfigBase` 关键字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `side` | `"left"` | 左/右 |
| `ik_hz` | `90` | XR 轮询 + IK 更新频率 |
| `translation_scale` / `rotation_scale` | `1.0` | 手柄→末端平移/旋转缩放 |
| `xr_yaw_quadrants` | `0` | 额外绕 Z 轴 90° 旋转的象限数 |
| `trigger_gripper_scale` | `0.5` | trigger→夹爪开度缩放（1.0=全程，0.5=半程） |
| `home_button` | `"Y"` | 回零按键 |
| `urdf_path` | 见源码 | IK 用 URDF（按仓库相对路径自动解析，一般无需改） |

---

## 脚本说明

### run_bi_teleop.py — 快速遥操作自检

端到端的双臂遥操作 runner：同时启动左右臂、连接 Pico 手柄、跑一个高频动作循环直到 Ctrl-C。
**不录制数据**，用于正式采集前的快速上电自检与手感调试。

```bash
python scripts/nero_teleop/run_bi_teleop.py
# 例：把 XR 偏航映射旋转一个象限
python scripts/nero_teleop/run_bi_teleop.py --xr-yaw-quadrants 1
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--firmware` | `auto` | 固件版本 `auto`/`default`/`v111` |
| `--speed` | `60` | 电机速度百分比 |
| `--move-mode` | `mit` | 控制模式 `j`/`js`/`mit` |
| `--solver-dt` | `0.1` | IK 求解步长 |
| `--rotation-scale` | `1.0` | 旋转灵敏度 |
| `--xr-yaw-quadrants` | `0` | XR 偏航象限映射 {0,1,2,3} |
| `--trigger-gripper-scale` | `1.0` | trigger→夹爪开度缩放 |
| `--smoother-alpha` | `0.2` | 关节目标的 EMA 平滑系数 |
| `--mit-gravity-factor` | `1.0` | 重力补偿系数 |
| `--control-hz` | `90.0` | 控制循环频率 |
| `--obs-stride` | `3` | 观测打印步长 |
| `--viz` / `--viz-ip` / `--viz-port` / `--viz-session` | — | 启用 Rerun 可视化监控 |

---

### lerobot-hil-record — 直接生成 LeRobotDataset

实时录制并**直接落盘成 `LeRobotDataset`**（写到 `$HF_LEROBOT_HOME/{repo_id}`）。
支持三种 `--mode`：

- `teleop`：纯人工遥操作。
- `policy`：纯策略自动执行。
- `hil`：人在回路（Human-in-the-Loop），录制中可在策略/人工之间切换、暂停、做纠正。

你给的双臂遥操作采集示例：

```bash
lerobot-hil-record \
  --mode=teleop \
  --robot.type=bi_nero_follower \
  --teleop.type=bi_pico_nero_teleop \
  --display_data=true \
  --dataset.repo_id=<HF_USER>/<DATASET_REPO_ID> \
  --dataset.push_to_hub=false \
  --control_multiplier=3
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mode` | `hil` | `teleop` / `policy` / `hil` |
| `--robot.type` | 必填 | 机器人类型，如 `bi_nero_follower` |
| `--teleop.type` | mode 含 teleop/hil 时必填 | 遥操作器类型，如 `bi_pico_nero_teleop` |
| `--policy.path` | mode 含 policy/hil 时必填 | 预训练策略路径 |
| `--dataset.repo_id` | 必填 | 数据集 repo_id |
| `--dataset.fps` | `30` | 录制帧率 |
| `--dataset.push_to_hub` | `false` | 录完是否直接推 Hub（**建议 false**，单独用 push 脚本上传） |
| `--display_data` | `true` | 是否开 Rerun 可视化 |
| `--control_multiplier` | `3` | 控制频率 = `fps × multiplier` |
| `--smoother_alpha` | `0.2` | 关节 EMA 平滑（夹爪不平滑） |
| `--policy_gripper_max_width_m` | `0.1` | 夹爪最大开度上限 (0, 0.1] |

键盘控制（录制时）：`→` 开始/确认准备，`Enter` 保存当前 episode，`←` 取消当前 episode，
`Space` 暂停（HIL），`q` 切到策略（HIL），`e` 切到人工纠正（HIL），`Esc` 退出。

---

### 原始数据工作流：raw-record → annotate-subtask → build-dataset

当需要保留原始数据、做特征筛选或 subtask 标注时，走这条路线。

#### lerobot-raw-record — 录制原始数据

和 `lerobot-hil-record` 参数基本一致（同样有 `teleop`/`policy`/`hil` 三种模式），
但**不直接生成 LeRobotDataset**，而是把原始数据按 episode 落盘，便于逐条检查/删除/标注：

```
<root>/raw/<repo_id>/
  run_meta.json            # 特征 schema + 本次 run 配置
  ep_000000/
    info.json              # 单 episode 元数据
    frames.parquet         # 标量特征（关节、夹爪等）
    events.jsonl           # 源切换 / 暂停 / subtask 标记等事件
    <cam_key>/000000.png   # 每个相机一个目录，逐帧 PNG
  ep_000001/ ...
```

```bash
lerobot-raw-record \
  --mode=teleop \
  --robot.type=bi_nero_follower \
  --teleop.type=bi_pico_nero_teleop \
  --dataset.repo_id=<RAW_NAME> \
  --control_multiplier=3
```

额外能力：`h` 键可在录制前把机器人归到已知 home 位姿（`--home_joints_rad` / `--home_speed_rad_s`），
逐帧事件日志，异步 PNG 写盘。

#### lerobot-annotate-subtask — 标注 subtask（可选）

本地网页标注器，对某个原始 run 目录做逐帧/分段的 subtask 标注，导出为 `extras.parquet` 列，
随后会被 `lerobot-build-dataset` 自动发现并合并进数据集。

```bash
lerobot-annotate-subtask --root <root>/raw/<repo_id>
# 然后浏览器打开 http://127.0.0.1:8000
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--root` | 必填 | 原始 run 目录 |
| `--host` | `127.0.0.1` | 服务地址 |
| `--port` | `8000` | 端口 |
| `--no-browser` | `false` | 不自动打开浏览器 |

#### lerobot-build-dataset — 构建 LeRobotDataset

把一个或多个原始 run 目录合并构建为 `LeRobotDataset`（含视频编码、特征筛选、subtask 合并）：

```bash
lerobot-build-dataset \
  --runs <root>/raw/<repo_id>/ \
  --output_repo_id <HF_USER>/<DATASET_REPO_ID> \
  --video true
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--runs` | 必填 | 一个或多个原始 run 目录 |
| `--output_repo_id` | 必填 | 输出数据集 repo_id |
| `--output_root` | `None` | 输出位置（默认 `$HF_LEROBOT_HOME/{repo_id}`） |
| `--video` | `true` | 图像编码为视频 |
| `--include_features` / `--exclude_features` | `*` / 空 | 特征通配筛选 |
| `--task_override` | `None` | 覆盖每条 episode 的任务描述 |
| `--push_to_hub` | `false` | 构建后是否推 Hub |
| `--dry_run` | `false` | 仅预览不写盘 |

> 它会校验所有 run 的 fps / robot_type / 特征 schema 一致，并保留/合并 `extras.parquet`（如 subtask 标注）。

---

### lerobot-push-dataset — 上传到 HuggingFace

把本地录好的 `LeRobotDataset` 上传到 HuggingFace Hub：

```bash
lerobot-push-dataset \
  --repo_id <HF_USER>/<DATASET_REPO_ID> \
  --upload-large-folder
```

> 公司网络内若需走代理，可加 `--proxy http://<代理地址>:<端口>`；部分代理还需配合 `--tls-max-1-2`。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--repo_id` | 必填 | HF 数据集 repo，如 `<HF_USER>/<DATASET_REPO_ID>` |
| `--root` | `None` | 本地数据集目录（默认 `$HF_LEROBOT_HOME/{repo_id}`） |
| `--branch` | `None` | 上传到指定分支 |
| `--private` | `false` | 私有数据集 |
| `--proxy` | `None` | HTTP/SOCKS 代理 URL（公司网络内常用） |
| `--disable-xet` | `false` | 禁用 Xet，代理网络不稳定时改用标准 HTTP/LFS 上传 |
| `--tls-max-1-2` | `false` | 限制 TLS 最高 1.2（部分代理兼容需要） |
| `--upload-large-folder` | `false` | 用 `upload_large_folder`，适合超大数据集 |
| `--num-workers` | `None` | 大文件夹上传的并发线程数 |
| `--dry-run` | `false` | 仅加载并汇总，不真正上传 |
| `--no-push-videos` | `false` | 跳过 `videos/` 目录 |

---

### nero_candle_pi0_relative.sh — 下载 / 统计 / 训练

针对 `strike_match_3_subtask` 数据集、**relative action（相对动作）** 的 Pi0 / Pi0.5 训练一站式脚本，
封装了数据/模型下载、统计重算、以及 Accelerate 多卡训练。

```bash
# 下载数据集
./scripts/nero_teleop/nero_candle_pi0_relative.sh download

# 查看数据集信息 / 统计 / 校验统计
./scripts/nero_teleop/nero_candle_pi0_relative.sh info
./scripts/nero_teleop/nero_candle_pi0_relative.sh stats
./scripts/nero_teleop/nero_candle_pi0_relative.sh verify-stats

# 训练（可用环境变量覆盖默认配置）
DATASET_REPO_ID=<HF_USER>/<DATASET_REPO_ID> \
./scripts/nero_teleop/nero_candle_pi0_relative.sh train
```

> 脚本里的 `DATASET_REPO_ID` 和 `DATASTORE_ROOT` 用的是通用占位默认值，
> 实际使用时通过上面的环境变量覆盖成你自己的数据集与存储路径即可（无需改脚本）。

支持的子命令：

| 子命令 | 作用 |
| --- | --- |
| `env` | 打印解析后的环境与数据集路径 |
| `check` | 校验 Python 环境能否 import LeRobot |
| `download` | 下载/物化数据集 |
| `download-policy` | 缓存预训练策略 |
| `download-tokenizer` | 缓存 tokenizer |
| `download-all` | 上面三项一起做 |
| `info` | 显示数据集元数据与特征信息 |
| `stats` | **用相对动作变换重算 action 统计** |
| `verify-stats` | 显示 `meta/stats.json` 里算好的 action 统计 |
| `train-command` | 只打印 Accelerate 启动命令模板（不执行） |
| `train` | 执行完整 Pi0 / Pi0.5 训练 |

> **为什么 relative 要重算 stats**：使用相对动作（`use_relative_actions=true`）时，
> action 的数值分布与绝对动作不同，必须用 `stats` 重新计算归一化统计，否则归一化会错。
> 因此 relative 流程的标准顺序是：`download` → `stats` → `verify-stats` → `train`。

常用环境变量（均有默认值，可在命令前覆盖）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATASET_REPO_ID` | `ming326/strike_match_3_subtask` | 数据集 repo |
| `POLICY_TYPE` | `pi05` | 策略类型；可覆盖为 `pi0` |
| `POLICY_PRETRAINED_PATH` | `lerobot/pi05_base` | 预训练权重；`POLICY_TYPE=pi0` 时默认用 `lerobot/pi0_base` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5` | 默认只使用前 6 张卡 |
| `GLOBAL_BATCH_SIZE` | `192` | 全局 batch（6 卡时每卡 32） |
| `NUM_GPUS` | `6` | GPU 数 |
| `POLICY_COMPILE` | `true` | 正式训练默认开启；20 步 smoke 建议覆盖为 `false` |
| `SUBTASK_MAX_TOKENS` / `SUBTASK_MAX_DECODE_TOKENS` | `16` / `16` | 当前 subtask 标签较短，避免 48 token padding 造成额外计算 |
| `STEPS` | `20000` | 训练步数 |
| `SAVE_FREQ` / `LOG_FREQ` | `1000` / `50` | 保存/日志频率 |
| `RELATIVE_EXCLUDE_JOINTS` | `['gripper']` | 相对动作里排除的关节（夹爪用绝对值） |
| `DATASTORE_ROOT` | `/datastore01/hongming` | 数据/缓存/输出根目录，用时覆盖成自己的 |
| `OUTPUT_DIR` | `${DATASTORE_ROOT}/lerobot_outputs/${JOB_NAME}` | 训练输出目录 |

---

### lerobot-policy-deploy — 部署（含 RTC）

把训练好的策略上机部署，使用 **RTC（Real-Time Chunking，实时分块推理）** + 键盘控制。
你给的部署示例：

```bash
lerobot-policy-deploy \
  --robot.type=bi_nero_follower \
  --policy.path=<OUTPUT_DIR>/<JOB_NAME>/checkpoints/<STEP>/pretrained_model \
  --rtc.execution_horizon=10 \
  --rtc_queue_threshold=40 \
  --dataset.repo_id=ming326/strike_match_3_subtask \
  --dataset.root=/home/zenbot-robot/.cache/huggingface/lerobot/ming326/strike_match_3_subtask \
  --dataset.fps=30 \
  --dataset.task="Pick up the match in front, strike it to light it, then use it to light the small candle on the cake in front." \
  --policy_gripper_max_width_m=0.05 \
  --policy.dtype=bfloat16 \
  --policy.compile_model=false
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--robot.type` | 必填 | 机器人类型，如 `bi_nero_follower` |
| `--policy.path` | 必填 | 训练产物 `pretrained_model` 目录 |
| `--dataset.fps` | `30` | 观测/控制频率（Hz） |
| `--dataset.task` | — | 任务描述（语言条件策略需要） |
| `--dataset.repo_id` | `None` | 参考数据集（用于校验元数据） |
| `--dataset.root` | `None` | 本地参考数据集路径；elapsed-time 部署使用本地数据时应显式提供 |
| `--use_subtask_time_conditioning` | `None` | 默认跟随 checkpoint；只能用 `false` 关闭做 ablation，不能给旧 checkpoint 强制开启 |
| `--subtask_time_deployment_margin_seconds` | `5.0` | 每个 subtask 的部署时间上限为数据集最大真实持续时间加该 margin |
| `--control_multiplier` | `3` | 控制频率 = `fps × multiplier` |
| `--policy_gripper_max_width_m` | `0.1` | 夹爪最大开度上限 (0, 0.1] |
| `--policy.dtype` | — | 推理精度，如 `bfloat16` |
| `--policy.compile_model` | — | 是否 `torch.compile`（首次部署调试建议 `false`） |
| `--interpolation_multiplier` | `1` | 动作上采样倍数 |
| `--smoother_alpha` | `1.0` | EMA 平滑（1.0=关闭） |

键盘控制：`→` 启动策略，`Space` 暂停，`h`（暂停时）回 home，`Esc` 退出。

当 checkpoint 启用了 subtask elapsed-time conditioning 时，部署会从 `--dataset.repo_id` 指定的逐帧
`subtask` 标注中严格提取固定顺序和各段最大持续时间，不在代码中硬编码任务文本。启动或 home 后，状态面板先显示
`[TIME] waiting-for-first-subtask`；模型成功提交序列第一项后才启动 monotonic timer。`Space` 会清空旧动作队列、观测和
policy/processor runtime cache，同时冻结并保留当前 subtask 时间；再次按 `→` 从冻结值继续，暂停墙钟时间不累计。
暂停后按 `h` 会执行 full reset 并清空 tracker，home 完成后必须重新从序列第一项开始。面板中的 `raw` 是实际 active
elapsed，`input` 是最近一次送给模型的截断值，`cap` 是数据集 maximum 加 margin。

#### RTC（实时分块）参数说明

RTC 来自 Physical Intelligence，把动作生成当作"修复/补全"问题：策略每次只生成一小段
带重叠时间步的动作"块"，用 prefix attention 条件于之前的输出，从而**降低推理延迟、避免长序列等待**。
部署时 RTC 默认开启。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--rtc.execution_horizon` | `10` | 每次推理生成/执行的动作步数（执行视野） |
| `--rtc_queue_threshold` | `40` | 动作队列上限；超过则暂停推理，防止队列无限增长 |
| `--rtc.prefix_attention_schedule` | `LINEAR` | 重叠区前缀注意力的权重调度 |
| `--rtc.max_guidance_weight` | `10.0` | 引导权重上限（稳定性） |

工作方式：策略在后台线程跑（`RTCInferenceEngine`），主循环按 `control_hz` 取动作，
`ActionInterpolator` 把动作上采样到控制频率；队列满时推理线程暂停"等"主循环消费。
部署时每秒会打印一次频率、队列大小、推理延迟等诊断信息。
