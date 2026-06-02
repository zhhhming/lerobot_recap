# Subtask 标注器

给 `lerobot-raw-record` 生成的 raw run 标注 **subtask**（每帧一个字符串标签），
导出后可被 `lerobot-build-dataset` 直接合并进 LeRobotDataset 作为每帧 feature 用于训练。

## 运行

```bash
conda activate lerobot-main
cd tools/subtask_annotator
python server.py --root ~/.cache/huggingface/lerobot/raw/user/my_raw_data
# 浏览器自动打开 http://127.0.0.1:8000
```

参数：`--port`（默认 8000）、`--host`（默认 127.0.0.1）、`--no-browser`（不自动开浏览器）。
只用到 Python 标准库 + pyarrow（lerobot 环境已自带），无需 pip 安装。

## 数据流

```
raw run/                              标注器写入                   build_dataset 读取
  ep_000000/
    frames.parquet  ── 读取关节/动作 ──┐
    left_wrist/*.png ─ 读取展示 ───────┤
    ...                               │
    extras.parquet  ←── 导出写入 ──────┘  ←──── 合并成 LeRobotDataset 的 subtask feature
  annotation_config.json  ←── subtask 调色板 / feature 名（整个 run 共用）
  annotations.json        ←── 每个 episode 的关键帧 + 每帧标签
```

- **`annotation_config.json`**：`feature_name`（默认 `subtask`）、`default_value`（未标注帧的填充值，默认空串）、`subtasks`（`[{name,color}]`，整个数据集共用）。
- **`annotations.json`**：`{ "<episode_idx>": { keyframes:[...], labels:[每帧标签] } }`，是可读、可手动编辑的标注真值。
- **导出**：点右上角「导出 extras.parquet」，给**每个** episode 写 `extras.parquet`（满足 build_dataset 的「全有或全无」约束）；未标注帧填 `default_value`。

导出后照常构建数据集：

```bash
python src/lerobot/scripts/lerobot_build_dataset.py \
  --runs '["~/.cache/huggingface/lerobot/raw/user/my_raw_data"]' \
  --output_repo_id user/my_dataset
```

最终数据集每帧会多出一个 `subtask` 字符串 feature。

## 操作说明

**Subtask 列表（左上）**：整个数据集共用、可复用到所有 episode。
- 输入名字点「添加」新建；点色块改颜色；双击名字重命名；✕ 删除。

**两种标注模式**：
- **区间(关键帧)**：把某个 subtask 色块**拖到时间轴**的某一段，即填充该段两个关键帧之间（含两端）的所有帧；也可先单击某段选中、再点 subtask。
  - episode 开头和结尾默认是关键帧。在「关键帧」条上双击可新增/删除关键帧，也可拖动关键帧移动边界。
  - 例：选中末端关键帧前面那一段拖入 subtask，则从开头到该关键帧整段都被标注。
- **单帧**：把 subtask 拖到某一帧，或选中后按数字键，只标注当前播放头所在帧。

**时间轴 / 曲线（中部画布）**：x 轴为整段 episode；黄色竖线为当前帧。
- 单击空白处或拖动可移动播放头（scrub）；三张相机图随之更新。
- subtask 段按颜色显示在最上面一条，未标注为灰底。

**关节显示（左下）**：每个关节可单独开启 `A`(action) / `O`(observation) 曲线，
各曲线独立自动缩放 y 轴并显示当前帧数值。「只看 gripper」快速选择左右夹爪。

**快捷键**：`←/→` 上/下一帧，`空格` 播放/暂停，`k` 或 `Ctrl+Enter` 当前帧加/删关键帧
（用 `←/→` 把黄线移到位再按一下即可打点），
`1`–`9` 选择对应 subtask 并按当前模式应用，`0` 选择「清除」。

标注会自动保存（450ms 防抖）到 `annotations.json`，右上角显示保存状态。
