# pick_cube：chunk-relative 训练与 Piper 自带夹爪推理

从工程根目录运行以下命令。这次使用 `local/datasets/pick_cube`：40 集、16,523 帧、30 Hz、两路 640×480 视频。任务文本直接读取数据集：

```text
<control mode> end effector <control mode> pick up the yellow cube and drop it into the box
```

夹爪已经是 `gripper_width_mm`，保留毫米数值，不做角度映射、开度重标定或二值化。原始 `action[t]` 是下一帧的绝对 TCP 和夹爪宽度；准备脚本把 TCP 对齐到当前帧、将 rotation-6D 转为 rotvec，并以当前帧 `observation.state` 作为 `state` 和 `actions[:, 6]`。第 0 帧 TCP 沿用第一条可用位姿。采样窗口在训练时转换为 `inv(T_actions[0]) @ T_actions[k]`，夹爪仍为绝对毫米宽度。

## 本机数据与统计

本次已生成以下目录，不需要重复执行。原始数据保持不变，视频优先硬链接到新目录。

```bash
uv run examples/piper/chunk_relative/prepare_lightumi_chunk_relative.py \
  --source local/datasets/pick_cube \
  --output local/datasets/pick_cube_chunk_relative

uv run examples/piper/chunk_relative/compute_norm_stats.py \
  --dataset local/datasets/pick_cube_chunk_relative \
  --config-name pi05_piper_pick_cube_chunk_relative
```

输出数据约 1.3 GB；统计位于 `assets/pi05_piper_pick_cube_chunk_relative/pick_cube_chunk_relative/`，清单记录 `frame_count=16523`、`valid_action_rows=777150`。训练仍使用模型所需的归一化与反归一化；这里的“不处理夹爪”指不额外改变原始夹爪数值的物理含义。

## 上传到远程服务器

服务器应已有与本机兼容的 openpi 工程和依赖环境。在本机修改 SSH 地址及服务器工程路径后执行：

```bash
SERVER=user@host
REMOTE_DIR=/srv/openpi
ssh "$SERVER" "mkdir -p '$REMOTE_DIR'"
rsync -aR --info=progress2 \
  src/openpi/training/config.py \
  src/openpi/policies/piper_chunk_relative.py \
  local/datasets/pick_cube_chunk_relative \
  assets/pi05_piper_pick_cube_chunk_relative \
  "$SERVER:$REMOTE_DIR/"
```

`-R` 保留工程相对路径。数据和 assets 被 Git 忽略，单独推送代码不会上传它们。上面的命令会更新服务器对应代码文件，先保存服务器已有改动。

## 服务器训练

登录服务器，从工程根目录执行；将 `/srv/openpi` 改成实际路径：

```bash
cd /srv/openpi
uv sync --frozen
python - <<'PY'
import hashlib
import json
from pathlib import Path
info_path = Path('local/datasets/pick_cube_chunk_relative/meta/info.json')
info = json.loads(info_path.read_bytes())
manifest = json.loads(Path('assets/pi05_piper_pick_cube_chunk_relative/pick_cube_chunk_relative/norm_stats_manifest.json').read_text())
assert info['total_frames'] == manifest['frame_count'] == 16523
assert manifest['valid_action_rows'] == 777150
assert manifest['dataset_info_sha256'] == hashlib.sha256(info_path.read_bytes()).hexdigest()
assert info['gripper_mapping']['source_unit'] == 'mm'
assert info['gripper_mapping']['mapping_stage'] == 'none'
print('pick_cube dataset and normalization stats are ready')
PY
mkdir -p local/runs/wandb
WANDB_MODE=offline WANDB_DIR=local/runs/wandb XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_piper_pick_cube_chunk_relative --exp-name=pick_cube_native
```

配置为 pi0.5 全量微调，batch size 8、动作窗口 50、30,000 步，每 2,000 步保存检查点：
`checkpoints/pi05_piper_pick_cube_chunk_relative/pick_cube_native/<step>`。
使用这次新增的 `pi05_piper_pick_cube_chunk_relative` 配置；旧的 `pi05_piper_eef_chunk_relative` 使用不同相机键。

初始权重为 `gs://openpi-assets/checkpoints/pi05_base/params`，服务器需要可访问或已缓存。需要断开 SSH 后继续训练时，在 `tmux` 或服务器作业调度器中运行。恢复已有训练，在训练命令末尾加 `--resume`。若显存不足，可先将 `--batch-size=4` 加到训练命令中。

## 训练后启动模型服务

在服务器检查实际存在的检查点目录，再将 `<step>` 替换为目录名：

```bash
ls checkpoints/pi05_piper_pick_cube_chunk_relative/pick_cube_native
uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
  --policy.config=pi05_piper_pick_cube_chunk_relative \
  --policy.dir=checkpoints/pi05_piper_pick_cube_chunk_relative/pick_cube_native/<step>
```

检查点内应包含 `assets/pick_cube_chunk_relative/norm_stats.json`，迁移检查点时一并复制。

## 机械臂机器推理

新入口为 `infer_pick_cube.py`，复用 `infer_test.py` 的相机预览、裁剪、TCP 插值与推理日志。它读取真实夹爪反馈作为 1D 状态，通过同一个 Piper CAN 控制自带夹爪，不连接 DM 串口，也不使用固定零状态。模型输出第 7 维直接作为毫米命令，预测倍率为 1、连续控制。SDK 的单位换算是 `raw/1000` 和 `round(mm*1000)`；发送命令时限制到物理范围 `0..70 mm`。训练数据中约 `-0.593 mm` 的最小读数按原值保留。

推理机器需要这次更新的脚本和现有运行环境，以及准备后数据的 `meta/info.json`、`meta/tasks.jsonl`。相机预览要同时保存训练参考图时，还需要对应视频。

本次相机对应关系如下：

| 用途 | 设备和参数 | 对应训练数据 |
| --- | --- | --- |
| 全局相机 | D435，`--base-camera <RealSense序列号>` | `observation.images.global`，640×480 |
| 腕部相机 | USB，`--wrist-camera 0` 表示 `/dev/video0` | `observation.images.camera`，640×480 |

USB 腕部的处理顺序为：**采集原图 → 按 `--wrist-crop X Y W H` 裁剪 → 缩放到训练尺寸 640×480 → 等比缩放并填充到模型输入 224×224**。裁剪坐标以原始 USB 帧为准。正式推理必须显式传入裁剪框，缺失或越界会报错。

本次 `pick_cube` 的训练参考帧保留鱼眼圆边，与已保存的 USB 640×480 原图取景范围接近。若仍使用相同 USB 相机和输出模式，先用 `--wrist-crop 0 0 640 480` 保留整帧；不要仅为去掉黑边而额外裁剪。旧的预览快照不能代替当前设备核对，先生成新的预览：

```bash
uv run examples/piper/chunk_relative/infer_pick_cube.py \
  --dataset local/datasets/pick_cube_chunk_relative \
  --wrist-camera 0 --wrist-capture-size 640 480 --wrist-crop 0 0 640 480 \
  --preview-wrist local/runs/outputs/pick_cube_wrist_preview
```

比较输出的 `raw.png`（USB 原图）、`prepared.png`（裁剪后的训练尺寸画面）、`model_input.png` 与 `reference_dataset.png`、`reference_model_input.png`。参考图读取本次 `pick_cube` 视频第 0 集第 30 帧。重点核对鱼眼视场、工具在图中的位置和物体比例；`preview_report.json` 记录实际采集尺寸和裁剪框。

若原始 USB 画面需要局部裁剪，修改 `--wrist-capture-size` 和 `--wrist-crop` 后重新预览。例如原图为 1280×720 时，`--wrist-crop 100 50 960 640` 表示从 `(100,50)` 开始取 960×640，再缩放为 640×480；这只是参数示例，实际区域应与训练画面一致。**预览与正式推理使用相同的采集尺寸、裁剪、曝光和白平衡参数。**

若通过 SSH 隧道访问服务器，在机械臂机器另开终端执行：

```bash
ssh -N -L 6006:localhost:6006 user@host
```

随后运行推理；将相机序列号与 `--tcp-offset-m` 改为实测值。偏移为 J6 法兰到当前夹爪工具 TCP 的平移，单位米；更换为自带夹爪后需按实际工具设置，示例中的 `0 0 0.22` 只是工程默认值。

```bash
uv run examples/piper/chunk_relative/infer_pick_cube.py \
  --dataset local/datasets/pick_cube_chunk_relative \
  --host localhost --port 6006 --can-name can0 \
  --base-camera 112222070761 --wrist-camera 0 \
  --wrist-capture-size 640 480 --wrist-crop 0 0 640 480 --wrist-fourcc YUYV \
  --steps 1 --tcp-offset-m 0 0 0.22 --visualize-trajectory
```

启动会使能机械臂和夹爪；沿用现有控制器的原生夹爪初始化，会下发零开度命令。按 Enter 开始推理，`s` 暂停、`q` 退出。每次跳过模型第 0 个同帧动作，执行 `--steps` 个未来目标，再以实测 TCP 重新锚定；可选范围为 `1..49`。默认 30 Hz 插值，平移/旋转速度限制为 `0.05 m/s`、`0.3 rad/s`。无桌面环境时省略 `--visualize-trajectory`。

日志默认写入 `local/runs/outputs/pick_cube_inference_logs/<timestamp_pid>/`，包含原始模型动作、模型输入图像和发送的 TCP 命令。此改动已做数据与无硬件回归验证，尚未做远程 GPU 训练或真机运行。
