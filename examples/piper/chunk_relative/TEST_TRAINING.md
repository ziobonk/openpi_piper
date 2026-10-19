# 用 `local/datasets/test` 在服务器训练 chunk-relative 策略

源数据为 30 Hz lightUMI 数据：40 集、14,011 帧、两路视频。`action` 是下一对齐帧的 9D 绝对 TCP（xyz + rotation-6D），没有夹爪编码器。准备脚本将下一帧 TCP 移到同帧并转为 rotvec；第 0 帧使用第一条可用 TCP。训练数据的 `state` 和 `actions[:, 6]` **全部为 0**，只训练 TCP 位姿，不提供可用的夹爪预测。原始 `local/datasets/test` 保持不变。

## 本机准备与核对

在工程根目录执行。已生成的目录存在时，准备脚本会拒绝覆盖；无需重复执行。

```bash
uv run examples/piper/chunk_relative/prepare_lightumi_chunk_relative.py \
  --source local/datasets/test \
  --output local/datasets/test_chunk_relative

uv run examples/piper/chunk_relative/compute_norm_stats.py \
  --dataset local/datasets/test_chunk_relative \
  --config-name pi05_piper_test_chunk_relative
```

准备后的数据在 `local/datasets/test_chunk_relative`，归一化文件在
`assets/pi05_piper_test_chunk_relative/test_chunk_relative/`。统计只使用各 episode 的有效动作窗口；夹爪零通道使用稳定的分位范围。当前统计清单应记录 `frame_count=14011`、`valid_action_rows=651550`。

## 同步到服务器

在**本机工程根目录**填入 SSH 地址和服务器上已有的 openpi 工程目录，再执行：

```bash
SERVER=user@host
REMOTE_DIR=/srv/openpi
ssh "$SERVER" "mkdir -p '$REMOTE_DIR'"
rsync -aR --info=progress2 \
  examples/piper/chunk_relative/prepare_lightumi_chunk_relative.py \
  examples/piper/chunk_relative/compute_norm_stats.py \
  src/openpi/policies/piper_chunk_relative.py \
  src/openpi/training/config.py \
  local/datasets/test_chunk_relative \
  assets/pi05_piper_test_chunk_relative \
  "$SERVER:$REMOTE_DIR/"
```

`rsync -R` 保留工程内相对路径。约 790 MB 的准备后数据包含 80 个视频；无需上传原始 `test`。这些目录被 Git 忽略，因此仅推送代码分支不会传输数据或统计。同步的代码文件应与服务器工程其余代码版本兼容；同步前不要覆盖服务器上未保存的改动。

## 服务器启动训练

登录服务器，在工程根目录执行：

```bash
cd /srv/openpi
uv sync --frozen
python - <<'PY'
import json
from pathlib import Path
info = json.loads(Path('local/datasets/test_chunk_relative/meta/info.json').read_text())
manifest = json.loads(Path('assets/pi05_piper_test_chunk_relative/test_chunk_relative/norm_stats_manifest.json').read_text())
assert info['total_frames'] == manifest['frame_count'] == 14011
assert manifest['valid_action_rows'] == 651550
assert info['gripper_mapping']['available'] is False
print('Dataset, gripper placeholder, and normalization manifest are ready')
PY
mkdir -p local/runs/wandb local/runs/logs
WANDB_MODE=offline WANDB_DIR=local/runs/wandb XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_piper_test_chunk_relative --exp-name=test
```

命令在前台训练 30,000 步，每 2,000 步保存一次检查点，目录为
`checkpoints/pi05_piper_test_chunk_relative/test/<step>`。若需要断开 SSH 后继续训练，在服务器用 `tmux`/`screen` 启动同一命令，或按服务器的作业调度器提交。已有检查点目录时，加 `--resume`；不要加 `--overwrite`，除非确实要删除旧检查点。

配置从 `gs://openpi-assets/checkpoints/pi05_base/params` 载入初始权重，服务器首次运行需要能访问或预先缓存该权重。`WANDB_MODE=offline` 将日志留在服务器本地。

训练时第 0 个 TCP 动作是同帧基准，模型采样的其余 49 步才是未来动作。这个数据集没有夹爪信号，推理端必须单独控制或固定夹爪；`infer_pick_place.py` 需要真实夹爪角度元数据，不能直接用于此检查点。

## 训练后推理

先在训练服务器确认实际保存的步数，再从工程根目录启动策略服务：

```bash
ls checkpoints/pi05_piper_test_chunk_relative/test
uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
  --policy.config=pi05_piper_test_chunk_relative \
  --policy.dir=checkpoints/pi05_piper_test_chunk_relative/test/<step>
```

`<step>` 改为实际存在的检查点目录名。服务器会读取检查点内的
`assets/test_chunk_relative/norm_stats.json`；只复制模型参数而不复制统计文件无法正确推理。

在连接 Piper CAN 和两路相机的机器上使用专用入口。该机器只需准备后数据集的
`meta/info.json` 和 `meta/tasks.jsonl`，用来读取原任务文本、30 Hz 和相机分辨率；不需要 790 MB 视频。若服务器端口不直接可达，可先在机械臂机器建立 SSH 隧道 `ssh -L 6006:localhost:6006 user@server`，再用 `--host localhost`。首次运行例如：

**先核对腕部相机画面。** 训练视频 `observation.images.camera` 是经过裁剪的 640×480 画面；原始采集分辨率和裁剪框没有记录在数据集元信息里。旧推理路径直接将 USB 相机整帧缩放到 224×224，只有 USB 输出本身已是同一裁剪画面时才可能一致。先把训练视频也放在机械臂机器上，运行相机预览（不连接 CAN 或策略服务）：

相机 0 的现场 `v4l2-ctl` 信息显示当前为 640×480、YUYV、30 FPS，自动曝光和自动白平衡均开启；`exposure_time_absolute=166` 和 `white_balance_temperature=4600` 标记为 `inactive`，是自动模式下的默认控制值，**不能当成训练时的实测曝光/色温**。`zoom_absolute=0`，V4L2 裁剪范围是整张 640×480；这也不能排除录制软件做了后处理裁剪。训练视频未保存曝光、增益、白平衡、gamma、锐度或原相机的控制值。

在连接相机 0 的机器上先采集原图和相机控制快照，不连接机械臂：

```bash
uv run examples/piper/chunk_relative/capture_wrist_diagnostics.py \
  --camera 0 --device /dev/video0 --width 640 --height 480 --fps 30 \
  --output-dir local/runs/outputs/wrist_capture
```

输出 `raw_00.png` 至 `raw_04.png` 和 `capture_report.json`（内含打开相机前、采集中 `v4l2-ctl --all`、`--list-ctrls`、`--list-formats-ext` 结果、实际 OpenCV 格式及像素统计）。数据集视频存在时还会写出 `dataset_reference.png`；若只复制了 `meta`，仍会保存原图和设备信息。将 `raw_00.png` 和 `capture_report.json` 与训练视频帧一起对比，才能确定现场裁剪框和色彩差异。曝光/白平衡的像素均值只有在场景与照明一致时才有比较意义。

```bash
uv run examples/piper/chunk_relative/infer_test.py \
  --dataset local/datasets/test_chunk_relative \
  --wrist-camera 0 --wrist-capture-size 640 480 --wrist-crop 0 0 640 480 \
  --preview-wrist local/runs/outputs/wrist_preview
```

预览目录中 `raw.png` 是相机实际输出，`prepared.png` 是待送入模型前的 640×480 画面，`model_input.png` 是模型看到的 224×224 画面；若数据集视频存在，还会保存 `reference_dataset.png` 和 `reference_model_input.png`（第 0 集第 30 帧）。比较取景范围、视角和物体比例，确定裁剪框后加 `--wrist-crop X Y W H` 再预览。`X Y W H` 是**原始相机帧**中的像素坐标，程序会将此区域缩放到训练视频的 640×480。预览会打印实际帧尺寸；设备若未按请求输出指定分辨率，应按实际尺寸重新确定裁剪框。

相机 0 的 `raw_00.png` 与训练视频第 0 集第 30 帧实测都是 640×480，鱼眼圆边拟合约为 `(中心x, 中心y, 半径)=(322,238,261)` 与 `(320,238,262)` 像素；**当前 640×480 输出已经包含与训练视频几乎相同的鱼眼视场，不需要再次裁切**，使用 `--wrist-crop 0 0 640 480`。采集报告显示 OpenCV 实际使用 YUYV、30 FPS；推理入口默认请求 YUYV 并检查实际格式、尺寸和帧率。两帧的相机姿态、桌面摆设和照明不同，所以不能从像素均值差推算训练时的曝光或白平衡设定。

专用入口默认显式开启 USB 相机的自动曝光和自动白平衡；若根据同场景对比确认训练采集使用了固定值，可在**预览和真机推理中使用相同参数** `--wrist-exposure-ms <毫秒> --wrist-white-balance-k <色温K>`。相机 0 支持的手动色温范围为 2800 至 6500 K。手动曝光会写入实际请求值（V4L2 单位为 0.1 ms），不再固定写入 100。`capture_report.json` 可检查参数是否生效；不要把自动模式下标记 `inactive` 的 166 或 4600 当成训练参数。

现场预览 `wrist_preview_check/preview_report.json` 揭示了亮度差异的直接原因：旧代码请求 OpenCV 自动曝光值 `0.75`，但这台 V4L2 相机实际读回 `1`（手动模式），曝光时间固定为 `166`（16.6 ms）；初次未改相机控制的诊断报告则读回 `3`（自动模式）。旧预览桌面区域灰度均值约 `62`，姿态相近的训练第 37 集第 90 帧约 `176`。现在推理和预览都请求 V4L2 自动曝光模式 `3` 并核对读回；若读回仍不是 `3`，程序直接报错。白平衡仍为自动模式 `1`。两路画面的鱼眼取景范围接近，但训练时的曝光/白平衡控制值未保存，不能仅凭不同场景的像素差还原。预览和真机推理都会等待腕部相机取得 60 个有效帧；修改代码后应重新生成预览并检查报告中的 `auto_exposure=3` 及新图像，再决定是否需要手动校准。

确认裁剪框后再运行真机推理：

```bash
uv run examples/piper/chunk_relative/infer_test.py \
  --dataset local/datasets/test_chunk_relative \
  --host <服务器地址> --port 6006 --can-name can0 \
  --base-camera <RealSense序列号> --wrist-camera 0 \
  --wrist-capture-size 640 480 --wrist-crop 0 0 640 480 --wrist-fourcc YUYV \
  --steps 1 --tcp-offset-m X Y Z --visualize-trajectory
```

`X Y Z` 是现场实测的 J6 法兰到工具 TCP 平移，单位米；省略该参数时使用工程默认值 `0 0 0.22`，仅在当前工具确实符合时才省略。启动时程序使能机械臂并连接相机、策略服务，按 Enter 才开始发送运动目标；`s` 暂停，`q` 退出。首次每次查询只执行 1 个未来 TCP 目标，再读取当前 Piper TCP 重新锚定；模型第 0 步是同帧动作，会跳过。默认以 30 Hz 插值，并按 `0.05 m/s`、`0.3 rad/s` 限制插值速度；确认方向与工作空间后再增大 `--steps`，范围为 1..49。

专用入口向模型发送固定 `state=[0]`，不读取夹爪反馈用于模型状态，也不会使能或下发夹爪命令；实体夹爪保持当前状态。它从训练元数据读取原始任务文本，不使用其他 Piper 入口内置的提示词。相机应对应训练时的 `observation.images.global`（基座）和 `observation.images.camera`（腕部），视角与工具 TCP 定义须一致。无图形桌面时去掉 `--visualize-trajectory`。可加 `--trajectory-log local/runs/outputs/test_infer.npz` 保存目标及实测 TCP，目标文件必须尚不存在。程序退出后机械臂仍保持使能。

## 在 Piper 上回放数据集 TCP

在连接 Piper CAN 的机器上，从工程根目录使用原始 9D 数据集和专用回放入口。它只下发 TCP 位姿，不读取训练检查点，也不控制夹爪；`--exec-horizon 20` 表示每 20 个未来目标用实测 Piper TCP 重新锚定。`--tcp-offset-m` 是实测 J6 法兰到工具 TCP 的平移，单位米；省略时使用工程默认值 `0 0 0.22`，仅当它与当前工具一致时才可省略。

```bash
# 离线检查前 20 个目标，不连接机械臂
uv run examples/piper/data_tools/replay_lightumi_tcp_relative.py \
  --data-dir local/datasets/test --episode 0 \
  --exec-horizon 20 --max-frames 20 --dry-run

# 首次真机逐帧试播；把 X Y Z 换成实际工具 TCP 偏移
uv run examples/piper/data_tools/replay_lightumi_tcp_relative.py \
  --data-dir local/datasets/test --episode 0 \
  --exec-horizon 20 --max-frames 20 --step --speed 0.25 \
  --speed-pct 20 --readback --tcp-offset-m X Y Z
```

episode 0 的离线检查结果：前 20 个目标相对首目标最大偏移约 3.1 mm，最大单步约 0.5 mm；整集有 440 个未来目标、22 个 20 步块，最大相对偏移约 304 mm。确认 TCP 原点和轴方向、现场工作空间及运动方向后，去掉 `--max-frames 20 --step` 可回放整集；可用 `--workspace XMIN XMAX YMIN YMAX ZMIN ZMAX` 限制 Robot Base 下的目标范围。每次运行在开始前会等待确认；不要传 `--yes`。按 Ctrl+C 可中断，脚本结束后机械臂仍保持使能。完成至少一个目标后会在 `local/runs/outputs/` 保存实测与目标轨迹的 `.npz` 和 `.gif`。

原始数据的最后一个重复动作会丢弃，第一个动作只作基准。若只有准备后的 `test_chunk_relative`，其第 0、1 行 TCP 相同；本节的原始 9D 入口会直接处理时间对齐和末尾重复动作，适合真机回放。
