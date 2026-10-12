# Piper SDK 真机原始采集与 chunk-relative 训练

采集端从 Piper SDK 读取 J6 法兰位姿与原生夹爪反馈，把法兰坐标系下的 `--tcp-offset-m` 旋转到基座坐标系后得到 TCP。采集进程不使能、不回零、不下发机械臂或夹爪命令。演示运动和夹爪开合须由外部设备或人工完成；人工移动前按现场机械臂操作规程处理使能状态。

原始数据集每帧保存两路相机图像、7 维实测绝对 TCP 与原生夹爪开度，并额外保存 6 维 SDK 法兰位姿 `flange_pose`。`state` 与 `actions` 是同帧、相同的绝对 TCP 反馈，不是下一帧命令，也没有做 chunk-relative 转换。要用于预测未来动作的其他模型，请按该模型的时序定义从原始轨迹取未来帧，避免把同帧 `actions` 误当成下一帧标签。

如需训练本工程的 chunk-relative 策略，再从原始数据另建 1 维夹爪状态的数据集；训练时每个动作块转换为 `inv(T_action[0]) @ T_action[k]`，夹爪保持绝对开度。采集、训练、推理必须使用相同工具轴、TCP 偏移、相机安装与原生夹爪行程。

## 采集

先激活 CAN，安装 `piper_sdk`、`pyrealsense2`、OpenCV 与项目依赖。可用 `python examples/piper/runtime/camera_utils.py --list` 查询 D435i 序列号。下面的 `X Y Z` 必须换成实测 J6 到工具 TCP 的米制偏移；全局相机必须是 D435i，腕部从 D435i 或 USB 中选一个。

```bash
.venv/bin/python examples/piper/chunk_relative/collect_piper_sdk.py \
  --task "pick up the cube" --base-camera 112222070761 \
  --wrist-rs-serial 231122071797 --tcp-offset-m 0 0 0

# 腕部使用 D435i 时改用：--wrist-rs-serial <wrist_d435i_serial>
```

终端或预览窗口按 `c` 开始、`s` 保存、`d` 丢弃、`q` 退出。反馈或图像过期时，当前 episode 会自动保存并暂停录制。无桌面可加 `--no-preview`。默认**原始**数据集路径是 `local/datasets/piper_sdk_raw_eef`，默认 20 Hz、全局 1280×720、腕部 640×480、原生夹爪 0–70 mm。夹爪物理行程不同可用 `--gripper-open-width-mm` 修改；追加录制时帧率、图像尺寸、TCP 偏移和夹爪行程必须一致。每集建议不少于模型的 50 帧动作窗口。

原始 Parquet 中，`state` 和 `actions` 均为 `[tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper_width_mm]`，前三维是米，旋转是 rotvec 弧度；`flange_pose` 保存补偿前的 J6 位姿 `[x, y, z, rx, ry, rz]`，单位相同。`meta/info.json` 记录 `tcp_offset_m`、帧率、相机尺寸和夹爪行程。新默认路径不会覆盖此前的 `local/datasets/piper_sdk_chunk_relative`。

若最后一集录错，可先预览、再删除。例如原始数据集的 ep3：

```bash
uv run examples/piper/collection/delete_last_raw_episode.py \
  --dataset local/datasets/piper_sdk_raw_eef --episode 3
uv run examples/piper/collection/delete_last_raw_episode.py \
  --dataset local/datasets/piper_sdk_raw_eef --episode 3 --apply
```

工具只允许删除原始数据集的最后一集，会同步更新 Parquet、episode 列表、总帧数和统计；不会修改派生数据集。若已从原始数据生成派生数据集，删除后需从新原始数据重新生成派生数据及归一化统计。

如果提示反馈过期，先运行以下只读诊断，无需启动相机：

```bash
uv run examples/piper/chunk_relative/collect_piper_sdk.py --probe-feedback --can-name can0
```

输出分别给出 TCP、夹爪的最新时间戳与帧率，以及 CAN 总帧率。`CAN: 0` 表示当前接口没有收到可解析的 Piper CAN 帧；只有 `gripper: never received` 时，检查原生夹爪是否接入、上电及是否在该 CAN 总线上发送反馈。采集不会为了获取反馈而使能或移动夹爪。

## 回放已采集的 episode

先确认 episode 已按 `s` 保存。专用脚本默认以录制首帧为相对基准，把整集轨迹锚定到回放时的当前机器人 TCP。`action[0]` 是同帧基准，会跳过。先用 `--dry-run` 检查轨迹范围，此模式不连接机械臂：

```bash
uv run examples/piper/data_tools/replay_piper_sdk_raw.py \
  --dataset local/datasets/piper_sdk_raw_eef --episode 0 --dry-run
```

首次真机回放可逐帧确认，默认不控制夹爪：

```bash
uv run examples/piper/data_tools/replay_piper_sdk_raw.py \
  --dataset local/datasets/piper_sdk_raw_eef --episode 0 \
  --step --speed 0.25 --readback
```

若要复现采集时的 Robot Base 绝对位姿，使用 `--mode absolute`。确认启动提示后，脚本先从当前 TCP 分步移动到录制首帧，并通过真机反馈检查到位，然后才回放后续帧；这段移动不控制夹爪。默认允许的起点差距为 100 mm、45°，接近指令每步不超过 5 mm、2°，以 20 Hz 发送，到位误差需在 10 mm、5° 以内。超出起点范围或到位超时会停止回放。可通过 `--max-start-error-mm`、`--max-start-rotation-deg`、`--approach-step-mm`、`--approach-step-rotation-deg`、`--approach-hz`、`--approach-timeout-s`、`--arrival-position-mm` 和 `--arrival-rotation-deg` 调整这些值；`--workspace XMIN XMAX YMIN YMAX ZMIN ZMAX` 同时约束接近路径和回放路径。确认 TCP 方向和工作空间后，若要同时回放原生 Piper 夹爪，在命令末尾加 `--control-gripper`。脚本读取元数据中的 `tcp_offset_m`、帧率和夹爪行程，无需达妙串口参数。Ctrl+C 可中断；机器人在已使能的情况下于停止后保持使能。

## 派生 chunk-relative 数据、训练、服务

以下转换另建数据集，保留原始目录不变；输出目录若已存在，脚本会拒绝覆盖。Parquet 中的 `actions` 仍是同帧绝对 TCP，只有 `state` 缩为 1 维；真正的块相对化发生在训练采样动作块之后。

```bash
uv run examples/piper/chunk_relative/prepare_chunk_relative_eef.py \
  --source local/datasets/piper_sdk_raw_eef \
  --output local/datasets/piper_sdk_raw_prepared \
  --source-kind piper_sdk

.venv/bin/python examples/piper/chunk_relative/compute_norm_stats.py \
  --dataset local/datasets/piper_sdk_raw_prepared \
  --config-name pi05_piper_sdk_raw_prepared

uv run scripts/train.py pi05_piper_sdk_raw_prepared --exp-name=real_piper

uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
  --policy.config=pi05_piper_sdk_raw_prepared \
  --policy.dir=checkpoints/pi05_piper_sdk_raw_prepared/real_piper/<step>
```

## 真机推理

```bash
.venv/bin/python examples/piper/chunk_relative/infer_piper_sdk.py \
  --base-camera <global_d435i_serial> --wrist-usb 0 --steps 3

# 腕部使用 D435i 时改用：--wrist-rs-serial <wrist_d435i_serial>
```

推理入口从数据集元数据读取提示词、帧率、相机尺寸、TCP 偏移和夹爪行程，使用 Piper SDK 读取当前 TCP，执行相对动作块，并通过 Piper 原生夹爪反馈与 `GripperCtrl` 控制夹爪。它不需要达妙夹爪串口。训练后的策略服务与推理机器需能读取对应数据集元数据。

此前已采集的 `local/datasets/piper_sdk_chunk_relative` 保持原样。若仍使用旧检查点或旧数据集，可在推理命令中显式传 `--dataset local/datasets/piper_sdk_chunk_relative`。
