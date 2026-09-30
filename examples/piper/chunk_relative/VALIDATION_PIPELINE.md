# UMI / PicoDual → Piper 验证流程

从仓库根目录运行以下命令。数据集与设备输出放在 `local/`，代码和文档由 Git 管理。整个流程使用 20 Hz、TCP 位姿 `[x,y,z,rx,ry,rz]`（m、rad）与夹爪宽度（mm）。

## 1. 检查数据入口

**PicoDual / UMI 离线轨迹**：原始 Zarr 的 `data/tcp_action` 必须是同帧、绝对的 PICO World TCP；按 [训练流程](TRAINING_PIPELINE.md) 转成 LeRobot，再运行：

```bash
.venv/bin/python examples/piper/chunk_relative/prepare_chunk_relative_eef.py \
  --source local/datasets/pick_cube_0928 \
  --output local/datasets/pick_cube_0928_chunk_relative \
  --robot-open-width-mm 20
```

**lightumi UDP 遥操轨迹**：先用 `collect_eef_udp.py` 录制。UDP 包格式是 `lightumi.teleop_sender` v1，默认选 `arms[index=1]`。采集数据中 `state` 是下发指令前的 Piper 实测 TCP，`actions` 是下发目标，二者有控制时序差。转换时显式选择 `--source-kind udp`，以每帧实测 `state` 建立同帧绝对动作；夹爪已是 mm，保留原值。新录制数据带 `pose_coordinate_frame=robot_base_tcp` 元数据；旧录制数据没有该字段，须先确认来源和坐标系，再补上元数据。

```bash
.venv/bin/python examples/piper/chunk_relative/prepare_chunk_relative_eef.py \
  --source local/datasets/umi_udp_eef \
  --output local/datasets/umi_udp_chunk_relative \
  --source-kind udp --robot-open-width-mm 20
```

两个产物都在 Parquet 中保存**同帧绝对 TCP**。模型采样动作块后才计算 `D[k] = inv(T[0]) @ T[k]`。动作 0 是观测同帧基准，执行时跳过。Pico 数据为 `pico_world_tcp`；UDP 实测数据为 `robot_base_tcp`。回放只使用块内相对位姿，因此两种全局坐标均可，但虚拟 TCP 与真机 TCP 的原点和轴必须一致。

## 2. 离线检查和回放

```bash
.venv/bin/python examples/piper/data_tools/replay_picodual_lerobot.py \
  --data_dir local/datasets/pick_cube_0928_chunk_relative --episode 0 --dry_run

# UDP 录制数据转换后使用相同入口
.venv/bin/python examples/piper/data_tools/replay_picodual_lerobot.py \
  --data_dir local/datasets/umi_udp_chunk_relative --episode 0 --dry_run
```

检查输出中的帧数、块数、单步平移/旋转和夹爪范围。首次真机回放逐帧确认，先只控制 TCP，限制工作空间并降低时间倍率：

```bash
.venv/bin/python examples/piper/data_tools/replay_picodual_lerobot.py \
  --data_dir local/datasets/pick_cube_0928_chunk_relative --episode 0 \
  --exec_horizon 20 --step --speed 0.25 --readback \
  --workspace 0.20 0.60 -0.30 0.30 0.05 0.55
```

`--workspace` 的六个值是示例，按真实工作空间修改。每个块先读取 Piper 当前 TCP，再执行 `T_robot @ inv(T_data[start]) @ T_data[start+k]`，下个块重新锚定。确认运动方向与夹爪宽度后，才按需加入 `--control_gripper`。机器人会保持使能；现场操作人员负责停机。

## 3. UDP 遥操验证与录制

先在发送端运行 UMI/lightumi 发送程序，再仅检查 UDP 接收（不连接机械臂或相机）：

```bash
.venv/bin/python examples/piper/collection/collect_eef_udp.py \
  --udp_test --print_udp --udp_host 0.0.0.0 --udp_port 5005 --arm_index 1
```

确认 `valid` 位姿、所选手臂、方向与夹爪角度后，启动采集；下面的相机和串口参数须按设备修改：

```bash
.venv/bin/python examples/piper/collection/collect_eef_udp.py \
  --output local/datasets/umi_udp_eef \
  --rs2_base <base_serial> --usb_wrist <wrist_index> \
  --can_name can0 --gripper_port /dev/ttyACM0
```

按 `e` 开关遥操，`z` 重置参考，`c` 开始录制，`s` 保存本集，`q` 退出。UDP 位姿超时、相机过期或无有效位姿会暂停遥操。原始 UDP 录制的 commanded `actions` 适合诊断控制误差；转换后的 measured 轨迹适合 chunk-relative 回放和同帧训练数据准备。

## 4. 训练和策略推理

Pico 数据的归一化、训练配置、服务与推理命令见 [README](README.md) 和 [训练流程](TRAINING_PIPELINE.md)。配置 `pi05_piper_pick_cube_0928_chunk_relative` 固定读取 `pick_cube_0928_chunk_relative`；如果要用 UDP 转换的数据训练，需单独建立对应的数据配置与归一化统计，不能把不同坐标语义或夹爪校准的数据混到该数据集中。

## 5. 可复现性

代码改动和本文档由 Git 提交。`local/`、`assets/` 与检查点通常被忽略；换机器时需单独同步数据集及其 `meta/`、归一化统计和检查点内的统计副本。可用下面的命令检查代码版本与工作区：

```bash
git status --short
git log -1 --oneline
```
