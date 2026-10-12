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

`--workspace` 的六个值是示例，按真实工作空间修改。每个块先读取 Piper 当前 TCP，再执行 `T_robot @ inv(T_data[start]) @ T_data[start+k]`，下个块重新锚定。`--speed 0.25` 会把原始 20 Hz 数据的每步时长扩展为 0.2 秒，并在相邻数据帧间插入 4 个 TCP 指令，保持约 20 Hz 的下发节奏。每块会打印实测 TCP 与上一目标的跟踪偏差；偏差大时应暂停检查跟踪和速度设置。Piper SDK 的 `EndPoseCtrl` 接收欧拉角，代码会选择与上一命令连续的等价角度，避免 ±180° 分支产生 360° 的数值跳变。确认运动方向与夹爪宽度后，才按需加入 `--control_gripper`。机器人会保持使能；现场操作人员负责停机。

## 3. UDP 遥操验证与录制

先在发送端运行 UMI/lightumi 发送程序，再仅检查 UDP 接收（不连接机械臂或相机）：

```bash
.venv/bin/python examples/piper/collection/collect_eef_udp.py \
  --udp_test --print_udp --udp_host 0.0.0.0 --udp_port 5005 --arm_index 1
```

遥操按 `e` 开始或按 `z` 重置时读取当前实测 Piper TCP；随后将 UMI 相邻位姿的局部相对运动映射到当前目标 TCP 的局部轴，并用右乘 SE(3) 增量更新目标。TCP 旋转后，平移方向随 TCP 轴转动。转换后的目标仍以 Piper 基座坐标发送与记录。确认 `valid` 位姿、所选手臂、方向与夹爪角度后，启动采集；下面的相机和串口参数须按设备修改：

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

## lightUMI 9D/10D 数据的 TCP-relative 回放

`local/datasets/lightumi_20261012_213232` 有两集 30 Hz、9D TCP 动作，没有夹爪反馈。可直接用专用入口做按块重新锚定的 chunk-relative 回放，无需生成带虚拟夹爪状态的派生数据集。`--exec-horizon 20` 表示每执行 20 个未来目标就读取一次 Piper 当前 TCP，下一块按 `T_robot_now @ inv(T_action[start]) @ T_action[k]` 生成目标；每块首个 action 只作锚点，不下发。数据源的末尾重复 action 会丢弃，全程不控制夹爪。

```bash
uv run examples/piper/data_tools/replay_lightumi_tcp_relative.py \
  --data-dir local/datasets/lightumi_20261012_213232 \
  --episode 0 --exec-horizon 20 --dry-run

# 首次真机试播：确认 TCP 偏移和工作空间后，逐帧执行前 20 个目标
uv run examples/piper/data_tools/replay_lightumi_tcp_relative.py \
  --data-dir local/datasets/lightumi_20261012_213232 \
  --episode 0 --exec-horizon 20 --max-frames 20 \
  --step --speed 0.25 --readback
```

离线校验得到 episode 0 的 354 个未来 TCP 目标（18 块），episode 1 的 582 个目标（30 块）。若要回放第 1 集，改成 `--episode 1`；完整回放时去掉 `--max-frames` 和 `--step`。`--tcp-offset-m X Y Z` 应设置为实际 J6 到工具 TCP 的偏移；可用 `--workspace` 限制目标范围。按块回放生成的比较轨迹以每块实际下发的目标为基准，以便与 Piper 实测轨迹对齐。

`lightumi_20261009_212049` 是单路 30 Hz LeRobot 数据：`action` 为下一对齐帧的绝对 PICO World TCP `[xyz, rotation-6D]`，旋转 6D 按旋转矩阵前两列分组保存；没有夹爪动作或 TCP 状态。专用入口只读取 Parquet 的 action 列，不解码视频。它以第一个 action 为数据基准，以启动时实测 Piper TCP 为机器人基准，执行 `T_robot_start @ inv(T_action[0]) @ T_action[k]`。首个 action 只作基准，末尾重复 action 被丢弃，夹爪保持原状态。此模式从启动到结束只锚定一次，要求两端 TCP 原点与轴方向一致。

先离线验证（不会连接机械臂）：

```bash
.venv/bin/python examples/piper/data_tools/replay_lightumi_tcp_relative.py \
  --data-dir local/datasets/lightumi_20261009_212049 --episode 0 --dry-run
```

本地数据检查得到 282 个不重复的 action 位姿、281 个未来 TCP 目标；相对首目标最大平移约 155 mm，最大单步平移约 3.6 mm、旋转约 0.24°。首次在有 `can0` 的机械臂端只试前几步，逐帧确认：

```bash
.venv/bin/python examples/piper/data_tools/replay_lightumi_tcp_relative.py \
  --data-dir local/datasets/lightumi_20261009_212049 --episode 0 \
  --max-frames 3 --step --speed 0.25 --readback
```

`local/datasets/pick_place` 的 action 是 10D：前 9 维是同样编码的绝对 TCP，最后一维 `gripper_angle_rad` 是下一帧源夹爪角度。此入口只回放 TCP，**忽略第 10 维且不控制 Piper 夹爪**；源角度没有对应 Piper 夹爪宽度的标定，不能直接当毫米命令。episode 3 离线检查得到 275 个有效位姿、274 个目标，最大相对位移约 303 mm，最大单步平移约 8.7 mm。对应命令如下，反斜杠后不能有空格：

```bash
uv run examples/piper/data_tools/replay_lightumi_tcp_relative.py \
  --data-dir local/datasets/pick_place \
  --episode 3 --speed 0.5 --readback --tcp-offset-m 0 0 0
```

这里的 `--tcp-offset-m 0 0 0` 将 J6 法兰原点当作 Piper TCP；如果数据对应爪尖，应改成实测的 J6 到爪尖偏移。首次真机试播可加 `--max-frames 3 --step`。

每个目标帧下发完成后都会读取并保存 Piper SDK 反馈的法兰位姿经 TCP 偏移换算得到的实测 TCP，`--readback` 只控制终端是否打印追踪误差统计。回放结束或中途按 Ctrl+C 后，只要完成过至少一个目标帧，脚本会在 `local/runs/outputs/` 自动保存同名 `.npz` 和 `.gif`；可用 `--comparison-gif local/runs/outputs/my_trial.gif` 指定路径。GIF 的 XY、XZ、YZ 三个投影同时显示数据集与 Piper 的相对 TCP 轨迹，右下角显示相对位置误差随回放时间变化，标题显示当前帧的位置和姿态误差。绘图在机械臂运动结束后执行，不占用控制循环。

两条轨迹分别以数据集首个 action TCP 和回放开始时的 Piper TCP 为原点，并用各自**初始 TCP 的坐标轴**表示。两端 TCP 轴方向需要一致，才能把它们的坐标与误差直接比较。`.npz` 保存 `dataset_relative_tcp`、`robot_relative_tcp`、`robot_absolute_tcp`、`target_absolute_tcp`、`elapsed_s` 和锚点；数组首行是锚点，后续行与已完成的目标帧逐行对齐。`.npz` 中的位姿单位为米和旋转向量弧度，`elapsed_s` 为秒；GIF 位置和误差显示为毫米。`--gif-fps` 可调整 GIF 播放帧率，不影响回放速度。`--dry-run` 只验证数据，不产生机械臂实测轨迹或 GIF。

Piper 的 TCP 在本工程中由 `examples/piper/runtime/inference_eef.py` 的 `GRIPPER_TCP_OFFSET_M` 定义：默认 `[0, 0, 0.22]` 米，表示法兰 J6 到爪尖 TCP 的平移，数值应按实际工具测量。读位姿时用 `p_tcp = p_J6 + R_J6 @ offset`；下发目标时用 `p_J6 = p_tcp - R_target @ offset`。本入口可加 `--tcp-offset-m X Y Z` 临时覆盖，例如 `--tcp-offset-m 0 0 0.22`，只影响本次回放。该补偿假定 TCP 与 J6 轴方向一致；若工具坐标轴有固定旋转，还需完整的法兰到工具 SE(3) 标定。

可先用 `examples/piper/hardware/test_tcp_control.py --offset 0 0 0.22 --step 0.005` 在有足够空间的真机上微调验证爪尖是否在旋转时保持原地。确认方向与工作空间后去掉 `--max-frames` 和 `--step` 可回放全程。可用 `--workspace XMIN XMAX YMIN YMAX ZMIN ZMAX` 按现场实际范围限制 Piper 基座坐标目标。默认回放时间倍率是 0.25，数据帧间插值后约以 30 Hz 下发 TCP 命令。真机开始前脚本会打印锚点并等待 Enter；按 Ctrl+C 中断。执行完机器人仍保持使能。本数据格式不适用于 `replay_picodual_lerobot.py` 的 7D chunk-relative 入口。
