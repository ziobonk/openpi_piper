# `pick_cube_0928_chunk_relative`：数据、训练与推理全流程

`pick_place` 的 30 Hz lightUMI 数据准备和服务器训练使用 [PICK_PLACE_TRAINING.md](PICK_PLACE_TRAINING.md)。

本文对应训练配置 `pi05_piper_pick_cube_0928_chunk_relative` 和目录
`local/datasets/pick_cube_0928_chunk_relative`。命令从工程根目录执行。核心约定是：
**磁盘上保存同帧、绝对的 PICO World TCP 位姿；训练时对每个采样窗口单独转成相对首步的动作；真机推理时以当前 Piper TCP 为基座恢复目标。**
位姿是 `[x, y, z, rx, ry, rz]`，平移单位 m，旋转是 rotvec、单位 rad；第 7 维是夹爪开度、单位 mm。当前流程没有使用 rot6d。

## 1. 数据从哪里来

完整的数据链如下：

```text
PicoDual Zarr: data/tcp_action，同帧绝对 PICO World TCP
  ↓ convert_picodual_to_lerobot.py --raw-tcp-action
local/datasets/pick_cube_0928: state/action 均为同帧 7D 绝对位姿
  ↓ prepare_chunk_relative_eef.py
local/datasets/pick_cube_0928_chunk_relative:
    state = [gripper_width_mm]              (1D)
    actions = [PICO World TCP, gripper_mm]   (7D，位姿仍为绝对值)
  ↓ LeRobot 按观测时刻采样 50 帧动作，再由 ChunkRelativeEEFActions 转换
训练输入: 图像 + 夹爪状态 + 指令；监督目标: 相对首步的 50×7 动作块
```

当前本地数据集的 `meta/info.json` 标识为 20 Hz、50 个 episode、11,465 帧，
`pose_coordinate_frame=pico_world_tcp`、`coordinate_transform_applied=false`。
这两个字段是数据准备脚本和统计脚本的输入检查条件。基座图像来自
`global_camera_img`，腕部图像来自 `left_img`，转成 LeRobot 的
`image`、`wrist_image` 字段。

如果要从原始 Zarr 重建第一阶段，使用实际 Zarr 路径：

```bash
.venv/bin/python examples/piper/data_tools/convert_picodual_to_lerobot.py \
  --input local/zarr/<原始数据>.zarr \
  --output local/datasets/pick_cube_0928 \
  --raw-tcp-action --fps 20 \
  --task '<control mode> end effector <control mode>pick up the blue cube and place it on the yellow cube.'
```

`--raw-tcp-action` 读取 `data/tcp_action`；代码中的关键分支是：

```python
ep_state = _fill_episode_controller_poses(episode_actions).astype(np.float32)
ep_actions = ep_state.copy()  # action[t] 与 state[t] 同帧
```

因此这一分支**不做时序前移**，也不再乘控制器到 TCP 的固定安装变换。
缺失的位姿在 episode 内用最近有效位姿填补；此原始 TCP 分支将 NaN 夹爪值设为 0，其他夹爪值保留。相机 JPEG 按帧写入数据集。
如果输入已经是 `pick_cube_0928`，从下一步开始即可，不需要重复转换。转换器会提示不足 2 帧的 episode 与恒定夹爪开度；前者没有未来 TCP 动作，后者无法做本流程的夹爪范围映射，需先检查原始 Zarr 的有效帧筛选结果。

## 2. 生成训练专用数据集

```bash
.venv/bin/python examples/piper/chunk_relative/prepare_chunk_relative_eef.py \
  --source local/datasets/pick_cube_0928 \
  --output local/datasets/pick_cube_0928_chunk_relative \
  --robot-open-width-mm 20
```

脚本首先要求源数据为同帧的 7D `state/actions`，且两列数值相同。它从源数据的
`meta/stats.json` 读取全局夹爪范围，并执行：

```python
grip_mm = clip((source_grip - grip_min) / (grip_max - grip_min), 0, 1) * 20
new_state = grip_mm[:, None]       # (N, 1)
new_actions = actions.copy()       # (N, 7)
new_actions[:, 6] = grip_mm        # 前 6 维绝对 TCP 位姿不变
```

当前本地源数据记录的 `grip_min=0`、`grip_max≈29.4`，映射到 Piper 的
`0..20 mm`。这是**数据单位转换**，不是模型归一化。脚本只改 Parquet 的两列和
对应的元信息/逐 episode 统计；图像、指令和时间索引沿用原数据。
目标目录已存在时脚本会报错，避免覆盖数据。产物中的 `meta/stats.json` 描述
**磁盘上绝对动作**的统计范围，不能当作模型的归一化文件。

此时第 `t` 行的 `actions[t]` 仍是 `^W T_TCP(t)`，不是相对首帧的位姿。
保留绝对动作的理由是模型每次训练都从不同的观测帧开始采样窗口；
若提前相对整条 episode 首帧计算，就会给后续窗口错误的基座。

## 3. 每个训练动作块怎样相对化

训练配置在 `src/openpi/training/config.py` 中指定
`LeRobotPiperChunkRelativeDataConfig`。数据加载器按 20 Hz 为每个观测
采样 `t=0..49` 的 50 个动作；数据变换按以下顺序运行：

```text
LeRobot 字段
→ RepackTransform（image、wrist_image、state、actions、prompt）
→ PiperInputs（两路图像；第三路补零并 mask；state/action 转 float32）
→ ChunkRelativeEEFActions
→ Normalize（见下一节）
→ ResizeImages(224)、TokenizePrompt、PadStatesAndActions(32)
→ π0.5
```

设一个窗口的绝对位姿为 `T₀, T₁, …, T₄₉`。监督目标为
`Dₖ = T₀⁻¹ Tₖ`，而夹爪目标仍是第 `k` 帧的绝对开度 `gₖ`。
`D₀` 是单位位姿。代码在
`src/openpi/policies/piper_chunk_relative.py` 中使用真正的 SE(3) 运算：

```python
R0 = Rotation.from_rotvec(actions[0, 3:6])
Rk = Rotation.from_rotvec(actions[:, 3:6])
relative[:, :3] = R0.inv().apply(actions[:, :3] - actions[0, :3])
relative[:, 3:6] = (R0.inv() * Rk).as_rotvec()
relative[0, :6] = 0.0
# relative[:, 6] 保留原夹爪开度
```

因此旋转采用旋转群组合，不是逐元素减 rotvec。第 0 步和当前观测同帧，
推理执行时必须跳过；50 步输出最多只有 49 个未来动作可执行。
由于输入状态只有 1 维夹爪，模型不会收到绝对 PICO 或 Piper 位姿。
相对动作消除了全局世界原点的依赖，但训练虚拟 TCP 与真机 TCP 的工具坐标定义、
相机视角和夹爪开度方向仍需一致。

## 4. 归一化统计如何计算

本流程使用专用脚本，而不是通用的 `scripts/compute_norm_stats.py`：

```bash
.venv/bin/python examples/piper/chunk_relative/compute_norm_stats.py
```

专用脚本只读每个 episode Parquet 的 `state`、`actions` 两列，不解码图像。
它先对所有真实帧的 1D 状态更新统计，再对每个起点 `t` 和有效偏移
`k ∈ [0, min(49, L-1-t)]` 计算 `T_t⁻¹ T_{t+k}` 与绝对夹爪 `g_{t+k}`。
等价的关键循环为：

```python
for offset in range(min(50, len(actions))):
    count = len(actions) - offset
    base = rotations[:count]
    rows[:, :3] = base.inv().apply(actions[offset:, :3] - actions[:count, :3])
    rows[:, 3:6] = (base.inv() * rotations[offset:]).as_rotvec()
    rows[:, 6] = actions[offset:, 6]
    action_stats.update(rows)
```

这样覆盖所有**有效窗口元素**；episode 结尾由 LeRobot 为训练采样补出的重复动作
没有进入统计。当前本地统计清单为 11,465 个状态样本和 512,000 行有效动作。
脚本使用 `RunningStats` 求 `mean/std/q01/q99`，分位数由直方图近似计算；
近静止维度的 `q99-q01` 至少取
`[0.001,0.001,0.001,0.01,0.01,0.01,0.1]`，
状态夹爪至少取 `0.1`。`std` 也会设为不小于稳定后的分位区间除以
`4.6527`。实际 π0.5 使用的是 **q01/q99 分位数归一化**：

```text
z = 2 · (x - q01) / (q99 - q01 + 1e-6) - 1
x = (z + 1) / 2 · (q99 - q01 + 1e-6) + q01
```

公式来自 `src/openpi/transforms.py`。代码没有对超出分位范围的值截断，
所以 `z` 可能超出 `[-1,1]`。相对化发生在归一化**之前**；夹爪虽然
保持绝对物理开度，仍作为动作第 7 维参与数值归一化。

输出位置为：

```text
assets/pi05_piper_pick_cube_0928_chunk_relative/
└── pick_cube_0928_chunk_relative/
    ├── norm_stats.json
    └── norm_stats_manifest.json
```

`manifest` 记录数据集 `info.json` 的 SHA-256、horizon 和有效动作行数，
用于核对统计来源。训练配置按 `assets/<配置名>/<数据集名>/norm_stats.json`
读取它；统计文件和数据集均被 Git 忽略，换机器训练时须单独同步。
训练保存检查点时会把统计复制到
`checkpoints/<配置名>/<实验名>/<step>/assets/pick_cube_0928_chunk_relative/`；
推理服务从检查点读取该副本，不能只拷贝参数文件。

## 5. 训练配置和命令

配置的关键字段如下（节选自 `src/openpi/training/config.py`）：

```python
TrainConfig(
    name="pi05_piper_pick_cube_0928_chunk_relative",
    model=Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        discrete_state_input=False,
    ),
    data=LeRobotPiperChunkRelativeDataConfig(
        repo_id="pick_cube_0928_chunk_relative",
        local_data_dir="./local/datasets/pick_cube_0928_chunk_relative",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=8,
    num_train_steps=30_000,
)
```

`action_dim=32` 是模型接口宽度：实际状态 1 维、动作 7 维，均在模型变换末尾
补零到 32 维；输出只取前 7 维。语言指令由数据集任务元信息注入。
训练初始化权重配置指向 `pi05_base` 检查点，首次运行需能访问它或提前准备
对应权重。训练命令：

```bash
WANDB_DIR=local/runs/wandb XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_piper_pick_cube_0928_chunk_relative \
  --exp-name=pick_cube_0928
```

检查点路径为
`checkpoints/pi05_piper_pick_cube_0928_chunk_relative/pick_cube_0928/<step>`。
`<step>` 用实际保存的步数替换。首次开训前确认数据集目录和上一节的统计文件都存在。

## 6. 服务端和同步推理

使用训练配置、检查点参数及其归一化统计启动服务：

```bash
uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
  --policy.config=pi05_piper_pick_cube_0928_chunk_relative \
  --policy.dir=checkpoints/pi05_piper_pick_cube_0928_chunk_relative/pick_cube_0928/<step>
```

服务端对观测应用与训练相同的 `PiperInputs → Normalize → 模型变换`；
模型输出经过 `Unnormalize → PiperOutputs` 后是 `(50,7)` 相对动作，
其中位姿单位为 m/rad，夹爪为 mm。真机同步入口示例：

```bash
.venv/bin/python examples/piper/chunk_relative/infer.py \
  --host localhost --port 6006 \
  --base-camera <基座相机序列号> --wrist-camera 0 --steps 3
```

客户端在观测时同时读取当前 Piper TCP `T_robot`、夹爪宽度、基座和腕部图像。
发给模型的 `observation/state` 只有 `[gripper_mm]`；`T_robot` 留在客户端。
收到动作后跳过第 0 步，对后续每步执行：

```python
T_target_robot = T_robot @ D_pred[k]   # k = 1..steps
gripper_target_mm = clip(g_pred[k], 0, 20)
```

实际代码使用 `_compose_pose6` 处理 SE(3)，每次重新查询策略时重新读取
`T_robot` 作为该块基座。该模式默认连续夹爪控制、预测缩放 1.0。
`--steps` 是每次查询后最多执行的**未来动作数**，范围 1..49；
同步模式执行完这几步才请求下一块。真机客户端无需读取 PICO 数据集，
也无需 PICO World 到 Robot Base 的外参。

## 7. 异步、融合与 RTC

异步入口共用 `async_runner.py` 和 `runtime/inference_eef_async.py`：

```bash
.venv/bin/python examples/piper/chunk_relative/async_infer.py \
  --base-camera <基座相机序列号> --wrist-camera 0 --steps 20
.venv/bin/python examples/piper/chunk_relative/temporal_smoothing_infer.py \
  --base-camera <基座相机序列号> --wrist-camera 0 --steps 20
.venv/bin/python examples/piper/chunk_relative/temporal_ensembling_infer.py \
  --base-camera <基座相机序列号> --wrist-camera 0 --steps 20
```

推理线程采样观测并请求新块；20 Hz 控制线程消费缓冲区。
新块到达时先丢弃观测以后已执行的过期动作，再把相对位姿全部转换成
Robot Base 绝对目标，最后才覆盖、平滑或融合。因此不同观测对应的动作
在同一个坐标系内处理。三种缓冲策略分别为：最新块覆盖待执行队列、
新旧块在未执行的重叠区平滑、按相同控制时间点对多次预测加权融合。
旋转融合使用 SO(3)，不是直接平均 rotvec。当前入口默认
`--inference-hz 5`、`--steps 20`；`--steps` 很小时，缓存耗尽后
机械臂保持上一目标直到下一次结果返回。

RTC 另用 `rtc_infer.py`，服务端须换成
`pi05_piper_pick_cube_0928_chunk_relative_rtc` 配置，并使用**同一个 JAX 检查点**
路径。RTC 客户端把上一整块 Robot Base 动作按已执行步数对齐，连同
`inference_delay`、`execute_horizon` 发给服务端；服务端对这块动作再次执行
相对首步转换和归一化，在采样阶段用它引导新块。首次请求没有历史块，
直接采样。RTC 配置继承普通 π0.5 的参数结构，只改变采样逻辑；
PyTorch 检查点会被拒绝。完整启动命令见 [README](README.md)。

## 8. 核对要点与源码入口

- 数据集 `meta/info.json` 必须标识 `pico_world_tcp` 和未预先变换；否则专用脚本会拒绝。
- 核对 `meta/tasks.jsonl` 与推理脚本的 `PROMPT`。当前本地任务文本是
  `pick up the blue cube and place it on the yellow cube.`；
  当前工作区的同步和异步入口文本与它不同，正式推理前应统一。
- 核对虚拟 TCP 与 Piper TCP 的工具原点/轴方向、两路相机顺序、夹爪开闭方向。
- 可以先运行 `tests/test_piper_chunk_relative.py` 和
  `tests/test_piper_async_modes.py` 的无硬件检查；真实检查点和机械臂执行需另行验证。

| 环节 | 关键代码 |
| --- | --- |
| 原始 Zarr → LeRobot | `examples/piper/data_tools/convert_picodual_to_lerobot.py` |
| 夹爪映射、1D 状态 | `examples/piper/chunk_relative/prepare_chunk_relative_eef.py` |
| 有效窗口归一化统计 | `examples/piper/chunk_relative/compute_norm_stats.py` |
| 训练配置、变换顺序 | `src/openpi/training/config.py` |
| 每块相对化 | `src/openpi/policies/piper_chunk_relative.py` |
| 服务端归一化/反归一化 | `src/openpi/policies/policy_config.py`、`src/openpi/transforms.py` |
| 同步推理与坐标恢复 | `examples/piper/chunk_relative/infer.py`、`examples/piper/runtime/inference_eef.py` |
| 异步/RTC | `examples/piper/chunk_relative/async_runner.py`、`examples/piper/runtime/inference_eef_async.py`、`src/openpi/models/pi0_rtc.py` |
