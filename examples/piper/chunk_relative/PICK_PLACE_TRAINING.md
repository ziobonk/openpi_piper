# `pick_place` 的 chunk-relative 数据与服务器训练

本流程在本地准备数据，在服务器计算模型归一化统计并训练。源数据是
`local/datasets/pick_place`：50 集、17,526 帧、30 Hz、两路视频。`action[t]`
的前 9 维是**下一对齐帧**的绝对 source-tracking-world TCP，最后一维是下一帧夹爪角度；
`observation.state[t]` 是当前帧夹爪编码器角度。角度越大，夹爪越张开。

## 本地数据准备

准备产物已生成在 `local/datasets/pick_place_chunk_relative`。若从源数据重建，先选一个不存在的输出目录：

```bash
uv run examples/piper/chunk_relative/prepare_lightumi_chunk_relative.py \
  --source local/datasets/pick_place \
  --output local/datasets/pick_place_chunk_relative
```

转换器为每帧写入 `state=[当前编码器角度(rad)]` 和
`actions=[同帧绝对 TCP xyz(m), rotvec(rad), 当前编码器角度(rad)]`。
它把源 `action[t-1]` 的下一帧 TCP 对齐到第 `t` 行；每集第 0 行没有可恢复的同帧
TCP，用第一条可用 TCP 填补，此事实写入 `meta/info.json`。训练时才对采样动作块
计算 `inv(T_actions[0]) @ T_actions[k]`，不会提前把整集转换成相对首帧。视频保留在
产物目录，使用本地硬链接或必要时复制，传到服务器时仍是完整文件。

**数据准备不映射夹爪。** 新增的 `state` 和 `actions[:,6]` 均保留源角度。
`meta/info.json` 的 `gripper_mapping` 只记录推理映射所需的端点和方向：当前数据观测范围为
`0.012217..0.792379 rad`，建议分别作为闭合、张开端点，对应 Piper 的 `0..20 mm`。
这是从观测范围推定的端点；若实际夹爪未覆盖完整行程，推理前应按实测标定修正。
`meta/stats.json` 记录磁盘上的原始绝对动作和角度，**不是模型归一化统计**。

本地可检查产物的一集，不连接机械臂：

```bash
uv run examples/piper/data_tools/replay_picodual_lerobot.py \
  --data_dir local/datasets/pick_place_chunk_relative --episode 3 --dry_run
```

该回放入口默认只验证 TCP。数据中的夹爪角度仍为 rad，入口会拒绝
`--control_gripper`，避免把角度误当作毫米下发。

## 服务器计算归一化与训练

将本分支代码和整个准备后数据集传到服务器同一工程目录。例如服务器工程位于
`/srv/openpi` 时：

```bash
rsync -a local/datasets/pick_place_chunk_relative/ \
  user@server:/srv/openpi/local/datasets/pick_place_chunk_relative/
```

服务器上从工程根目录依次执行：

```bash
uv run examples/piper/chunk_relative/compute_norm_stats.py \
  --dataset local/datasets/pick_place_chunk_relative \
  --config-name pi05_piper_pick_place_chunk_relative

PIPER_PICK_PLACE_DATA_DIR=local/datasets/pick_place_chunk_relative \
WANDB_DIR=local/runs/wandb XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_piper_pick_place_chunk_relative \
  --exp-name=pick_place
```

若服务器把数据放在 `data/pick_place_chunk_relative`，将上面命令中的
`PIPER_PICK_PLACE_DATA_DIR` 设为该目录。代码默认路径仍是
`local/datasets/pick_place_chunk_relative`。

专用统计脚本只读 Parquet 的 `state/actions`，不解码视频；对所有有效的 50 帧窗口
先作 chunk-relative 变换，再统计相对 TCP 和原始夹爪角度。结果位于
`assets/pi05_piper_pick_place_chunk_relative/pick_place_chunk_relative/`，包含
`norm_stats.json` 和带数据集哈希的 `norm_stats_manifest.json`。训练检查点会带上统计副本。
训练配置读取 `observation.images.global` 作为基座图像、
`observation.images.camera` 作为腕部图像，使用原任务文本“把方块放到纸盒子里”。
此配置固定使用 PyAV 解码训练视频。服务器的 TorchCodec 若因缺少 FFmpeg 动态库而
无法加载，也不会影响这套数据的训练。服务器更新代码后，可先检查 PyAV 是否可用：

```bash
uv run python -c 'import av; print("PyAV", av.__version__)'
```

若 PyAV 未安装，先同步项目依赖（`uv sync`）；`av` 是锁定依赖中的 LeRobot 依赖。
重试训练时无需重新计算归一化统计。若上一次运行已创建检查点目录，
在训练命令末尾加 `--resume`；没有已保存步数时训练脚本会从头开始。
`--overwrite` 会删除该目录及其中已有的检查点，只有明确要丢弃旧训练时才使用。

## 推理时映射夹爪

服务器启动已训练的策略；将 `<step>` 换成实际检查点步数：

```bash
uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
  --policy.config=pi05_piper_pick_place_chunk_relative \
  --policy.dir=checkpoints/pi05_piper_pick_place_chunk_relative/pick_place/<step>
```

机器人端使用准备后数据集的 `meta/info.json` 读取角度端点和 30 Hz 频率：

```bash
uv run examples/piper/chunk_relative/infer_pick_place.py \
  --host <server> --port 6006 \
  --base-camera <realsense_serial> --wrist-camera <opencv_index> \
  --steps 3
```

运行时将 Piper 夹爪读回的 `0..20 mm` 映射为模型状态的源角度，再把模型预测角度
映射回 Piper 的毫米开度。位姿仍按 chunk-relative 规则，以每次推理时的实测 TCP
为基准。命令可加 `--tcp-offset-m X Y Z` 指定实测 J6 到 TCP 偏移；
`0 0 0` 表示以 J6 法兰原点为 TCP。数据集 TCP 轴、相机视角和真实工具定义应与推理现场一致。

`local/`、`assets/` 和检查点不随 Git 提交；服务器需分别收到准备后数据、
服务器生成的统计以及训练代码。无需把原始 `pick_place` 目录传到服务器训练。
