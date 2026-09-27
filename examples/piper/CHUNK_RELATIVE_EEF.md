# Piper EEF：每个动作块首帧为基准

训练数据来自 `pick_cube_raw_action`，其中每帧 `actions` 是 PICO World 下的绝对虚拟 TCP 位姿，且 `state` 与 `actions` 同帧。先生成单独的数据集，再计算新配置的归一化统计：

```bash
.venv/bin/python examples/piper/prepare_chunk_relative_eef.py \
  --source pick_cube_raw_action --output pick_cube_chunk_relative \
  --robot-open-width-mm 20
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piper_eef_chunk_relative
```

新数据集的 `state` 只有夹爪开度，单位毫米。`actions` 在 Parquet 中仍保留绝对 TCP 位姿；训练数据变换对**每一个采样得到的 50 帧动作块**分别计算 `D[k] = inv(T_actions[0]) @ T_actions[k]`。这一步必须放在动作块采样之后，不能预先对整条 episode 只算一次。`D[0]` 是单位位姿，推理执行时跳过该同帧动作。夹爪目标是绝对开度，未做位姿差分；导出时按源数据全局夹爪范围线性映射到机械臂的 `0..20 mm`。

数据和代码应同步到训练服务器。示例中把服务器地址和工作目录替换为实际值；数据集已被 `.gitignore` 忽略，因此单独同步。归一化统计可以在服务器端计算，或把本地生成的 `assets/pi05_piper_eef_chunk_relative/pick_cube_chunk_relative/` 一起同步。

```bash
# 本机：提交并推送代码分支后，同步数据集（或用其他文件传输方式）
rsync -a --info=progress2 pick_cube_chunk_relative/ \
  <user>@<server>:/path/to/openpi/pick_cube_chunk_relative/

# 服务器：在相同代码分支与项目目录中执行
uv sync
uv run scripts/compute_norm_stats.py --config-name pi05_piper_eef_chunk_relative
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_piper_eef_chunk_relative --exp-name=piper_chunk_relative

# 服务器：训练完成后启动策略服务，监听端口按网络环境设置
uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
  --policy.config=pi05_piper_eef_chunk_relative \
  --policy.dir=checkpoints/pi05_piper_eef_chunk_relative/piper_chunk_relative/30000

.venv/bin/python examples/piper/inference_eef.py --host <服务器地址> --port 6006 \
  --model_action_frame chunk_relative --exec_horizon 20 \
  --rs2_base <相机序列号> --usb_wrist <相机编号>
```

异步控制可换用 `examples/piper/inference_eef_async.py`，保留相同的 `--model_action_frame chunk_relative --exec_horizon 20` 参数。推理时每次采样图像和夹爪状态的同时，读取机械臂当前 TCP 位姿 `T_robot`，整块目标都按 `T_robot @ D[k]` 转成 Robot Base 位姿。下一次请求重新采样 TCP。异步客户端在缓冲和平滑前已完成此转换，所以新旧动作块使用同一坐标系。推理端不读取训练数据集，也不读取 PICO 世界原点。

使用前仍须核对两端的**虚拟 TCP 与机械臂 TCP 的原点和轴方向一致**，相机视角与训练数据一致，并确认夹爪的 0 mm/20 mm 对应关系。若工具坐标定义不同，需要先对局部位姿作工具坐标变换；本方案只消除了全局世界坐标对齐需求。建议先在离线评估或低速状态下检查动作方向：

```bash
.venv/bin/python examples/piper/inference_eef.py \
  --dataset pick_cube_chunk_relative --episode 0 --max_frames 100 \
  --model_action_frame chunk_relative --no_show
```
