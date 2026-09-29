# PicoDual → Piper：chunk-relative pi05

本流程使用 `pick_cube_0928_chunk_relative` 数据集：采样频率为 20 Hz，包含基座和腕部图像、以毫米为单位的 1 维夹爪状态，以及 PICO World 坐标系下 7 维的绝对 TCP 动作。训练时，每个 50 步动作块都转换为相对首步的位姿 `inv(T_action[0]) @ T_action[k]`；夹爪目标仍是绝对开度。第 0 步是与观测同帧的动作，真机推理时会跳过。旋转仍使用 rotvec 表示。

## 数据与归一化统计

数据集在本地生成，Git 会忽略它。如果尚未生成，可从本地的 `pick_cube_0928` 创建：

```bash
.venv/bin/python examples/piper/chunk_relative/prepare_chunk_relative_eef.py \
  --source pick_cube_0928 --output pick_cube_0928_chunk_relative \
  --robot-open-width-mm 20
```

下面的脚本只读取有效动作和状态，不解码图像，也不把 episode 末尾的填充动作计入统计。统计文件和说明文件写入 `assets/`，同样由 Git 忽略。迁移检查点时需要一并保留对应统计，或从同一份数据重新计算。

```bash
.venv/bin/python examples/piper/chunk_relative/compute_norm_stats.py
```

专用训练配置为 `pi05_piper_pick_cube_0928_chunk_relative`，不会复用旧数据集 `pick_cube_chunk_relative` 的统计。在其他机器训练时，还需单独同步数据集。

## 训练与启动服务

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_piper_pick_cube_0928_chunk_relative --exp-name=pick_cube_0928

uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
  --policy.config=pi05_piper_pick_cube_0928_chunk_relative \
  --policy.dir=checkpoints/pi05_piper_pick_cube_0928_chunk_relative/pick_cube_0928/<step>
```

将 `<step>` 换成已保存的检查点步数。上真机前，先通过现有离线评估或空跑回放，核对相机顺序、TCP 坐标轴与原点、夹爪反馈及目标运动方向。

## 真机推理

专用入口固定了任务指令、动作坐标系和模型的 50 步 horizon；命令行只需设置服务器、相机和每次执行步数：

```bash
.venv/bin/python examples/piper/chunk_relative/infer.py \
  --base-camera <serial> --wrist-camera <index> --steps 3
```

先用较少的执行步数观察运动，确认正确后再增加 `--steps`。需要更多硬件参数或离线评估时，直接使用 `examples/piper/runtime/inference_eef.py`。
