# zarr2parquet

High-performance C++ zarr → LeRobot v2 parquet converter.

## 构建

```bash
# 安装依赖 (vcpkg)
$VCPKG_ROOT/vcpkg install blosc "arrow[parquet,core]" nlohmann-json libjpeg-turbo cli11 zlib

# 构建
cd examples/piper/cpp_convert
mkdir -p build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 用法

```bash
./zarr2parquet \
    --input /home/rhr/openpi/data/test.zarr \
    --output ./pick_place \
    --mode eef \
    --task "pick up the black block and place it into the cup." \
    --threads 16
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input, -i` | (必需) | zarr replay buffer 路径 |
| `--output, -o` | (必需) | 输出 LeRobot 数据集目录 |
| `--mode, -m` | `joint` | 控制模式: `joint` 或 `eef` |
| `--task` | `pick up the block` | 任务描述文本，写入 tasks.jsonl |
| `--fps` | `50` | 帧率，用于生成时间戳 |
| `--quality, -q` | `90` | JPEG 质量 (1-100) |
| `--threads, -t` | CPU 核数 | 总线程数 |
| `--parallel, -p` | threads/2 | 同时处理的 episode 数 |

### 示例

```bash
# 基础转换
./zarr2parquet \
    -i /home/rhr/diffusion_policy_piper/data/dual_demo/replay_buffer.zarr \
    -o ./dual_piper_joint_lerobot

# 自定义任务和线程
./zarr2parquet \
    -i /home/rhr/diffusion_policy_piper/data/dual_demo/replay_buffer.zarr \
    -o ./dual_piper_joint_lerobot \
    -m joint \
    --task "Both arms fold the T-shirt" \
    --threads 24 --parallel 12
```

## 输出结构

```
dual_piper_joint_lerobot/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── meta/
    ├── info.json              # 数据集元信息
    ├── episodes.jsonl         # episode 列表
    ├── episodes_stats.jsonl   # 逐 episode 统计
    ├── tasks.jsonl            # 任务定义
    └── stats.json             # 全局统计
```

## 性能

51 episodes / 45,833 frames / 3 cameras:

| 工具 | 耗时 (估算) | 说明 |
|---|---|---|
| C++ 版 | ~2-3 分钟 | libjpeg-turbo + Arrow 原生 parquet + 多线程 |
| Python 版 | ~8-12 分钟 | turbojpeg + multiprocessing |

## 与 Python 版对比

| | C++ | Python |
|---|---|---|
| zarr 读取 | 原生 blosc 解压 | zarr 库 |
| JPEG 编码 | libjpeg-turbo | turbojpeg |
| Parquet 写入 | Apache Arrow C++ | pandas → Arrow |
| 并行 | 真正多线程 | 多进程 (spawn overhead) |
