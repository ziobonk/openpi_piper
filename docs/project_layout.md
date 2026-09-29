# 工程目录

| 目录 | 内容 |
|------|------|
| `src/openpi/models/`、`src/openpi/models_pytorch/` | π₀/π₀.₅ 模型实现 |
| `src/openpi/policies/` | 机器人数据映射与策略接口 |
| `src/openpi/training/` | 训练配置、数据加载与检查点 |
| `src/openpi/serving/`、`packages/openpi-client/` | 服务端与客户端 |
| `scripts/` | 通用训练、统计和服务入口 |
| `examples/piper/` | Piper 专用采集、数据工具、位姿变换和推理；见 [Piper 目录说明](../examples/piper/README.md) |
| `examples/aloha_real/`、`examples/aloha_sim/`、`examples/droid/`、`examples/libero/`、`examples/ur5/` | 其他机器人或数据集示例 |
| `tools/` | 独立辅助脚本 |
| `local/` | 本地数据、压缩包、媒体、运行输出和参考文件；整个目录由 Git 忽略 |
| `tests/` | 工程级测试 |
| `third_party/` | 第三方源码 |

本地文件统一放在 `local/datasets/`、`local/zarr/`、`local/archives/`、`local/media/`、`local/runs/` 和 `local/reference/`。训练统计仍放在 `assets/`，检查点仍放在 `checkpoints/`，这两个目录也由 Git 忽略。运行示例命令时，以仓库根目录为当前工作目录。
