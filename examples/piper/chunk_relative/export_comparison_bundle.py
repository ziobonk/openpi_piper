#!/usr/bin/env python3
"""Export the current Piper chunk-relative workflow as a small review bundle."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "local/share/pick_cube_0928_chunk_relative_comparison"

# Keep repository-relative paths so the training document points to the same code.
SOURCES = (
    "examples/piper/chunk_relative/TRAINING_PIPELINE.md",
    "examples/piper/chunk_relative/README.md",
    "examples/piper/chunk_relative/export_comparison_bundle.py",
    "examples/piper/chunk_relative/prepare_chunk_relative_eef.py",
    "examples/piper/chunk_relative/compute_norm_stats.py",
    "examples/piper/chunk_relative/infer.py",
    "examples/piper/chunk_relative/async_runner.py",
    "examples/piper/chunk_relative/async_infer.py",
    "examples/piper/chunk_relative/temporal_smoothing_infer.py",
    "examples/piper/chunk_relative/temporal_ensembling_infer.py",
    "examples/piper/chunk_relative/rtc_infer.py",
    "examples/piper/data_tools/convert_picodual_to_lerobot.py",
    "examples/piper/runtime/inference_eef.py",
    "examples/piper/runtime/inference_eef_async.py",
    "examples/piper/runtime/temporal_buffers.py",
    "examples/piper/runtime/camera_utils.py",
    "examples/piper/transforms/pico_relative_se3.py",
    "examples/piper/transforms/pico_arm_transform.py",
    "src/openpi/policies/piper_chunk_relative.py",
    "src/openpi/policies/piper_policy.py",
    "src/openpi/policies/policy.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/training/config.py",
    "src/openpi/training/data_loader.py",
    "src/openpi/training/checkpoints.py",
    "src/openpi/models/pi0_config.py",
    "src/openpi/models/pi0_rtc.py",
    "src/openpi/transforms.py",
    "src/openpi/shared/normalize.py",
    "scripts/train.py",
    "scripts/serve_policy.py",
    "tests/test_piper_chunk_relative.py",
    "tests/test_piper_async_modes.py",
    "tests/test_piper_chunk_relative_replay.py",
)

METADATA = {
    "dataset/meta/info.json": ROOT / "local/datasets/pick_cube_0928_chunk_relative/meta/info.json",
    "dataset/meta/tasks.jsonl": ROOT / "local/datasets/pick_cube_0928_chunk_relative/meta/tasks.jsonl",
    "normalization/norm_stats.json": ROOT / "assets/pi05_piper_pick_cube_0928_chunk_relative/pick_cube_0928_chunk_relative/norm_stats.json",
    "normalization/norm_stats_manifest.json": ROOT / "assets/pi05_piper_pick_cube_0928_chunk_relative/pick_cube_0928_chunk_relative/norm_stats_manifest.json",
}

INTRO = """# pick_cube_0928_chunk_relative 对比包

先读 [训练全流程](code/examples/piper/chunk_relative/TRAINING_PIPELINE.md)，
再按下面的路径查看源代码。本包按导出时的**当前工作区**复制，包含尚未提交的代码修改。
`MANIFEST.json` 记录原路径、字节数和 SHA-256，便于确认双方查看的是同一版本。

| 对比环节 | 文件 |
| --- | --- |
| Zarr 到同帧绝对 TCP 数据 | `code/examples/piper/data_tools/convert_picodual_to_lerobot.py` |
| 夹爪映射、1D state | `code/examples/piper/chunk_relative/prepare_chunk_relative_eef.py` |
| 有效窗口归一化统计 | `code/examples/piper/chunk_relative/compute_norm_stats.py` |
| 训练配置、数据加载 | `code/src/openpi/training/config.py`、`data_loader.py` |
| 动作块相对化 | `code/src/openpi/policies/piper_chunk_relative.py` |
| 模型输入/输出、归一化 | `code/src/openpi/policies/piper_policy.py`、`policy.py`、`code/src/openpi/transforms.py` |
| 同步真机推理 | `code/examples/piper/chunk_relative/infer.py`、`code/examples/piper/runtime/inference_eef.py` |
| 异步、平滑、融合 | `code/examples/piper/chunk_relative/async_runner.py`、`code/examples/piper/runtime/inference_eef_async.py`、`temporal_buffers.py` |
| RTC 采样 | `code/src/openpi/models/pi0_rtc.py`、`code/examples/piper/chunk_relative/rtc_infer.py` |

`dataset/meta/` 只含结构及任务描述；`normalization/` 只含数值统计。
此包**没有** Parquet、相机图像、Zarr、检查点或硬件依赖；
用于代码与方法对比，单独拷贝后不能直接训练或控制机械臂。
"""


def main() -> None:
    if OUTPUT.exists() or OUTPUT.with_suffix(".zip").exists():
        raise FileExistsError(f"导出文件已存在，请先移动旧包再重试: {OUTPUT}")
    missing = [source for source in SOURCES if not (ROOT / source).is_file()]
    missing.extend(str(path) for path in METADATA.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"缺少对比包所需文件: {missing}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".comparison-staging-", dir=OUTPUT.parent))
    try:
        entries = []
        for source in SOURCES:
            destination = Path("code") / source
            origin = ROOT / source
            target = staging / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            entries.append((str(destination), str(origin.relative_to(ROOT)), target))
        for destination, origin in METADATA.items():
            target = staging / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            entries.append((destination, str(origin.relative_to(ROOT)), target))

        manifest = {
            "workflow": "pi05_piper_pick_cube_0928_chunk_relative",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source": "current working tree, including uncommitted edits",
            "files": [
                {
                    "bundle_path": destination,
                    "source_path": source,
                    "bytes": target.stat().st_size,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
                for destination, source, target in entries
            ],
        }
        (staging / "START_HERE.md").write_text(INTRO, encoding="utf-8")
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        staging.rename(OUTPUT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    archive = shutil.make_archive(str(OUTPUT), "zip", root_dir=OUTPUT.parent, base_dir=OUTPUT.name)
    print(f"已导出 {len(entries)} 个文件: {OUTPUT}")
    print(f"可直接发送压缩包: {archive}")


if __name__ == "__main__":
    main()
