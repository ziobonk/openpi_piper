#!/usr/bin/env python3
"""
将 diffusion_policy_piper 的 zarr replay buffer 转换为 LeRobot 格式数据集。

两种模式:
    eef   — 末端位姿 (笛卡尔空间)
    joint — 关节角 (关节空间)

用法:
    # EEF 模式 (默认)
    python examples/piper/data_tools/convert_dual_demo_to_lerobot.py

    # 关节模式
    python examples/piper/data_tools/convert_dual_demo_to_lerobot.py --mode joint

    # 指定输入输出
    python examples/piper/data_tools/convert_dual_demo_to_lerobot.py \
        --input /home/rhr/diffusion_policy_piper/data/vr_fold/replay_buffer.zarr \
        --output data/vr_fold \
        --mode joint
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
FPS = 50
TASK_TEXT = "Move the closet T-shirt to the center.Both arms fold the T-shirt.Then right arm moves the T-shirt to the right."
MODE_EEF = "eef"
MODE_JOINT = "joint"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


# 优先使用 turbojpeg（比 cv2.imencode 快 2-3 倍）
try:
    from turbojpeg import compress as _tj_compress, PF, SAMP

    def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
        return _tj_compress(img, quality=quality, pixelformat=PF.RGB, subsamp=SAMP.Y420)
except ImportError:
    def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
        import cv2
        bgr = img[..., ::-1]
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()


def joint_state_names() -> list[str]:
    # 6-DOF 双臂关节角 + 夹爪: [left_j1..j6, left_gripper, right_j1..j6, right_gripper]
    return [
        "left_joint_1", "left_joint_2", "left_joint_3",
        "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
        "right_joint_1", "right_joint_2", "right_joint_3",
        "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper",
    ]


def eef_state_names() -> list[str]:
    return [
        "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper",
        "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
    ]

# ---------------------------------------------------------------------------
# 主转换
# ---------------------------------------------------------------------------


def _process_episode_worker(args: tuple) -> dict:
    """多进程 worker：独立打开 zarr，处理单个 episode，写 parquet，返回统计信息。"""
    zarr_path, mode, ep_idx, ep_start, ep_end, output_dir, task_text, fps = args

    import numpy as np
    import pandas as pd
    import zarr
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor

    z = zarr.open(zarr_path, mode="r")
    data = z["data"]

    if mode == "eef":
        left_src = data["left_robot_eef_pose"]
        right_src = data["right_robot_eef_pose"]
        action_src = data["action"]
    else:
        left_src = data["left_robot_joint"]
        right_src = data["right_robot_joint"]
        action_src = None

    left_grip_src = data["left_gripper_angle"]
    right_grip_src = data["right_gripper_angle"]
    img0_src = data["img_camera_0"]
    img1_src = data["img_camera_1"]
    img2_src = data["img_camera_2"]

    n_frames = ep_end - ep_start
    sl = slice(ep_start, ep_end)

    # 按切片读取
    ep_left = left_src[sl].astype(np.float32)
    ep_right = right_src[sl].astype(np.float32)
    ep_lg = left_grip_src[sl].astype(np.float32)
    ep_rg = right_grip_src[sl].astype(np.float32)
    ep_ts = (np.arange(n_frames, dtype=np.float32) / fps)
    ep_img0 = img0_src[sl]
    ep_img1 = img1_src[sl]
    ep_img2 = img2_src[sl]

    # 组装 14 维 state
    ep_state = np.zeros((n_frames, 14), dtype=np.float32)
    if mode == "eef":
        # [left_pose(6), left_gripper(1), right_pose(6), right_gripper(1)]
        ep_state[:, 0:6] = ep_left
        ep_state[:, 6] = ep_lg.reshape(-1)
        ep_state[:, 7:13] = ep_right
        ep_state[:, 13] = ep_rg.reshape(-1)
    else:
        # joint: [left_j1..j6, left_gripper, right_j1..j6, right_gripper]
        ep_state[:, 0:6] = ep_left
        ep_state[:, 6] = ep_lg.reshape(-1)
        ep_state[:, 7:13] = ep_right
        ep_state[:, 13] = ep_rg.reshape(-1)

    # actions
    if action_src is not None:
        ep_actions = action_src[sl].astype(np.float32)
    else:
        ep_actions = ep_state.astype(np.float32)

    # 并行 JPEG 编码
    def encode_row(i: int) -> dict:
        return {
            "state": ep_state[i],
            "actions": ep_actions[i],
            "timestamp": ep_ts[i],
            "frame_index": i,
            "episode_index": ep_idx,
            "index": ep_start + i,
            "task_index": 0,
            "image": {"bytes": encode_jpeg(ep_img0[i]), "path": f"frame_{i:06d}.jpg"},
            "wrist_image": {"bytes": encode_jpeg(ep_img1[i]), "path": f"frame_{i:06d}.jpg"},
            "wrist_image_right": {"bytes": encode_jpeg(ep_img2[i]), "path": f"frame_{i:06d}.jpg"},
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(encode_row, range(n_frames)))

    # 写 parquet（带 HuggingFace schema metadata，确保 Image 列被正确识别）
    import datasets
    from datasets import Image as HFImage, Sequence, Value, Features

    df = pd.DataFrame(frames)
    data_dir = Path(output_dir) / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"

    hf_features = Features({
        "state": Sequence(length=14, feature=Value(dtype="float32")),
        "actions": Sequence(length=14, feature=Value(dtype="float32")),
        "timestamp": Value(dtype="float32"),
        "frame_index": Value(dtype="int64"),
        "episode_index": Value(dtype="int64"),
        "index": Value(dtype="int64"),
        "task_index": Value(dtype="int64"),
        "image": HFImage(),
        "wrist_image": HFImage(),
        "wrist_image_right": HFImage(),
    })
    hf_ds = datasets.Dataset.from_pandas(df, features=hf_features)
    hf_ds.to_parquet(str(parquet_path))
    print(f"  → Episode {ep_idx}: {n_frames} frames → {parquet_path.name}")

    # LeRobot v2 per-feature stats
    def _feature_stats(arr: np.ndarray) -> dict:
        return {
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "count": np.array([n_frames], dtype=np.int64),
        }

    def _image_stats(img: np.ndarray) -> dict:
        n = img.shape[0]
        c = img.shape[3]
        batch_size = 64
        ch_min = np.full(c, 255, dtype=np.float64)
        ch_max = np.full(c, 0, dtype=np.float64)
        ch_sum = np.zeros(c, dtype=np.float64)
        ch_sq = np.zeros(c, dtype=np.float64)
        total = 0
        for b in range(0, n, batch_size):
            batch = img[b:b + batch_size].reshape(-1, c).astype(np.float64)
            ch_min = np.minimum(ch_min, batch.min(axis=0))
            ch_max = np.maximum(ch_max, batch.max(axis=0))
            ch_sum += batch.sum(axis=0)
            ch_sq += (batch ** 2).sum(axis=0)
            total += batch.shape[0]
        mean = (ch_sum / total).reshape(3, 1, 1).astype(np.float32)
        var = (ch_sq / total - (ch_sum / total) ** 2).reshape(3, 1, 1).astype(np.float32)
        return {
            "min": ch_min.reshape(3, 1, 1).astype(np.float32),
            "max": ch_max.reshape(3, 1, 1).astype(np.float32),
            "mean": mean,
            "std": np.sqrt(np.maximum(var, 0)).astype(np.float32),
            "count": np.array([n], dtype=np.int64),
        }

    ep_stats = {
        "state": _feature_stats(ep_state),
        "actions": _feature_stats(ep_actions),
        "image": _image_stats(ep_img0),
        "wrist_image": _image_stats(ep_img1),
        "wrist_image_right": _image_stats(ep_img2),
    }

    return {
        "episode_index": ep_idx,
        "tasks": [task_text],
        "length": n_frames,
        "stats": ep_stats,
    }


def main():
    p = argparse.ArgumentParser(description="Convert dual-piper zarr to LeRobot format")
    p.add_argument("--input", default="data/dual_demo/replay_buffer.zarr",
                   help="输入 zarr 路径 (默认: data/dual_demo/replay_buffer.zarr)")
    p.add_argument("--output", default=None,
                   help="输出目录 (默认: data/dual_piper_{mode}_lerobot)")
    p.add_argument("--mode", choices=[MODE_EEF, MODE_JOINT], default=MODE_EEF,
                   help="控制模式: eef (末端位姿) | joint (关节角)")
    p.add_argument("--fps", type=int, default=FPS, help=f"帧率 (默认: {FPS})")
    p.add_argument("--task", default=TASK_TEXT, help="任务指令文本")
    args = p.parse_args()

    zarr_path = Path(args.input)
    if not zarr_path.exists():
        print(f"[ERROR] 找不到 {zarr_path}")
        sys.exit(1)

    mode = args.mode
    output_dir = args.output or f"data/dual_piper_{mode}_lerobot"
    robot_type = f"piper_dual_{mode}"

    print(f"[INFO] 模式: {mode}")
    print(f"[INFO] 输入: {zarr_path}")
    print(f"[INFO] 输出: {output_dir}")

    z = zarr.open(str(zarr_path), mode="r")

    ep_ends = z["meta/episode_ends"][:]
    n_episodes = len(ep_ends)
    total_frames = int(ep_ends[-1])

    # ---- 获取图像尺寸（只读第一帧） ----
    img_h0, img_w0 = z["data/img_camera_0"][0].shape[:2]
    img_h1, img_w1 = z["data/img_camera_1"][0].shape[:2]
    img_h2, img_w2 = z["data/img_camera_2"][0].shape[:2]
    print(f"[INFO] 相机尺寸: base=({img_h0},{img_w0}), wrist=({img_h1},{img_w1}), wrist_r=({img_h2},{img_w2})")

    print(f"[INFO] 总帧数: {total_frames}, Episodes: {n_episodes}")
    ep_lengths = [int(ep_ends[i] - (ep_ends[i - 1] if i > 0 else 0)) for i in range(n_episodes)]
    print(f"[INFO] 各 episode 帧数: {ep_lengths}")

    # ---- 特征名 ----
    if mode == MODE_JOINT:
        s_names = joint_state_names()
        a_names = joint_state_names()
    else:
        s_names = eef_state_names()
        a_names = eef_state_names()

    # ---- 多进程并行处理 episode ----
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    data_dir = out / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- 多进程：每个 worker 独立打开 zarr，处理分配的 episode ----
    from concurrent.futures import ProcessPoolExecutor, as_completed

    task_text = args.task
    zarr_path_str = str(zarr_path)
    output_dir_str = str(output_dir)

    ep_ranges = []
    ep_start = 0
    for ep_idx in range(n_episodes):
        ep_end = int(ep_ends[ep_idx])
        ep_ranges.append((ep_idx, ep_start, ep_end))
        ep_start = ep_end

    # 准备 worker 参数
    worker_args = [
        (zarr_path_str, mode, ep_idx, start, end, output_dir_str, task_text, args.fps)
        for ep_idx, start, end in ep_ranges
    ]

    print(f"[INFO] 启动 {min(len(worker_args), 8)} 个进程并行处理...")
    episode_info = [None] * n_episodes

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process_episode_worker, w): w[2] for w in worker_args}
        for future in as_completed(futures):
            result = future.result()
            episode_info[result["episode_index"]] = result

    # 聚合逐 episode 统计量为全局统计

    # ---- 写 meta 文件 ----
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "state": {"dtype": "float32", "shape": [14], "names": s_names},
            "actions": {"dtype": "float32", "shape": [14], "names": a_names},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "image": {"dtype": "image", "shape": [img_h0, img_w0, 3], "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": [img_h1, img_w1, 3], "names": ["height", "width", "channel"]},
            "wrist_image_right": {"dtype": "image", "shape": [img_h2, img_w2, 3], "names": ["height", "width", "channel"]},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # episodes.jsonl（只保留元信息，不含 stats）
    with open(out / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episode_info:
            f.write(json.dumps({
                "episode_index": ep["episode_index"],
                "tasks": ep["tasks"],
                "length": ep["length"],
            }, ensure_ascii=False) + "\n")

    # tasks.jsonl
    tasks = [{"task_index": 0, "task": args.task}]
    (out / "meta" / "tasks.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n",
        encoding="utf-8",
    )

    # episodes_stats.jsonl — LeRobot v2 嵌套格式
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    with open(out / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for ep in episode_info:
            f.write(json.dumps({
                "episode_index": ep["episode_index"],
                "stats": ep["stats"],
            }, cls=NumpyEncoder) + "\n")

    # stats.json — 从逐 episode 统计聚合为 LeRobot v2 格式
    from lerobot.common.datasets.compute_stats import aggregate_stats
    all_ep_stats = [ep["stats"] for ep in episode_info]
    global_stats = aggregate_stats(all_ep_stats)
    (out / "meta" / "stats.json").write_text(json.dumps(global_stats, cls=NumpyEncoder, indent=2), encoding="utf-8")

    print(f"\n[DONE] 数据集已输出到 {output_dir}/")
    print(f"  Mode:         {mode}")
    print(f"  Robot type:   {robot_type}")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration:     {total_frames / args.fps:.1f}s @ {args.fps}fps")


if __name__ == "__main__":
    main()
