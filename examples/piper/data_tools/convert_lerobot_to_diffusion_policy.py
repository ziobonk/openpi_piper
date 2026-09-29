#!/usr/bin/env python3
"""
将 LeRobot v2.1 格式的 dual_piper 数据集转换为 diffusion_policy_piper 的 JSON+JPEG 格式。

只保留 left 机械臂数据和 base + left wrist 相机。

自动检测数据集类型:
    - piper_dual_eef:   state = [x, y, z, rx, ry, rz, gripper, ...] → 直接使用 eef_pose
    - piper_dual_joint: state = [j1..j6, gripper, ...] → eef_pose 用占位符

用法:
    # joint 模式数据集 (eef_pose 为占位符)
    python examples/piper/data_tools/convert_lerobot_to_diffusion_policy.py \
        --input data/dual_piper_lerobot \
        --output /home/rhr/diffusion_policy_piper/data/test

    # eef 模式数据集 (真实的 xyz/rpy)
    python examples/piper/data_tools/convert_lerobot_to_diffusion_policy.py \
        --input data/dual_piper_eef_lerobot \
        --output /home/rhr/diffusion_policy_piper/data/test_eef

    # 只转换前 3 个 episode (测试用)
    python examples/piper/data_tools/convert_lerobot_to_diffusion_policy.py \
        --input data/dual_piper_eef_lerobot \
        --output /tmp/test_convert -n 3

输出格式:
    <output>/
    ├── min_max.json
    └── episode_N/
        ├── {i}_base.jpg
        ├── {i}_wrist.jpg
        ├── states_ee.json
        └── end_efec.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def save_json(json_path: Path, data):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def decode_image(image_struct) -> np.ndarray:
    """从 LeRobot parquet 的 image struct 列解码出 RGB numpy 数组。"""
    img_data = image_struct.as_py()
    img_bytes = img_data["bytes"]
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def process_episode(
    parquet_path: Path,
    episode_dir: Path,
    episode_idx: int,
    is_eef: bool,
    task_text: str,
) -> tuple[list[dict], list[list[float]]]:
    """
    处理单个 episode 的 parquet 文件，写入图片和 JSON。

    is_eef=True 时 state[:7] = [x, y, z, rx, ry, rz, gripper]
    is_eef=False 时 state[:7] = [j1, j2, j3, j4, j5, j6, gripper]

    返回 (states_ee_data, end_efec_data) 用于后续计算 min_max。
    """
    table = pq.read_table(parquet_path)
    n_frames = len(table)

    # 提取 left 机械臂数据
    states = np.array(table.column("state").to_pylist())  # (N, 14)
    left_states = states[:, :7]  # (N, 7)

    # 提取图片列
    images_base = table.column("image")
    images_wrist = table.column("wrist_image")  # left wrist

    episode_dir.mkdir(parents=True, exist_ok=True)

    states_ee_data = []
    end_efec_data = []

    for i in range(n_frames):
        # 保存 base 相机图片 (resize to 224x224)
        img_base = decode_image(images_base[i])
        img_base = cv2.resize(img_base, (224, 224), interpolation=cv2.INTER_AREA)
        img_base_bgr = cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(episode_dir / f"{i}_base.jpg"), img_base_bgr)

        # 保存 wrist 相机图片 (left, already 224x224)
        img_wrist = decode_image(images_wrist[i])
        img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(episode_dir / f"{i}_wrist.jpg"), img_wrist_bgr)

        gripper = float(left_states[i, 6])

        if is_eef:
            # EEF 模式: state = [x, y, z, rx, ry, rz, gripper]
            xyz = left_states[i, :3].tolist()
            rpy = left_states[i, 3:6].tolist()
            joints = [0.0] * 6  # 无法从 eef 反算
            # end_efec: [x, y, z, rx, ry, rz, gripper]
            eef_flat = xyz + rpy + [gripper]
        else:
            # Joint 模式: state = [j1..j6, gripper]
            joints = left_states[i, :6].tolist()
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            # end_efec 直接用关节角: [j1, j2, j3, j4, j5, j6, gripper]
            eef_flat = joints + [gripper]

        # states_ee.json 格式
        state_dict = {
            "joints": joints,
            "gripper_position": 123,  # placeholder, 与参考脚本一致
            "gripper_position_echo": gripper,
            "action": task_text,
            "objects_to_track": {
                "EE": {
                    "xyz": xyz,
                    "rpy": rpy,
                }
            },
        }
        states_ee_data.append(state_dict)
        end_efec_data.append(eef_flat)

    # 写 JSON
    save_json(episode_dir / "states_ee.json", states_ee_data)
    save_json(episode_dir / "end_efec.json", end_efec_data)

    return states_ee_data, end_efec_data


def main():
    p = argparse.ArgumentParser(
        description="Convert LeRobot dual_piper dataset to diffusion_policy JSON+JPEG format (left arm only)"
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="输入 LeRobot 数据集路径 (如 data/dual_piper_lerobot)"
    )
    p.add_argument(
        "-o", "--output", required=True,
        help="输出目录 (如 /home/rhr/diffusion_policy_piper/data/test)"
    )
    p.add_argument(
        "-n", "--limit", type=int, default=None,
        help="只转换前 N 个 episode (用于测试)"
    )
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] 输入路径不存在: {input_dir}")
        sys.exit(1)

    # 读取 meta
    meta_dir = input_dir / "meta"
    with open(meta_dir / "info.json") as f:
        info = json.load(f)

    # 自动检测数据集类型
    robot_type = info.get("robot_type", "")
    is_eef = "eef" in robot_type
    mode_str = "EEF (末端位姿)" if is_eef else "Joint (关节角)"
    task_text = "fold_tshirt"

    # 读取任务描述
    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path) as f:
            first_line = f.readline().strip()
            if first_line:
                task_info = json.loads(first_line)
                task_text = task_info.get("task", task_text)

    n_episodes = info["total_episodes"]
    if args.limit is not None:
        n_episodes = min(n_episodes, args.limit)

    print(f"[INFO] 数据集: {robot_type}")
    print(f"[INFO] 模式: {mode_str}")
    print(f"[INFO] Episodes: {n_episodes}, 总帧数: {info['total_frames']}")
    print(f"[INFO] 输入: {input_dir}")
    print(f"[INFO] 输出: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个 episode
    all_end_efec = []

    for ep_idx in tqdm(range(n_episodes), desc="Converting episodes"):
        chunk_idx = ep_idx // info.get("chunks_size", 1000)
        parquet_path = input_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"

        if not parquet_path.exists():
            tqdm.write(f"[WARN] 跳过 episode {ep_idx}: 文件不存在 {parquet_path}")
            continue

        episode_dir = output_dir / f"episode_{ep_idx}"
        _, end_efec = process_episode(parquet_path, episode_dir, ep_idx, is_eef, task_text)
        all_end_efec.extend(end_efec)

    # 计算全局 min_max
    all_end_efec = np.array(all_end_efec)  # (total_frames, 7)
    max_vals = np.max(all_end_efec, axis=0).tolist()
    min_vals = np.min(all_end_efec, axis=0).tolist()

    min_max = {"max": max_vals, "min": min_vals}
    save_json(output_dir / "min_max.json", min_max)

    print(f"\n[DONE] 转换完成!")
    print(f"  模式: {mode_str}")
    print(f"  Episodes: {n_episodes}")
    print(f"  输出目录: {output_dir}")
    print(f"  min_max.json:")
    print(f"    max: {[round(v, 6) for v in max_vals]}")
    print(f"    min: {[round(v, 6) for v in min_vals]}")

    if is_eef:
        print(f"\n[INFO] end_efec.json 格式: [x, y, z, rx, ry, rz, gripper]")
    else:
        print(f"\n[INFO] end_efec.json 格式: [j1, j2, j3, j4, j5, j6, gripper] (关节角)")


if __name__ == "__main__":
    main()
