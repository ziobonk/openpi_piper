#!/usr/bin/env python3
"""Live TCP-only inference for the zero-gripper lightUMI chunk-relative dataset."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PIPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPER_DIR / "runtime"))
import inference_eef  # noqa: E402


def training_metadata(dataset: Path) -> tuple[float, tuple[int, int], str]:
    info = json.loads((dataset / "meta/info.json").read_text())
    mapping = info.get("gripper_mapping", {})
    if (
        info.get("robot_type") != "piper_chunk_relative_eef"
        or info.get("source_kind") != "lightumi"
        or info.get("pose_coordinate_frame") != "source_tracking_world_tcp"
        or info.get("coordinate_transform_applied") is not False
        or mapping.get("available") is not False
        or mapping.get("placeholder_value") != 0.0
        or info.get("features", {}).get("state", {}).get("shape") != [1]
        or info.get("features", {}).get("actions", {}).get("shape") != [7]
    ):
        raise ValueError("expected prepared TCP-only lightUMI chunk-relative metadata")
    fps = float(info["fps"])
    shape = info["features"]["observation.images.global"]["shape"]
    if not np.isfinite(fps) or fps <= 0 or len(shape) != 3 or shape[2] != 3 or any(v <= 0 for v in shape):
        raise ValueError("invalid training FPS or base camera shape")
    tasks = [json.loads(line)["task"] for line in (dataset / "meta/tasks.jsonl").read_text().splitlines() if line]
    if len(tasks) != 1 or not tasks[0].strip():
        raise ValueError("TCP-only inference requires exactly one task prompt")
    return fps, (int(shape[1]), int(shape[0])), tasks[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("local/datasets/test_chunk_relative"))
    parser.add_argument("--host", default="localhost", help="policy server host")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--base-camera", required=True, help="RealSense base camera serial")
    parser.add_argument("--wrist-camera", required=True, type=int, help="OpenCV wrist camera index")
    parser.add_argument("--steps", type=int, default=1, help="future TCP targets per query, 1..49")
    parser.add_argument("--tcp-offset-m", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--max-pos-speed", type=float, default=0.05, help="TCP interpolation limit in m/s")
    parser.add_argument("--max-rot-speed", type=float, default=0.3, help="TCP interpolation limit in rad/s")
    parser.add_argument("--visualize-trajectory", action="store_true")
    parser.add_argument("--trajectory-log", default=None, help="write measured TCP and targets to a new .npz file")
    args = parser.parse_args()
    if not 1 <= args.steps <= 49 or not 1 <= args.port <= 65535:
        parser.error("steps must be 1..49 and port 1..65535")
    if not np.isfinite([args.max_pos_speed, args.max_rot_speed]).all() or min(args.max_pos_speed, args.max_rot_speed) <= 0:
        parser.error("max-pos-speed and max-rot-speed must be positive and finite")
    if args.tcp_offset_m is not None and (
        not np.isfinite(args.tcp_offset_m).all() or np.linalg.norm(args.tcp_offset_m) > 1.0
    ):
        parser.error("tcp-offset-m must be finite meters with length at most 1 m")

    fps, base_size, prompt = training_metadata(args.dataset)
    runtime_argv = [
        "--host", args.host,
        "--port", str(args.port),
        "--can_name", args.can_name,
        "--rs2_base", args.base_camera,
        "--usb_wrist", str(args.wrist_camera),
        "--action_horizon", "50",
        "--exec_horizon", str(args.steps),
        "--model_action_frame", "chunk_relative",
        "--model_state_value", "0",
        "--no_gripper_control",
        "--max_pos_speed", str(args.max_pos_speed),
        "--max_rot_speed", str(args.max_rot_speed),
        "--interp_freq", "30",
        "--control_freq", str(fps),
        "--base_rs_size", str(base_size[0]), str(base_size[1]),
        "--prompt", prompt,
    ]
    if args.tcp_offset_m is not None:
        runtime_argv.extend(("--tcp_offset_m", *(str(value) for value in args.tcp_offset_m)))
    if args.visualize_trajectory:
        runtime_argv.append("--visualize_trajectory")
    if args.trajectory_log is not None:
        runtime_argv.extend(("--trajectory_log", args.trajectory_log))
    print(f"[test] {fps:g} Hz, base camera {base_size[0]}x{base_size[1]}, {args.steps} future steps/query")
    print("[test] model state = 0; physical gripper is not enabled or commanded")
    inference_eef.main(inference_eef._parse_args(argv=runtime_argv))  # noqa: SLF001


if __name__ == "__main__":
    main()
