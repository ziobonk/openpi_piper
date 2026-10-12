#!/usr/bin/env python3
"""Live chunk-relative inference for pick_place, mapping source radians at runtime."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PIPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPER_DIR / "runtime"))
import inference_eef  # noqa: E402


def training_metadata(dataset: Path) -> tuple[float, tuple[float, float], float, str]:
    info = json.loads((dataset / "meta/info.json").read_text())
    mapping = info.get("gripper_mapping", {})
    if (
        info.get("robot_type") != "piper_chunk_relative_eef"
        or info.get("source_kind") != "lightumi"
        or info.get("pose_coordinate_frame") != "source_tracking_world_tcp"
        or info.get("coordinate_transform_applied") is not False
        or mapping.get("applied_to_dataset") is not False
        or mapping.get("source_unit") != "rad"
        or mapping.get("larger_angle_means_open") is not True
    ):
        raise ValueError("expected prepared lightUMI chunk-relative dataset with unmapped source angles")
    closed = float(mapping["source_closed_rad"])
    opened = float(mapping["source_open_rad"])
    open_width_mm = float(mapping["robot_open_width_mm"])
    fps = float(info["fps"])
    if not np.isfinite([closed, opened, open_width_mm, fps]).all() or opened <= closed or open_width_mm <= 0 or fps <= 0:
        raise ValueError("invalid source gripper range, robot width or dataset FPS")
    tasks = [json.loads(line)["task"] for line in (dataset / "meta/tasks.jsonl").read_text().splitlines() if line]
    if len(tasks) != 1:
        raise ValueError("pick_place inference requires one task prompt")
    return fps, (closed, opened), open_width_mm, tasks[0]


def base_camera_size(dataset: Path) -> tuple[int, int]:
    info = json.loads((dataset / "meta/info.json").read_text())
    shape = info.get("features", {}).get("observation.images.global", {}).get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or any(not isinstance(v, int) or v <= 0 for v in shape):
        raise ValueError("dataset must declare a valid base camera image shape")
    return shape[1], shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("local/datasets/pick_place_chunk_relative"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--base-camera", required=True, help="RealSense base camera serial")
    parser.add_argument("--wrist-camera", type=int, required=True, help="OpenCV wrist camera index")
    parser.add_argument("--steps", type=int, default=3, help="future steps per query, 1..49")
    parser.add_argument("--tcp-offset-m", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--gripper-port", default=inference_eef.GRIPPER_PORT, help="DM USB2CAN serial port")
    parser.add_argument("--gripper-open-rad", type=float, default=-10.0, help="measured DM fully open position (rad)")
    parser.add_argument("--gripper-close-rad", type=float, default=0.0, help="measured DM fully closed position (rad)")
    parser.add_argument(
        "--visualize-trajectory", action=argparse.BooleanOptionalAction, default=True,
        help="show upcoming TCP targets in Robot Base during inference (default: on)",
    )
    parser.add_argument(
        "--trajectory-view", choices=("3d", "2d"), default="3d",
        help="interactive 3D window (default) or XY/XZ/YZ projections",
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 49 or not 1 <= args.port <= 65535:
        parser.error("steps must be 1..49 and port 1..65535")
    if not args.gripper_port:
        parser.error("pick_place inference requires a DM gripper serial port")
    if (
        not np.isfinite([args.gripper_open_rad, args.gripper_close_rad]).all()
        or args.gripper_open_rad >= args.gripper_close_rad
    ):
        parser.error("gripper endpoints must be finite, with --gripper-open-rad < --gripper-close-rad")
    if not inference_eef._HAS_DM_GRIPPER:
        parser.error("DM gripper driver is unavailable")
    fps, (closed, opened), open_width_mm, prompt = training_metadata(args.dataset)
    base_width, base_height = base_camera_size(args.dataset)
    runtime_argv = [
        "--host", args.host,
        "--port", str(args.port),
        "--rs2_base", args.base_camera,
        "--usb_wrist", str(args.wrist_camera),
        "--exec_horizon", str(args.steps),
        "--model_action_frame", "chunk_relative",
        "--model_gripper_range", str(closed), str(opened),
        "--gripper_open_width_mm", str(open_width_mm),
        "--gripper_port", args.gripper_port,
        "--gripper_open_rad", str(args.gripper_open_rad),
        "--gripper_close_rad", str(args.gripper_close_rad),
        "--gripper_speed", "10",
        "--control_freq", str(fps),
        "--base_rs_size", str(base_width), str(base_height),
        "--prompt", prompt,
    ]
    if args.tcp_offset_m is not None:
        runtime_argv.extend(("--tcp_offset_m", *(str(value) for value in args.tcp_offset_m)))
    if args.visualize_trajectory:
        runtime_argv.extend(("--visualize_trajectory", "--trajectory_view", args.trajectory_view))
    print(f"[pick_place] encoder {closed:.6f}..{opened:.6f} rad -> Piper 0..{open_width_mm:g} mm at inference")
    print(f"[pick_place] DM open={args.gripper_open_rad:g} rad, closed={args.gripper_close_rad:g} rad")
    inference_eef.main(inference_eef._parse_args(argv=runtime_argv))  # noqa: SLF001


if __name__ == "__main__":
    main()
