#!/usr/bin/env python3
"""Live chunk-relative inference for read-only Piper SDK demonstrations."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import inference_eef


def dataset_settings(dataset: Path) -> tuple[float, float, tuple[int, int], tuple[int, int], str, np.ndarray]:
    info = json.loads((dataset / "meta/info.json").read_text())
    if (info.get("robot_type") != "piper_chunk_relative_eef"
            or info.get("source_kind") != "piper_sdk_read_only"
            or info.get("pose_coordinate_frame") != "robot_base_tcp"
            or info.get("coordinate_transform_applied") is not False
            or info.get("gripper_mapping", {}).get("applied_to_dataset") is not True
            or info["features"]["state"]["shape"] != [1]
            or info["features"]["actions"]["shape"] != [7]):
        raise ValueError("expected read-only Piper SDK chunk-relative dataset")
    fps = float(info["fps"])
    width = float(info["gripper_mapping"]["robot_open_width_mm"])
    offset = np.asarray(info["tcp_offset_m"], dtype=np.float64)
    shape = info["features"]["image"]["shape"]
    wrist_shape = info["features"]["wrist_image"]["shape"]
    if (not np.isfinite([fps, width]).all() or min(fps, width) <= 0
            or offset.shape != (3,) or not np.isfinite(offset).all()
            or len(shape) != 3 or len(wrist_shape) != 3
            or any(not isinstance(v, int) or v <= 0 for v in (*shape, *wrist_shape))):
        raise ValueError("invalid dataset FPS, gripper width, TCP offset or image shape")
    tasks = [json.loads(line)["task"] for line in (dataset / "meta/tasks.jsonl").read_text().splitlines() if line]
    if len(tasks) != 1:
        raise ValueError("select a dataset with exactly one task prompt")
    return fps, width, (shape[1], shape[0]), (wrist_shape[1], wrist_shape[0]), tasks[0], offset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("local/datasets/piper_sdk_raw_prepared"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--base-camera", required=True, help="global D435i serial")
    wrist = parser.add_mutually_exclusive_group(required=True)
    wrist.add_argument("--wrist-rs-serial", help="wrist D435i serial")
    wrist.add_argument("--wrist-usb", type=int, help="wrist USB/OpenCV device index")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.steps <= 49 or not 1 <= args.port <= 65535:
        parser.error("steps must be 1..49 and port 1..65535")
    fps, width, base_size, wrist_size, prompt, offset = dataset_settings(args.dataset)
    runtime_argv = [
        "--host", args.host, "--port", str(args.port), "--can_name", args.can_name,
        "--rs2_base", args.base_camera, "--gripper_port", "",
        "--gripper_open_width_mm", str(width), "--control_freq", str(fps),
        "--base_rs_size", *(str(v) for v in base_size),
        "--wrist_rs_size", *(str(v) for v in wrist_size),
        "--tcp_offset_m", *(str(v) for v in offset),
        "--model_action_frame", "chunk_relative", "--exec_horizon", str(args.steps),
        "--prompt", prompt,
    ]
    runtime_argv.extend(("--rs2_wrist", args.wrist_rs_serial) if args.wrist_rs_serial else
                        ("--usb_wrist", str(args.wrist_usb)))
    inference_eef.main(inference_eef._parse_args(argv=runtime_argv))  # noqa: SLF001


if __name__ == "__main__":
    main()
