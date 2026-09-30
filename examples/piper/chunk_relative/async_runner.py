#!/usr/bin/env python3
"""Piper chunk-relative π0.5 asynchronous inference entry point."""

import argparse
from pathlib import Path
import sys

PIPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPER_DIR / "runtime"))
import inference_eef  # noqa: E402
import inference_eef_async  # noqa: E402

PROMPT = "<control mode> end effector <control mode>pick up the blue cube and place it on the yellow cube."


def main(mode: str) -> None:
    parser = argparse.ArgumentParser(description=f"Piper chunk-relative π0.5 {mode} inference")
    parser.add_argument("--host", default="localhost", help="策略服务器地址")
    parser.add_argument("--port", type=int, default=6006, help="策略服务器端口")
    parser.add_argument("--base-camera", required=True, help="基座 RealSense 序列号")
    parser.add_argument("--wrist-camera", required=True, type=int, help="腕部 OpenCV 相机编号")
    parser.add_argument("--steps", type=int, default=20, help="每块最多执行的未来动作数，1..49")
    parser.add_argument("--inference-hz", type=float, default=1.0, help="推理频率上限")
    args = parser.parse_args()
    if not 1 <= args.steps <= 49:
        parser.error("--steps 必须在 1..49 之间")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1..65535 之间")
    if args.inference_hz <= 0:
        parser.error("--inference-hz 必须大于 0")

    runtime_args = inference_eef._parse_args(argv=[  # noqa: SLF001
        "--host", args.host,
        "--port", str(args.port),
        "--rs2_base", args.base_camera,
        "--usb_wrist", str(args.wrist_camera),
        "--action_horizon", "50",
        "--exec_horizon", str(args.steps),
        "--model_action_frame", "chunk_relative",
        "--prompt", PROMPT,
    ])
    runtime_args.inference_rate = args.inference_hz
    runtime_args.latency_k = 8
    runtime_args.min_smooth_steps = 8
    inference_eef_async.main(runtime_args, mode=mode)


if __name__ == "__main__":
    main("async")
