#!/usr/bin/env python3
"""Small live client for the pick_cube_0928 chunk-relative checkpoint."""

import argparse
from pathlib import Path
import sys

PIPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPER_DIR))
import inference_eef  # noqa: E402

PROMPT = "<control mode> end effector <control mode>pick up the blue cube and place it on the yellow cube."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="policy server host")
    parser.add_argument("--port", type=int, default=6006, help="policy server port")
    parser.add_argument("--base-camera", required=True, help="RealSense base camera serial")
    parser.add_argument("--wrist-camera", required=True, type=int, help="OpenCV wrist camera index")
    parser.add_argument("--steps", type=int, default=3, help="future steps to execute per query (1..49)")
    args = parser.parse_args()
    if not 1 <= args.steps <= 49:
        parser.error("--steps must be between 1 and 49")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    runtime_args = inference_eef._parse_args(argv=[  # noqa: SLF001 - reuse the existing validated CLI defaults
        "--host", args.host,
        "--port", str(args.port),
        "--rs2_base", args.base_camera,
        "--usb_wrist", str(args.wrist_camera),
        "--exec_horizon", str(args.steps),
        "--model_action_frame", "chunk_relative",
        "--prompt", PROMPT,
    ])
    inference_eef.main(runtime_args)


if __name__ == "__main__":
    main()
