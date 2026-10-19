#!/usr/bin/env python3
"""Capture USB wrist frames and V4L2 controls without connecting to the robot."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


def v4l2_report(device: str) -> dict[str, dict[str, object]]:
    if shutil.which("v4l2-ctl") is None:
        return {"error": {"message": "v4l2-ctl not installed"}}
    result = {}
    for label, flag in (("all", "--all"), ("controls", "--list-ctrls"), ("formats", "--list-formats-ext")):
        completed = subprocess.run(
            ["v4l2-ctl", "--device", device, flag],
            capture_output=True, text=True, check=False,
        )
        result[label] = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    return result


def image_stats(frame: np.ndarray) -> dict[str, object]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    middle = rgb[rgb.shape[0] // 4 : 3 * rgb.shape[0] // 4,
                 rgb.shape[1] // 4 : 3 * rgb.shape[1] // 4]
    return {
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "rgb_mean": np.round(rgb.mean(axis=(0, 1)), 2).tolist(),
        "rgb_center_mean": np.round(middle.mean(axis=(0, 1)), 2).tolist(),
        "rgb_center_p05": np.percentile(middle, 5, axis=(0, 1)).tolist(),
        "rgb_center_p95": np.percentile(middle, 95, axis=(0, 1)).tolist(),
        "center_dark_fraction": float(np.mean(np.max(middle, axis=2) <= 8)),
        "center_clipped_fraction": float(np.mean(np.max(middle, axis=2) >= 250)),
    }


def capture(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or f"/dev/video{args.camera}"
    before = v4l2_report(device)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera} ({device})")
    frames = []
    try:
        if args.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        for _ in range(args.warmup_frames):
            cap.read()
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"camera read failed on frame {i}")
            frames.append(frame)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        properties = {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "fourcc": "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_white_balance": cap.get(cv2.CAP_PROP_AUTO_WB),
            "white_balance_temperature": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
            "gain": cap.get(cv2.CAP_PROP_GAIN),
        }
        during = v4l2_report(device)
    finally:
        cap.release()

    for i, frame in enumerate(frames):
        if not cv2.imwrite(str(output_dir / f"raw_{i:02d}.png"), frame):
            raise RuntimeError(f"could not save raw_{i:02d}.png")
    reference_stats = None
    if args.dataset is not None:
        video = args.dataset / f"videos/chunk-{args.episode // 1000:03d}/observation.images.camera/episode_{args.episode:06d}.mp4"
        if video.exists():
            reference = cv2.VideoCapture(str(video))
            try:
                reference.set(cv2.CAP_PROP_POS_FRAMES, args.reference_frame)
                ok, ref_frame = reference.read()
            finally:
                reference.release()
            if ok:
                cv2.imwrite(str(output_dir / "dataset_reference.png"), ref_frame)
                reference_stats = image_stats(ref_frame)
            else:
                print(f"Warning: could not read frame {args.reference_frame} from {video}")
        else:
            print(f"Warning: training video missing at {video}; raw capture and controls were still saved")

    report = {
        "camera_index": args.camera,
        "device": device,
        "requested": {"width": args.width, "height": args.height, "fps": args.fps, "fourcc": args.fourcc},
        "opencv_actual": properties,
        "frames": {f"raw_{i:02d}.png": image_stats(frame) for i, frame in enumerate(frames)},
        "dataset_reference": reference_stats,
        "v4l2_before": before,
        "v4l2_during": during,
        "note": "Image statistics require the same scene and lighting for exposure/white-balance comparison. Video cannot recover original V4L2 controls.",
    }
    (output_dir / "capture_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(frames)} raw frames, V4L2 controls, and statistics to {output_dir}")
    print(f"Actual: {properties['width']:g}x{properties['height']:g} @ {properties['fps']:g} FPS, {properties['fourcc']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--device", help="corresponding V4L2 device, default /dev/video<camera>")
    parser.add_argument("--output-dir", type=Path, default=Path("local/runs/outputs/wrist_capture"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", help="optional camera pixel format, e.g. YUYV or MJPG")
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=Path("local/datasets/test_chunk_relative"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--reference-frame", type=int, default=30)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.frames) <= 0 or args.warmup_frames < 0:
        parser.error("width, height, fps and frames must be positive; warmup-frames must be nonnegative")
    if args.fourcc is not None and len(args.fourcc) != 4:
        parser.error("--fourcc must have four characters")
    if args.episode < 0 or args.reference_frame < 0 or args.camera < 0:
        parser.error("camera, episode and reference-frame must be nonnegative")
    capture(args)


if __name__ == "__main__":
    main()
