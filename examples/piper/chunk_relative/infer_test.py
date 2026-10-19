#!/usr/bin/env python3
"""Live TCP-only inference for the zero-gripper lightUMI chunk-relative dataset."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PIPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPER_DIR / "runtime"))
from camera_utils import OpenCVCameras  # noqa: E402
from camera_utils import _resize_pad_bgr  # noqa: E402
from camera_utils import crop_resize_bgr  # noqa: E402
import inference_eef  # noqa: E402


def training_metadata(dataset: Path) -> tuple[float, tuple[int, int], tuple[int, int], str]:
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
    base_shape = info["features"]["observation.images.global"]["shape"]
    wrist_shape = info["features"]["observation.images.camera"]["shape"]
    if not np.isfinite(fps) or fps <= 0 or any(
        len(shape) != 3 or shape[2] != 3 or any(v <= 0 for v in shape)
        for shape in (base_shape, wrist_shape)
    ):
        raise ValueError("invalid training FPS or camera shape")
    tasks = [json.loads(line)["task"] for line in (dataset / "meta/tasks.jsonl").read_text().splitlines() if line]
    if len(tasks) != 1 or not tasks[0].strip():
        raise ValueError("TCP-only inference requires exactly one task prompt")
    return fps, (int(base_shape[1]), int(base_shape[0])), (int(wrist_shape[1]), int(wrist_shape[0])), tasks[0]


def preview_wrist(
    dataset: Path,
    camera_id: int,
    capture_size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    wrist_size: tuple[int, int],
    output_dir: Path,
    exposure_ms: float | None = None,
    white_balance_k: int | None = None,
) -> None:
    """Save the raw and prepared USB frames without opening CAN or a policy client."""
    cap = cv2.VideoCapture(camera_id)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open wrist camera {camera_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        if exposure_ms is None:
            OpenCVCameras._set_auto_exposure(cap, "wrist")  # noqa: SLF001
        else:
            OpenCVCameras._set_exposure(cap, "wrist", exposure_ms)  # noqa: SLF001
        if white_balance_k is None:
            OpenCVCameras._set_auto_white_balance(cap, "wrist")  # noqa: SLF001
        else:
            OpenCVCameras._set_white_balance(cap, "wrist", white_balance_k)  # noqa: SLF001
        frame = None
        for _ in range(10):
            ok, candidate = cap.read()
            if ok:
                frame = candidate
        if frame is None:
            raise RuntimeError(f"wrist camera {camera_id} returned no frame")
    finally:
        cap.release()

    actual_size = (frame.shape[1], frame.shape[0])
    effective_crop = crop or (0, 0, *actual_size)
    prepared = crop_resize_bgr(frame, effective_crop, wrist_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "raw.png"), frame)
    cv2.imwrite(str(output_dir / "prepared.png"), prepared)
    cv2.imwrite(str(output_dir / "model_input.png"), _resize_pad_bgr(prepared, 224, 224))
    print(f"[wrist preview] requested {capture_size}, actual {actual_size}; crop {effective_crop} -> {wrist_size}")

    video = dataset / "videos/chunk-000/observation.images.camera/episode_000000.mp4"
    if video.exists():
        reference = cv2.VideoCapture(str(video))
        try:
            reference.set(cv2.CAP_PROP_POS_FRAMES, 30)
            ok, dataset_frame = reference.read()
        finally:
            reference.release()
        if ok:
            cv2.imwrite(str(output_dir / "reference_dataset.png"), dataset_frame)
            cv2.imwrite(str(output_dir / "reference_model_input.png"), _resize_pad_bgr(dataset_frame, 224, 224))
            print("[wrist preview] saved training video frame 30 for field-of-view comparison")
    print(f"[wrist preview] images saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("local/datasets/test_chunk_relative"))
    parser.add_argument("--host", default="localhost", help="policy server host")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--base-camera", help="RealSense base camera serial; required for live inference")
    parser.add_argument("--wrist-camera", required=True, type=int, help="OpenCV wrist camera index")
    parser.add_argument("--wrist-capture-size", type=int, nargs=2, default=(640, 480), metavar=("WIDTH", "HEIGHT"),
                        help="requested raw USB resolution before crop")
    parser.add_argument("--wrist-crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="crop rectangle in raw USB pixels; use 0 0 W H if USB output is already cropped")
    parser.add_argument("--preview-wrist", type=Path, metavar="OUTPUT_DIR",
                        help="save raw, cropped and model images without connecting the robot")
    parser.add_argument("--wrist-exposure-ms", type=float, help="manual USB exposure in milliseconds; default auto")
    parser.add_argument("--wrist-white-balance-k", type=int, help="manual USB white balance temperature; default auto")
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
    capture_size = tuple(args.wrist_capture_size)
    crop = tuple(args.wrist_crop) if args.wrist_crop is not None else None
    if min(capture_size) <= 0 or (crop is not None and (min(crop[:2]) < 0 or min(crop[2:]) <= 0)):
        parser.error("wrist capture size and crop dimensions must be positive; crop origin must be nonnegative")
    if args.wrist_exposure_ms is not None and (not np.isfinite(args.wrist_exposure_ms) or args.wrist_exposure_ms <= 0):
        parser.error("--wrist-exposure-ms must be finite and positive")
    if args.wrist_white_balance_k is not None and args.wrist_white_balance_k <= 0:
        parser.error("--wrist-white-balance-k must be positive")

    fps, base_size, wrist_size, prompt = training_metadata(args.dataset)
    if args.preview_wrist is not None:
        preview_wrist(
            args.dataset, args.wrist_camera, capture_size, crop, wrist_size, args.preview_wrist,
            args.wrist_exposure_ms, args.wrist_white_balance_k,
        )
        return
    if args.base_camera is None:
        parser.error("--base-camera is required for live inference")
    if crop is None:
        parser.error("--wrist-crop is required for live inference; use --preview-wrist to inspect the camera first")
    if crop[0] + crop[2] > capture_size[0] or crop[1] + crop[3] > capture_size[1]:
        parser.error("--wrist-crop exceeds --wrist-capture-size")
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
        "--wrist_cv_size", str(capture_size[0]), str(capture_size[1]),
        "--wrist_cv_crop", *(str(value) for value in crop),
        "--wrist_cv_output_size", str(wrist_size[0]), str(wrist_size[1]),
        "--prompt", prompt,
    ]
    if args.tcp_offset_m is not None:
        runtime_argv.extend(("--tcp_offset_m", *(str(value) for value in args.tcp_offset_m)))
    if args.visualize_trajectory:
        runtime_argv.append("--visualize_trajectory")
    if args.trajectory_log is not None:
        runtime_argv.extend(("--trajectory_log", args.trajectory_log))
    if args.wrist_exposure_ms is None:
        runtime_argv.append("--wrist_cv_force_auto_exposure")
    else:
        runtime_argv.extend(("--wrist_exposure", str(args.wrist_exposure_ms)))
    if args.wrist_white_balance_k is None:
        runtime_argv.append("--wrist_cv_force_auto_white_balance")
    else:
        runtime_argv.extend(("--wrist_cv_white_balance_k", str(args.wrist_white_balance_k)))
    print(f"[test] {fps:g} Hz, base {base_size}, wrist raw {capture_size}, crop {crop} -> {wrist_size}")
    print("[test] model state = 0; physical gripper is not enabled or commanded")
    inference_eef.main(inference_eef._parse_args(argv=runtime_argv))  # noqa: SLF001


if __name__ == "__main__":
    main()
