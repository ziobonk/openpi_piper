#!/usr/bin/env python3
"""Replay absolute virtual TCP data in chunk-relative Piper TCP coordinates.

Each window of H dataset actions is transformed by inv(T[0]) @ T[k].  The
window is anchored at fresh robot TCP feedback; action zero is the same-frame
observation and is never sent. Consecutive windows overlap by one frame.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def relative_pose(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(reference[3:6])
    return np.r_[
        rotation.inv().apply(target[:3] - reference[:3]),
        (rotation.inv() * Rotation.from_rotvec(target[3:6])).as_rotvec(),
    ]


def compose_pose(reference: np.ndarray, local: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(reference[3:6])
    return np.r_[reference[:3] + rotation.apply(local[:3]), (rotation * Rotation.from_rotvec(local[3:6])).as_rotvec()]


def load_episode(root: Path, episode: int) -> tuple[np.ndarray, float]:
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("robot_type") != "piper_chunk_relative_eef":
        raise ValueError("expected dataset from prepare_chunk_relative_eef.py")
    if info["features"]["state"]["shape"] != [1] or info["features"]["actions"]["shape"] != [7]:
        raise ValueError("expected gripper-only state and 7D absolute TCP actions")
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("dataset fps must be positive")
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    rows = []
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "frame_index", "actions"])
        ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float64)
        rows.extend((int(index), action) for index, action in zip(indices[ids == episode], actions[ids == episode]))
    rows.sort(key=lambda row: row[0])
    if len(rows) < 2 or [index for index, _ in rows] != list(range(len(rows))):
        raise ValueError(f"episode {episode} needs at least two contiguous frames starting at zero")
    actions = np.stack([action for _, action in rows])
    if actions.shape != (len(rows), 7) or not np.isfinite(actions).all():
        raise ValueError("actions must be finite 7D TCP poses")
    return actions, fps


def action_chunks(actions: np.ndarray, horizon: int):
    """Yield (frame index, local target) once for each frame after frame zero."""
    if horizon < 2:
        raise ValueError("horizon must be at least two")
    for start in range(0, len(actions) - 1, horizon - 1):
        window = actions[start : start + horizon]
        for offset in range(1, len(window)):
            yield start, start + offset, np.r_[relative_pose(window[0], window[offset]), window[offset, 6]]


def check_target(target: np.ndarray, previous: np.ndarray, initial: np.ndarray, args: argparse.Namespace) -> None:
    if not np.isfinite(target).all() or not 0 <= target[6] <= args.gripper_open_width_mm:
        raise ValueError("nonfinite target or gripper width outside configured range")
    step = relative_pose(previous, target)
    excursion = relative_pose(initial, target)
    if np.linalg.norm(step[:3]) > args.max_step_m or np.linalg.norm(step[3:6]) > args.max_step_rad:
        raise ValueError("target exceeds per-frame TCP motion limit")
    if np.linalg.norm(excursion[:3]) > args.max_excursion_m or np.linalg.norm(excursion[3:6]) > args.max_excursion_rad:
        raise ValueError("target exceeds excursion from initial robot TCP")


def replay(actions: np.ndarray, fps: float, robot, args: argparse.Namespace) -> list[np.ndarray]:
    period = 1.0 / (fps * args.rate)
    initial = np.asarray(robot.get_eef_pose(), dtype=np.float64)
    if initial.shape != (6,) or not np.isfinite(initial).all():
        raise ValueError("invalid initial robot TCP feedback")
    previous = initial.copy()
    base = initial.copy()
    targets = []
    current_chunk = -1
    next_tick = time.monotonic()
    for chunk_start, frame, local in action_chunks(actions, args.horizon):
        if chunk_start != current_chunk:
            base = np.asarray(robot.get_eef_pose(), dtype=np.float64)
            if base.shape != (6,) or not np.isfinite(base).all():
                raise ValueError("invalid robot TCP feedback at chunk boundary")
            residual = relative_pose(previous, base)
            if (
                np.linalg.norm(residual[:3]) > args.max_feedback_error_m
                or np.linalg.norm(residual[3:6]) > args.max_feedback_error_rad
            ):
                raise ValueError(f"robot TCP feedback differs from last target at chunk {chunk_start}")
            current_chunk = chunk_start
        target = np.r_[compose_pose(base, local[:6]), local[6]]
        check_target(target, previous, initial, args)
        robot.send_eef_command(target[:6])
        if args.gripper:
            robot.send_gripper_command(float(target[6]))
        targets.append(target)
        previous = target[:6]
        if not args.dry_run:
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
        if args.verbose:
            print(f"frame={frame} chunk={chunk_start} tcp={np.round(target[:6], 5)} grip={target[6]:.2f}")
    return targets


class DryRunRobot:
    def __init__(self, pose: np.ndarray):
        self.pose = np.asarray(pose, dtype=np.float64).copy()

    def get_eef_pose(self) -> np.ndarray:
        return self.pose.copy()

    def send_eef_command(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, dtype=np.float64).copy()

    def send_gripper_command(self, width: float) -> None:
        pass


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50, help="dataset frames per chunk, including same-frame action")
    parser.add_argument("--rate", type=float, default=1.0, help="playback rate multiplier")
    parser.add_argument(
        "--start-pose",
        nargs=6,
        type=float,
        default=[0.4, 0, 0.3, 0, 0, 0],
        help="dry-run robot TCP pose: x y z rx ry rz",
    )
    parser.add_argument("--execute", action="store_true", help="connect to Piper and issue commands")
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--speed-pct", type=int, default=20)
    parser.add_argument("--gripper", action="store_true", help="also command the native Piper gripper")
    parser.add_argument("--gripper-open-width-mm", type=float, default=20.0)
    parser.add_argument("--max-step-m", type=float, default=0.03)
    parser.add_argument("--max-step-rad", type=float, default=0.15)
    parser.add_argument("--max-excursion-m", type=float, default=0.15)
    parser.add_argument("--max-excursion-rad", type=float, default=0.8)
    parser.add_argument("--max-feedback-error-m", type=float, default=0.03)
    parser.add_argument("--max-feedback-error-rad", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.dry_run = not args.execute
    limits = (
        args.rate,
        args.gripper_open_width_mm,
        args.max_step_m,
        args.max_step_rad,
        args.max_excursion_m,
        args.max_excursion_rad,
        args.max_feedback_error_m,
        args.max_feedback_error_rad,
    )
    if args.horizon < 2 or args.episode < 0 or not all(np.isfinite(limits)) or min(limits) <= 0:
        parser.error("episode must be nonnegative; horizon >= 2; rate and limits must be finite and positive")
    if not 20 <= args.speed_pct <= 100 or not np.isfinite(args.start_pose).all():
        parser.error("speed-pct must be 20..100 and start-pose finite")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    actions, fps = load_episode(args.dataset, args.episode)
    if args.execute:
        from inference_eef import PiperEEFController

        robot = PiperEEFController(
            args.can_name, control_gripper=args.gripper, gripper_command_max_mm=args.gripper_open_width_mm
        )
        try:
            if not robot.enable(speed_pct=args.speed_pct):
                raise RuntimeError("Piper enable failed")
            targets = replay(actions, fps, robot, args)
        finally:
            robot.close_gripper_transport()
    else:
        targets = replay(actions, fps, DryRunRobot(args.start_pose), args)
    print(
        f"{'Executed' if args.execute else 'Validated'} {len(targets)} targets from episode {args.episode}; "
        f"first dataset frame skipped, horizon={args.horizon}, fps={fps:g}"
    )


if __name__ == "__main__":
    main()
