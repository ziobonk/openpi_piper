#!/usr/bin/env python3
"""Replay measured TCP poses from local/datasets/piper_sdk_raw_eef.

The dataset stores same-frame absolute observations, not motor commands. By
default, reproduce its motion relative to the robot's current TCP. In
--mode absolute, first move to the recorded first pose at a bounded rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.piper.data_tools.replay_picodual_lerobot import _check_workspace  # noqa: E402
from examples.piper.data_tools.replay_picodual_lerobot import _episode_relative_actions  # noqa: E402
from examples.piper.data_tools.replay_picodual_lerobot import _matrix_to_pose6  # noqa: E402
from examples.piper.data_tools.replay_picodual_lerobot import _pose6_to_matrix  # noqa: E402
from examples.piper.data_tools.replay_picodual_lerobot import _validate_trajectory  # noqa: E402
from examples.piper.data_tools.replay_picodual_lerobot import interpolate_action  # noqa: E402


def load_episode(dataset: Path, episode: int, max_frames: int = 0) -> tuple[dict, np.ndarray]:
    info = json.loads((dataset / "meta/info.json").read_text())
    if (info.get("source_kind") != "piper_sdk_raw" or info.get("robot_type") != "piper_eef"
            or info.get("pose_coordinate_frame") != "robot_base_tcp"
            or info.get("coordinate_transform_applied") is not False
            or info["features"]["state"]["shape"] != [7]
            or info["features"]["actions"]["shape"] != [7]):
        raise ValueError("expected an untransformed Piper SDK raw EEF dataset")
    offset = np.asarray(info.get("tcp_offset_m"), dtype=np.float64)
    width = float(info["gripper_mapping"]["robot_open_width_mm"])
    fps = float(info["fps"])
    if (offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > 1.0
            or not np.isfinite([width, fps]).all() or min(width, fps) <= 0):
        raise ValueError("invalid TCP offset, gripper width or FPS in dataset metadata")
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line]
    selected = next((row for row in episodes if row["episode_index"] == episode), None)
    if selected is None:
        raise ValueError(f"episode {episode} does not exist; available: {[row['episode_index'] for row in episodes]}")
    path = dataset / info["data_path"].format(
        episode_chunk=episode // int(info["chunks_size"]), episode_index=episode,
    )
    table = pq.read_table(path, columns=["frame_index", "state", "actions"])
    if len(table) != selected["length"]:
        raise ValueError("Parquet row count differs from episode metadata")
    count = min(len(table), max_frames or len(table))
    if count < 2:
        raise ValueError("replay needs at least two frames")
    indices = np.asarray(table["frame_index"].to_pylist()[:count], dtype=np.int64)
    state = np.asarray(table["state"].to_pylist()[:count], dtype=np.float64)
    actions = np.asarray(table["actions"].to_pylist()[:count], dtype=np.float64)
    if (not np.array_equal(indices, np.arange(count)) or state.shape != (count, 7)
            or actions.shape != (count, 7) or not np.isfinite(state).all()
            or not np.isfinite(actions).all() or not np.allclose(state, actions, atol=1e-6)):
        raise ValueError("raw episode must contain consecutive, finite, same-frame 7D state/actions")
    if actions[:, 6].min() < -1e-3 or actions[:, 6].max() > width + 1e-3:
        raise ValueError("recorded gripper width exceeds metadata range")
    return info, actions


def targets_from_start(actions: np.ndarray, start_tcp: np.ndarray, mode: str) -> np.ndarray:
    targets = actions.copy()
    if mode == "relative":
        transform = _pose6_to_matrix(start_tcp) @ np.linalg.inv(_pose6_to_matrix(actions[0, :6]))
        for target in targets:
            target[:6] = _matrix_to_pose6(transform @ _pose6_to_matrix(target[:6]))
    elif mode != "absolute":
        raise ValueError(f"unknown replay mode: {mode}")
    return targets


def _fresh_pose(robot, *, timeout: float = 3.0) -> np.ndarray:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        feedback = robot._piper.GetArmEndPoseMsgs()  # noqa: SLF001 - SDK feedback freshness
        if feedback.time_stamp > 0 and feedback.Hz > 0:
            return robot.get_eef_pose().astype(np.float64)
        time.sleep(0.02)
    raise TimeoutError("no fresh Piper TCP feedback")


def _fresh_gripper_raw(robot, *, timeout: float = 3.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        feedback = robot._piper.GetArmGripperMsgs()  # noqa: SLF001 - SDK feedback freshness
        if feedback.time_stamp > 0 and feedback.Hz > 0:
            raw = int(feedback.gripper_state.grippers_angle)
            if 0 <= raw <= 70000:
                return raw
            raise ValueError(f"native gripper feedback out of range: {raw}")
        time.sleep(0.02)
    raise TimeoutError("no fresh Piper native gripper feedback")


def _pose_gap(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    distance_mm = float(np.linalg.norm(current[:3] - target[:3]) * 1000)
    angle_deg = float(np.rad2deg(
        (Rotation.from_rotvec(current[3:6]).inv() * Rotation.from_rotvec(target[3:6])).magnitude()
    ))
    return distance_mm, angle_deg


def plan_absolute_approach(start_tcp: np.ndarray, recorded_tcp: np.ndarray, args) -> np.ndarray:
    """Plan TCP waypoints before sending any absolute-mode arm command."""
    start_tcp = np.asarray(start_tcp, dtype=np.float64)
    recorded_tcp = np.asarray(recorded_tcp, dtype=np.float64)
    if (start_tcp.shape != (6,) or recorded_tcp.shape != (6,)
            or not np.isfinite(start_tcp).all() or not np.isfinite(recorded_tcp).all()):
        raise ValueError("approach poses must be finite 6D TCP poses")
    distance_mm, angle_deg = _pose_gap(start_tcp, recorded_tcp)
    print(f"[Start] recorded/current TCP gap: {distance_mm:.1f} mm, {angle_deg:.1f} deg")
    if distance_mm > args.max_start_error_mm or angle_deg > args.max_start_rotation_deg:
        raise ValueError("recorded first pose exceeds absolute approach limits")
    steps = max(1, int(np.ceil(distance_mm / args.approach_step_mm)),
                int(np.ceil(angle_deg / args.approach_step_rotation_deg)))
    start = np.r_[start_tcp, 0.0]
    target = np.r_[recorded_tcp, 0.0]
    waypoints = np.array([interpolate_action(start, target, part / steps)[:6]
                          for part in range(1, steps + 1)])
    _check_workspace(waypoints, args.workspace)
    return waypoints


def move_to_absolute_start(robot, recorded_tcp: np.ndarray, args) -> np.ndarray:
    """Send bounded waypoints, then require measured arrival before replay."""
    start_tcp = _fresh_pose(robot)
    waypoints = plan_absolute_approach(start_tcp, recorded_tcp, args)
    print(f"[Approach] {len(waypoints)} TCP commands at {args.approach_hz:g} Hz")
    for target in waypoints:
        robot.send_eef_command(target)
        time.sleep(1.0 / args.approach_hz)
    deadline = time.monotonic() + args.approach_timeout_s
    while time.monotonic() < deadline:
        actual = _fresh_pose(robot)
        distance_mm, angle_deg = _pose_gap(actual, recorded_tcp)
        if distance_mm <= args.arrival_position_mm and angle_deg <= args.arrival_rotation_deg:
            print(f"[Approach] first pose reached: {distance_mm:.1f} mm, {angle_deg:.1f} deg")
            return actual
        robot.send_eef_command(recorded_tcp)
        time.sleep(0.1)
    raise TimeoutError(
        f"recorded first pose not reached within {args.approach_timeout_s:g} s; "
        f"last gap {distance_mm:.1f} mm, {angle_deg:.1f} deg"
    )


def replay(args) -> None:
    info, actions = load_episode(args.dataset, args.episode, args.max_frames)
    relative = _episode_relative_actions(actions)
    _validate_trajectory(relative, args)
    print(f"[Dataset] episode={args.episode}, frames={len(actions)}, FPS={info['fps']:g}, "
          f"gripper={actions[:, 6].min():.1f}..{actions[:, 6].max():.1f} mm, "
          f"TCP offset={info['tcp_offset_m']} m")
    if args.mode == "absolute":
        print("[Replay] mode=absolute; action[0] is the initial TCP target before frame replay")
    else:
        print("[Replay] mode=relative; action[0] is the same-frame anchor and is not commanded")
    if args.mode == "absolute":
        _check_workspace(actions, args.workspace)
    if args.dry_run:
        print("[DryRun] Dataset validated; no robot connection or motion")
        return

    from examples.piper.runtime import inference_eef as eef

    eef.GRIPPER_TCP_OFFSET_M[:] = np.asarray(info["tcp_offset_m"], dtype=np.float64)
    width = float(info["gripper_mapping"]["robot_open_width_mm"])
    robot = eef.PiperEEFController(
        can_name=args.can_name, control_gripper=False, gripper_command_max_mm=width,
    )
    arm_enabled = False
    try:
        start_tcp = _fresh_pose(robot)
        if args.mode == "absolute":
            approach = plan_absolute_approach(start_tcp, actions[0, :6], args)
            print(f"[Approach] will move to recorded first pose in {len(approach)} steps")
        targets = targets_from_start(actions, start_tcp, args.mode)
        _check_workspace(targets, args.workspace)
        print(f"[Start] current TCP={start_tcp.round(4)}; first target={targets[1, :6].round(4)}")
        if not args.yes:
            prompt = ("Press Enter to enable the arm, move to the first pose, and replay; Ctrl+C cancels..."
                      if args.mode == "absolute" else
                      "Press Enter to enable the arm and begin replay; Ctrl+C cancels...")
            input(prompt)
        if not robot.enable(speed_pct=args.speed_pct):
            raise RuntimeError("Piper arm enable failed")
        arm_enabled = True
        # The arm may settle while enabling; re-anchor the complete relative trajectory.
        if args.mode == "absolute":
            start_tcp = move_to_absolute_start(robot, actions[0, :6], args)
        else:
            start_tcp = _fresh_pose(robot)
        targets = targets_from_start(actions, start_tcp, args.mode)
        _check_workspace(targets, args.workspace)
        initial_grip = actions[0, 6]
        if args.control_gripper:
            initial_raw = _fresh_gripper_raw(robot)
            initial_grip = initial_raw / 70000.0 * width
            # Hold the present opening while enabling the native gripper.
            robot._piper.GripperCtrl(initial_raw, eef.GRIPPER_EFFORT, 0x02, 0)  # noqa: SLF001
            robot._piper.GripperCtrl(initial_raw, eef.GRIPPER_EFFORT, 0x01, 0)  # noqa: SLF001

        subdivisions = max(1, round(1.0 / args.speed))
        interval = 1.0 / float(info["fps"]) / args.speed / subdivisions
        previous = np.r_[start_tcp, initial_grip]
        for frame in range(1, len(targets)):
            target = targets[frame]
            if args.step:
                input(f"[Step] {frame}/{len(targets)-1}: press Enter to command...")
            for part in range(1, subdivisions + 1):
                command = interpolate_action(previous, target, part / subdivisions)
                _check_workspace(command[None, :], args.workspace)
                robot.send_eef_command(command[:6])
                if args.control_gripper:
                    robot.send_gripper_command(float(command[6]))
                time.sleep(interval)
            previous = target
            if args.readback and (args.step or frame % 25 == 0 or frame == len(targets) - 1):
                actual = _fresh_pose(robot)
                error_mm = np.linalg.norm(actual[:3] - target[:3]) * 1000
                print(f"[Readback] frame {frame}: TCP position error {error_mm:.1f} mm")
        print(f"[Replay] completed {len(targets)-1} target frames")
    except KeyboardInterrupt:
        print("[Replay] interrupted; no further commands sent")
    finally:
        robot._piper.DisconnectPort()  # noqa: SLF001
        if arm_enabled:
            print("[Replay] arm remains enabled at the last target")
        else:
            print("[Replay] arm was not enabled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("local/datasets/piper_sdk_raw_eef"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--speed", type=float, default=0.25, help="time multiplier; 0.25 is quarter speed")
    parser.add_argument("--speed-pct", type=int, default=30)
    parser.add_argument("--step", action="store_true", help="wait for Enter before every target")
    parser.add_argument("--readback", action="store_true")
    parser.add_argument("--control-gripper", action="store_true", help="also command the native Piper gripper")
    parser.add_argument("--yes", action="store_true", help="skip the start prompt")
    parser.add_argument("--max-total-translation-m", type=float, default=0.5)
    parser.add_argument("--max-step-translation-m", type=float, default=0.05)
    parser.add_argument("--max-total-rotation-deg", type=float, default=120.0)
    parser.add_argument("--max-step-rotation-deg", type=float, default=20.0)
    parser.add_argument("--max-start-error-mm", type=float, default=100.0,
                        help="maximum distance to recorded first pose before approach")
    parser.add_argument("--max-start-rotation-deg", type=float, default=45.0,
                        help="maximum rotation to recorded first pose before approach")
    parser.add_argument("--approach-step-mm", type=float, default=5.0)
    parser.add_argument("--approach-step-rotation-deg", type=float, default=2.0)
    parser.add_argument("--approach-hz", type=float, default=20.0)
    parser.add_argument("--approach-timeout-s", type=float, default=10.0)
    parser.add_argument("--arrival-position-mm", type=float, default=10.0)
    parser.add_argument("--arrival-rotation-deg", type=float, default=5.0)
    parser.add_argument("--workspace", type=float, nargs=6,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    args = parser.parse_args()
    if (args.episode < 0 or args.max_frames < 0 or not np.isfinite(args.speed) or args.speed <= 0
            or not 20 <= args.speed_pct <= 100):
        parser.error("episode/max-frames must be nonnegative; speed > 0; speed-pct 20..100")
    limits = [
        args.max_total_translation_m, args.max_step_translation_m,
        args.max_total_rotation_deg, args.max_step_rotation_deg,
        args.max_start_error_mm, args.max_start_rotation_deg,
        args.approach_step_mm, args.approach_step_rotation_deg, args.approach_hz,
        args.approach_timeout_s, args.arrival_position_mm, args.arrival_rotation_deg,
    ]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        parser.error("all motion and start-error limits must be positive and finite")
    if args.workspace is not None and not np.isfinite(args.workspace).all():
        parser.error("workspace coordinates must be finite")
    if args.workspace is not None and any(args.workspace[i] >= args.workspace[i + 1] for i in (0, 2, 4)):
        parser.error("workspace minimum must be below maximum on every axis")
    replay(args)


if __name__ == "__main__":
    main()
