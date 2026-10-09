#!/usr/bin/env python3
# ruff: noqa: E402, SLF001
"""Replay lightUMI 9D absolute TCP actions relative to the current Piper TCP.

The dataset action is the *next* aligned PICO-world TCP pose, encoded as xyz
and the first two rotation-matrix columns (columns grouped). There is no TCP
state or gripper action. The first action is used as a pose anchor, and the
repeated last action is dropped. For every remaining action k, command
T_piper_initial @ inv(T_lightumi_action_0) @ T_lightumi_action_k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.piper.data_tools import replay_picodual_lerobot as replay

DEFAULT_DATA_DIR = "local/datasets/lightumi_20261009_212049"
ROTATION_NAMES = [
    "left_tcp_rot_r00",
    "left_tcp_rot_r10",
    "left_tcp_rot_r20",
    "left_tcp_rot_r01",
    "left_tcp_rot_r11",
    "left_tcp_rot_r21",
]


def decode_tcp_actions(actions: np.ndarray) -> np.ndarray:
    """Decode column-grouped rotation-6D into [xyz, rotvec] TCP poses."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 9 or len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError("lightUMI actions must be finite (N, 9)")
    first = actions[:, 3:6]
    second = actions[:, 6:9]
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    dot = np.sum(first * second, axis=1)
    if np.any(np.abs(first_norm - 1) > 0.05) or np.any(np.abs(second_norm - 1) > 0.05) or np.any(np.abs(dot) > 0.05):
        raise ValueError("lightUMI rotation-6D columns are not near orthonormal")
    x_axis = first / first_norm[:, None]
    y_axis = second - np.sum(second * x_axis, axis=1)[:, None] * x_axis
    y_axis /= np.linalg.norm(y_axis, axis=1)[:, None]
    z_axis = np.cross(x_axis, y_axis)
    rotations = np.stack((x_axis, y_axis, z_axis), axis=-1)
    return np.column_stack((actions[:, :3], Rotation.from_matrix(rotations).as_rotvec()))


def load_episode(data_dir: Path, episode: int) -> tuple[float, np.ndarray]:
    """Read only one episode's action column; no video decoding or robot access."""
    import pandas as pd

    info = json.loads((data_dir / "meta/info.json").read_text())
    if info.get("robot_type") != "lightumi" or info.get("features", {}).get("action", {}).get("shape") != [9]:
        raise ValueError("expected lightUMI 9D TCP action dataset")
    if info["features"]["action"].get("names", [])[3:] != ROTATION_NAMES:
        raise ValueError("unexpected rotation-6D column order")
    conversion = info.get("lightumi_conversion", {})
    if "next aligned absolute PICO-world TCP" not in conversion.get("action", ""):
        raise ValueError("expected next-aligned absolute PICO-world TCP actions")
    if conversion.get("rotation_6d") != "columns_grouped:[r00,r10,r20,r01,r11,r21]":
        raise ValueError("unexpected rotation-6D encoding")
    episodes = {
        int(row["episode_index"]): int(row["length"])
        for line in (data_dir / "meta/episodes.jsonl").read_text().splitlines()
        if line
        for row in [json.loads(line)]
    }
    if episode not in episodes:
        raise ValueError(f"episode {episode} is unavailable; choices: {sorted(episodes)}")
    sources = {
        int(row["episode_index"]): row
        for line in (data_dir / "meta/lightumi_sources.jsonl").read_text().splitlines()
        if line
        for row in [json.loads(line)]
    }
    source = sources.get(episode, {})
    if source.get("action_is_next_aligned_tcp") is not True or source.get("last_action_repeated") is not True:
        raise ValueError("episode must have next-aligned TCP actions and a repeated final action")
    parquet = data_dir / info["data_path"].format(
        episode_chunk=episode // int(info["chunks_size"]), episode_index=episode
    )
    frame = pd.read_parquet(parquet, columns=["frame_index", "action"]).sort_values("frame_index")
    if len(frame) != episodes[episode] or not np.array_equal(frame["frame_index"], np.arange(len(frame))):
        raise ValueError("episode frame indices or length do not match metadata")
    actions = np.asarray(frame["action"].to_list(), dtype=np.float64)
    if len(actions) < 3 or not np.allclose(actions[-1], actions[-2], rtol=0, atol=1e-6):
        raise ValueError("expected at least two distinct TCP actions followed by a repeated final action")
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("dataset FPS must be positive")
    return fps, decode_tcp_actions(actions[:-1])


def make_targets(poses: np.ndarray, robot_base: np.ndarray) -> np.ndarray:
    """Anchor the complete TCP-relative path at the measured starting TCP."""
    actions = np.column_stack((poses, np.zeros(len(poses))))
    return replay.chunk_relative_targets(actions, 0, len(actions) - 1, robot_base)


def run(args: argparse.Namespace) -> None:
    fps, poses = load_episode(args.data_dir, args.episode)
    count = len(poses) - 1 if args.max_frames is None else min(args.max_frames, len(poses) - 1)
    poses = poses[: count + 1]
    relative = make_targets(poses, np.zeros(6))
    reference = np.vstack((np.zeros(7), relative))
    replay._validate_trajectory(reference, args)
    pos_steps, rot_steps = replay._trajectory_steps(reference)
    print(
        f"[lightUMI] episode {args.episode}: {len(poses)} unique action poses, {count} future TCP targets, {fps:g} Hz"
    )
    print("[lightUMI] action[0] is the data anchor; the repeated final action and gripper are not commanded")
    print(
        f"[lightUMI] max relative displacement {np.max(np.linalg.norm(relative[:, :3], axis=1)) * 1000:.1f} mm; "
        f"max step {pos_steps.max() * 1000:.1f} mm, {np.rad2deg(rot_steps.max()):.2f} deg"
    )
    if args.dry_run:
        print("[DryRun] No robot connection or motion")
        return

    from examples.piper.runtime import inference_eef as eef

    if args.tcp_offset_m is not None:
        eef.GRIPPER_TCP_OFFSET_M[:] = args.tcp_offset_m
    print(f"[TCP] J6-to-TCP offset in flange axes: {eef.GRIPPER_TCP_OFFSET_M.tolist()} m")
    robot = eef.PiperEEFController(args.can_name, control_gripper=False)
    if not robot.enable(speed_pct=args.speed_pct):
        raise RuntimeError("Piper enable failed")
    base = robot.get_eef_pose().astype(np.float64)
    targets = make_targets(poses, base)
    replay._check_workspace(targets, args.workspace)
    subdivisions = max(1, round(1 / args.speed))
    period = 1 / fps / args.speed / subdivisions
    previous = np.r_[base, 0.0]
    print(
        f"[Replay] Piper TCP anchor {base.round(4)}, {subdivisions} commands/data frame, {period * 1000:.1f} ms interval"
    )
    errors = []
    try:
        if not args.yes:
            input("Check TCP axes, workspace, and clear path; press Enter to replay or Ctrl+C to cancel...")
        for frame_index, target in enumerate(targets, start=1):
            if args.step:
                input(f"[Step] action {frame_index}/{count}; press Enter to send...")
            for substep in range(1, subdivisions + 1):
                command = replay.interpolate_action(previous, target, substep / subdivisions)
                replay._check_workspace(command[None, :], args.workspace)
                robot.send_eef_command(command[:6])
                time.sleep(period)
            previous = target
            if args.readback:
                actual = robot.get_eef_pose()
                error = np.linalg.inv(replay._pose6_to_matrix(target[:6])) @ replay._pose6_to_matrix(actual)
                errors.append((np.linalg.norm(error[:3, 3]), Rotation.from_matrix(error[:3, :3]).magnitude()))
            if args.step or frame_index % 25 == 0 or frame_index == count:
                print(f"  target {frame_index}/{count}: xyz={target[:3].round(4)}, rotvec={target[3:6].round(3)}")
        print(f"[Replay] Completed {count} TCP targets")
        if errors:
            values = np.asarray(errors)
            print(
                f"[Readback] translation mean/max {values[:, 0].mean() * 1000:.1f}/{values[:, 0].max() * 1000:.1f} mm; "
                f"rotation mean/max {np.rad2deg(values[:, 1].mean()):.2f}/{np.rad2deg(values[:, 1].max()):.2f} deg"
            )
    except KeyboardInterrupt:
        print("[Replay] Interrupted")
    finally:
        print("[Replay] Robot remains enabled")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay lightUMI 9D TCP actions in Piper TCP-relative mode")
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="validate the episode without connecting to Piper")
    parser.add_argument("--max-frames", type=int, default=None, help="limit future TCP targets for initial validation")
    parser.add_argument("--speed", type=float, default=0.25, help="dataset playback time scale; default 0.25")
    parser.add_argument("--speed-pct", type=int, default=30, help="Piper Cartesian speed percentage")
    parser.add_argument("--can-name", default="can0")
    parser.add_argument(
        "--tcp-offset-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="J6 flange to TCP translation in flange axes (m); overrides the project default for this run",
    )
    parser.add_argument("--step", action="store_true", help="confirm each data-frame target")
    parser.add_argument("--readback", action="store_true", help="summarize observed TCP tracking error")
    parser.add_argument("--yes", action="store_true", help="skip the start prompt")
    parser.add_argument("--max-total-translation-m", type=float, default=0.5)
    parser.add_argument("--max-step-translation-m", type=float, default=0.05)
    parser.add_argument("--max-total-rotation-deg", type=float, default=120)
    parser.add_argument("--max-step-rotation-deg", type=float, default=20)
    parser.add_argument(
        "--workspace", type=float, nargs=6, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX")
    )
    args = parser.parse_args()
    if args.speed <= 0 or not 20 <= args.speed_pct <= 100 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("speed must be positive, speed-pct 20..100, and max-frames >= 1")
    if args.tcp_offset_m is not None and (
        not np.isfinite(args.tcp_offset_m).all() or np.linalg.norm(args.tcp_offset_m) > 1.0
    ):
        parser.error("tcp-offset-m must be finite meters with length <= 1 m")
    if args.workspace is not None and any(args.workspace[i] >= args.workspace[i + 1] for i in (0, 2, 4)):
        parser.error("workspace minimum must be less than maximum on each axis")
    run(args)


if __name__ == "__main__":
    main()
