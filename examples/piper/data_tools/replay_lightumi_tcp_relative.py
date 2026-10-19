#!/usr/bin/env python3
# ruff: noqa: E402, SLF001
"""Replay raw or prepared lightUMI TCP actions relative to the current Piper TCP.

The raw dataset action is the *next* aligned absolute tracking-world TCP pose,
encoded as xyz and the first two rotation-matrix columns (columns grouped).
The prepared dataset has same-frame 7D actions with a duplicated initial pose.
The raw terminal duplicate or prepared initial duplicate is excluded; both
formats then yield the same pose sequence. Any gripper channel is ignored.
For every remaining action k, command
T_piper_initial @ inv(T_lightumi_action_0) @ T_lightumi_action_k.
"""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
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
    if info.get("robot_type") == "piper_chunk_relative_eef" and info.get("source_kind") == "lightumi":
        return load_prepared_episode(data_dir, episode, info)
    action_feature = info.get("features", {}).get("action", {})
    action_shape = action_feature.get("shape")
    if info.get("robot_type") != "lightumi" or action_shape not in ([9], [10]):
        raise ValueError("expected lightUMI 9D/10D TCP action dataset")
    names = action_feature.get("names", [])
    expected_names = ROTATION_NAMES + (["gripper_angle_rad"] if action_shape == [10] else [])
    if names[3:] != expected_names:
        raise ValueError("unexpected TCP rotation-6D or gripper column order")
    conversion = info.get("lightumi_conversion", {})
    action_description = conversion.get("action", "")
    if not any(
        f"next aligned absolute {world} TCP" in action_description
        for world in ("PICO-world", "source-tracking-world")
    ):
        raise ValueError("expected next-aligned absolute tracking-world TCP actions")
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
    return fps, decode_tcp_actions(actions[:-1, :9])


def load_prepared_episode(data_dir: Path, episode: int, info: dict) -> tuple[float, np.ndarray]:
    """Read the same-frame 7D actions, excluding their duplicated first pose."""
    import pandas as pd

    if (
        info.get("features", {}).get("actions", {}).get("shape") != [7]
        or info.get("features", {}).get("state", {}).get("shape") != [1]
        or info.get("first_frame_tcp_fill") != "next_frame_pose"
        or info.get("coordinate_transform_applied") is not False
        or info.get("gripper_mapping", {}).get("available") is not False
    ):
        raise ValueError("expected prepared TCP-only lightUMI same-frame actions")
    episodes = {
        int(row["episode_index"]): int(row["length"])
        for line in (data_dir / "meta/episodes.jsonl").read_text().splitlines()
        if line
        for row in [json.loads(line)]
    }
    if episode not in episodes:
        raise ValueError(f"episode {episode} is unavailable; choices: {sorted(episodes)}")
    parquet = data_dir / info["data_path"].format(
        episode_chunk=episode // int(info["chunks_size"]), episode_index=episode
    )
    frame = pd.read_parquet(parquet, columns=["frame_index", "actions", "state"]).sort_values("frame_index")
    if len(frame) != episodes[episode] or not np.array_equal(frame["frame_index"], np.arange(len(frame))):
        raise ValueError("episode frame indices or length do not match metadata")
    actions = np.asarray(frame["actions"].to_list(), dtype=np.float64)
    state = np.asarray(frame["state"].to_list(), dtype=np.float64)
    if (
        actions.shape != (len(frame), 7)
        or state.shape != (len(frame), 1)
        or len(actions) < 3
        or not np.isfinite(actions).all()
        or not np.isfinite(state).all()
        or not np.allclose(actions[:, 6], 0, rtol=0, atol=1e-6)
        or not np.allclose(state, 0, rtol=0, atol=1e-6)
        or not np.allclose(actions[0, :6], actions[1, :6], rtol=0, atol=1e-6)
    ):
        raise ValueError("expected finite prepared TCP poses, zero gripper, and duplicated first pose")
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("dataset FPS must be positive")
    # Prepared actions are [a0, a0, a1, ..., aN-2]. The raw replay loads
    # [a0, a1, ..., aN-2] after dropping the source's repeated final action.
    return fps, actions[1:, :6]


def make_targets(poses: np.ndarray, robot_base: np.ndarray) -> np.ndarray:
    """Anchor the complete TCP-relative path at the measured starting TCP."""
    actions = np.column_stack((poses, np.zeros(len(poses))))
    return replay.chunk_relative_targets(actions, 0, len(actions) - 1, robot_base)


def make_chunk_targets(poses: np.ndarray, start: int, horizon: int, robot_base: np.ndarray) -> np.ndarray:
    """Anchor one action chunk at the Piper TCP measured at that chunk's start."""
    actions = np.column_stack((poses, np.zeros(len(poses))))
    return replay.chunk_relative_targets(actions, start, horizon, robot_base)


def relative_poses(poses: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Express absolute TCP poses in their measured initial TCP frame."""
    anchor_inverse = np.linalg.inv(replay._pose6_to_matrix(anchor))
    return np.stack(
        [replay._matrix_to_pose6(anchor_inverse @ replay._pose6_to_matrix(pose)) for pose in poses]
    )


def save_comparison(
    gif_path: Path,
    dataset_relative: np.ndarray,
    robot_absolute: np.ndarray,
    target_absolute: np.ndarray,
    elapsed_s: np.ndarray,
    dataset_anchor: np.ndarray,
    robot_anchor: np.ndarray,
    *,
    episode: int,
    dataset_fps: float,
    gif_fps: float,
) -> None:
    """Persist every measured TCP sample and render a shared relative-path GIF."""
    robot_relative = relative_poses(robot_absolute, robot_anchor)
    if not (len(dataset_relative) == len(robot_relative) == len(target_absolute) == len(elapsed_s)):
        raise ValueError("comparison traces must have matching frame counts")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = gif_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        dataset_relative_tcp=dataset_relative,
        robot_relative_tcp=robot_relative,
        robot_absolute_tcp=robot_absolute,
        target_absolute_tcp=target_absolute,
        elapsed_s=elapsed_s,
        dataset_anchor=dataset_anchor,
        robot_anchor=robot_anchor,
        episode=episode,
        dataset_fps=dataset_fps,
    )
    print(f"[Comparison] Saved {len(robot_relative)} aligned TCP samples: {npz_path}")
    render_comparison_gif(gif_path, dataset_relative, robot_relative, elapsed_s, gif_fps=gif_fps)
    print(f"[Comparison] Saved relative TCP trajectory GIF: {gif_path}")


def render_comparison_gif(
    gif_path: Path,
    dataset_relative: np.ndarray,
    robot_relative: np.ndarray,
    elapsed_s: np.ndarray,
    *,
    gif_fps: float,
) -> None:
    """Draw dataset and measured TCP in their respective starting TCP axes."""
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import PillowWriter
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    desired_mm = dataset_relative[:, :3] * 1000
    actual_mm = robot_relative[:, :3] * 1000
    position_error_mm = np.linalg.norm(desired_mm - actual_mm, axis=1)
    desired_rotation = Rotation.from_rotvec(dataset_relative[:, 3:6])
    actual_rotation = Rotation.from_rotvec(robot_relative[:, 3:6])
    rotation_error_deg = np.rad2deg((desired_rotation.inv() * actual_rotation).magnitude())
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    moving_lines = []
    for axis, (i, j, label) in zip(axes.flat[:3], projections, strict=True):
        axis.plot(desired_mm[:, i], desired_mm[:, j], color="tab:blue", alpha=0.2, linewidth=1)
        desired_line, = axis.plot([], [], color="tab:blue", label="Dataset TCP")
        actual_line, = axis.plot([], [], color="tab:orange", label="Piper TCP")
        desired_dot, = axis.plot([], [], "o", color="tab:blue", markersize=4)
        actual_dot, = axis.plot([], [], "o", color="tab:orange", markersize=4)
        moving_lines.append((desired_line, actual_line, desired_dot, actual_dot, i, j))
        values = np.r_[desired_mm[:, [i, j]], actual_mm[:, [i, j]]]
        low = values.min(axis=0)
        high = values.max(axis=0)
        middle = (low + high) / 2
        half_span = max(float(np.max(high - low)) * 0.57, 5.0)
        axis.set_xlim(middle[0] - half_span, middle[0] + half_span)
        axis.set_ylim(middle[1] - half_span, middle[1] + half_span)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"{label[0]} relative to start (mm)")
        axis.set_ylabel(f"{label[1]} relative to start (mm)")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(loc="best", fontsize=8)
    error_axis = axes.flat[3]
    position_line, = error_axis.plot([], [], color="tab:red", label="Position error (mm)")
    error_axis.set_xlim(0, max(float(elapsed_s[-1]), 0.1))
    error_axis.set_ylim(0, max(float(position_error_mm.max()) * 1.15, 1.0))
    error_axis.set_xlabel("Elapsed replay time (s)")
    error_axis.set_ylabel("Relative position error (mm)")
    error_axis.grid(alpha=0.25)
    error_axis.legend(loc="best", fontsize=8)

    def update(frame: int):
        for desired_line, actual_line, desired_dot, actual_dot, i, j in moving_lines:
            desired_line.set_data(desired_mm[: frame + 1, i], desired_mm[: frame + 1, j])
            actual_line.set_data(actual_mm[: frame + 1, i], actual_mm[: frame + 1, j])
            desired_dot.set_data([desired_mm[frame, i]], [desired_mm[frame, j]])
            actual_dot.set_data([actual_mm[frame, i]], [actual_mm[frame, j]])
        position_line.set_data(elapsed_s[: frame + 1], position_error_mm[: frame + 1])
        fig.suptitle(
            f"TCP relative to each start | frame {frame}/{len(desired_mm) - 1} | "
            f"position {position_error_mm[frame]:.1f} mm | rotation {rotation_error_deg[frame]:.1f} deg",
            fontsize=10,
        )
        return ()

    animation = FuncAnimation(fig, update, frames=len(desired_mm), interval=1000 / gif_fps)
    try:
        animation.save(gif_path, writer=PillowWriter(fps=gif_fps), dpi=80)
    finally:
        plt.close(fig)


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
    print("[lightUMI] first TCP pose is the data anchor; conversion duplicate is excluded; gripper is ignored")
    exec_horizon = getattr(args, "exec_horizon", None)
    if exec_horizon is not None:
        print(f"[ChunkRelative] re-anchor to measured Piper TCP every {exec_horizon} future targets; "
              f"{len(range(0, count, exec_horizon))} chunks")
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
    if exec_horizon is None:
        replay._check_workspace(make_targets(poses, base), args.workspace)
    subdivisions = max(1, round(1 / args.speed))
    period = 1 / fps / args.speed / subdivisions
    print(
        f"[Replay] Piper TCP anchor {base.round(4)}, {subdivisions} commands/data frame, {period * 1000:.1f} ms interval"
    )
    errors = []
    measured = [base.copy()]
    commanded = [base.copy()]
    elapsed = [0.0]
    try:
        if not args.yes:
            input("Check TCP axes, workspace, and clear path; press Enter to replay or Ctrl+C to cancel...")
        started_at = time.monotonic()
        chunk_size = exec_horizon or count
        for start in range(0, count, chunk_size):
            chunk_base = base if start == 0 else robot.get_eef_pose().astype(np.float64)
            targets = make_chunk_targets(poses, start, min(chunk_size, count - start), chunk_base)
            replay._check_workspace(targets, args.workspace)
            if exec_horizon is not None:
                print(f"[Chunk {start}] Piper TCP anchor {chunk_base.round(4)}; {len(targets)} targets")
            previous = np.r_[chunk_base, 0.0]
            for frame_index, target in enumerate(targets, start=start + 1):
                if args.step:
                    input(f"[Step] action {frame_index}/{count}; press Enter to send...")
                for substep in range(1, subdivisions + 1):
                    command = replay.interpolate_action(previous, target, substep / subdivisions)
                    replay._check_workspace(command[None, :], args.workspace)
                    robot.send_eef_command(command[:6])
                    time.sleep(period)
                previous = target
                actual = robot.get_eef_pose().astype(np.float64)
                measured.append(actual)
                commanded.append(target[:6].copy())
                elapsed.append(time.monotonic() - started_at)
                if args.readback:
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
        if len(measured) > 1:
            gif_path = args.comparison_gif or (
                Path("local/runs/outputs")
                / f"{args.data_dir.name}_episode{args.episode:03d}_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}_tcp_comparison.gif"
            )
            dataset_relative = (relative_poses(np.stack(commanded), base) if exec_horizon is not None
                                else np.vstack((np.zeros(6), relative[: len(measured) - 1, :6])))
            gif_fps = args.gif_fps or max(1, min(30, round(fps * args.speed)))
            save_comparison(
                gif_path,
                dataset_relative,
                np.stack(measured),
                np.stack(commanded),
                np.asarray(elapsed),
                base if exec_horizon is not None else poses[0],
                base,
                episode=args.episode,
                dataset_fps=fps,
                gif_fps=gif_fps,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay raw or prepared lightUMI TCP actions in Piper TCP-relative mode")
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="validate the episode without connecting to Piper")
    parser.add_argument("--max-frames", type=int, default=None, help="limit future TCP targets for initial validation")
    parser.add_argument("--exec-horizon", type=int, default=None,
                        help="chunk-relative mode: re-anchor at current Piper TCP after this many future targets")
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
    parser.add_argument("--comparison-gif", type=Path, default=None, help="output GIF path; a matching .npz trace is saved")
    parser.add_argument("--gif-fps", type=float, default=None, help="comparison GIF FPS; default follows replay time scale")
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
    if args.exec_horizon is not None and args.exec_horizon < 1:
        parser.error("exec-horizon must be >= 1")
    if args.comparison_gif is not None and args.comparison_gif.suffix.lower() != ".gif":
        parser.error("comparison-gif must end in .gif")
    if args.gif_fps is not None and (not np.isfinite(args.gif_fps) or args.gif_fps <= 0):
        parser.error("gif-fps must be positive and finite")
    if args.tcp_offset_m is not None and (
        not np.isfinite(args.tcp_offset_m).all() or np.linalg.norm(args.tcp_offset_m) > 1.0
    ):
        parser.error("tcp-offset-m must be finite meters with length <= 1 m")
    if args.workspace is not None and any(args.workspace[i] >= args.workspace[i + 1] for i in (0, 2, 4)):
        parser.error("workspace minimum must be less than maximum on each axis")
    run(args)


if __name__ == "__main__":
    main()
