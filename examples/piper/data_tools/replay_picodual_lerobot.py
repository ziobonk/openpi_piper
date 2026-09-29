#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""回放 PicoDual 单臂 EEF 数据集，包括 ``pick_cube_chunk_relative``。

对 chunk-relative 数据，每次查询从当前机械臂 TCP 锚定该块：
``T_target[k] = T_robot_now @ inv(T_actions[start]) @ T_actions[start+k]``。
首个 action 与观测同帧，只作基准不执行；默认每次执行后续 20 步，再读取新的
机械臂 TCP。数据集 state 只有夹爪开度，位姿从 actions 读取。无需 PICO World
到 Robot Base 的外参，但两端 TCP 原点和轴方向必须一致。

    python examples/piper/data_tools/replay_picodual_lerobot.py \
        --data_dir ./pick_cube_chunk_relative --episode 0 --dry_run
    python examples/piper/data_tools/replay_picodual_lerobot.py \
        --data_dir ./pick_cube_chunk_relative --episode 0 --step --speed 0.25

以下为旧 ``pick_cube`` 数据集的约定。

数据约定
--------
转换脚本写出的 ``state[t]`` 是数据集所用虚拟 TCP 在 episode 首帧坐标系中的
SE(3) 位姿 ``D_dataset(t)``。若数据集安装关系与当前 ``pico_arm_transform.py``
不同，本脚本从元数据读取 ``^C T_E_dataset``，计算：

    H = inv(^C T_E_dataset) @ ^C T_E_current
    D_current(t) = inv(H) @ D_dataset(t) @ H

记录机械臂开始回放时的真实 TCP ``^B T_R(0)``，每帧目标固定为：

    ^B T_target(t) = ^B T_R(0) @ D_current(t)

该锚定方式不会累计数值漂移，适合验证坐标轴方向、旋转轴和 TCP 杠杆臂。
不需要 PICO World 到 Robot Base 的外参。夹爪默认不执行，需显式开启。

用法：
    # 先离线检查旧 pick_cube 的转换范围
    python examples/piper/data_tools/replay_picodual_lerobot.py --data_dir ./pick_cube \
        --episode 0 --adapt_dataset_tcp --dry_run

    # 上机逐帧确认（默认不控制夹爪）
    python examples/piper/data_tools/replay_picodual_lerobot.py --data_dir ./pick_cube \
        --episode 0 --adapt_dataset_tcp --step --speed 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PIPER_DIR = HERE.parent
REPO_ROOT = HERE.parents[2]
for import_path in (HERE, PIPER_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


DEFAULT_DATA_DIR = "./pick_cube_chunk_relative"
DEFAULT_SPEED_PCT = 30
DEFAULT_GRIPPER_PORT = "/dev/ttyACM1"
DEFAULT_GRIPPER_CLOSE_RAD = 10.0
DEFAULT_GRIPPER_OPEN_WIDTH_MM = 23.0
DEFAULT_GRIPPER_CURRENT_LIMIT = 0.3
DEFAULT_GRIPPER_SPEED_RAD_S = 2.0


def _load_replay_helpers():
    """延迟导入，避免 ``--dry_run`` 或 ``--info_only`` 依赖机械臂 SDK。"""
    from replay_dataset_eef import get_dataset_fps
    from replay_dataset_eef import get_episode_task
    from replay_dataset_eef import load_eef_dataset

    return {
        "get_dataset_fps": get_dataset_fps,
        "get_episode_task": get_episode_task,
        "load_eef_dataset": load_eef_dataset,
    }


def load_chunk_relative_episode(data_dir: str, episode: int | None):
    """Read only the requested episode; the full dataset contains large images."""
    import pandas as pd

    root = Path(data_dir)
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("robot_type") != "piper_chunk_relative_eef":
        raise ValueError("expected a piper_chunk_relative_eef dataset")
    if info.get("pose_coordinate_frame") != "pico_world_tcp" or info.get("coordinate_transform_applied") is not False:
        raise ValueError("expected untransformed absolute PICO World TCP actions")
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    if not lengths:
        raise ValueError("dataset contains no episodes")
    selected = min(lengths) if episode is None else episode
    if selected not in lengths:
        raise ValueError(f"episode {selected} does not exist; available: {sorted(lengths)}")
    path = root / info["data_path"].format(
        episode_chunk=selected // int(info["chunks_size"]), episode_index=selected
    )
    frame = pd.read_parquet(path).sort_values("frame_index").reset_index(drop=True)
    actions = np.stack(frame["actions"].values).astype(np.float64)
    gripper = np.stack(frame["state"].values).astype(np.float64)
    if actions.shape != (lengths[selected], 7) or gripper.shape != (lengths[selected], 1):
        raise ValueError(f"invalid action/state shape: {actions.shape}, {gripper.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(gripper).all():
        raise ValueError("nonfinite action or state in dataset")
    if not np.allclose(actions[:, 6], gripper[:, 0], atol=1e-4):
        raise ValueError("same-frame action gripper and state gripper disagree")
    return info, frame, selected, lengths, actions


def chunk_relative_targets(actions: np.ndarray, start: int, horizon: int, base_tcp: np.ndarray) -> np.ndarray:
    """Decode one dataset chunk as ``T_robot_at_query @ inv(A[start]) @ A[k]``.

    The same-frame action at ``start`` is not sent to the robot. The caller
    captures ``base_tcp`` once per chunk, before sending any of its targets.
    """
    actions = np.asarray(actions, dtype=np.float64)
    base_tcp = np.asarray(base_tcp, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("actions must be finite (N, 7)")
    if base_tcp.shape != (6,) or not np.isfinite(base_tcp).all():
        raise ValueError("base_tcp must be a finite 6D pose")
    if not 0 <= start < len(actions) or horizon < 1:
        raise ValueError("invalid chunk start or horizon")
    base_inverse = np.linalg.inv(_pose6_to_matrix(actions[start, :6]))
    robot_base = _pose6_to_matrix(base_tcp)
    end = min(start + horizon + 1, len(actions))
    targets = actions[start + 1 : end].copy()
    for target in targets:
        target[:6] = _matrix_to_pose6(robot_base @ base_inverse @ _pose6_to_matrix(target[:6]))
    return targets


def _episode_relative_actions(actions: np.ndarray) -> np.ndarray:
    """Absolute dataset poses relative to its first action for validation."""
    relative = actions.copy()
    first_inverse = np.linalg.inv(_pose6_to_matrix(actions[0, :6]))
    for target in relative:
        target[:6] = _matrix_to_pose6(first_inverse @ _pose6_to_matrix(target[:6]))
    return relative


def _check_workspace(targets: np.ndarray, workspace: list[float] | None) -> None:
    if workspace is None or len(targets) == 0:
        return
    xmin, xmax, ymin, ymax, zmin, zmax = workspace
    inside = (
        (targets[:, 0] >= xmin) & (targets[:, 0] <= xmax)
        & (targets[:, 1] >= ymin) & (targets[:, 1] <= ymax)
        & (targets[:, 2] >= zmin) & (targets[:, 2] <= zmax)
    )
    if not inside.all():
        bad = int(np.flatnonzero(~inside)[0])
        raise ValueError(f"target {bad} outside --workspace: {targets[bad, :3]}")


def replay_chunk_relative(args) -> None:
    """Replay data using exactly the training/inference chunk-local TCP rule."""
    info, _, episode, lengths, actions = load_chunk_relative_episode(args.data_dir, args.episode)
    if args.info_only:
        print(f"Dataset: {args.data_dir}; FPS: {info['fps']}; episodes: {lengths}")
        return
    if args.adapt_dataset_tcp:
        raise ValueError("chunk-relative data has no controller_to_tcp; match the physical TCP definitions")
    if args.action_horizon < 2 or not 1 <= args.exec_horizon < args.action_horizon:
        raise ValueError("require action_horizon >= 2 and 1 <= exec_horizon < action_horizon")
    expected_width = float(info["gripper_mapping"]["robot_open_width_mm"])
    if args.gripper_open_width_mm is None:
        args.gripper_open_width_mm = expected_width
    elif args.control_gripper and not np.isclose(args.gripper_open_width_mm, expected_width):
        raise ValueError(f"dataset gripper width is 0..{expected_width:g} mm; --gripper_open_width_mm differs")
    ideal = _episode_relative_actions(actions)
    _validate_trajectory(ideal, args)
    starts = range(0, len(actions) - 1, args.exec_horizon)
    print(f"Episode {episode}: {len(actions)} frames, {len(starts)} chunks, "
          f"query stride={args.exec_horizon}, action horizon={args.action_horizon}")
    print("Each chunk uses its first absolute TCP action as T_base; action[0] is skipped.")
    pos, rot = _trajectory_steps(ideal)
    print(f"Ideal trajectory: max step {pos.max()*1000:.1f} mm, "
          f"{np.rad2deg(rot.max()):.2f} deg; gripper "
          f"{actions[:,6].min():.1f}..{actions[:,6].max():.1f} mm")
    if args.dry_run:
        print("[DryRun] No robot connection or motion.")
        return

    from examples.piper.runtime import inference_eef as eef

    robot = _build_gripper_and_robot(args, eef)
    if not robot.enable(speed_pct=args.speed_pct):
        raise RuntimeError("Piper enable failed")
    period = 1.0 / float(info["fps"]) / args.speed
    sent = 0
    position_errors = []
    rotation_errors = []
    try:
        if not args.yes:
            input("Confirm TCP definitions and workspace, then press Enter to start; Ctrl+C cancels...")
        for start in starts:
            # A fresh measured robot TCP is the base for this entire chunk.
            robot_base = robot.get_eef_pose().astype(np.float64)
            targets = chunk_relative_targets(actions, start, args.exec_horizon, robot_base)
            _check_workspace(targets, args.workspace)
            print(f"[Chunk {start}] robot base={robot_base.round(4)}; {len(targets)} commands")
            for offset, target in enumerate(targets, start=1):
                frame = start + offset
                if args.step:
                    input(f"[Step] frame {frame}/{len(actions)-1}, press Enter to send...")
                robot.execute_action(target)
                sent += 1
                time.sleep(period)
                if args.readback:
                    actual = robot.get_eef_pose().astype(np.float64)
                    error = np.linalg.inv(_pose6_to_matrix(target[:6])) @ _pose6_to_matrix(actual)
                    position_errors.append(np.linalg.norm(error[:3, 3]))
                    rotation_errors.append(Rotation.from_matrix(error[:3, :3]).magnitude())
                if args.step or frame % 25 == 0 or frame == len(actions)-1:
                    print(f"  frame {frame}: xyz={target[:3].round(4)} "
                          f"rotvec={target[3:6].round(3)} grip={target[6]:.1f}mm")
        print(f"[Replay] Completed {sent} target frames")
        if position_errors:
            print(f"[Readback] position mean/max={np.mean(position_errors)*1000:.1f}/"
                  f"{np.max(position_errors)*1000:.1f}mm, rotation mean/max="
                  f"{np.rad2deg(np.mean(rotation_errors)):.2f}/"
                  f"{np.rad2deg(np.max(rotation_errors)):.2f}deg")
    except KeyboardInterrupt:
        print("[Replay] Interrupted by user")
    finally:
        print("[Replay] Robot remains enabled")


def _select_episode(episode_indices: list[int], frames: dict[int, object]) -> int:
    print("\n可用 episodes:")
    for i in range(0, len(episode_indices), 8):
        chunk = episode_indices[i : i + 8]
        parts = [f"{ep:>2}:{len(frames[ep]):>4}帧" for ep in chunk]
        print("    " + "  ".join(parts))

    first = episode_indices[0]
    last = episode_indices[-1]
    while True:
        raw = input(f"选择 episode [{first}-{last}]，回车默认 {first}: ").strip()
        if raw == "":
            return first
        try:
            ep = int(raw)
        except ValueError:
            print(f"  [WARNING] 无效输入: {raw}")
            continue
        if ep in episode_indices:
            return ep
        print(f"  [WARNING] episode {ep} 不存在")


def _build_gripper_and_robot(args, eef):
    gripper = None
    has_dm_gripper = getattr(eef, "_HAS_DM_GRIPPER", False)
    if args.control_gripper and has_dm_gripper and args.gripper_port:
        print(f"[Gripper] 使用达妙 DM-J4310-2EC 夹爪 ({args.gripper_port})")
        gripper = eef.DmGripperController(
            port=args.gripper_port,
            close_rad=args.gripper_close_rad,
            open_width_mm=args.gripper_open_width_mm,
            current_limit=args.gripper_current,
            speed_rad_s=args.gripper_speed,
        )
    elif args.control_gripper and has_dm_gripper:
        print("[Gripper] 未指定夹爪串口，回退原生夹爪")
    elif args.control_gripper:
        print("[Gripper] 未加载达妙夹爪驱动，回退原生夹爪")
    else:
        print("[Gripper] 不控制夹爪（默认；传 --control_gripper 才启用）")

    return eef.PiperEEFController(
        args.can_name,
        gripper=gripper,
        control_gripper=args.control_gripper,
        gripper_command_max_mm=args.gripper_open_width_mm,
    )


def _pose6_to_matrix(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _matrix_to_pose6(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]
    )


def _current_controller_to_tcp() -> np.ndarray:
    """Derive current ``^C T_E`` from pico_arm_transform.py calibration."""
    from examples.piper.transforms.pico_arm_transform import C_ARM_FROM_PICO
    from examples.piper.transforms.pico_arm_transform import PICO_FROM_TCP_ARM_REFERENCE_METERS
    from examples.piper.transforms.pico_arm_transform import Q_PICO_REFERENCE_XYZW
    from examples.piper.transforms.pico_arm_transform import R_ARM_REFERENCE
    from examples.piper.transforms.pico_arm_transform import quaternion_to_matrix

    alignment = (
        quaternion_to_matrix(Q_PICO_REFERENCE_XYZW).T
        @ C_ARM_FROM_PICO.T
        @ R_ARM_REFERENCE
    )
    mount = np.eye(4, dtype=np.float64)
    mount[:3, :3] = alignment
    mount[:3, 3] = (
        -alignment @ R_ARM_REFERENCE.T @ PICO_FROM_TCP_ARM_REFERENCE_METERS
    )
    return mount


def adapt_episode_states(
    states: np.ndarray, dataset_controller_to_tcp: np.ndarray, *, adapt: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return poses in current TCP coordinates and ``^E_dataset T_E_current``."""
    current_mount = _current_controller_to_tcp()
    dataset_mount = np.asarray(dataset_controller_to_tcp, dtype=np.float64)
    if dataset_mount.shape != (4, 4):
        raise ValueError("数据集 meta/info.json 缺少有效的 4x4 controller_to_tcp")
    frame_change = np.linalg.inv(dataset_mount) @ current_mount
    rotation_difference = Rotation.from_matrix(frame_change[:3, :3]).magnitude()
    translation_difference = np.linalg.norm(frame_change[:3, 3])
    mounts_match = translation_difference < 1e-6 and rotation_difference < 1e-6
    if not mounts_match and not adapt:
        raise ValueError(
            "数据集 TCP 与当前 TCP 不一致："
            f"原点差={translation_difference * 1000:.1f}mm，"
            f"旋转差={np.rad2deg(rotation_difference):.2f}deg；"
            "请传 --adapt_dataset_tcp 做完整 SE(3) 共轭转换"
        )

    converted = np.asarray(states, dtype=np.float64).copy()
    if adapt:
        frame_change_inverse = np.linalg.inv(frame_change)
        for index, pose in enumerate(states):
            dataset_pose = _pose6_to_matrix(pose[:6])
            current_pose = frame_change_inverse @ dataset_pose @ frame_change
            converted[index, :6] = _matrix_to_pose6(current_pose)
    return converted, frame_change


def _trajectory_steps(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.zeros(len(states), dtype=np.float64)
    rotation = np.zeros(len(states), dtype=np.float64)
    matrices = [_pose6_to_matrix(pose[:6]) for pose in states]
    for index in range(1, len(states)):
        step = np.linalg.inv(matrices[index - 1]) @ matrices[index]
        translation[index] = np.linalg.norm(step[:3, 3])
        rotation[index] = Rotation.from_matrix(step[:3, :3]).magnitude()
    return translation, rotation


def _validate_trajectory(states: np.ndarray, args) -> None:
    translation_steps, rotation_steps = _trajectory_steps(states)
    total_translation = np.linalg.norm(states[:, :3], axis=1)
    total_rotation = np.array(
        [Rotation.from_rotvec(pose[3:6]).magnitude() for pose in states]
    )
    violations = []
    if total_translation.max() > args.max_total_translation_m:
        violations.append(
            f"相对起点最大平移 {total_translation.max():.3f}m > {args.max_total_translation_m:.3f}m"
        )
    if translation_steps.max() > args.max_step_translation_m:
        violations.append(
            f"最大单步平移 {translation_steps.max():.3f}m > {args.max_step_translation_m:.3f}m"
        )
    if np.rad2deg(rotation_steps.max()) > args.max_step_rotation_deg:
        violations.append(
            f"最大单步旋转 {np.rad2deg(rotation_steps.max()):.2f}deg > {args.max_step_rotation_deg:.2f}deg"
        )
    if np.rad2deg(total_rotation.max()) > args.max_total_rotation_deg:
        violations.append(
            f"相对起点最大旋转 {np.rad2deg(total_rotation.max()):.2f}deg > {args.max_total_rotation_deg:.2f}deg"
        )
    if violations:
        raise ValueError("轨迹超过安全阈值：\n  - " + "\n  - ".join(violations))


def _print_episode_info(
    ep: int, states: np.ndarray, frame_change: np.ndarray, fps: float, speed: float
) -> None:
    print(f"\nEpisode {ep}: {len(states)} 帧, FPS={fps:.1f}, 回放速度={speed:.2f}x")
    for axis, idx in zip("xyz", range(3), strict=True):
        print(f"  {axis}: {states[:, idx].min():.3f} ~ {states[:, idx].max():.3f} m")
    print(f"  gripper: {states[:, 6].min():.1f} ~ {states[:, 6].max():.1f} mm")
    pos, rot = _trajectory_steps(states)
    print(f"  delta pos: mean={pos.mean() * 1000:.1f}mm max={pos.max() * 1000:.1f}mm")
    print(f"  delta rot: mean={np.rad2deg(rot.mean()):.2f}deg max={np.rad2deg(rot.max()):.2f}deg")
    print(
        "  dataset TCP -> current TCP: "
        f"origin={np.linalg.norm(frame_change[:3, 3]) * 1000:.1f}mm, "
        f"rotation={np.rad2deg(Rotation.from_matrix(frame_change[:3, :3]).magnitude()):.2f}deg"
    )


def _show_info(args, helpers) -> None:
    meta, frames = helpers["load_eef_dataset"](args.data_dir)
    fps = helpers["get_dataset_fps"](meta)
    print(f"数据集: {args.data_dir}")
    print(f"FPS:     {fps:.1f}")
    print(f"Episodes: {sorted(frames.keys())}")
    for ep in sorted(frames.keys()):
        episode_df = frames[ep]
        states = np.stack(episode_df["state"].values).astype(np.float64)
        print(f"\n  Episode {ep}: {len(episode_df)} 帧")
        for axis, idx in zip("xyz", range(3), strict=True):
            print(f"    {axis}: {states[:, idx].min():.3f} ~ {states[:, idx].max():.3f} m")
        print(f"    gripper: {states[:, 6].min():.1f} ~ {states[:, 6].max():.1f} mm")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="回放 Piper EEF 数据集（支持 pick_cube_chunk_relative）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help=f"数据集目录 (默认: {DEFAULT_DATA_DIR})")
    parser.add_argument("--episode", type=int, default=None, help="回放的 episode；缺省时交互选择")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率 (默认: 1.0)")
    parser.add_argument("--action_horizon", type=int, default=50, help="训练动作块长度")
    parser.add_argument("--exec_horizon", type=int, default=20, help="每次锚定后执行的动作数")
    parser.add_argument("--step", action="store_true", help="每一帧发送前等待 Enter，推荐首次验证时使用")
    parser.add_argument("--readback", action="store_true", help="每帧发送后读取实际 TCP 并统计跟踪误差")
    parser.add_argument(
        "--adapt_dataset_tcp",
        action="store_true",
        help="将数据集保存的旧虚拟 TCP 用完整 SE(3) 共轭变换到当前 TCP",
    )
    parser.add_argument("--can_name", default="can0", help="CAN 端口名称 (默认: can0)")
    parser.add_argument("--speed_pct", type=int, default=DEFAULT_SPEED_PCT, help="末端控制速度档百分比")
    parser.add_argument("--control_gripper", action="store_true", help="显式启用夹爪；默认只验证 TCP 位姿")
    parser.add_argument("--gripper_port", default=DEFAULT_GRIPPER_PORT, help="达妙夹爪 USB2CAN 串口")
    parser.add_argument("--gripper_close_rad", type=float, default=DEFAULT_GRIPPER_CLOSE_RAD)
    parser.add_argument("--gripper_open_width_mm", type=float, default=None,
                        help="机械臂全开宽度；chunk-relative 默认使用数据集元数据值")
    parser.add_argument("--gripper_current", type=float, default=DEFAULT_GRIPPER_CURRENT_LIMIT)
    parser.add_argument("--gripper_speed", type=float, default=DEFAULT_GRIPPER_SPEED_RAD_S)
    parser.add_argument("--dry_run", action="store_true", help="只加载数据并打印 delta，不连接机械臂")
    parser.add_argument("--info_only", action="store_true", help="仅显示数据集概要")
    parser.add_argument("--yes", action="store_true", help="跳过开始前的确认提示")
    parser.add_argument("--max_total_translation_m", type=float, default=0.5)
    parser.add_argument("--max_step_translation_m", type=float, default=0.05)
    parser.add_argument("--max_total_rotation_deg", type=float, default=120.0)
    parser.add_argument("--max_step_rotation_deg", type=float, default=20.0)
    parser.add_argument(
        "--workspace",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=None,
        help="可选 Robot Base 工作空间限制（米）",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed 必须大于 0")

    info_path = Path(args.data_dir) / "meta/info.json"
    if info_path.exists() and json.loads(info_path.read_text()).get("robot_type") == "piper_chunk_relative_eef":
        replay_chunk_relative(args)
        return
    if args.gripper_open_width_mm is None:
        args.gripper_open_width_mm = DEFAULT_GRIPPER_OPEN_WIDTH_MM

    helpers = _load_replay_helpers()
    if args.info_only:
        _show_info(args, helpers)
        return

    meta, frames = helpers["load_eef_dataset"](args.data_dir)
    fps = helpers["get_dataset_fps"](meta)
    episode_indices = sorted(frames.keys())
    if not episode_indices:
        raise SystemExit("数据集中没有 episode")

    episode = args.episode
    if episode is None:
        episode = _select_episode(episode_indices, frames)
    if episode not in frames:
        raise SystemExit(f"episode {episode} 不存在")

    episode_df = frames[episode]
    dataset_states = np.stack(episode_df["state"].values).astype(np.float64)
    dataset_mount = meta.get("info", {}).get("controller_to_tcp")
    if dataset_mount is None:
        raise SystemExit("数据集 meta/info.json 没有 controller_to_tcp，无法验证 TCP 坐标关系")
    try:
        states, frame_change = adapt_episode_states(
            dataset_states,
            np.asarray(dataset_mount, dtype=np.float64),
            adapt=args.adapt_dataset_tcp,
        )
        _validate_trajectory(states, args)
    except ValueError as exc:
        raise SystemExit(f"[SAFETY] {exc}") from exc
    task = helpers["get_episode_task"](meta, episode)
    print(f"\n任务指令: {task}")
    _print_episode_info(episode, states, frame_change, fps, args.speed)

    if args.dry_run:
        print("\n[DryRun] 已跳过机械臂连接和运动执行。")
        return

    from examples.piper.runtime import inference_eef as eef

    robot = _build_gripper_and_robot(args, eef)
    if not robot.enable(speed_pct=args.speed_pct):
        raise SystemExit("[ERROR] 机械臂使能失败")

    try:
        period = 1.0 / fps / args.speed
        robot_start = robot.get_eef_pose().astype(np.float64)
        robot_start_matrix = _pose6_to_matrix(robot_start)
        targets = []
        for state in states:
            target = np.empty(7, dtype=np.float64)
            target[:6] = _matrix_to_pose6(robot_start_matrix @ _pose6_to_matrix(state[:6]))
            target[6] = state[6]
            targets.append(target)
        targets = np.asarray(targets)

        if args.workspace is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = args.workspace
            inside = (
                (targets[:, 0] >= xmin)
                & (targets[:, 0] <= xmax)
                & (targets[:, 1] >= ymin)
                & (targets[:, 1] <= ymax)
                & (targets[:, 2] >= zmin)
                & (targets[:, 2] <= zmax)
            )
            if not inside.all():
                bad = int(np.flatnonzero(~inside)[0])
                raise RuntimeError(f"目标 frame {bad} 超出 --workspace: {targets[bad, :3]}")

        print(f"[Align] Robot start TCP: {robot_start.round(4)}")
        print(f"[Align] First target:    {targets[0, :6].round(4)}")
        print(f"[Align] Last target:     {targets[-1, :6].round(4)}")
        if not args.yes:
            input("确认起点和目标范围安全后，按 Enter 开始；Ctrl+C 取消...")

        position_errors = []
        rotation_errors = []
        for frame, target in enumerate(targets):
            if args.step:
                input(f"[Step] frame {frame}/{len(targets) - 1}，按 Enter 发送；Ctrl+C 取消...")
            robot.execute_action(target)
            time.sleep(period)
            if args.readback:
                actual = robot.get_eef_pose().astype(np.float64)
                error = np.linalg.inv(_pose6_to_matrix(target[:6])) @ _pose6_to_matrix(actual)
                position_errors.append(np.linalg.norm(error[:3, 3]))
                rotation_errors.append(Rotation.from_matrix(error[:3, :3]).magnitude())

            if frame % 25 == 0 or frame == len(targets) - 1 or args.step:
                pos = ", ".join(f"{value:.4f}" for value in target[:3])
                rot = ", ".join(f"{value:.3f}" for value in target[3:6])
                print(
                    f"\r[Ep{episode}] {frame + 1}/{len(targets)} | "
                    f"pos=[{pos}] rot=[{rot}] grip={target[6]:.1f}mm",
                    end="\n" if args.step else "",
                    flush=True,
                )

        print(f"\n[Replay] Episode {episode} 坐标转换回放完成")
        if position_errors:
            print(
                "[Readback] tracking error: "
                f"position mean/max={np.mean(position_errors) * 1000:.1f}/"
                f"{np.max(position_errors) * 1000:.1f}mm, "
                f"rotation mean/max={np.rad2deg(np.mean(rotation_errors)):.2f}/"
                f"{np.rad2deg(np.max(rotation_errors)):.2f}deg"
            )
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        print("[INFO] 已退出 (机械臂和夹爪保持使能)")


if __name__ == "__main__":
    main()
