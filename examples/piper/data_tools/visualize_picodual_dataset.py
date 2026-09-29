#!/usr/bin/env python3
# ruff: noqa: E402, RUF001, RUF002
"""可视化 PicoDual EEF 数据集，包括 ``pick_cube_chunk_relative``。

对 chunk-relative 数据，模型 state 只有夹爪值；这里从同帧 actions 重建
PICO World TCP 轨迹，并按训练时的 ``inv(T_actions[0]) @ T_actions[k]``
显示每个 50 步动作块。可用 ``--tcp_start_pose`` 查看映射到 Robot Base 后的目标。

    python examples/piper/data_tools/visualize_picodual_dataset.py \
        --data_dir ./local/datasets/pick_cube_chunk_relative --episode 0 --view training_chunk



同时显示基座图像、腕部图像、episode-relative TCP 三维轨迹、当前 TCP
坐标轴、位置/旋转曲线和夹爪宽度。交互窗口中可拖动三维轨迹旋转视角，
并用底部滑块切换帧；切换帧时保留当前视角。默认显示数据集原始 TCP 坐标；使用
``--tcp_frame current`` 可把旧数据用完整 SE(3) 共轭变换到当前
``pico_arm_transform.py`` 定义的 TCP 后再显示。若数据集位姿是未经转换的
PICO-relative 轨迹，使用 ``--model_action_frame pico`` 可按
``inference_action_transform.py`` 的标定转换到当前 TCP-relative 后再显示。

示例：
    python examples/piper/data_tools/visualize_picodual_dataset.py --data_dir ./local/datasets/pick_cube --episode 0

    # 同时查看 50 步当前 chunk、已拼接历史 chunk 和灰色完整轨迹
    python examples/piper/data_tools/visualize_picodual_dataset.py --data_dir ./local/datasets/pick_cube \
        --episode 0 --view training_chunk

    # 原始 PICO 数据：用机器人初始 TCP 位姿显示 Robot Base 下的训练轨迹
    python examples/piper/data_tools/visualize_picodual_dataset.py \
        --data_dir ./local/datasets/pick_cube_raw_action --episode 0 --view training_chunk \
        --tcp_start_pose 0.3404 0.0074 0.2504 -0.5464 1.7926 -0.024

    # 在连接 Piper 的机器上直接读取当前 TCP 位姿作为 Robot Base 起点
    python examples/piper/data_tools/visualize_picodual_dataset.py \
        --data_dir ./local/datasets/pick_cube_raw_action --episode 0 --view training_chunk \
        --tcp_start_from_piper --can_name can0

    # 仅导出第 2 个 chunk（起点为第 100 帧）
    python examples/piper/data_tools/visualize_picodual_dataset.py --data_dir ./local/datasets/pick_cube \
        --episode 0 --view training_chunk --start 100 --end 101 \
        --output chunk2.gif --no_show

    # 仅用于 state/actions 是 PICO-relative 的数据集
    python examples/piper/data_tools/visualize_picodual_dataset.py --data_dir ./local/datasets/pick_cube \
        --episode 0 --model_action_frame pico

    python examples/piper/data_tools/visualize_picodual_dataset.py --data_dir ./local/datasets/pick_cube \
        --episode 0 --tcp_frame current --output pick_cube_ep0.gif --no_show

    # 使用回放日志中的 Robot start TCP，在 Robot Base 坐标系中显示目标轨迹
    python examples/piper/data_tools/visualize_picodual_dataset.py --episode 0 \
        --tcp_frame current --tcp_start_pose 0.35 0.0 0.25 0.0 1.57 0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PIPER_DIR = HERE.parent
REPO_ROOT = HERE.parents[2]
for import_path in (HERE, PIPER_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from examples.piper.transforms.inference_action_transform import pico_relative_actions_to_tcp_relative


def read_piper_tcp_pose(can_name: str, timeout: float = 3.0) -> np.ndarray:
    """只读 Piper 反馈，返回 Robot Base 下的爪尖 TCP [xyz(m), rotvec(rad)]。"""
    sdk_path = REPO_ROOT / "piper_sdk"
    if str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise RuntimeError("无法导入 piper_sdk；请在 Piper 端安装仓库中的 piper_sdk") from exc

    piper = C_PiperInterface_V2(
        can_name=can_name, judge_flag=False, can_auto_init=True, dh_is_offset=1
    )
    try:
        # 仅接收 CAN 反馈; 不使能或移动机械臂。
        piper.ConnectPort(piper_init=False)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            feedback = piper.GetArmEndPoseMsgs()
            # Hz 只有在 XY、ZRX、RYRZ 三类位姿帧都有反馈时才非零。
            if feedback.time_stamp > 0 and feedback.Hz > 0:
                ep = feedback.end_pose
                position = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=np.float64) * 1e-6
                euler = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], dtype=np.float64)
                rotation = Rotation.from_euler("xyz", euler * np.pi / 180000.0)
                position += rotation.apply([0.0, 0.0, 0.22])
                pose = np.concatenate([position, rotation.as_rotvec()])
                if not np.all(np.isfinite(pose)):
                    raise RuntimeError("Piper TCP 反馈包含无效数值")
                return pose
            time.sleep(0.02)
        raise TimeoutError(f"{timeout:g} 秒内未收到完整的 Piper TCP 位姿反馈（CAN: {can_name}）")
    finally:
        piper.DisconnectPort()


def load_episode(data_dir: str, episode: int | None):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("需要 pandas 和 parquet 支持，请安装项目数据依赖") from exc

    root = Path(data_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"找不到数据集元数据: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    tasks = []
    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line]

    episode_rows_path = root / "meta/episodes.jsonl"
    if episode_rows_path.exists():
        episode_rows = [json.loads(line) for line in episode_rows_path.read_text().splitlines() if line]
        available = sorted(int(row["episode_index"]) for row in episode_rows)
    else:
        available = sorted(int(path.stem.split("_")[-1]) for path in (root / "data").rglob("episode_*.parquet"))
    if not available:
        raise ValueError("数据集中没有 episode")
    selected = available[0] if episode is None else episode
    if selected not in available:
        raise ValueError(f"episode {selected} 不存在，可用值: {available}")
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    path = root / template.format(episode_chunk=selected // int(info.get("chunks_size", 1000)), episode_index=selected)
    if not path.exists():
        raise FileNotFoundError(f"找不到 episode parquet: {path}")
    frame = pd.read_parquet(path).sort_values("frame_index").reset_index(drop=True)
    task = tasks[0].get("task", "") if tasks else ""
    return info, frame, selected, task


def decode_rgb(value) -> np.ndarray:
    """Decode a Hugging Face Image parquet cell to RGB uint8."""
    if isinstance(value, np.ndarray):
        image = value
        if image.ndim == 3 and image.shape[-1] == 3:
            return image.astype(np.uint8, copy=False)
    if hasattr(value, "convert"):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)

    payload = value
    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload is None and value.get("path"):
            payload = Path(value["path"]).read_bytes()
    if isinstance(payload, bytes | bytearray | memoryview):
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.zeros((224, 224, 3), dtype=np.uint8)


def adapt_model_action_frame(states: np.ndarray, model_action_frame: str) -> np.ndarray:
    """Convert model/dataset-relative poses to current TCP-relative poses when needed."""
    if model_action_frame == "tcp":
        return np.asarray(states, dtype=np.float64).copy()
    if model_action_frame == "pico":
        return pico_relative_actions_to_tcp_relative(states)
    raise ValueError(
        f"model_action_frame 必须为 'tcp' 或 'pico'，实际为 {model_action_frame!r}"
    )


def adapt_tcp_frame(states: np.ndarray, info: dict, tcp_frame: str):
    if tcp_frame == "dataset":
        return states.copy(), np.eye(4)
    mount = info.get("controller_to_tcp")
    if mount is None:
        raise ValueError("meta/info.json 没有 controller_to_tcp，不能转换到当前 TCP")
    from replay_picodual_lerobot import adapt_episode_states

    return adapt_episode_states(states, np.asarray(mount, dtype=np.float64), adapt=True)


def _pose6_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"TCP 位姿必须是 6 维 [x y z rx ry rz]，实际为 {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _matrix_to_pose6(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]
    )


def anchor_states_to_tcp_start(states: np.ndarray, tcp_start_pose: np.ndarray) -> np.ndarray:
    """Map episode-relative poses into Robot Base with ``T_start @ D(t)``."""
    start = _pose6_to_matrix(tcp_start_pose)
    anchored = np.asarray(states, dtype=np.float64).copy()
    for index, state in enumerate(states):
        anchored[index, :6] = _matrix_to_pose6(start @ _pose6_to_matrix(state[:6]))
    return anchored


def _equal_3d_bounds(points: np.ndarray) -> tuple[np.ndarray, float]:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float((high - low).max()) / 2.0, 0.01)
    return center, radius * 1.12


def training_chunk_actions(
    states: np.ndarray, actions: np.ndarray, index: int, horizon: int, *, chunk_relative: bool = False
) -> np.ndarray:
    """Return the unnormalized action target used by pi05_piper_eef for one sample.

    LeRobot pads timestamps past the episode end with its last row. The first
    target is either the current or next frame, according to dataset metadata.
    """
    if horizon < 1 or not 0 <= index < len(states) or len(states) != len(actions):
        raise ValueError("invalid training sample, action array, or action horizon")
    from openpi.transforms import DeltaEEFPoseActions
    from openpi.policies.piper_chunk_relative import ChunkRelativeEEFActions

    target_indices = np.minimum(index + np.arange(horizon), len(actions) - 1)
    sample = {
        "state": np.array([states[index, 6]]) if chunk_relative else states[index].copy(),
        "actions": actions[target_indices].copy(),
    }
    if chunk_relative:
        return ChunkRelativeEEFActions()(sample)["actions"]
    return DeltaEEFPoseActions(position_dims=3, rotation_dims=3)(sample)["actions"]


def display_chunk_poses(
    poses: np.ndarray, initial_pose: np.ndarray, tcp_start_pose: np.ndarray | None, pose_frame: str
) -> np.ndarray:
    """Map training poses to Robot Base TCP only for display, leaving labels intact."""
    if tcp_start_pose is None:
        return poses
    start = _pose6_to_matrix(tcp_start_pose)
    if pose_frame == "pico_teleop":
        # Raw data/action is an absolute PICO pose. First form PICO motion,
        # then conjugate it into the calibrated TCP frame before anchoring.
        relative_pico = np.linalg.inv(initial_pose) @ poses
        relative_pico_poses = np.stack([_matrix_to_pose6(pose) for pose in relative_pico])
        relative_tcp_poses = pico_relative_actions_to_tcp_relative(relative_pico_poses)
        relative_tcp = np.stack([_pose6_to_matrix(pose) for pose in relative_tcp_poses])
        return start @ relative_tcp
    if pose_frame == "pico_world_tcp":
        return start @ (np.linalg.inv(initial_pose) @ poses)
    if pose_frame == "episode_initial_virtual_tcp":
        return start @ poses
    raise ValueError(f"cannot anchor unsupported pose frame {pose_frame!r} to Robot Base TCP")


def stitched_frame_indices(length: int, horizon: int, *, same_frame: bool) -> np.ndarray:
    """Frame number of every stitched pose, including observed chunk seams."""
    indices = [0]
    target_offset = 0 if same_frame else 1
    for start in range(0, length, horizon):
        if same_frame and start:
            indices.append(start)
        indices.extend(start + target_offset + np.arange(horizon))
    return np.asarray(indices, dtype=np.int64)


def stitch_training_chunks(
    states: np.ndarray, actions: np.ndarray, horizon: int, *, same_frame: bool = False,
    chunk_relative: bool = False,
) -> np.ndarray:
    """Place non-overlapping training chunks in the episode frame.

    Same-frame labels stop at state[start+horizon-1]. Their next chunk starts
    at the observed state[start+horizon], so include that boundary step.
    """
    if len(states) == 0:
        raise ValueError("cannot stitch an empty episode")
    anchor = _pose6_to_matrix(states[0, :6])
    stitched = [anchor.copy()]
    for start in range(0, len(states), horizon):
        if same_frame and start:
            anchor = _pose6_to_matrix(states[start, :6])
            stitched.append(anchor.copy())
        deltas = training_chunk_actions(
            states, actions, start, horizon, chunk_relative=chunk_relative
        )
        chunk_poses = [anchor @ _pose6_to_matrix(delta[:6]) for delta in deltas]
        stitched.extend(chunk_poses)
        anchor = chunk_poses[-1]
    return np.stack(stitched)


class DatasetViewer:
    def __init__(
        self,
        frame,
        states: np.ndarray,
        *,
        episode: int,
        task: str,
        fps: float,
        coordinate_label: str,
        anchored_to_robot_base: bool,
        view: str = "episode",
        training_states: np.ndarray | None = None,
        training_actions: np.ndarray | None = None,
        sample_indices: np.ndarray | None = None,
        action_horizon: int = 50,
        pose_frame: str = "episode_initial_virtual_tcp",
        tcp_start_pose: np.ndarray | None = None,
        chunk_relative: bool = False,
    ):
        self.frame = frame
        self.states = states
        self.episode = episode
        self.task = task
        self.fps = fps
        self.coordinate_label = coordinate_label
        self.anchored_to_robot_base = anchored_to_robot_base
        self.view = view
        self.training_states = training_states
        self.training_actions = training_actions
        self.sample_indices = sample_indices
        self.action_horizon = action_horizon
        self.pose_frame = pose_frame
        self.chunk_relative = chunk_relative
        self.same_frame_actions = pose_frame in {"pico_teleop", "pico_world_tcp"}
        self.chunk_stride = action_horizon + int(self.same_frame_actions)
        self.stitched_frame_indices = None
        self.tcp_start_pose = tcp_start_pose
        self.display_full_poses = None
        self.display_stitched_poses = None
        self.training_pose_matrices = None
        self.stitched_gripper = None
        self.stitched_poses = None
        self.stitch_error_mm = None
        if view == "training_chunk":
            if training_states is None or training_actions is None:
                raise ValueError("training_chunk view requires full episode state/actions")
            self.training_pose_matrices = np.stack(
                [_pose6_to_matrix(state[:6]) for state in training_states]
            )
            gripper = [training_states[0, 6]]
            for start in range(0, len(training_states), action_horizon):
                if self.same_frame_actions and start:
                    gripper.append(training_states[start, 6])
                target_indices = np.minimum(start + np.arange(action_horizon), len(training_actions) - 1)
                gripper.extend(training_actions[target_indices, 6])
            self.stitched_gripper = np.asarray(gripper)
            self.stitched_poses = stitch_training_chunks(
                training_states, training_actions, action_horizon,
                same_frame=self.same_frame_actions, chunk_relative=self.chunk_relative
            )
            self.stitched_frame_indices = stitched_frame_indices(
                len(training_states), action_horizon, same_frame=self.same_frame_actions
            )
            initial_pose = self.training_pose_matrices[0]
            self.display_full_poses = display_chunk_poses(
                self.training_pose_matrices, initial_pose, tcp_start_pose, pose_frame
            )
            self.display_stitched_poses = display_chunk_poses(
                self.stitched_poses, initial_pose, tcp_start_pose, pose_frame
            )
            valid = self.stitched_frame_indices < len(training_states)
            point_errors = (
                self.display_stitched_poses[valid, :3, 3]
                - self.display_full_poses[self.stitched_frame_indices[valid], :3, 3]
            )
            self.stitch_error_mm = float(np.max(np.linalg.norm(point_errors, axis=1)) * 1000.0)
        self.image_cache: dict[tuple[int, str], np.ndarray] = {}

        self.xyz_center, self.xyz_radius = _equal_3d_bounds(states[:, :3])
        span = np.ptp(states[:, :3], axis=0).max()
        self.axis_length = max(0.015, float(span) * 0.12)
        initial_rotation = Rotation.from_rotvec(states[0, 3:6])
        self.rotation_angle = np.array([
            (initial_rotation.inv() * Rotation.from_rotvec(pose[3:6])).magnitude()
            for pose in states
        ])

    def _image(self, index: int, key: str) -> np.ndarray:
        cache_key = (index, key)
        if cache_key not in self.image_cache:
            if key not in self.frame.columns:
                self.image_cache[cache_key] = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                self.image_cache[cache_key] = decode_rgb(self.frame.iloc[index][key])
        return self.image_cache[cache_key]

    @staticmethod
    def make_figure():
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(16, 8.5), dpi=110)
        grid = figure.add_gridspec(2, 3, width_ratios=(1.1, 1.1, 1.35), hspace=0.30, wspace=0.22)
        rotation_axis = figure.add_subplot(grid[1, 1])
        axes = {
            "base": figure.add_subplot(grid[0, 0]),
            "wrist": figure.add_subplot(grid[0, 1]),
            "trajectory": figure.add_subplot(grid[:, 2], projection="3d"),
            "position": figure.add_subplot(grid[1, 0]),
            "rotation": rotation_axis,
            "gripper": rotation_axis.twinx(),
        }
        return figure, axes

    def draw(self, axes: dict, index: int, *, preserve_view: bool = False) -> None:
        index = int(np.clip(index, 0, len(self.states) - 1))
        trajectory = axes["trajectory"]
        view = (trajectory.elev, trajectory.azim, trajectory.roll) if preserve_view else None
        for axis in axes.values():
            axis.clear()

        axes["base"].imshow(self._image(index, "image"))
        axes["base"].set_title("Base / global camera")
        axes["base"].axis("off")
        axes["wrist"].imshow(self._image(index, "wrist_image"))
        axes["wrist"].set_title("Wrist / left camera")
        axes["wrist"].axis("off")

        if self.view == "training_chunk":
            self._draw_training_chunk(axes, index)
            if view is not None:
                trajectory.view_init(elev=view[0], azim=view[1], roll=view[2])
            return

        states = self.states
        trajectory.plot(
            states[:, 0], states[:, 1], states[:, 2], color="0.78", linewidth=1.0, label="full trajectory"
        )
        trajectory.plot(
            states[: index + 1, 0],
            states[: index + 1, 1],
            states[: index + 1, 2],
            color="#7b2cbf",
            linewidth=2.2,
            label="replayed",
        )
        trajectory.scatter(*states[0, :3], color="black", marker="o", s=35, label="episode start")
        trajectory.scatter(*states[index, :3], color="#ff7f0e", marker="o", s=55, label="current TCP")
        rotation = Rotation.from_rotvec(states[index, 3:6]).as_matrix()
        for axis_index, color, label in zip(range(3), ("r", "g", "b"), ("TCP x", "TCP y", "TCP z"), strict=True):
            endpoint = states[index, :3] + rotation[:, axis_index] * self.axis_length
            trajectory.plot(
                [states[index, 0], endpoint[0]],
                [states[index, 1], endpoint[1]],
                [states[index, 2], endpoint[2]],
                color=color,
                linewidth=2.4,
                label=label,
            )
        center, radius = self.xyz_center, self.xyz_radius
        trajectory.set_xlim(center[0] - radius, center[0] + radius)
        trajectory.set_ylim(center[1] - radius, center[1] + radius)
        trajectory.set_zlim(center[2] - radius, center[2] + radius)
        if view is not None:
            trajectory.view_init(elev=view[0], azim=view[1], roll=view[2])
        trajectory.set_xlabel("x (m)")
        trajectory.set_ylabel("y (m)")
        trajectory.set_zlabel("z (m)")
        trajectory.set_title(self.coordinate_label)
        trajectory.legend(loc="upper left", fontsize=7)

        time_axis = np.arange(len(states)) / self.fps
        position = axes["position"]
        for component, color in zip(range(3), ("r", "g", "b"), strict=True):
            position.plot(time_axis, states[:, component] * 1000.0, color=color, label="xyz"[component])
        position.axvline(time_axis[index], color="black", linestyle="--", linewidth=1.0)
        position.set_xlabel("time (s)")
        position.set_ylabel("position (mm)")
        position.set_title(
            "Robot Base TCP position" if self.anchored_to_robot_base else "Episode-relative translation"
        )
        position.grid(alpha=0.25)
        position.legend(fontsize=8, ncol=3)

        rotation_axis = axes["rotation"]
        rotation_axis.plot(time_axis, np.rad2deg(self.rotation_angle), color="#7b2cbf", label="rotation angle")
        rotation_axis.axvline(time_axis[index], color="black", linestyle="--", linewidth=1.0)
        rotation_axis.set_xlabel("time (s)")
        rotation_axis.set_ylabel("rotation angle (deg)", color="#7b2cbf")
        rotation_axis.tick_params(axis="y", labelcolor="#7b2cbf")
        rotation_axis.grid(alpha=0.25)
        gripper_axis = axes["gripper"]
        gripper_axis.plot(time_axis, states[:, 6], color="#ff7f0e", alpha=0.85, label="gripper")
        gripper_axis.set_ylabel("gripper (mm)", color="#ff7f0e")
        gripper_axis.tick_params(axis="y", labelcolor="#ff7f0e")
        rotation_axis.set_title("Rotation from episode start and gripper")

        pose = states[index]
        title = (
            f"Episode {self.episode} | frame {index}/{len(states) - 1} | "
            f"t={time_axis[index]:.2f}s | xyz={np.round(pose[:3], 4)} m | "
            f"rotvec={np.round(pose[3:6], 3)} rad | gripper={pose[6]:.1f} mm"
        )
        axes["base"].figure.suptitle(title, fontsize=11)

    def _draw_training_chunk(self, axes: dict, index: int) -> None:
        """Overlay a 50-action chunk and stitched history on the full episode path."""
        if (
            self.training_states is None or self.training_actions is None
            or self.sample_indices is None or self.stitched_poses is None
            or self.training_pose_matrices is None or self.stitched_gripper is None
            or self.display_full_poses is None or self.display_stitched_poses is None
            or self.stitched_frame_indices is None
        ):
            raise ValueError("training_chunk view requires full episode state/actions and sample indices")
        sample_index = int(self.sample_indices[index])
        if sample_index % self.action_horizon:
            raise ValueError("training_chunk samples must start on action-horizon boundaries")
        chunk_number = sample_index // self.action_horizon
        actions = training_chunk_actions(
            self.training_states, self.training_actions, sample_index, self.action_horizon,
            chunk_relative=self.chunk_relative
        )
        trajectory = axes["trajectory"]
        full_points = self.display_full_poses[:, :3, 3]
        stitched_points = self.display_stitched_poses[:, :3, 3]
        segment_start = chunk_number * self.chunk_stride
        history = stitched_points[: segment_start + 1]
        current = stitched_points[segment_start : segment_start + self.action_horizon + 1]
        trajectory.plot(*full_points.T, color="0.4", alpha=0.3, linewidth=2,
                        label="full episode (reference)")
        if len(history) > 1:
            trajectory.plot(*history.T, color="#7b2cbf", linewidth=2.2,
                            label="stitched history")
        trajectory.plot(*current.T, color="#ff7f0e", linewidth=2.8,
                        label="current 50-action chunk")
        trajectory.scatter(*current[0], color="black", s=55, label="current state")
        trajectory.scatter(*current[-1], color="#ff7f0e", s=45, label="chunk end")
        for step in range(10, self.action_horizon + 1, 10):
            trajectory.text(*current[step], str(step), fontsize=8)

        center, radius = _equal_3d_bounds(np.vstack((full_points, stitched_points)))
        axis_length = min(max(radius * 0.10, 0.005), 0.025)
        for pose, line_width in ((self.display_stitched_poses[segment_start], 2.2),
                                 (self.display_stitched_poses[segment_start + self.action_horizon], 1.3)):
            origin = pose[:3, 3]
            for axis_index, color in enumerate(("r", "g", "b")):
                endpoint = origin + pose[:3, axis_index] * axis_length
                trajectory.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]],
                                [origin[2], endpoint[2]], color=color, linewidth=line_width)
        trajectory.set_xlim(center[0] - radius, center[0] + radius)
        trajectory.set_ylim(center[1] - radius, center[1] + radius)
        trajectory.set_zlim(center[2] - radius, center[2] + radius)
        display_frame = (
            "Robot Base TCP" if self.tcp_start_pose is not None
            else "PICO tracking" if self.pose_frame == "pico_teleop"
            else "PICO World TCP" if self.pose_frame == "pico_world_tcp" else "episode TCP"
        )
        trajectory.set_xlabel(f"{display_frame} x (m)")
        trajectory.set_ylabel(f"{display_frame} y (m)")
        trajectory.set_zlabel(f"{display_frame} z (m)")
        first_target = sample_index + int(not self.same_frame_actions)
        trajectory.set_title(
            f"Chunk {chunk_number} | target frames {first_target}–{first_target + self.action_horizon - 1}"
        )
        trajectory.legend(loc="upper left", fontsize=7)

        # XYZ spans the episode in the same display frame as the 3D plot.
        # Rotation and gripper stay on the selected chunk's 0..horizon axis.
        steps = np.arange(self.action_horizon + 1)
        target_offset = int(not self.same_frame_actions)
        reference_indices = np.minimum(
            sample_index + target_offset + np.arange(self.action_horizon),
            len(self.training_states) - 1,
        )
        state_inverse = np.linalg.inv(self.training_pose_matrices[sample_index])
        reference_poses = np.concatenate(
            (self.training_pose_matrices[sample_index : sample_index + 1],
             self.training_pose_matrices[reference_indices]), axis=0
        )
        reference_relative = state_inverse @ reference_poses
        reference_rotvec = Rotation.from_matrix(reference_relative[:, :3, :3]).as_rotvec()
        current_rotvec = np.vstack((np.zeros(3), actions[:, 3:6]))

        history_rotvec = []
        history_gripper = []
        for past_chunk in range(chunk_number):
            offset = past_chunk * self.chunk_stride
            past_poses = self.stitched_poses[offset : offset + self.action_horizon + 1]
            past_relative = np.linalg.inv(past_poses[0]) @ past_poses
            history_rotvec.append(Rotation.from_matrix(past_relative[:, :3, :3]).as_rotvec())
            history_gripper.append(self.stitched_gripper[offset : offset + self.action_horizon + 1])

        def plot_components(axis, reference, history, current_values, *, scale, labels, ylabel, title):
            from matplotlib.lines import Line2D

            styles = ("-", "--", ":")
            for component, style in enumerate(styles):
                axis.plot(steps, reference[:, component] * scale, color="0.4",
                          linestyle=style, alpha=0.38, linewidth=3)
                for past in history:
                    axis.plot(steps, past[:, component] * scale, color="#7b2cbf",
                              linestyle=style, alpha=0.24, linewidth=1.2)
                axis.plot(steps, current_values[:, component] * scale, color="#ff7f0e",
                          linestyle=style, linewidth=1.9)
            axis.set_xlim(0, self.action_horizon)
            axis.set_xticks(np.arange(0, self.action_horizon + 1, 10))
            axis.set_xlabel("action step within chunk")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(alpha=0.25)
            handles = [
                Line2D([0], [0], color=color, alpha=alpha, linewidth=2, label=label)
                for color, alpha, label in (
                    ("0.4", 0.4, "reference"), ("#7b2cbf", 0.6, "past chunks"),
                    ("#ff7f0e", 1.0, "current"),
                )
            ]
            handles += [
                Line2D([0], [0], color="black", linestyle=style, label=label)
                for style, label in zip(styles, labels, strict=True)
            ]
            axis.legend(handles=handles, fontsize=7, ncol=3, loc="upper left")

        from matplotlib.lines import Line2D

        position = axes["position"]
        episode_steps = np.arange(len(self.training_states))
        history_steps = self.stitched_frame_indices[: segment_start + 1]
        current_steps = self.stitched_frame_indices[segment_start : segment_start + self.action_horizon + 1]
        valid_current = current_steps < len(self.training_states)
        styles = ("-", "--", ":")
        for component, style in enumerate(styles):
            position.plot(episode_steps, full_points[:, component] * 1000.0,
                          color="0.4", linestyle=style, alpha=0.38, linewidth=2.8)
            position.plot(history_steps, history[:, component] * 1000.0,
                          color="#7b2cbf", linestyle=style, linewidth=1.8)
            position.plot(current_steps[valid_current], current[valid_current, component] * 1000.0,
                          color="#ff7f0e", linestyle=style, linewidth=2.1)
        position.axvspan(sample_index, min(sample_index + self.action_horizon, len(self.training_states) - 1),
                         color="#ff7f0e", alpha=0.06)
        position.set_xlim(0, len(self.training_states) - 1)
        position.set_xlabel("episode frame")
        position.set_ylabel(f"{display_frame} position (mm)")
        position.set_title("XYZ: full episode / stitched history / current chunk")
        position.grid(alpha=0.25)
        handles = [
            Line2D([0], [0], color=color, alpha=alpha, linewidth=2, label=label)
            for color, alpha, label in (
                ("0.4", 0.4, "full"), ("#7b2cbf", 1.0, "history"),
                ("#ff7f0e", 1.0, "current"),
            )
        ]
        handles += [
            Line2D([0], [0], color="black", linestyle=style, label=label)
            for style, label in zip(styles, ("x", "y", "z"), strict=True)
        ]
        position.legend(handles=handles, fontsize=7, ncol=3, loc="upper left")

        action_frame = "PICO" if self.pose_frame == "pico_teleop" else "TCP"
        plot_components(
            axes["rotation"], reference_rotvec, history_rotvec, current_rotvec,
            scale=180.0 / np.pi, labels=("rx", "ry", "rz"),
            ylabel=f"rotvec in chunk state {action_frame} (deg)",
            title="Rotation: current 50-action chunk",
        )
        gripper = axes["gripper"]
        gripper.plot(steps, np.r_[self.training_states[sample_index, 6],
                                  self.training_states[reference_indices, 6]],
                     color="0.4", alpha=0.38, linewidth=3)
        for past in history_gripper:
            gripper.plot(steps, past, color="#7b2cbf", alpha=0.24, linewidth=1.2)
        gripper.plot(steps, np.r_[self.training_states[sample_index, 6], actions[:, 6]],
                     color="#ff7f0e", linewidth=1.9)
        gripper.yaxis.set_label_position("right")
        gripper.yaxis.tick_right()
        gripper.set_ylabel("gripper (mm)", color="#ff7f0e")
        gripper.tick_params(axis="y", labelcolor="#ff7f0e")

        axes["base"].figure.suptitle(
            f"Episode {self.episode} | chunk {chunk_number} | state frame {sample_index} | "
            f"stitched vs full max error: {self.stitch_error_mm:.3g} mm",
            fontsize=11,
        )

    def show(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        figure, axes = self.make_figure()
        figure.subplots_adjust(bottom=0.10, top=0.91)
        figure.text(0.99, 0.025, "Drag 3D trajectory to rotate", ha="right", fontsize=8)
        slider_axis = figure.add_axes((0.20, 0.025, 0.53, 0.028))
        slider = Slider(slider_axis, "chunk" if self.view == "training_chunk" else "frame",
                        0, len(self.states) - 1, valinit=0, valstep=1)

        def update(value):
            self.draw(axes, int(value), preserve_view=True)
            figure.canvas.draw_idle()

        slider.on_changed(update)
        update(0)
        plt.show()

    def save(self, output: str, output_fps: float) -> None:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = self.make_figure()
        figure.subplots_adjust(top=0.91)

        def update(index):
            self.draw(axes, index)
            return []

        movie = animation.FuncAnimation(
            figure, update, frames=len(self.states), interval=1000.0 / output_fps, blit=False
        )
        print(f"[Save] {len(self.states)} 帧 → {output_path}")
        if output_path.suffix.lower() == ".gif":
            movie.save(output_path, writer=animation.PillowWriter(fps=output_fps))
        elif output_path.suffix.lower() == ".mp4":
            movie.save(output_path, writer=animation.FFMpegWriter(fps=output_fps, bitrate=2500))
        else:
            raise ValueError("--output 仅支持 .gif 或 .mp4")
        plt.close(figure)
        print(f"[Save] 完成: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="可视化 PicoDual → LeRobot EEF 数据集")
    parser.add_argument("--data_dir", default="./local/datasets/pick_cube", help="LeRobot 数据集目录")
    parser.add_argument("--episode", type=int, default=None, help="episode 编号，默认第一个")
    parser.add_argument("--view", choices=("episode", "training_chunk"), default="episode",
                        help="episode 全轨迹，或叠加当前 chunk、拼接历史与完整轨迹")
    parser.add_argument("--action_horizon", type=int, default=50,
                        help="training_chunk 模式的动作长度，pi05_piper_eef 默认为 50")
    parser.add_argument(
        "--model_action_frame",
        choices=("tcp", "pico"),
        default="tcp",
        help=(
            "数据集 state/actions 的相对坐标系（默认: tcp）。"
            "使用 data/tcp_relative_action 转换的数据保持 tcp；"
            "未经 TCP 转换的 PICO-relative 数据选择 pico"
        ),
    )
    parser.add_argument(
        "--tcp_frame",
        choices=("dataset", "current"),
        default="dataset",
        help="显示数据集原 TCP，或转换到当前 pico_arm_transform.py TCP",
    )
    tcp_start_group = parser.add_mutually_exclusive_group()
    tcp_start_group.add_argument(
        "--tcp_start_pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        default=None,
        help=(
            "可选 Robot Base 下的 episode 初始 TCP [x y z(m) rx ry rz(rotvec rad)]；"
            "training_chunk 对原始 PICO 数据先转 TCP 再映射到 Robot Base"
        ),
    )
    tcp_start_group.add_argument(
        "--tcp_start_from_piper",
        action="store_true",
        help="从 --can_name 指定的 Piper 读取当前爪尖 TCP，作为 Robot Base 起点",
    )
    parser.add_argument("--can_name", default="can0", help="Piper CAN 端口名称，默认 can0")
    parser.add_argument("--start", type=int, default=0, help="起始帧（包含）")
    parser.add_argument("--end", type=int, default=None, help="结束帧（不包含）")
    parser.add_argument("--stride", type=int, default=1, help="帧采样步长；training_chunk 模式下为 chunk 步长")
    parser.add_argument("--output", default=None, help="可选导出路径：.gif 或 .mp4")
    parser.add_argument("--output_fps", type=float, default=None, help="导出帧率，默认数据 FPS / stride")
    parser.add_argument("--no_show", action="store_true", help="不打开交互窗口，适合仅导出")
    args = parser.parse_args()
    if args.action_horizon < 1:
        parser.error("--action_horizon 必须 >= 1")
    if args.view == "training_chunk" and (
        args.model_action_frame != "tcp" or args.tcp_frame != "dataset"
    ):
        parser.error("training_chunk 的训练曲线使用数据原坐标，不能叠加 --model_action_frame 或 --tcp_frame")
    if args.stride < 1:
        parser.error("--stride 必须 >= 1")
    if args.output_fps is not None and args.output_fps <= 0:
        parser.error("--output_fps 必须 > 0")
    if args.no_show and not args.output:
        parser.error("--no_show 必须与 --output 一起使用")
    if args.model_action_frame == "pico" and args.tcp_frame == "current":
        parser.error(
            "--model_action_frame pico 已转换到当前 TCP，不能再与 --tcp_frame current 叠加"
        )
    return args


def main() -> None:
    args = parse_args()
    info, frame, episode, task = load_episode(args.data_dir, args.episode)
    dataset_states = np.stack(frame["state"].values).astype(np.float64)
    chunk_relative = info.get("robot_type") == "piper_chunk_relative_eef"
    if chunk_relative:
        if args.model_action_frame != "tcp":
            raise SystemExit("chunk-relative 数据集已使用虚拟 TCP 位姿，不能选 --model_action_frame pico")
        absolute_actions = np.stack(frame["actions"].values).astype(np.float64)
        if dataset_states.shape != (len(frame), 1) or absolute_actions.shape != (len(frame), 7):
            raise SystemExit("chunk-relative 数据集要求 state (N,1)、actions (N,7)")
        if not np.allclose(dataset_states[:, 0], absolute_actions[:, 6], atol=1e-4):
            raise SystemExit("数据集同帧 state 和 action 的夹爪开度不一致")
        # Pose is stored only in actions. Reconstruct a display state without
        # changing the model's actual one-dimensional observation.
        training_states = absolute_actions.copy()
        training_actions = absolute_actions if args.view == "training_chunk" else None
    else:
        training_states = dataset_states
        training_actions = np.stack(frame["actions"].values).astype(np.float64) if args.view == "training_chunk" else None
    states = training_states.copy()
    if states.ndim != 2 or states.shape[1] != 7 or not np.isfinite(states).all():
        raise SystemExit(f"期望完整 7 维可视化位姿，收到 {states.shape}")
    states = adapt_model_action_frame(states, args.model_action_frame)
    states, frame_change = adapt_tcp_frame(states, info, args.tcp_frame)
    if args.tcp_start_from_piper:
        try:
            tcp_start_pose = read_piper_tcp_pose(args.can_name)
        except (RuntimeError, TimeoutError, OSError) as exc:
            raise SystemExit(f"读取 Piper TCP 失败: {exc}") from exc
    else:
        tcp_start_pose = (
            np.asarray(args.tcp_start_pose, dtype=np.float64)
            if args.tcp_start_pose is not None else None
        )
    anchored_to_robot_base = tcp_start_pose is not None
    if anchored_to_robot_base and args.view != "training_chunk":
        if info.get("pose_coordinate_frame") == "pico_world_tcp":
            initial = _pose6_to_matrix(states[0, :6])
            relative = states.copy()
            for row in relative:
                row[:6] = _matrix_to_pose6(np.linalg.inv(initial) @ _pose6_to_matrix(row[:6]))
            states = anchor_states_to_tcp_start(relative, tcp_start_pose)
        else:
            states = anchor_states_to_tcp_start(states, tcp_start_pose)

    stop = len(frame) if args.end is None else min(args.end, len(frame))
    if args.start < 0 or args.start >= stop:
        raise SystemExit(f"无效帧范围: start={args.start}, end={args.end}, episode 帧数={len(frame)}")
    if args.view == "training_chunk":
        # Keep chunk starts aligned to the episode, even when --start/--end filter the display.
        indices = np.arange(0, len(frame), args.action_horizon)
        indices = indices[(indices >= args.start) & (indices < stop)][:: args.stride]
        if len(indices) == 0:
            raise SystemExit("所选帧范围内没有 chunk 起点")
        fps = float(info.get("fps", 20.0)) / (args.action_horizon * args.stride)
    else:
        indices = np.arange(args.start, stop, args.stride)
        fps = float(info.get("fps", 20.0)) / args.stride
    frame = frame.iloc[indices].reset_index(drop=True)
    states = states[indices]

    if args.view == "training_chunk":
        source_frame = info.get("pose_coordinate_frame", "unknown")
        display_frame = (
            "Robot Base TCP" if anchored_to_robot_base
            else "PICO tracking" if source_frame == "pico_teleop"
            else "PICO World TCP" if source_frame == "pico_world_tcp" else "episode TCP"
        )
        print(f"数据集: {args.data_dir}")
        print(f"Episode: {episode}, 显示 {len(states)} 个 chunk, 每个 {args.action_horizon} 个 action")
        print(f"训练数据位姿坐标系: {source_frame}; 3D 显示坐标系: {display_frame}")
    else:
        display_tcp_frame = (
            "current (converted from PICO-relative)"
            if args.model_action_frame == "pico"
            else args.tcp_frame
        )
        print(f"数据集: {args.data_dir}")
        print(f"Episode: {episode}, 显示 {len(states)} 帧, FPS={fps:.2f}, TCP frame={display_tcp_frame}")
        print(f"模型/数据位姿坐标系: {args.model_action_frame.upper()}-relative")
    print(f"任务: {task or '(无)'}")
    if chunk_relative:
        print("模型 state: 仅夹爪宽度；可视化 TCP 位姿来自同帧 actions")
    if args.model_action_frame == "pico":
        print("PICO-relative -> current TCP-relative: 已应用 inference_action_transform")
    if args.tcp_frame == "current":
        angle = Rotation.from_matrix(frame_change[:3, :3]).magnitude()
        print(
            "dataset TCP -> current TCP: "
            f"origin={np.linalg.norm(frame_change[:3, 3]) * 1000:.1f}mm, "
            f"rotation={np.rad2deg(angle):.2f}deg"
        )
    if anchored_to_robot_base:
        print(f"Robot Base TCP start [xyz(m), rotvec(rad)]: {tcp_start_pose}")
        if args.tcp_start_from_piper:
            print("--tcp_start_pose " + " ".join(f"{value:.8f}" for value in tcp_start_pose))
        if args.view == "training_chunk" and info.get("pose_coordinate_frame") == "pico_teleop":
            print("3D 显示: 原始 PICO 位姿 → 首帧相对 PICO → 标定 TCP → Robot Base")
        elif info.get("pose_coordinate_frame") == "pico_world_tcp":
            print("3D 显示: 原始 TCP 位姿 → 首帧相对 TCP → Robot Base")
        else:
            print("显示关系: ^B T_target(t) = ^B T_start @ D(t)")

    viewer = DatasetViewer(
        frame,
        states,
        episode=episode,
        task=task,
        fps=fps,
        coordinate_label=(
            "Robot Base TCP target"
            if anchored_to_robot_base
            else (
                "Episode-relative TCP (converted from PICO-relative)"
                if args.model_action_frame == "pico"
                else "PICO World TCP (stored actions)" if chunk_relative
                else f"Episode-relative TCP ({args.tcp_frame} frame)"
            )
        ),
        anchored_to_robot_base=anchored_to_robot_base,
        view=args.view,
        training_states=training_states if args.view == "training_chunk" else None,
        training_actions=training_actions,
        sample_indices=indices if args.view == "training_chunk" else None,
        action_horizon=args.action_horizon,
        pose_frame=info.get("pose_coordinate_frame", "episode_initial_virtual_tcp"),
        tcp_start_pose=tcp_start_pose if args.view == "training_chunk" else None,
        chunk_relative=chunk_relative,
    )
    if args.output:
        viewer.save(args.output, args.output_fps or fps)
    if not args.no_show:
        viewer.show()


if __name__ == "__main__":
    main()
