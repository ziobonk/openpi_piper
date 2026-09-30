"""Offline validation of UDP conversion and chunk-relative robot targets."""

import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "piper"))

from prepare_chunk_relative_eef import convert
from replay_chunk_relative_eef import DryRunRobot
from replay_chunk_relative_eef import action_chunks
from replay_chunk_relative_eef import load_episode
from replay_chunk_relative_eef import parse_args
from replay_chunk_relative_eef import relative_pose
from replay_chunk_relative_eef import replay


def _dataset(root: Path, states: np.ndarray, actions: np.ndarray, *, kind: str) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "robot_type": "piper_eef" if kind == "udp" else "pico",
        "pose_coordinate_frame": "piper_base_tcp" if kind == "udp" else "pico_world_tcp",
        "coordinate_transform_applied": False,
        "features": {"state": {"shape": [7]}, "actions": {"shape": [7]}},
        "fps": 20,
        "total_frames": len(states),
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    stats = {
        name: {"min": data.min(axis=0).tolist(), "max": data.max(axis=0).tolist()}
        for name, data in (("state", states), ("actions", actions))
    }
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    (root / "meta" / "episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": stats}) + "\n")
    table = pa.table(
        {
            "state": pa.array(states.tolist()),
            "actions": pa.array(actions.tolist()),
            "episode_index": pa.array([0] * len(states)),
            "frame_index": pa.array(list(range(len(states)))),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")


def test_udp_dataset_conversion_and_replay_rebases_each_chunk(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    states = np.array([[0, 0, 0, 0, 0, 0, 8]] * 5, dtype=np.float32)
    actions = np.array([[0.01 * i, 0, 0, 0, 0, 0, 9] for i in range(5)], dtype=np.float32)
    _dataset(source, states, actions, kind="udp")
    convert(source, output, source_kind="udp")
    stored = pq.read_table(next((output / "data").rglob("*.parquet")))
    np.testing.assert_allclose(stored["state"].to_pylist(), [[8]] * 5)
    np.testing.assert_allclose(stored["actions"].to_pylist(), actions)
    loaded, fps = load_episode(output, 0)
    args = parse_args(["--dataset", str(output), "--horizon", "3"])
    robot = DryRunRobot(np.array([0.4, 0, 0.3, 0, 0, 0]))
    targets = replay(loaded, fps, robot, args)
    assert len(targets) == 4
    np.testing.assert_allclose(np.stack(targets)[:, 0], [0.41, 0.42, 0.43, 0.44], atol=1e-7)
    assert [start for start, _, _ in action_chunks(loaded, 3)] == [0, 0, 2, 2]


def test_rotated_virtual_tcp_maps_to_robot_local_axes():
    actions = np.array([[0, 0, 0, 0, 0, 0, 10], [0.01, 0, 0, 0, 0, 0, 9]])
    args = parse_args(["--dataset", "unused"])
    robot = DryRunRobot(np.r_[0.4, 0, 0.3, Rotation.from_euler("z", 90, degrees=True).as_rotvec()])
    targets = replay(actions, 20, robot, args)
    np.testing.assert_allclose(targets[0][:3], [0.4, 0.01, 0.3], atol=1e-7)
    np.testing.assert_allclose(relative_pose(robot.pose, robot.pose), 0, atol=1e-7)


def test_step_limit_stops_before_command():
    actions = np.array([[0, 0, 0, 0, 0, 0, 10], [0.1, 0, 0, 0, 0, 0, 10]])
    args = parse_args(["--dataset", "unused"])
    robot = DryRunRobot(np.array([0.4, 0, 0.3, 0, 0, 0]))
    with pytest.raises(ValueError, match="per-frame"):
        replay(actions, 20, robot, args)
    np.testing.assert_allclose(robot.pose, [0.4, 0, 0.3, 0, 0, 0])


def test_feedback_error_stops_at_chunk_boundary():
    actions = np.array([[0.01 * i, 0, 0, 0, 0, 0, 10] for i in range(4)])
    args = parse_args(["--dataset", "unused", "--horizon", "2", "--max-feedback-error-m", "0.005"])

    class LaggingRobot(DryRunRobot):
        def __init__(self):
            super().__init__(np.array([0.4, 0, 0.3, 0, 0, 0]))
            self.commands = 0

        def send_eef_command(self, pose):
            self.commands += 1  # Feedback remains at the initial pose.

    robot = LaggingRobot()
    with pytest.raises(ValueError, match="feedback differs"):
        replay(actions, 20, robot, args)
    assert robot.commands == 1


def test_pico_conversion_preserves_absolute_tcp_and_maps_gripper(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    poses = np.array([[0, 0, 0, 0, 0, 0, 0], [0.01, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    _dataset(source, poses, poses, kind="pico")
    convert(source, output)
    actions, _ = load_episode(output, 0)
    np.testing.assert_allclose(actions[:, 6], [0, 20])
