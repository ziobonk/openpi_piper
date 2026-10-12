"""Raw Piper SDK replay keeps absolute and anchored trajectories distinct."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.piper.data_tools.replay_piper_sdk_raw import load_episode
from examples.piper.data_tools.replay_piper_sdk_raw import move_to_absolute_start
from examples.piper.data_tools.replay_piper_sdk_raw import plan_absolute_approach
from examples.piper.data_tools.replay_piper_sdk_raw import replay
from examples.piper.data_tools.replay_piper_sdk_raw import targets_from_start


def _dataset(tmp_path):
    root = tmp_path / "raw"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    actions = np.array([
        [0.4, 0, 0.3, 0, 0, 0, 10],
        [0.41, 0, 0.3, 0, 0, 0, 12],
        [0.42, 0, 0.3, 0, 0, 0, 14],
    ], dtype=np.float32)
    info = {
        "source_kind": "piper_sdk_raw", "robot_type": "piper_eef",
        "pose_coordinate_frame": "robot_base_tcp", "coordinate_transform_applied": False,
        "features": {"state": {"shape": [7]}, "actions": {"shape": [7]}},
        "tcp_offset_m": [0, 0, 0.1], "gripper_mapping": {"robot_open_width_mm": 70},
        "fps": 20, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")
    pd.DataFrame({"frame_index": range(3), "state": list(actions), "actions": list(actions)}).to_parquet(
        root / "data/chunk-000/episode_000000.parquet"
    )
    return root, actions


def test_raw_replay_loads_same_frame_pose_and_anchors_to_current_tcp(tmp_path):
    root, actions = _dataset(tmp_path)
    info, loaded = load_episode(root, 0)
    assert info["tcp_offset_m"] == [0, 0, 0.1]
    np.testing.assert_allclose(loaded, actions)
    start = np.array([0.5, 0.1, 0.3, 0, 0, 0])
    relative = targets_from_start(loaded, start, "relative")
    absolute = targets_from_start(loaded, start, "absolute")
    np.testing.assert_allclose(relative[:, :3], [[0.5, 0.1, 0.3], [0.51, 0.1, 0.3], [0.52, 0.1, 0.3]])
    np.testing.assert_allclose(absolute[:, :3], actions[:, :3])
    np.testing.assert_allclose(relative[:, 6], actions[:, 6])


def _approach_args():
    return SimpleNamespace(
        max_start_error_mm=300, max_start_rotation_deg=90,
        approach_step_mm=5, approach_step_rotation_deg=2, approach_hz=20,
        approach_timeout_s=1, arrival_position_mm=10, arrival_rotation_deg=5,
        workspace=None,
    )


def test_absolute_approach_interpolates_translation_and_rotation():
    args = _approach_args()
    start = np.array([0.4, 0, 0.3, 0, 0, 0])
    recorded = np.array([0.4145, 0, 0.3, 0, 0, np.deg2rad(15.2)])
    waypoints = plan_absolute_approach(start, recorded, args)
    assert len(waypoints) == 8
    np.testing.assert_allclose(waypoints[-1], recorded)
    assert np.max(np.linalg.norm(np.diff(np.vstack([start[:3], waypoints[:, :3]]), axis=0), axis=1)) <= 0.005
    assert np.max(np.abs(np.diff(np.r_[0, waypoints[:, 5]]))) <= np.deg2rad(2) + 1e-9


def test_absolute_approach_rejects_large_gap_and_workspace():
    args = _approach_args()
    start = np.array([0.4, 0, 0.3, 0, 0, 0])
    with pytest.raises(ValueError, match="exceeds absolute approach limits"):
        plan_absolute_approach(start, np.array([0.8, 0, 0.3, 0, 0, 0]), args)
    args.workspace = [0.41, 0.5, -0.1, 0.1, 0.2, 0.4]
    with pytest.raises(ValueError, match="outside --workspace"):
        plan_absolute_approach(start, np.array([0.42, 0, 0.3, 0, 0, 0]), args)


def test_absolute_approach_requires_measured_arrival(monkeypatch):
    args = _approach_args()
    start = np.array([0.4, 0, 0.3, 0, 0, 0])
    recorded = np.array([0.415, 0, 0.3, 0, 0, 0])

    class Robot:
        def __init__(self):
            self.commands = []

        def send_eef_command(self, pose):
            self.commands.append(pose.copy())

    robot = Robot()
    reads = iter([start, start, recorded])
    monkeypatch.setattr("examples.piper.data_tools.replay_piper_sdk_raw._fresh_pose", lambda robot: next(reads))
    monkeypatch.setattr("examples.piper.data_tools.replay_piper_sdk_raw.time.sleep", lambda _: None)
    actual = move_to_absolute_start(robot, recorded, args)
    np.testing.assert_allclose(actual, recorded)
    np.testing.assert_allclose(robot.commands[-1], recorded)
    assert len(robot.commands) == 4


def test_dry_run_only_reads_dataset(tmp_path):
    root, _ = _dataset(tmp_path)
    args = SimpleNamespace(
        dataset=root, episode=0, max_frames=0, mode="relative", dry_run=True, workspace=None,
        max_total_translation_m=0.5, max_step_translation_m=0.05,
        max_total_rotation_deg=120, max_step_rotation_deg=20,
    )
    replay(args)
