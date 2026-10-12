"""Dataset replay and visualization keep the chunk-local TCP convention."""
# ruff: noqa: E402, SLF001

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.piper.data_tools.replay_picodual_lerobot import chunk_relative_targets, load_chunk_relative_episode
from examples.piper.data_tools.visualize_picodual_dataset import load_episode, stitch_training_chunks, training_chunk_actions


def _matrix(pose):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _actions():
    result = np.zeros((8, 7), dtype=np.float64)
    result[:, 0] = np.linspace(0.1, 0.24, len(result))
    result[:, 1] = np.linspace(-0.2, -0.1, len(result))
    result[:, 3:6] = Rotation.from_euler("z", np.linspace(0.2, 0.55, len(result))).as_rotvec()
    result[:, 6] = np.arange(len(result))
    return result


def test_replay_uses_current_robot_tcp_for_each_chunk_and_skips_same_frame():
    actions = _actions()
    first_robot_tcp = np.array([0.4, 0.0, 0.3, 0.1, -0.2, 0.3])
    first = chunk_relative_targets(actions, 0, 3, first_robot_tcp)
    assert first.shape == (3, 7)
    for k, target in enumerate(first, start=1):
        expected = _matrix(first_robot_tcp) @ np.linalg.inv(_matrix(actions[0, :6])) @ _matrix(actions[k, :6])
        np.testing.assert_allclose(_matrix(target[:6]), expected, atol=1e-12)
        assert target[6] == actions[k, 6]
    # A new observation at action[3] defines a fresh base. The next target is
    # action[4], so every dataset target after the same-frame action runs once.
    second_robot_tcp = first[-1, :6].copy()
    second = chunk_relative_targets(actions, 3, 3, second_robot_tcp)
    assert second.shape == (3, 7)
    np.testing.assert_allclose(
        _matrix(second[0, :6]),
        _matrix(second_robot_tcp) @ np.linalg.inv(_matrix(actions[3, :6])) @ _matrix(actions[4, :6]),
        atol=1e-12,
    )


def test_viewer_targets_match_training_transform_and_stitch_absolute_path():
    actions = _actions()
    for start in (0, 3):
        local = training_chunk_actions(actions, actions, start, 4, chunk_relative=True)
        np.testing.assert_allclose(local[0, :6], 0, atol=1e-7)
        for k in range(4):
            np.testing.assert_allclose(
                _matrix(local[k, :6]),
                np.linalg.inv(_matrix(actions[start, :6])) @ _matrix(actions[start + k, :6]),
                atol=1e-6,
            )
    stitched = stitch_training_chunks(actions, actions, 4, same_frame=True, chunk_relative=True)
    # Entries 0,1..4 are state0 + first action window; entry 5 is state4.
    for i, expected_index in enumerate((0, 0, 1, 2, 3, 4, 4, 5, 6, 7)):
        np.testing.assert_allclose(stitched[i], _matrix(actions[expected_index, :6]), atol=1e-6)


def test_episode_loaders_read_only_selected_chunk_relative_episode(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    data = root / "data/chunk-000"
    data.mkdir(parents=True)
    info = {
        "robot_type": "piper_chunk_relative_eef",
        "fps": 20,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "gripper_mapping": {"robot_open_width_mm": 20},
        "pose_coordinate_frame": "pico_world_tcp",
        "coordinate_transform_applied": False,
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 8}) + "\n")
    actions = _actions()
    pd.DataFrame({
        "episode_index": np.zeros(8, dtype=np.int64),
        "frame_index": np.arange(8),
        "state": [row[6:7] for row in actions],
        "actions": list(actions),
    }).to_parquet(data / "episode_000000.parquet")
    _, frame, episode, lengths, loaded = load_chunk_relative_episode(str(root), 0)
    assert episode == 0 and lengths == {0: 8} and len(frame) == 8
    np.testing.assert_allclose(loaded, actions)
    _, visual_frame, visual_episode, _ = load_episode(str(root), 0)
    assert visual_episode == 0 and len(visual_frame) == 8


def test_slow_replay_sends_interpolated_tcp_commands(monkeypatch):
    from types import SimpleNamespace

    from examples.piper.data_tools import replay_picodual_lerobot as replay

    actions = np.array([
        [0.40, 0, 0.30, 0, 0, 0, 10],
        [0.41, 0, 0.30, 0, 0, 0, 10],
    ], dtype=np.float64)
    sent = []
    sleeps = []

    class Robot:
        def enable(self, *, speed_pct):
            assert speed_pct == 30
            return True

        def get_eef_pose(self):
            return actions[0, :6].copy()

        def execute_action(self, action):
            sent.append(action.copy())

    monkeypatch.setattr(replay, "load_chunk_relative_episode", lambda *_: (
        {"fps": 20, "gripper_mapping": {"robot_open_width_mm": 20}}, None, 0, {0: 2}, actions
    ))
    monkeypatch.setattr(replay, "_validate_trajectory", lambda *_: None)
    monkeypatch.setattr(replay, "_build_gripper_and_robot", lambda *_: Robot())
    monkeypatch.setattr(replay.time, "sleep", sleeps.append)
    args = SimpleNamespace(
        data_dir="unused", episode=0, info_only=False, adapt_dataset_tcp=False,
        action_horizon=50, exec_horizon=20, gripper_open_width_mm=None,
        control_gripper=False, dry_run=False, speed_pct=30, speed=0.25,
        yes=True, step=False, readback=False, workspace=None,
    )
    replay.replay_chunk_relative(args)
    assert len(sent) == 4
    np.testing.assert_allclose([action[0] for action in sent], [0.4025, 0.405, 0.4075, 0.41])
    np.testing.assert_allclose(sleeps, [0.05] * 4)


def test_piper_sdk_replay_uses_recorded_tcp_offset_and_native_gripper(monkeypatch):
    from types import SimpleNamespace

    from examples.piper.data_tools import replay_picodual_lerobot as replay
    from examples.piper.runtime import inference_eef as eef

    actions = _actions()[:2]
    original_offset = eef.GRIPPER_TCP_OFFSET_M.copy()
    observed = {}

    class Robot:
        def enable(self, *, speed_pct):
            return True

        def get_eef_pose(self):
            return actions[0, :6].copy()

        def execute_action(self, action):
            observed["sent"] = action.copy()

    def build(args, module):
        observed["port"] = args.gripper_port
        observed["offset"] = module.GRIPPER_TCP_OFFSET_M.copy()
        return Robot()

    monkeypatch.setattr(replay, "load_chunk_relative_episode", lambda *_: (
        {"fps": 20, "source_kind": "piper_sdk_read_only", "tcp_offset_m": [0, 0, 0],
         "gripper_mapping": {"robot_open_width_mm": 70}}, None, 0, {0: 2}, actions,
    ))
    monkeypatch.setattr(replay, "_validate_trajectory", lambda *_: None)
    monkeypatch.setattr(replay, "_build_gripper_and_robot", build)
    monkeypatch.setattr(replay.time, "sleep", lambda *_: None)
    args = SimpleNamespace(
        data_dir="unused", episode=0, info_only=False, adapt_dataset_tcp=False,
        action_horizon=50, exec_horizon=1, gripper_open_width_mm=None,
        gripper_port="/dev/ttyACM1", control_gripper=True, dry_run=False,
        speed_pct=30, speed=1.0, yes=True, step=False, readback=False, workspace=None,
    )
    try:
        replay.replay_chunk_relative(args)
    finally:
        eef.GRIPPER_TCP_OFFSET_M[:] = original_offset
    assert observed["port"] == ""
    np.testing.assert_array_equal(observed["offset"], [0, 0, 0])
    assert "sent" in observed
