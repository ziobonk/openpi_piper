"""lightUMI rotation-6D actions replay as a TCP-relative path."""
# ruff: noqa: I001

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.piper.data_tools.replay_lightumi_tcp_relative import (  # noqa: E402
    ROTATION_NAMES,
    decode_tcp_actions,
    load_episode,
    make_targets,
    relative_poses,
)


def _action(position, rotation):
    matrix = rotation.as_matrix()
    return np.r_[position, matrix[:, 0], matrix[:, 1]]


def _matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def test_rotation_columns_and_tcp_anchor_use_full_se3():
    first = _action([1.0, 2.0, 3.0], Rotation.from_euler("z", 90, degrees=True))
    second = _action([1.1, 2.0, 3.0], Rotation.from_euler("z", 100, degrees=True))
    poses = decode_tcp_actions(np.stack([first, second]))
    base = np.array([0.4, 0.0, 0.3, 0.1, -0.2, 0.3])
    target = make_targets(poses, base)
    assert target.shape == (1, 7)
    expected = _matrix(base) @ np.linalg.inv(_matrix(poses[0])) @ _matrix(poses[1])
    np.testing.assert_allclose(_matrix(target[0]), expected, atol=1e-10)
    np.testing.assert_allclose(target[0, 6], 0)


@pytest.mark.parametrize("world", ["PICO-world", "source-tracking-world"])
def test_loader_drops_repeated_last_next_action(tmp_path, world):
    root = tmp_path / "lightumi"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "robot_type": "lightumi",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {"action": {"shape": [9], "names": ["x", "y", "z", *ROTATION_NAMES]}},
        "lightumi_conversion": {
            "action": f"next aligned absolute {world} TCP xyz + rotation-6D, without a gripper angle",
            "rotation_6d": "columns_grouped:[r00,r10,r20,r01,r11,r21]",
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")
    (root / "meta/lightumi_sources.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "action_is_next_aligned_tcp": True,
                "last_action_repeated": True,
            }
        )
        + "\n"
    )
    first = _action([1, 2, 3], Rotation.identity())
    second = _action([1.01, 2, 3], Rotation.from_euler("z", 0.1))
    pd.DataFrame({"frame_index": [0, 1, 2], "action": [first, second, second]}).to_parquet(
        root / "data/chunk-000/episode_000000.parquet"
    )
    fps, poses = load_episode(root, 0)
    assert fps == 30
    assert poses.shape == (2, 6)
    np.testing.assert_allclose(poses[1, :3], [1.01, 2, 3])
    with pytest.raises(ValueError, match="unavailable"):
        load_episode(root, 1)


def test_rotation_6d_rejects_bad_columns():
    with pytest.raises(ValueError, match="orthonormal"):
        decode_tcp_actions(np.array([[0, 0, 0, 1, 0, 0, 1, 0, 0]]))


def test_live_path_records_tcp_and_creates_gif(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from examples.piper.data_tools import replay_lightumi_tcp_relative as lightumi
    from examples.piper.runtime import inference_eef as eef

    poses = np.array([[1.0, 2.0, 3.0, 0, 0, 0], [1.01, 2.0, 3.0, 0, 0, 0]])
    sent = []
    reads = []

    class Robot:
        def __init__(self):
            self.pose = np.array([0.4, 0, 0.3, 0, 0, 0])

        def enable(self, *, speed_pct):
            assert speed_pct == 30
            return True

        def get_eef_pose(self):
            reads.append(True)
            return self.pose.copy()

        def send_eef_command(self, pose):
            self.pose = pose.copy()
            sent.append(pose.copy())

    monkeypatch.setattr(lightumi, "load_episode", lambda *_: (30.0, poses))
    monkeypatch.setattr(eef, "GRIPPER_TCP_OFFSET_M", np.array([0.0, 0.0, 0.22]))
    monkeypatch.setattr(eef, "PiperEEFController", lambda *_, **kwargs: Robot())
    monkeypatch.setattr(lightumi.time, "sleep", lambda _: None)
    args = SimpleNamespace(
        data_dir=Path("unused"),
        episode=0,
        dry_run=False,
        max_frames=1,
        speed=0.25,
        speed_pct=30,
        can_name="can0",
        tcp_offset_m=[0.01, 0.0, 0.25],
        step=False,
        readback=False,
        yes=True,
        max_total_translation_m=0.5,
        max_step_translation_m=0.05,
        max_total_rotation_deg=120,
        max_step_rotation_deg=20,
        workspace=None,
        comparison_gif=tmp_path / "comparison.gif",
        gif_fps=3,
    )
    lightumi.run(args)
    assert len(reads) == 2
    np.testing.assert_allclose(eef.GRIPPER_TCP_OFFSET_M, [0.01, 0.0, 0.25])
    assert len(sent) == 4
    np.testing.assert_allclose([pose[0] for pose in sent], [0.4025, 0.405, 0.4075, 0.41])

    from PIL import Image

    with np.load(tmp_path / "comparison.npz") as saved:
        assert saved["dataset_relative_tcp"].shape == (2, 6)
        assert saved["robot_absolute_tcp"].shape == (2, 6)
        np.testing.assert_allclose(saved["dataset_relative_tcp"][:, 0], [0, 0.01])
        np.testing.assert_allclose(saved["robot_relative_tcp"][:, 0], [0, 0.01])
        np.testing.assert_allclose(saved["robot_absolute_tcp"][:, 0], [0.4, 0.41])
        np.testing.assert_allclose(saved["target_absolute_tcp"][:, 0], [0.4, 0.41])
        assert saved["elapsed_s"][1] >= 0
    with Image.open(tmp_path / "comparison.gif") as gif:
        assert gif.n_frames == 2


def test_relative_pose_uses_initial_tcp_axes():
    anchor = np.array([0.4, 0, 0.3, 0, 0, np.pi / 2])
    second = np.array([0.4, 0.01, 0.3, 0, 0, np.pi / 2])
    relative = relative_poses(np.stack([anchor, second]), anchor)
    np.testing.assert_allclose(relative[0], np.zeros(6), atol=1e-10)
    np.testing.assert_allclose(relative[1], [0.01, 0, 0, 0, 0, 0], atol=1e-10)
