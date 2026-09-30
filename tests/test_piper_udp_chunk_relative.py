"""UDP measured trajectories can become same-frame chunk-relative replay data."""
# ruff: noqa: E402, I001

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.piper.chunk_relative.prepare_chunk_relative_eef import convert
from examples.piper.data_tools.replay_picodual_lerobot import chunk_relative_targets, load_chunk_relative_episode


def test_udp_conversion_uses_measured_pose_and_preserves_gripper_mm(tmp_path):
    source = tmp_path / "udp"
    (source / "meta").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    states = np.array(
        [
            [0.40, 0, 0.30, 0, 0, 0, 10],
            [0.41, 0, 0.30, 0, 0, 0, 11],
            [0.42, 0, 0.30, 0, 0, 0, 12],
        ],
        dtype=np.float32,
    )
    commands = states.copy()
    commands[:, 0] += 0.05  # Command leads the measured robot pose.
    info = {
        "robot_type": "piper_eef",
        "pose_coordinate_frame": "robot_base_tcp",
        "coordinate_transform_applied": False,
        "total_frames": 3,
        "features": {"state": {"shape": [7]}, "actions": {"shape": [7]}},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "chunks_size": 1000,
        "fps": 20,
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/stats.json").write_text(
        json.dumps({"state": {"min": states.min(axis=0).tolist(), "max": states.max(axis=0).tolist()}})
    )
    episode = {"episode_index": 0, "length": 3, "stats": {"state": {}, "actions": {}}}
    (source / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")
    (source / "meta/episodes_stats.jsonl").write_text(json.dumps(episode) + "\n")
    pd.DataFrame(
        {"episode_index": [0] * 3, "frame_index": range(3), "state": list(states), "actions": list(commands)}
    ).to_parquet(source / "data/chunk-000/episode_000000.parquet")

    target = tmp_path / "chunk"
    convert(source, target, source_kind="udp", robot_open_width_mm=20)
    result_info, _, _, _, actions = load_chunk_relative_episode(str(target), 0)
    assert result_info["pose_coordinate_frame"] == "robot_base_tcp"
    np.testing.assert_allclose(actions, states)
    decoded = chunk_relative_targets(actions, 0, 2, np.array([0.5, 0, 0.3, 0, 0, 0]))
    np.testing.assert_allclose(decoded[:, 0], [0.51, 0.52], atol=1e-6)
    np.testing.assert_allclose(decoded[:, 6], [11, 12])


def test_udp_conversion_rejects_unknown_frame(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text(json.dumps({"robot_type": "piper_eef"}))
    with pytest.raises(ValueError, match="robot_base_tcp"):
        convert(source, tmp_path / "output", source_kind="udp")
