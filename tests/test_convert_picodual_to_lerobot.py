"""Picodual conversion keeps every source frame without using action_valid."""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import numcodecs
import pytest


from examples.piper.data_tools.convert_picodual_to_lerobot import prepare_episode_actions
from examples.piper.data_tools.convert_picodual_to_lerobot import prepare_episode_pose_data
from examples.piper.data_tools.convert_picodual_to_lerobot import read_numeric_array


def test_numeric_zarr_chunks_keep_exact_row_count(tmp_path):
    array_dir = tmp_path / "action"
    array_dir.mkdir()
    codec = numcodecs.Blosc(cname="zstd", clevel=5)
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    (array_dir / ".zarray").write_text(
        json.dumps({"shape": [5, 2], "chunks": [3, 2], "dtype": "<f4", "compressor": codec.get_config()})
    )
    (array_dir / "0.0").write_bytes(codec.encode(values[:3].tobytes()))
    (array_dir / "1.0").write_bytes(codec.encode(values[3:].tobytes()))

    np.testing.assert_array_equal(read_numeric_array(array_dir), values)


def test_missing_pose_is_held_without_dropping_gripper_frames():
    missing = [np.nan] * 6
    raw = np.asarray(
        [
            [*missing, 1],
            [0.1, 0.2, 0.3, 0, 0, 0, 2],
            [*missing, 3],
            [0.4, 0.5, 0.6, 0, 0, 0, 4],
            [*missing, 5],
        ],
        dtype=np.float32,
    )

    state, action = prepare_episode_actions(raw)

    assert state.shape == action.shape == raw.shape
    assert np.isfinite(state).all()
    assert np.isfinite(action).all()
    np.testing.assert_array_equal(state[:, 6], [1, 2, 3, 4, 5])
    np.testing.assert_allclose(state[0, :6], state[1, :6])
    np.testing.assert_allclose(state[2, :6], state[1, :6])
    np.testing.assert_allclose(state[4, :6], state[3, :6])
    np.testing.assert_allclose(action[:-1], state[1:])
    np.testing.assert_allclose(action[-1], state[-1])


def test_episode_requires_at_least_one_pose():
    raw = np.full((2, 7), np.nan, dtype=np.float32)
    raw[:, 6] = 1
    with pytest.raises(ValueError, match="no finite pose"):
        prepare_episode_actions(raw)


def test_episode_pose_is_full_se3_relative_and_mount_is_right_multiplied():
    # Controller starts yawed +90 degrees, then moves +1 along PICO World X.
    # The relative translation must therefore be [0, -1, 0], not [1, 0, 0].
    raw = np.array(
        [
            [2.0, 3.0, 0.0, 0.0, 0.0, np.pi / 2, 4.0],
            [3.0, 3.0, 0.0, 0.0, 0.0, np.pi / 2, 5.0],
        ],
        dtype=np.float64,
    )
    mount = np.eye(4)
    mount[:3, 3] = [0.0, 0.0, 0.2]

    data = prepare_episode_pose_data(raw, mount)

    np.testing.assert_allclose(data.world_tcp, data.world_controller @ mount)
    np.testing.assert_allclose(data.state[0, :6], np.zeros(6), atol=1e-7)
    np.testing.assert_allclose(data.state[1, :3], [0.0, -1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(data.action[0], data.state[1])
    np.testing.assert_allclose(data.episode_relative_tcp[0], np.eye(4), atol=1e-7)


def test_episode_pose_rejects_tracking_origin_jump():
    raw = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="tracking-origin jump"):
        prepare_episode_pose_data(raw, np.eye(4), max_translation_step_m=0.2)
