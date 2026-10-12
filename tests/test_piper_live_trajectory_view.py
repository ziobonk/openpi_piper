"""Live preview must match the actual chunk-relative execution targets."""
# ruff: noqa: SLF001

from __future__ import annotations

import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.piper.chunk_relative import infer_pick_place
from examples.piper.runtime import inference_eef
from examples.piper.runtime.live_trajectory_view import LiveTrajectoryView
from examples.piper.runtime.live_trajectory_view import draw_3d_snapshot


def test_preview_matches_executed_robot_base_poses_with_read_only_actions():
    anchor = np.array([0.32, -0.08, 0.23, 0.2, 0.7, -0.3])
    actions = np.array(
        [
            [0.01, 0.02, -0.01, -0.1, 0.2, 0.1, 0.4],
            [0.03, -0.01, 0.02, 0.2, -0.1, 0.3, 0.5],
            [0.08, 0.05, 0.01, 0.4, 0.2, -0.2, 0.6],
        ],
        dtype=np.float32,
    )
    actions.setflags(write=False)
    view = LiveTrajectoryView()
    view.set_chunk(anchor, actions, execute_steps=2)
    assert view.targets.shape == (2, 6)
    for k in range(2):
        commanded = inference_eef._compose_pose6(anchor, actions[k, :6].astype(np.float64))
        np.testing.assert_allclose(view.targets[k, :3], commanded[:3], atol=1e-9)
        np.testing.assert_allclose(
            Rotation.from_rotvec(view.targets[k, 3:]).as_matrix(),
            Rotation.from_rotvec(commanded[3:]).as_matrix(),
            atol=1e-9,
        )
    first = view.render(0, 10.0)
    second = view.render(1, 20.0)
    assert first.shape == second.shape == (430, 1170, 3)
    assert first.dtype == np.uint8
    assert not np.array_equal(first, second)
    view.clear()
    assert view.targets is None


def test_sent_history_survives_new_chunks_and_pause_without_connecting_chunk_seams():
    view = LiveTrajectoryView()
    first_anchor = np.array([0.3, 0.0, 0.2, 0, 0, 0])
    first_actions = np.array([[0.01, 0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0, 0]])
    view.set_chunk(first_anchor, first_actions, 2)
    assert view.history_xyz.shape == (0, 3)  # predicted targets are not history
    view.record_sent(view.targets[0])
    np.testing.assert_allclose(view.history_xyz, [[0.3, 0, 0.2], [0.31, 0, 0.2]])
    view.clear()
    assert view.targets is None
    assert view.render(0).shape == (430, 1170, 3)  # history remains visible while paused

    second_anchor = np.array([0.42, 0.1, 0.25, 0, 0, 0])
    view.set_chunk(second_anchor, np.array([[0, 0.01, 0, 0, 0, 0]]), 1)
    view.record_sent(view.targets[0])
    history = view.history_xyz
    np.testing.assert_allclose(history[:2], [[0.3, 0, 0.2], [0.31, 0, 0.2]])
    assert np.isnan(history[2]).all()
    np.testing.assert_allclose(history[3:], [[0.42, 0.1, 0.25], [0.42, 0.11, 0.25]])
    assert view.render(1).shape == (430, 1170, 3)  # completed chunk is still drawable


def test_pick_place_preview_defaults_on_and_can_be_disabled(monkeypatch):
    parsed = []
    monkeypatch.setattr(infer_pick_place, "training_metadata", lambda _: (30.0, (0.0, 1.0), 20.0, "test"))
    monkeypatch.setattr(infer_pick_place, "base_camera_size", lambda _: (640, 480))
    monkeypatch.setattr(infer_pick_place.inference_eef, "_HAS_DM_GRIPPER", True)
    monkeypatch.setattr(infer_pick_place.inference_eef, "_parse_args", lambda argv: argv)
    monkeypatch.setattr(infer_pick_place.inference_eef, "main", lambda args: parsed.append(args))
    base_args = ["infer_pick_place.py", "--base-camera", "serial", "--wrist-camera", "0"]
    monkeypatch.setattr(sys, "argv", base_args)
    infer_pick_place.main()
    assert "--visualize_trajectory" in parsed[-1]
    assert parsed[-1][parsed[-1].index("--trajectory_view") + 1] == "3d"
    monkeypatch.setattr(sys, "argv", [*base_args, "--trajectory-view", "2d"])
    infer_pick_place.main()
    assert parsed[-1][parsed[-1].index("--trajectory_view") + 1] == "2d"
    monkeypatch.setattr(sys, "argv", [*base_args, "--no-visualize-trajectory"])
    infer_pick_place.main()
    assert "--visualize_trajectory" not in parsed[-1]
    monkeypatch.setattr(sys, "argv", [*base_args, "--gripper-open-rad", "-7.2", "--gripper-close-rad", "4.1"])
    infer_pick_place.main()
    assert parsed[-1][parsed[-1].index("--gripper_open_rad") + 1] == "-7.2"
    assert parsed[-1][parsed[-1].index("--gripper_close_rad") + 1] == "4.1"
    monkeypatch.setattr(sys, "argv", [*base_args, "--gripper-open-rad", "4.1", "--gripper-close-rad", "4.1"])
    with pytest.raises(SystemExit):
        infer_pick_place.main()


def test_3d_snapshot_draws_equal_axes_and_preserves_mouse_view():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    view = LiveTrajectoryView()
    anchor = np.array([0.3, 0.1, 0.2, 0.2, -0.1, 0.3])
    actions = np.array([[0.01, 0.02, 0.01, 0.1, 0.2, 0.1, 5], [0.03, 0.04, 0.02, -0.1, 0.3, 0.2, 10]], dtype=np.float32)
    view.set_chunk(anchor, actions, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=18, azim=42)
    draw_3d_snapshot(ax, view.anchor, view.targets, 0, 10.0)
    assert ax.elev == 18
    assert ax.azim == 42
    limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    np.testing.assert_allclose([hi - lo for lo, hi in limits], np.repeat(limits[0][1] - limits[0][0], 3))
    fig.canvas.draw()
    draw_3d_snapshot(ax, view.anchor, view.targets, 1, 20.0)
    assert ax.elev == 18
    assert ax.azim == 42
    view.record_sent(view.targets[0])
    view.clear()
    draw_3d_snapshot(ax, None, None, 0, None, view.history_xyz)
    assert len(ax.lines) == 1
    assert ax.elev == 18
    assert ax.azim == 42
    plt.close(fig)


def test_policy_query_previews_only_actions_selected_for_execution():
    client = inference_eef.PiperEEFInference.__new__(inference_eef.PiperEEFInference)
    client._trajectory_view = LiveTrajectoryView()
    client._action_cache = None
    client._chunk_tcp_base = None
    client._model_action_frame = "chunk_relative"
    client._exec_horizon = 2
    client._model_gripper_range = None
    client._gripper_prediction_scale = 1.0
    client._binary_gripper = False
    client._gripper_open_width_mm = 20.0
    anchor = np.array([0.3, 0.1, 0.2, 0.1, 0.2, -0.3])

    def observation():
        client._chunk_tcp_base = anchor.copy()
        return {"prompt": "test"}

    client._build_observation = observation
    chunk = np.array(
        [
            [0, 0, 0, 0, 0, 0, 10],
            [0.01, 0, 0, 0, 0, 0, 10],
            [0.02, 0, 0, 0, 0, 0, 10],
            [0.03, 0, 0, 0, 0, 0, 10],
        ],
        dtype=np.float32,
    )
    chunk.setflags(write=False)
    client._policy_client = SimpleNamespace(infer=lambda _: {"actions": chunk})
    client._query_policy()
    assert client._action_cache.shape == (3, 7)
    assert client._trajectory_view.targets.shape == (2, 6)
    for index in range(2):
        np.testing.assert_allclose(
            client._trajectory_view.targets[index, :3],
            inference_eef._compose_pose6(anchor, chunk[index + 1, :6].astype(np.float64))[:3],
        )


@pytest.mark.parametrize("command_succeeds", [True, False])
def test_history_records_target_only_after_command_returns(command_succeeds):
    client = inference_eef.PiperEEFInference.__new__(inference_eef.PiperEEFInference)
    client._running = True
    client._inferring = True
    client._input_queue = queue.Queue()
    client._action_cache = np.array([[0.02, 0, 0, 0, 0, 0, 5]], dtype=np.float32)
    client._cache_step = 0
    client._exec_horizon = 1
    client._period = 0
    client._model_action_frame = "chunk_relative"
    client._chunk_tcp_base = np.array([0.3, 0, 0.2, 0, 0, 0])
    client._model_gripper_range = None
    client._gripper_prediction_scale = 1.0
    client._binary_gripper = False
    client._gripper_open_width_mm = 20.0
    client._trajectory_recorder = None
    client._robot = SimpleNamespace(get_eef_pose=lambda: client._chunk_tcp_base.copy())
    client._display_trajectory = lambda *args: None
    client._trajectory_view = LiveTrajectoryView()
    client._trajectory_view.set_chunk(client._chunk_tcp_base, client._action_cache, 1)

    def send(*args):
        assert client._trajectory_view.history_xyz.size == 0
        client._running = False
        if not command_succeeds:
            raise RuntimeError("robot command failed")

    client._send_eef_cmd = send
    if command_succeeds:
        client._control_loop()
        np.testing.assert_allclose(client._trajectory_view.history_xyz[-1], [0.32, 0, 0.2])
    else:
        with pytest.raises(RuntimeError, match="robot command failed"):
            client._control_loop()
        assert client._trajectory_view.history_xyz.size == 0
