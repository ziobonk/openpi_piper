"""Live preview must match the actual chunk-relative execution targets."""
# ruff: noqa: SLF001

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
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
