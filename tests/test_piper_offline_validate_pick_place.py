"""Verify read-only TCP anchoring and same-frame chunk rollout."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from examples.piper.chunk_relative import offline_validate_pick_place as validation


def test_rollout_anchors_noncommuting_rotations_and_uses_dataset_observations():
    source = np.array(
        [
            [0.1, 0.2, 0.3, 0, 0, np.pi / 2, 0.2],
            [0.1, 0.3, 0.3, 0, 0, np.pi / 2, 0.3],
            [0.1, 0.4, 0.3, 0, 0, np.pi / 2, 0.4],
            [0.1, 0.5, 0.3, 0, 0, np.pi / 2, 0.5],
        ]
    )
    start = np.array([0.4, 0.1, 0.5, np.pi / 2, 0, 0])
    state = source[:, 6:7]

    class Video:
        def read(self, index):
            return np.full((8, 8, 3), index, np.uint8), np.zeros((8, 8, 3), np.uint8)

    class Policy:
        def __init__(self):
            self.observations = []

        def infer(self, observation):
            self.observations.append(observation)
            t = len(self.observations) - 1
            # A single-step chunk followed by a fresh query at frame 1.
            next_index = t + 1
            delta = np.linalg.inv(validation.pose_matrix(source[t, :6])) @ validation.pose_matrix(
                source[next_index, :6]
            )
            return {"actions": np.stack((np.zeros(7), np.r_[validation.matrix_pose(delta), source[next_index, 6]]))}

    policy = Policy()
    predicted, expected, input_index, images = validation.rollout(
        policy, state, source, Video(), start, "test prompt", steps=1
    )
    np.testing.assert_allclose(predicted[:, :3], expected[:, :3], atol=1e-8)
    np.testing.assert_allclose(
        Rotation.from_rotvec(predicted[:, 3:6]).as_matrix(),
        Rotation.from_rotvec(expected[:, 3:6]).as_matrix(),
        atol=1e-8,
    )
    assert input_index.tolist() == [0, 0, 1, 2]
    assert sorted(images) == [0, 1, 2]
    assert all(obs["observation/state"].shape == (1,) for obs in policy.observations)
    assert all(obs["prompt"] == "test prompt" for obs in policy.observations)


def test_read_piper_tcp_only_receives_feedback(monkeypatch):
    calls = []

    class FakePiper:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def ConnectPort(self, **kwargs):  # noqa: N802
            calls.append(("connect", kwargs))

        def GetArmEndPoseMsgs(self):  # noqa: N802
            calls.append(("read",))
            return SimpleNamespace(
                time_stamp=1,
                Hz=30,
                end_pose=SimpleNamespace(
                    X_axis=300_000,
                    Y_axis=0,
                    Z_axis=100_000,
                    RX_axis=0,
                    RY_axis=0,
                    RZ_axis=90_000,
                ),
            )

        def DisconnectPort(self):  # noqa: N802
            calls.append(("disconnect",))

    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface_V2=FakePiper))
    tcp = validation.read_piper_tcp("can0", np.array([0.1, 0, 0]))
    np.testing.assert_allclose(tcp[:3], [0.3, 0.1, 0.1], atol=1e-8)
    assert [entry[0] for entry in calls] == ["init", "connect", "read", "disconnect"]
    assert calls[1][1] == {"piper_init": False}
