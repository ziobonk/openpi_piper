"""Chunk-local TCP targets and gripper-only Piper observations."""
# ruff: noqa: E402, SLF001

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.piper.runtime import inference_eef as eef
from examples.piper.runtime.inference_eef_async import PiperEEFAsyncInference

from openpi.policies.piper_chunk_relative import ChunkRelativeEEFActions


def _pose_matrix(pose):
    return eef._pose6_to_matrix(np.asarray(pose))


def test_action_window_is_relative_to_its_first_pose_and_frame_invariant():
    poses = np.array([
        [0.2, -0.1, 0.4, 0.1, 0.3, -0.2, 12],
        [0.25, -0.08, 0.43, -0.2, 0.1, 0.3, 8],
        [0.31, 0.02, 0.38, 0.2, -0.1, 0.5, 4],
    ], dtype=np.float64)
    transform = ChunkRelativeEEFActions()
    result = transform({"state": np.array([12.0]), "actions": poses.copy()})["actions"]
    np.testing.assert_allclose(result[0, :6], 0)
    np.testing.assert_allclose(result[:, 6], poses[:, 6])
    for target, local in zip(poses, result, strict=True):
        np.testing.assert_allclose(_pose_matrix(local[:6]), np.linalg.inv(_pose_matrix(poses[0, :6])) @ _pose_matrix(target[:6]), atol=1e-6)

    world_change = np.eye(4)
    world_change[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.2, 0.1]).as_matrix()
    world_change[:3, 3] = [1, -2, 3]
    shifted = poses.copy()
    for original, target in zip(poses, shifted, strict=True):
        target[:6] = eef._matrix_to_pose6(world_change @ _pose_matrix(original[:6]))
    shifted_result = transform({"state": np.array([12.0]), "actions": shifted})["actions"]
    np.testing.assert_allclose(shifted_result, result, atol=1e-6)


def test_clients_send_only_gripper_and_capture_each_chunk_base():
    tcp = np.array([0.4, 0.0, 0.3, 0.0, 0.2, 0.1])
    robot_state = np.r_[tcp, 11.0]
    robot = SimpleNamespace(get_state=lambda: robot_state)
    for cls in (eef.PiperEEFInference, PiperEEFAsyncInference):
        client = cls.__new__(cls)
        client._model_action_frame = "chunk_relative"
        client._robot = robot
        client._camera = None
        client._default_prompt = "pick cube"
        client._interactive = False
        observation = client._build_observation()
        np.testing.assert_array_equal(observation["observation/state"], [11.0])
        np.testing.assert_allclose(client._chunk_tcp_base, tcp)
        robot_state[:6] += [0.1, 0, 0, 0, 0, 0]
        np.testing.assert_allclose(client._chunk_tcp_base, tcp)
        robot_state[:6] = tcp


def test_async_buffered_absolute_target_uses_captured_base():
    tcp = np.array([0.4, 0.0, 0.3, 0.0, 0.2, 0.1])
    local = np.array([0.02, 0.01, 0, 0, 0, 0.2])
    action = np.r_[eef._compose_pose6(tcp, local), 9.0]
    sent = []
    client = PiperEEFAsyncInference.__new__(PiperEEFAsyncInference)
    client._model_action_frame = "chunk_relative"
    client._robot = SimpleNamespace(
        send_eef_command=lambda pose: sent.append(np.asarray(pose)),
        send_gripper_command=lambda width: sent.append(width),
    )
    client._robot_lock = threading.Lock()
    client._buffer = SimpleNamespace(mark_executed=lambda _: None)
    client._use_interp = False
    client._gripper_prediction_scale = 1.0
    client._binary_gripper = False
    client._gripper_open_width_mm = 20.0
    client._model_gripper_range = None
    client._publish_action(action, np.zeros(6), generation=0)
    np.testing.assert_allclose(_pose_matrix(sent[0]), _pose_matrix(action[:6]))
    assert sent[1] == 9.0
    chunk = np.vstack([np.r_[np.zeros(6), 11.0], np.r_[local, 9.0]])
    np.testing.assert_array_equal(eef._execution_actions(chunk, "chunk_relative"), chunk[1:])
