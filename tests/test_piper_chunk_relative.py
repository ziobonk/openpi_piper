"""Chunk-local TCP targets and gripper-only Piper observations."""
# ruff: noqa: E402, SLF001

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
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


def test_training_delta_reanchors_in_measured_robot_tcp_frame():
    source = np.array([
        [0.2, -0.1, 0.4, 0.2, 0.1, -0.3, 0.1],
        [0.23, -0.08, 0.43, -0.1, 0.25, 0.4, 0.2],
    ])
    anchor = np.array([0.45, 0.05, 0.35, -0.2, 0.3, 0.1])
    local = ChunkRelativeEEFActions()({"state": np.array([0.1]), "actions": source.copy()})["actions"]
    future = eef._execution_actions(local, "chunk_relative")
    assert len(future) == 1
    target = eef._compose_pose6(anchor, future[0, :6])
    expected = _pose_matrix(anchor) @ np.linalg.inv(_pose_matrix(source[0, :6])) @ _pose_matrix(source[1, :6])
    np.testing.assert_allclose(_pose_matrix(target), expected, atol=1e-7)


def test_flange_tcp_offset_read_and_command_are_inverses(monkeypatch):
    offset = np.array([0.01, -0.02, 0.22])
    monkeypatch.setattr(eef, "GRIPPER_TCP_OFFSET_M", offset)
    flange = np.array([0.45, 0.1, 0.3])
    euler_deg = np.array([25.0, -10.0, 35.0])
    feedback = SimpleNamespace(
        X_axis=int(round(flange[0] * 1e6)),
        Y_axis=int(round(flange[1] * 1e6)),
        Z_axis=int(round(flange[2] * 1e6)),
        RX_axis=int(round(euler_deg[0] * 1000)),
        RY_axis=int(round(euler_deg[1] * 1000)),
        RZ_axis=int(round(euler_deg[2] * 1000)),
    )
    commands = []
    robot = eef.PiperEEFController.__new__(eef.PiperEEFController)
    robot._piper = SimpleNamespace(
        GetArmEndPoseMsgs=lambda: SimpleNamespace(end_pose=feedback),
        EndPoseCtrl=lambda *values: commands.append(values),
    )
    robot._last_command_euler_rad = None
    measured_tcp = robot.get_eef_pose()
    rotation = Rotation.from_euler("xyz", euler_deg, degrees=True)
    np.testing.assert_allclose(measured_tcp[:3], flange + rotation.apply(offset), atol=1e-6)
    robot.send_eef_command(measured_tcp)
    np.testing.assert_allclose(np.array(commands[0][:3]) / 1e6, flange, atol=1e-6)
    np.testing.assert_allclose(np.array(commands[0][3:]) / 1000, euler_deg, atol=1e-3)


def test_failed_policy_query_clears_old_chunk_and_tcp_anchor(monkeypatch):
    client = eef.PiperEEFInference.__new__(eef.PiperEEFInference)
    client._action_cache = np.ones((3, 7), dtype=np.float32)
    client._chunk_tcp_base = np.ones(6)
    client._trajectory_view = None
    client._model_action_frame = "chunk_relative"
    client._inferring = True
    client._build_observation = lambda: {"prompt": "test"}
    client._policy_client = SimpleNamespace(infer=lambda obs: (_ for _ in ()).throw(RuntimeError("offline")))
    client._connect_policy = lambda: None
    monkeypatch.setattr(eef.time, "sleep", lambda _: None)
    client._query_policy()
    assert client._action_cache is None
    assert client._chunk_tcp_base is None


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
        client._model_gripper_range = None
        observation = client._build_observation()
        np.testing.assert_array_equal(observation["observation/state"], [11.0])
        np.testing.assert_allclose(client._chunk_tcp_base, tcp)
        robot_state[:6] += [0.1, 0, 0, 0, 0, 0]
        np.testing.assert_allclose(client._chunk_tcp_base, tcp)
        robot_state[:6] = tcp


def test_chunk_relative_gripper_is_mapped_only_at_inference():
    state = np.r_[[0.4, 0, 0.3, 0, 0, 0], 10.0]
    for cls in (eef.PiperEEFInference, PiperEEFAsyncInference):
        client = cls.__new__(cls)
        client._model_action_frame = "chunk_relative"
        client._robot = SimpleNamespace(get_state=lambda: state)
        client._camera = None
        client._default_prompt = "pick cube"
        client._interactive = False
        client._model_gripper_range = (0.1, 0.9)
        client._gripper_open_width_mm = 20.0
        observation = client._build_observation()
        np.testing.assert_allclose(observation["observation/state"], [0.5])
        np.testing.assert_allclose(client._model_gripper_to_runtime(0.5), 10.0)


def test_dm_gripper_maps_width_to_negative_open_angle(monkeypatch):
    sent = []

    class FakeDmGripper:
        def __init__(self, motor, **kwargs):
            self.speed = kwargs["speed_rad_s"]

        def set_position(self, position):
            sent.append((position, self.speed))

        def read_feedback(self):
            return {"err": 1, "pos": -5.0}

    monkeypatch.setattr(eef, "Usb2Can", lambda port: SimpleNamespace(port=port))
    monkeypatch.setattr(eef, "DmJ4310", lambda bus, can_id: SimpleNamespace(bus=bus, can_id=can_id))
    monkeypatch.setattr(eef, "DmGripper", FakeDmGripper)
    gripper = eef.DmGripperController(open_rad=-10.0, close_rad=0.0, speed_rad_s=10.0)
    for width in (0, 10, 20):
        gripper.send(width)
    assert sent == [(0.0, 10.0), (-5.0, 10.0), (-10.0, 10.0)]
    assert gripper.read_mm() == 10.0
    with pytest.raises(ValueError, match="开闭角度必须不同"):
        eef.DmGripperController(open_rad=0.0, close_rad=0.0)


def test_dm_gripper_rejects_missing_disabled_and_uncalibrated_feedback():
    gripper = eef.DmGripperController.__new__(eef.DmGripperController)
    gripper._open_rad = -10.0
    gripper._close_rad = 0.0
    gripper._open_width_mm = 20.0
    for feedback, message in (
        (None, "无位置反馈"),
        ({"err": 0, "pos": -5.0}, "ERR=0"),
        ({"err": 1, "pos": 2.806}, "超出标定开闭区间"),
    ):
        gripper._gripper = SimpleNamespace(read_feedback=lambda value=feedback: value)
        with pytest.raises(eef.GripperFeedbackError, match=message):
            gripper.read_mm()


def test_piper_stops_startup_when_dm_gripper_enable_fails():
    controller = eef.PiperEEFController.__new__(eef.PiperEEFController)
    controller._piper = SimpleNamespace(EnablePiper=lambda: True, MotionCtrl_2=lambda *args: None)
    controller._control_gripper = True
    controller._gripper = SimpleNamespace(enable=lambda: False)
    assert controller.enable() is False

    disabled = []
    controller._gripper = SimpleNamespace(
        enable=lambda: True,
        read_mm=lambda: (_ for _ in ()).throw(eef.GripperFeedbackError("uncalibrated")),
        disable=lambda: disabled.append(True),
    )
    assert controller.enable() is False
    assert disabled == [True]


def test_gripper_feedback_loss_pauses_policy_without_reconnecting():
    client = eef.PiperEEFInference.__new__(eef.PiperEEFInference)
    client._action_cache = np.ones((3, 7), dtype=np.float32)
    client._chunk_tcp_base = np.ones(6)
    client._trajectory_view = None
    client._inferring = True
    client._build_observation = lambda: (_ for _ in ()).throw(eef.GripperFeedbackError("no feedback"))
    client._connect_policy = lambda: pytest.fail("should not reconnect policy on gripper feedback loss")
    client._query_policy()
    assert client._action_cache is None
    assert client._chunk_tcp_base is None
    assert client._inferring is False


def test_eef_camera_creation_accepts_dataset_resolution_without_hardware(monkeypatch):
    captured = []

    def fake_realsense(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace()

    camera_module = sys.modules[eef.create_cameras.__module__]
    monkeypatch.setattr(camera_module, "HAS_RS2", True)
    monkeypatch.setattr(camera_module, "RealSenseCameras", fake_realsense)
    monkeypatch.setattr(eef, "PiperEEFController", lambda *args, **kwargs: SimpleNamespace())
    inference = eef.PiperEEFInference(
        gripper_port=None, rs2_base_serial="camera-serial", base_rs_size=(640, 480)
    )
    assert inference._camera is not None
    assert captured[0]["width"] == 640
    assert captured[0]["height"] == 480


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


def test_piper_euler_commands_remain_continuous_across_180_degrees():
    commands = []
    feedback = SimpleNamespace(RX_axis=179800, RY_axis=55000, RZ_axis=90000)
    controller = eef.PiperEEFController.__new__(eef.PiperEEFController)
    controller._piper = SimpleNamespace(
        GetArmEndPoseMsgs=lambda: SimpleNamespace(end_pose=feedback),
        EndPoseCtrl=lambda *values: commands.append(values),
    )
    controller._last_command_euler_rad = None
    for rx_deg in (179.9, -179.9, -179.8):
        rotvec = Rotation.from_euler("xyz", [rx_deg, 55.0, 90.0], degrees=True).as_rotvec()
        controller.send_eef_command(np.r_[0.4, 0.0, 0.3, rotvec])
    rx_sent = np.array([row[3] for row in commands]) / 1000
    np.testing.assert_allclose(rx_sent, [179.9, 180.1, 180.2], atol=0.001)
    assert np.max(np.abs(np.diff(rx_sent))) < 1.0
