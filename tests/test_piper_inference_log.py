"""Live inference logs preserve model input/output and actual Piper commands."""
# ruff: noqa: SLF001

import json
from types import SimpleNamespace

import cv2
import numpy as np

from examples.piper.runtime import inference_eef as eef
from examples.piper.runtime.inference_log import InferenceLog


def test_query_images_full_model_output_and_interpolated_sdk_commands_are_linked(tmp_path, monkeypatch):
    log = InferenceLog(tmp_path / "run")
    client = eef.PiperEEFInference.__new__(eef.PiperEEFInference)
    client._inference_log = log
    client._trajectory_view = None
    client._action_cache = None
    client._chunk_tcp_base = None
    client._model_action_frame = "chunk_relative"
    client._exec_horizon = 1
    client._model_gripper_range = None
    client._gripper_prediction_scale = 1.0
    client._binary_gripper = False
    client._gripper_open_width_mm = 20.0
    client._inferring = True
    client._connect_policy = lambda: None
    global_rgb = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
    wrist_rgb = np.full((4, 6, 3), [70, 80, 90], dtype=np.uint8)
    anchor = np.array([0.4, 0, 0.3, 0, 0, 0], dtype=np.float64)

    def observation():
        client._chunk_tcp_base = anchor.copy()
        return {
            "observation/state": np.array([0], dtype=np.float32),
            "observation/image": global_rgb,
            "observation/wrist_image": wrist_rgb,
            "prompt": "touch the block",
        }

    raw_actions = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0.02, 0, 0, 0, 0, 0, 0],
        [0.03, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    client._build_observation = observation
    client._policy_client = SimpleNamespace(infer=lambda _: {"actions": raw_actions})
    client._query_policy()
    assert client._trace_query_id == 0
    np.testing.assert_array_equal(client._action_cache, raw_actions[1:])

    sdk_commands = []
    robot = eef.PiperEEFController.__new__(eef.PiperEEFController)
    robot._piper = SimpleNamespace(EndPoseCtrl=lambda *values: sdk_commands.append(values))
    robot._last_command_euler_rad = np.zeros(3)
    robot._control_gripper = False
    robot._end_pose_command_callback = client._record_sent_command
    client._robot = robot
    client._trace_execution_index = 0
    client._use_interp = True
    client._max_pos_speed = 0.1
    client._max_rot_speed = 1.0
    client._interp_freq = 10.0
    monkeypatch.setattr(eef.time, "sleep", lambda _: None)
    client._send_eef_cmd(anchor, anchor + np.array([0.02, 0, 0, 0, 0, 0]), 0, 0.1)
    log.close()

    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == ["query", "command", "command"]
    query = events[0]
    assert query["query_id"] == 0
    assert query["first_action_skipped"] is True
    assert query["execute_steps"] == 1
    assert query["model_actions"] == raw_actions.tolist()
    assert query["tcp_anchor"] == anchor.tolist()
    assert query["state"] == [0.0]
    for name, expected in (("global", global_rgb), ("wrist", wrist_rgb)):
        saved_bgr = cv2.imread(str(tmp_path / "run" / query["images"][name]))
        np.testing.assert_array_equal(cv2.cvtColor(saved_bgr, cv2.COLOR_BGR2RGB), expected)
    for index, (event, sdk) in enumerate(zip(events[1:], sdk_commands, strict=True), start=1):
        assert event["query_id"] == 0
        assert event["execution_index"] == 0
        assert event["model_action_index"] == 1
        assert event["interpolation_index"] == index
        assert event["interpolation_count"] == 2
        assert event["end_pose_ctrl"] == list(sdk)
    np.testing.assert_allclose([event["tcp_pose"][0] for event in events[1:]], [0.41, 0.42])
