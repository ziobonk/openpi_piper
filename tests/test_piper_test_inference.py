"""TCP-only inference uses the training prompt and never commands a gripper."""
# ruff: noqa: E402, SLF001

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.piper.chunk_relative import infer_test
from examples.piper.runtime import inference_eef as eef
from examples.piper.runtime.inference_eef_async import PiperEEFAsyncInference


def test_tcp_only_client_sends_zero_state_and_preserves_tcp_anchor():
    tcp = np.array([0.4, 0.1, 0.3, 0.0, 0.2, 0.1])
    for cls in (eef.PiperEEFInference, PiperEEFAsyncInference):
        client = cls.__new__(cls)
        client._model_action_frame = "chunk_relative"
        client._model_state_value = 0.0
        client._robot = SimpleNamespace(
            get_eef_pose=lambda: tcp.copy(),
            get_state=lambda: (_ for _ in ()).throw(AssertionError("gripper feedback must not be read")),
        )
        client._camera = None
        client._default_prompt = "touch the blocks"
        client._interactive = False
        observation = client._build_observation()
        np.testing.assert_array_equal(observation["observation/state"], [0.0])
        np.testing.assert_array_equal(client._chunk_tcp_base, tcp)
        assert observation["prompt"] == "touch the blocks"


def test_disabled_controller_gripper_sends_no_command():
    robot = eef.PiperEEFController.__new__(eef.PiperEEFController)
    robot._control_gripper = False
    robot._gripper = None
    robot._piper = SimpleNamespace(GripperCtrl=lambda *_: (_ for _ in ()).throw(AssertionError("gripper command sent")))
    robot.send_gripper_command(0.0)


def test_tcp_only_setup_never_opens_dm_gripper(monkeypatch):
    captured = {}
    monkeypatch.setattr(eef, "_HAS_DM_GRIPPER", True)
    monkeypatch.setattr(
        eef, "DmGripperController",
        lambda **_: (_ for _ in ()).throw(AssertionError("DM gripper was constructed")),
    )
    monkeypatch.setattr(eef, "PiperEEFController", lambda *_, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(eef, "create_cameras", lambda **_: None)
    eef.PiperEEFInference(
        model_action_frame="chunk_relative",
        model_state_value=0.0,
        control_gripper=False,
        gripper_port="/dev/ttyACM1",
    )
    assert captured["control_gripper"] is False
    assert captured["gripper"] is None


def test_test_entry_passes_training_metadata_and_disables_gripper(tmp_path, monkeypatch):
    dataset = tmp_path / "test_chunk_relative"
    (dataset / "meta").mkdir(parents=True)
    info = {
        "robot_type": "piper_chunk_relative_eef",
        "source_kind": "lightumi",
        "pose_coordinate_frame": "source_tracking_world_tcp",
        "coordinate_transform_applied": False,
        "fps": 30,
        "gripper_mapping": {"available": False, "placeholder_value": 0.0},
        "features": {
            "state": {"shape": [1]},
            "actions": {"shape": [7]},
            "observation.images.global": {"shape": [480, 640, 3]},
        },
    }
    (dataset / "meta/info.json").write_text(json.dumps(info))
    prompt = "<control mode> end effector <control mode> touch the blocks"
    (dataset / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")
    captured = []
    monkeypatch.setattr(infer_test.inference_eef, "main", captured.append)
    monkeypatch.setattr(sys, "argv", [
        "infer_test.py", "--dataset", str(dataset), "--base-camera", "serial", "--wrist-camera", "0",
        "--steps", "3", "--tcp-offset-m", "0", "0", "0.22",
    ])
    infer_test.main()
    assert len(captured) == 1
    args = captured[0]
    assert args.prompt == prompt
    assert args.model_action_frame == "chunk_relative"
    assert args.model_state_value == 0.0
    assert args.no_gripper_control
    assert args.control_freq == 30
    assert args.base_rs_size == [640, 480]
    assert args.tcp_offset_m == [0.0, 0.0, 0.22]
    assert args.interp_freq == 30
    assert args.max_pos_speed == 0.05
    assert args.max_rot_speed == 0.3
