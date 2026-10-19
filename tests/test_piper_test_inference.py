"""TCP-only inference uses the training prompt and never commands a gripper."""
# ruff: noqa: E402, SLF001

import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.piper.chunk_relative import capture_wrist_diagnostics
from examples.piper.chunk_relative import infer_test
from examples.piper.runtime import camera_utils
from examples.piper.runtime import inference_eef as eef
from examples.piper.runtime.camera_utils import OpenCVCameras
from examples.piper.runtime.camera_utils import crop_resize_bgr
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
            "observation.images.camera": {"shape": [480, 640, 3]},
        },
    }
    (dataset / "meta/info.json").write_text(json.dumps(info))
    prompt = "<control mode> end effector <control mode> touch the blocks"
    (dataset / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")
    captured = []
    monkeypatch.setattr(infer_test.inference_eef, "main", captured.append)
    monkeypatch.setattr(sys, "argv", [
        "infer_test.py", "--dataset", str(dataset), "--base-camera", "serial", "--wrist-camera", "0",
        "--wrist-capture-size", "1280", "720", "--wrist-crop", "100", "50", "960", "640",
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
    assert args.wrist_cv_size == [1280, 720]
    assert args.wrist_cv_crop == [100, 50, 960, 640]
    assert args.wrist_cv_output_size == [640, 480]
    assert args.wrist_cv_fourcc == "YUYV"
    assert args.wrist_cv_warmup_frames == 60
    assert args.wrist_cv_force_auto_exposure
    assert args.wrist_cv_force_auto_white_balance
    assert args.tcp_offset_m == [0.0, 0.0, 0.22]
    assert args.interp_freq == 30
    assert args.max_pos_speed == 0.05
    assert args.max_rot_speed == 0.3
    assert args.inference_log.startswith("local/runs/outputs/test_inference_logs/")
    monkeypatch.setattr(sys, "argv", [
        "infer_test.py", "--dataset", str(dataset), "--base-camera", "serial", "--wrist-camera", "0",
        "--wrist-crop", "0", "0", "640", "480",
        "--wrist-exposure-ms", "25", "--wrist-white-balance-k", "4600",
        "--inference-log", str(tmp_path / "custom_log"),
    ])
    infer_test.main()
    manual = captured[1]
    assert manual.wrist_exposure == 25.0
    assert manual.wrist_cv_white_balance_k == 4600
    assert not manual.wrist_cv_force_auto_exposure
    assert not manual.wrist_cv_force_auto_white_balance
    assert manual.inference_log == str(tmp_path / "custom_log")


def test_wrist_crop_uses_raw_frame_coordinates_and_dataset_size():
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    frame[1:5, 2:6] = (10, 20, 30)
    prepared = crop_resize_bgr(frame, (2, 1, 4, 4), (4, 4))
    assert prepared.shape == (4, 4, 3)
    np.testing.assert_array_equal(prepared, np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8))
    assert crop_resize_bgr(frame, (2, 1, 4, 4), (2, 2)).shape == (2, 2, 3)
    with np.testing.assert_raises_regex(ValueError, "exceeds captured frame"):
        crop_resize_bgr(frame, (5, 1, 4, 4), (4, 4))


def test_runtime_wrist_image_crops_before_model_resize():
    camera = OpenCVCameras.__new__(OpenCVCameras)
    camera._lock = threading.Lock()
    frame = np.full((6, 8, 3), (255, 0, 0), dtype=np.uint8)
    frame[1:5, 2:6] = (0, 0, 255)
    camera._latest = {"wrist": frame}
    camera._wrist_crop = (2, 1, 4, 4)
    camera._wrist_output_size = (4, 4)
    image = camera.get_wrist()
    np.testing.assert_array_equal(image[112, 112], (255, 0, 0))


def test_preview_only_reads_wrist_camera_and_writes_comparison_images(tmp_path, monkeypatch):
    class FakeCapture:
        def __init__(self):
            self.reads = 0
            self.values = {}

        def isOpened(self):  # noqa: N802 - OpenCV method name
            return True

        def set(self, prop, value):
            self.values[prop] = value
            return True

        def get(self, prop):
            return self.values.get(prop, 1.0)

        def read(self):
            self.reads += 1
            return True, np.full((6, 8, 3), (10, 20, 30), dtype=np.uint8)

        def release(self):
            pass

    cap = FakeCapture()
    monkeypatch.setattr(infer_test.cv2, "VideoCapture", lambda *_: cap)
    monkeypatch.setattr(infer_test.shutil, "which", lambda *_: None)
    infer_test.preview_wrist(tmp_path, 0, (8, 6), (2, 1, 4, 4), (4, 4), tmp_path / "preview")
    assert cap.reads == 60
    assert (tmp_path / "preview/raw.png").exists()
    assert (tmp_path / "preview/prepared.png").exists()
    assert (tmp_path / "preview/model_input.png").exists()
    report = json.loads((tmp_path / "preview/preview_report.json").read_text())
    assert report["actual_capture_size"] == [8, 6]
    assert report["v4l2_controls"] is None


def test_manual_camera_controls_apply_requested_values():
    class FakeCapture:
        def __init__(self):
            self.values = {}

        def set(self, prop, value):
            self.values[prop] = value
            return True

        def get(self, prop):
            return self.values[prop]

    cap = FakeCapture()
    OpenCVCameras._set_exposure(cap, "wrist", 25.0)
    OpenCVCameras._set_white_balance(cap, "wrist", 4600)
    assert cap.values[cv2.CAP_PROP_AUTO_EXPOSURE] == 1
    assert cap.values[cv2.CAP_PROP_EXPOSURE] == 250
    assert cap.values[cv2.CAP_PROP_AUTO_WB] == 0
    assert cap.values[cv2.CAP_PROP_WB_TEMPERATURE] == 4600


def test_auto_exposure_uses_v4l2_mode_and_checks_readback():
    class FakeCapture:
        def __init__(self):
            self.mode = 1

        def set(self, prop, value):
            assert prop == cv2.CAP_PROP_AUTO_EXPOSURE
            self.mode = value
            return True

        def get(self, prop):
            assert prop == cv2.CAP_PROP_AUTO_EXPOSURE
            return self.mode

    cap = FakeCapture()
    OpenCVCameras._set_auto_exposure(cap, "wrist")
    assert cap.mode == 3

    cap.get = lambda prop: 1
    with np.testing.assert_raises_regex(RuntimeError, "requested automatic exposure mode 3, got 1"):
        OpenCVCameras._set_auto_exposure(cap, "wrist")


def test_wrist_capture_requires_requested_pixel_format_and_resolution(monkeypatch):
    class FakeCapture:
        def __init__(self):
            self.values = {}

        def isOpened(self):  # noqa: N802 - OpenCV method name
            return True

        def set(self, prop, value):
            self.values[prop] = value
            return True

        def get(self, prop):
            return self.values.get(prop, 0)

        def read(self):
            return True, np.full((480, 640, 3), 120, dtype=np.uint8)

        def release(self):
            pass

    cap = FakeCapture()
    monkeypatch.setattr(camera_utils.cv2, "VideoCapture", lambda *_: cap)
    camera = OpenCVCameras(wrist_id=0, wrist_size=(640, 480), wrist_fourcc="YUYV", wrist_warmup_frames=3)
    assert cap.values[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"YUYV")
    assert cap.values[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert cap.values[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert cap.values[cv2.CAP_PROP_FPS] == 30
    camera.start()
    try:
        assert camera._wrist_frame_count >= 3
        assert np.any(camera.get_wrist())
    finally:
        camera.stop()


def test_capture_script_saves_raw_frame_and_report(tmp_path, monkeypatch):
    class FakeCapture:
        def isOpened(self):  # noqa: N802 - OpenCV method name
            return True

        def set(self, *_):
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
                    cv2.CAP_PROP_FPS: 30}.get(prop, 0)

        def read(self):
            return True, np.full((4, 6, 3), 120, dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(capture_wrist_diagnostics.cv2, "VideoCapture", lambda *_: FakeCapture())
    monkeypatch.setattr(capture_wrist_diagnostics, "v4l2_report", lambda *_: {"controls": {"stdout": "auto_exposure value=3"}})
    args = SimpleNamespace(
        camera=0, device=None, output_dir=tmp_path / "capture", fourcc=None,
        width=640, height=480, fps=30, warmup_frames=1, frames=2, dataset=None,
    )
    report = capture_wrist_diagnostics.capture(args)
    assert report["frames"]["raw_00.png"]["size"] == [6, 4]
    assert (args.output_dir / "raw_00.png").exists()
    assert (args.output_dir / "capture_report.json").exists()
