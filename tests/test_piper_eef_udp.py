"""UDP 单臂采集器的数据格式与坐标计算测试，无需连接硬件。"""

import errno
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow.parquet as parquet
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "piper"))

import collect_eef_udp as collector
from collect_eef_udp import BASE_SHAPE
from collect_eef_udp import WRIST_SHAPE
from collect_eef_udp import PickPlaceWriter
from collect_eef_udp import _record_pose
from collect_eef_udp import apply_delta
from collect_eef_udp import parse_arm_sample
from collect_eef_udp import tracker_delta
import inference_eef as eef


def _jpeg(shape):
    image = np.zeros(shape, dtype=np.uint8)
    image[..., 1] = 42
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


def test_packet_parser_selects_arm_and_rejects_invalid_pose():
    payload = {
        "type": "lightumi.teleop_sender",
        "version": 1,
        "sequence": 17,
        "arms": [
            {"index": 0, "pose": {"valid": 0}},
            {
                "index": 1,
                "pose": {"valid": 1, "position_m": [1, 2, 3], "quaternion_xyzw": [0, 0, 0, 2]},
                "gripper": {"valid": 1, "angle_deg": 12},
            },
        ],
    }
    data = json.dumps(payload).encode()
    assert parse_arm_sample(data, 0, 1.0) is None
    sample = parse_arm_sample(data, 1, 1.0)
    assert sample.sequence == 17
    np.testing.assert_allclose(sample.quaternion_xyzw, [0, 0, 0, 1])
    assert sample.gripper_angle_deg == 12
    payload["arms"][1]["pose"]["quaternion_xyzw"] = [0, 0, 0, 0]
    assert parse_arm_sample(json.dumps(payload).encode(), 1, 1.0) is None


def test_right_tracker_is_default_and_arm_can_be_overridden():
    assert collector.parse_args(["--udp_test"]).arm_index == 1
    assert collector.parse_args(["--udp_test", "--arm_index", "0"]).arm_index == 0


def test_udp_receiver_distinguishes_raw_packets_from_selected_pose(monkeypatch):
    def packet(index, sequence):
        return json.dumps(
            {
                "type": "lightumi.teleop_sender",
                "version": 1,
                "sequence": sequence,
                "arms": [
                    {
                        "index": index,
                        "pose": {"valid": 1, "position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
                    }
                ],
            }
        ).encode()

    class FakeSocket:
        def __init__(self):
            self.packets = [packet(0, 1), packet(1, 2)]

        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            pass

        def settimeout(self, _timeout):
            pass

        def recvfrom(self, _size):
            if self.packets:
                return self.packets.pop(0), ("sender", 38487)
            raise OSError(errno.EBADF, "done")

        def close(self):
            pass

    monkeypatch.setattr(collector.socket, "socket", lambda *_args: FakeSocket())
    receiver = collector.UDPPoseReceiver("0.0.0.0", 5005, 1)
    receiver.start()
    receiver._thread.join(timeout=1)
    assert receiver.latest().sequence == 2
    last_packet_at, packet_count, valid_count, dropped_print_count = receiver.stats()
    assert last_packet_at is not None
    assert (packet_count, valid_count, dropped_print_count) == (2, 1, 0)
    receiver.stop()


def test_tracker_delta_is_frame_mapped_and_speed_limited():
    dp, dr = tracker_delta(
        np.zeros(3),
        [0, 0, 0, 1],
        np.array([1.0, 0, 0]),
        Rotation.from_rotvec([1, 0, 0]).as_quat(),
        pos_scale=1,
        rot_scale=1,
        max_pos_step=0.01,
        max_rot_step=0.02,
        rotation_only=False,
    )
    np.testing.assert_allclose(dp, [0, 0.01, 0])
    np.testing.assert_allclose(dr, [0, 0.02, 0])
    pose = apply_delta(np.zeros(6), dp, dr)
    np.testing.assert_allclose(pose[:3], dp)
    np.testing.assert_allclose(pose[3:], dr)


def test_record_pose_keeps_compensated_robot_coordinates():
    arm_tcp_pose = np.array([0.12, -0.34, 0.56, 0.7, -0.8, 0.9])
    recorded = _record_pose(arm_tcp_pose, 12.3)
    np.testing.assert_allclose(recorded[:6], arm_tcp_pose)
    assert recorded[6] == pytest.approx(12.3)


def test_recording_uses_controller_tcp_offset():
    end_pose = SimpleNamespace(X_axis=0, Y_axis=0, Z_axis=0, RX_axis=0, RY_axis=0, RZ_axis=0)
    controller = eef.PiperEEFController.__new__(eef.PiperEEFController)
    controller._piper = SimpleNamespace(GetArmEndPoseMsgs=lambda: SimpleNamespace(end_pose=end_pose))
    recorded = _record_pose(controller.get_eef_pose(), 0)
    np.testing.assert_allclose(recorded[:3], eef.GRIPPER_TCP_OFFSET_M)


def test_image_stats_match_rgb_pixels_after_jpeg_decode():
    shape = (8, 10, 3)
    first = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    second = np.full(shape, [20, 80, 160], dtype=np.uint8)
    jpegs = []
    decoded_rgb = []
    for bgr in (first, second):
        ok, encoded = cv2.imencode(".jpg", bgr)
        assert ok
        data = encoded.tobytes()
        jpegs.append(data)
        decoded_rgb.append(
            cv2.cvtColor(cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        )

    pixels = np.concatenate([image.reshape(-1, 3) for image in decoded_rgb]).astype(np.float64)
    stats = collector._image_stats(jpegs, shape)
    for name, expected in (
        ("min", pixels.min(axis=0)),
        ("max", pixels.max(axis=0)),
        ("mean", pixels.mean(axis=0)),
        ("std", pixels.std(axis=0)),
    ):
        np.testing.assert_allclose(np.asarray(stats[name]).reshape(3), expected, atol=1e-10)
    assert stats["count"] == [2]


def test_refuses_to_mix_robot_frame_with_original_pick_place():
    root = Path(__file__).resolve().parents[1] / "pick_place"
    if not (root / "meta/info.json").exists():
        pytest.skip("本地未提供 pick_place 参考数据集")
    with pytest.raises(ValueError, match="机械臂基坐标系"):
        PickPlaceWriter(root, "test task")


def test_writer_matches_pick_place_schema_and_appends(tmp_path):
    root = tmp_path / "eef"
    base = _jpeg(BASE_SHAPE)
    wrist = _jpeg(WRIST_SHAPE)
    writer = PickPlaceWriter(root, "task one")
    assert not root.exists()
    writer.add_frame(np.zeros(7), np.ones(7), base, wrist)
    writer.add_frame(np.ones(7), np.full(7, 2), base, wrist)
    first = writer.save_episode()

    reference_path = Path(__file__).resolve().parents[1] / "pick_place/data/chunk-000/episode_000000.parquet"
    result = parquet.read_schema(first)
    if reference_path.exists():
        assert result.equals(parquet.read_schema(reference_path), check_metadata=False)
    else:
        assert result.names == [
            "state",
            "actions",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "image",
            "wrist_image",
        ]
    rows = parquet.read_table(first).to_pylist()
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[0]["actions"] == pytest.approx([1] * 7)
    assert rows[0]["image"]["bytes"].startswith(b"\xff\xd8")

    writer = PickPlaceWriter(root, "task two")
    writer.add_frame(np.full(7, 3), np.full(7, 4), base, wrist)
    second = writer.save_episode()
    second_row = parquet.read_table(second).to_pylist()[0]
    assert (second_row["episode_index"], second_row["index"], second_row["task_index"]) == (1, 2, 1)
    info = json.loads((root / "meta/info.json").read_text())
    reference_info = Path(__file__).resolve().parents[1] / "pick_place/meta/info.json"
    if reference_info.exists():
        assert info["features"] == json.loads(reference_info.read_text())["features"]
    assert info["robot_type"] == "piper_eef"
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"], info["fps"]) == (2, 3, 2, 20)
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["state"]["count"] == [3]
    assert stats["image"]["count"] == [3]


def test_background_save_allows_next_episode_and_flushes_before_exit(monkeypatch, tmp_path):
    root = tmp_path / "eef"
    writer = PickPlaceWriter(root, "task")
    saver = collector.BackgroundEpisodeSaver(writer)
    base = _jpeg(BASE_SHAPE)
    wrist = _jpeg(WRIST_SHAPE)
    started = threading.Event()
    release = threading.Event()
    real_write = writer.write_episode

    def delayed_write(frames):
        started.set()
        assert release.wait(timeout=5)
        return real_write(frames)

    monkeypatch.setattr(writer, "write_episode", delayed_write)
    try:
        writer.add_frame(np.zeros(7), np.ones(7), base, wrist)
        assert saver.save_episode() == 0
        assert started.wait(timeout=1)
        assert not (root / "meta/info.json").exists()

        writer.add_frame(np.ones(7), np.full(7, 2), base, wrist)
        assert saver.next_episode == 1
        assert writer.frame_count == 1
        assert saver.save_episode() == 1
    finally:
        release.set()
        saver.close()

    assert writer.next_episode == 2
    assert json.loads((root / "meta/info.json").read_text())["total_frames"] == 2


def test_no_udp_mode_records_feedback_without_socket_or_eef_commands(monkeypatch, tmp_path):
    args = collector.parse_args(
        [
            "--no_udp",
            "--output",
            str(tmp_path / "data"),
            "--base_cv_id",
            "0",
            "--usb_wrist",
            "1",
            "--no_preview",
            "--gripper_port",
            "",
        ]
    )
    assert args.gripper_source == "keys"
    frames = []
    sent_gripper = []
    base_jpeg = _jpeg(BASE_SHAPE)
    wrist_jpeg = _jpeg(WRIST_SHAPE)

    class FakeWriter:
        next_episode = 0
        task = "test"

        def __init__(self, *_args):
            self._staged = []

        def add_frame(self, state, action, _base, _wrist):
            frames.append((state, action))
            self._staged.append((state, action))

        def take_frames(self):
            staged = self._staged
            self._staged = []
            return staged

        def write_episode(self, _frames):
            pass

    class FakeRobot:
        def __init__(self, *_args, **_kwargs):
            pass

        def enable(self, **_kwargs):
            return True

        def get_gripper_mm(self):
            return 3.0

        def get_eef_pose(self):
            return np.array([0.1, 0.2, 0.3, 0, 0, 0])

        def get_state(self):
            return np.array([0.1, 0.2, 0.3, 0, 0, 0, 3.0])

        def send_eef_command(self, _pose):
            pytest.fail("--no_udp 不应下发 EEF 指令")

        def send_gripper_command(self, width):
            sent_gripper.append(width)

        def close_gripper_transport(self):
            pass

    class FakeCamera:
        def __init__(self, name, shape, **_kwargs):
            self.name = name
            self.frame = np.zeros(shape, dtype=np.uint8)

        def start(self):
            pass

        def latest(self):
            return self.frame, time.monotonic(), base_jpeg if self.name == "base" else wrist_jpeg

        def stop(self):
            pass

    class KeyEvents:
        def __init__(self):
            self.keys = ["c", None, "q", None]

        def get_nowait(self):
            key = self.keys.pop(0)
            if key is None:
                raise queue.Empty
            return key

    class FakeKeys:
        def __init__(self):
            self.events = KeyEvents()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    def reject_socket(*_args):
        pytest.fail("--no_udp 不应创建 UDP 接收器")

    monkeypatch.setattr(collector, "UDPPoseReceiver", reject_socket)
    monkeypatch.setattr(collector, "PickPlaceWriter", FakeWriter)
    monkeypatch.setattr(collector.eef, "PiperEEFController", FakeRobot)
    monkeypatch.setattr(collector, "LatestCamera", FakeCamera)
    monkeypatch.setattr(collector, "TerminalKeys", FakeKeys)
    monkeypatch.setattr(collector.cv2, "destroyAllWindows", lambda: None)
    collector.run(args)

    assert len(frames) == 1
    np.testing.assert_allclose(frames[0][0], frames[0][1])
    assert sent_gripper == [3.0]
