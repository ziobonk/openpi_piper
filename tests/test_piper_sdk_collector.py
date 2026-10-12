"""Read-only Piper SDK collector feedback behavior without robot hardware."""

from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from examples.piper.chunk_relative import collect_piper_sdk
from examples.piper.chunk_relative.infer_piper_sdk import dataset_settings
from examples.piper.chunk_relative.prepare_chunk_relative_eef import convert
from examples.piper.collection.collect_eef_udp import PickPlaceWriter
from examples.piper.collection.delete_last_raw_episode import delete_last_episode
from examples.piper.data_tools.replay_picodual_lerobot import load_chunk_relative_episode


class FakeArm:
    def __init__(self, **_kwargs):
        self.connect_kwargs = None
        self.pose_stamp = 1.0
        self.grip_stamp = 1.0

    def ConnectPort(self, **kwargs):  # noqa: N802 - match Piper SDK API
        self.connect_kwargs = kwargs

    def DisconnectPort(self):  # noqa: N802 - match Piper SDK API
        pass

    def GetArmEndPoseMsgs(self):  # noqa: N802 - match Piper SDK API
        return SimpleNamespace(
            time_stamp=self.pose_stamp, Hz=20.0,
            end_pose=SimpleNamespace(X_axis=100000, Y_axis=0, Z_axis=0, RX_axis=0, RY_axis=0, RZ_axis=90000),
        )

    def GetArmGripperMsgs(self):  # noqa: N802 - match Piper SDK API
        return SimpleNamespace(
            time_stamp=self.grip_stamp, Hz=20.0 if self.grip_stamp else 0.0,
            gripper_state=SimpleNamespace(grippers_angle=35000),
        )

    def GetCanFps(self):  # noqa: N802 - match Piper SDK API
        return 100.0


def test_read_only_connection_and_tcp_offset(monkeypatch):
    monkeypatch.setattr(collect_piper_sdk, "C_PiperInterface_V2", FakeArm)
    feedback = collect_piper_sdk.PiperFeedback("can0", np.array([0.1, 0, 0]), 70.0, 0.5)
    assert feedback.arm.connect_kwargs == {"piper_init": False}
    state, action = feedback.read()
    np.testing.assert_allclose(state, [35.0])
    np.testing.assert_allclose(action[:3], [0.1, 0.1, 0], atol=1e-6)
    np.testing.assert_allclose(feedback.last_flange_pose[:3], [0.1, 0, 0], atol=1e-6)
    feedback.close()


def test_raw_episode_preserves_flange_and_can_be_prepared_later(tmp_path):
    raw = tmp_path / "piper_sdk_raw_eef"
    writer = PickPlaceWriter(
        raw, "pick up the cube", state_dim=7, robot_type="piper_eef", raw_piper_sdk=True,
        tcp_offset_m=(0, 0, 0.1), gripper_open_width_mm=70.0,
        base_shape=(8, 8, 3), wrist_shape=(8, 8, 3),
    )
    ok, jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    for i in range(3):
        tcp = np.array([0.4 + i * 0.01, 0, 0.4, 0, 0, 0, 35 + i], dtype=np.float32)
        flange = tcp[:6].copy()
        flange[2] -= 0.1
        writer.add_frame(tcp, tcp, jpeg.tobytes(), jpeg.tobytes(), flange_pose=flange)
    writer.save_episode()
    raw_table = pq.read_table(raw / "data/chunk-000/episode_000000.parquet")
    np.testing.assert_allclose(raw_table["state"].to_pylist(), raw_table["actions"].to_pylist())
    np.testing.assert_allclose(np.asarray(raw_table["flange_pose"].to_pylist())[:, 2], 0.3)
    info, _, _, _, actions = load_chunk_relative_episode(str(raw), 0)
    assert info["source_kind"] == "piper_sdk_raw"
    assert actions.shape == (3, 7)

    prepared = tmp_path / "piper_sdk_chunk_relative"
    convert(raw, prepared, source_kind="piper_sdk")
    prepared_table = pq.read_table(prepared / "data/chunk-000/episode_000000.parquet")
    assert len(prepared_table["state"][0].as_py()) == 1
    np.testing.assert_allclose(prepared_table["actions"].to_pylist(), raw_table["actions"].to_pylist())
    import json

    prepared_info = json.loads((prepared / "meta/info.json").read_text())
    assert prepared_info["source_kind"] == "piper_sdk_read_only"
    assert prepared_info["tcp_offset_m"] == [0, 0, 0.1]
    fps, width, base_size, wrist_size, task, offset = dataset_settings(prepared)
    assert (fps, width, base_size, wrist_size, task) == (20.0, 70.0, (8, 8), (8, 8), "pick up the cube")
    np.testing.assert_allclose(offset, [0, 0, 0.1])


def test_missing_gripper_feedback_is_identified(monkeypatch):
    monkeypatch.setattr(collect_piper_sdk, "C_PiperInterface_V2", FakeArm)
    feedback = collect_piper_sdk.PiperFeedback("can0", np.zeros(3), 70.0, 0.5)
    feedback.arm.grip_stamp = 0.0
    with pytest.raises(RuntimeError, match=r"gripper: never received.*CAN: 100.0 fps"):
        feedback.read()
    feedback.close()


def test_delete_last_raw_episode_updates_counts_and_keeps_prior_episode(tmp_path):
    import json

    dataset = tmp_path / "raw"
    writer = PickPlaceWriter(
        dataset, "task", state_dim=7, robot_type="piper_eef", raw_piper_sdk=True,
        tcp_offset_m=(0, 0, 0), gripper_open_width_mm=70,
        base_shape=(8, 8, 3), wrist_shape=(8, 8, 3),
    )
    ok, jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok
    pose = np.array([0.4, 0, 0.3, 0, 0, 0, 35], dtype=np.float32)
    for _ in range(2):
        writer.add_frame(pose, pose, jpeg.tobytes(), jpeg.tobytes(), flange_pose=pose[:6])
        writer.save_episode()
    tail = dataset / "data/chunk-000/episode_000001.parquet"
    delete_last_episode(dataset, 1)
    assert tail.exists()
    delete_last_episode(dataset, 1, apply=True)
    assert not tail.exists()
    assert (dataset / "data/chunk-000/episode_000000.parquet").exists()
    info = json.loads((dataset / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["splits"]) == (1, 1, {"train": "0:1"})
    assert len((dataset / "meta/episodes.jsonl").read_text().splitlines()) == 1
