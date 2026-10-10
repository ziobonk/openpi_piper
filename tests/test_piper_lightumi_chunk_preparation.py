"""Prepare next-aligned lightUMI TCP actions without changing encoder units."""

import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.piper.chunk_relative import infer_pick_place  # noqa: E402
from examples.piper.chunk_relative.prepare_chunk_relative_eef import _fixed_list  # noqa: E402
from examples.piper.chunk_relative.prepare_lightumi_chunk_relative import convert  # noqa: E402
from examples.piper.data_tools.replay_lightumi_tcp_relative import ROTATION_NAMES  # noqa: E402


def test_next_frame_tcp_becomes_same_frame_and_gripper_stays_radians(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    (source / "meta").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "videos/chunk-000/observation.images.camera").mkdir(parents=True)
    video = source / "videos/chunk-000/observation.images.camera/episode_000000.mp4"
    video.write_bytes(b"video-placeholder")
    info = {
        "robot_type": "lightumi",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": 1,
        "total_frames": 3,
        "total_videos": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {"shape": [10], "names": ["left_tcp_x_m", "left_tcp_y_m", "left_tcp_z_m", *ROTATION_NAMES, "gripper_angle_rad"]},
            "observation.state": {"shape": [1], "names": ["encoder_angle_rad"]},
        },
        "lightumi_conversion": {
            "action": "next aligned absolute source-tracking-world TCP xyz + rotation-6D, then next encoder angle in radians",
            "rotation_6d": "columns_grouped:[r00,r10,r20,r01,r11,r21]",
        },
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/stats.json").write_text("{}")
    (source / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")
    (source / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "把方块放到纸盒子里"}) + "\n")
    (source / "meta/episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": {}}) + "\n")
    (source / "meta/lightumi_sources.jsonl").write_text(
        json.dumps({"episode_index": 0, "frames": 3, "action_is_next_aligned_tcp": True, "last_action_repeated": True}) + "\n"
    )
    next_actions = np.array([
        [1.0, 2, 3, 1, 0, 0, 0, 1, 0, 0.2],
        [1.1, 2, 3, 1, 0, 0, 0, 1, 0, 0.3],
        [1.1, 2, 3, 1, 0, 0, 0, 1, 0, 0.3],
    ], dtype=np.float32)
    encoder = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    table = pa.table({
        "frame_index": pa.array([0, 1, 2]),
        "episode_index": pa.array([0, 0, 0]),
        "action": _fixed_list(next_actions),
        "observation.state": _fixed_list(encoder),
    })
    pq.write_table(table, source / "data/chunk-000/episode_000000.parquet")

    convert(source, output)
    prepared = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    actions = np.asarray(prepared["actions"].to_pylist())
    states = np.asarray(prepared["state"].to_pylist())
    np.testing.assert_allclose(actions[:, :3], [[1, 2, 3], [1, 2, 3], [1.1, 2, 3]])
    np.testing.assert_allclose(actions[:, 6], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(states[:, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(np.asarray(prepared["action"].to_pylist()), next_actions)
    assert (output / video.relative_to(source)).read_bytes() == b"video-placeholder"
    prepared_info = json.loads((output / "meta/info.json").read_text())
    assert prepared_info["gripper_mapping"]["applied_to_dataset"] is False
    assert prepared_info["first_frame_tcp_fill"] == "next_frame_pose"
    fps, (closed, opened), open_width_mm, prompt = infer_pick_place.training_metadata(output)
    assert fps == 30
    np.testing.assert_allclose([closed, opened], [0.1, 0.3])
    assert open_width_mm == 20
    assert prompt == "把方块放到纸盒子里"

    captured = []
    monkeypatch.setattr(infer_pick_place.inference_eef, "main", captured.append)
    monkeypatch.setattr(
        sys, "argv",
        ["infer_pick_place.py", "--dataset", str(output), "--base-camera", "serial", "--wrist-camera", "0"],
    )
    infer_pick_place.main()
    assert len(captured) == 1
    assert captured[0].control_freq == 30
    np.testing.assert_allclose(captured[0].model_gripper_range, [0.1, 0.3])
    assert captured[0].gripper_open_width_mm == 20
    assert captured[0].prompt == "把方块放到纸盒子里"

    new_prompt = "<control mode> end effector <control mode>pick up the yellow cube then drop it into the box."
    retagged = tmp_path / "retagged"
    convert(source, retagged, task=new_prompt)
    assert json.loads((retagged / "meta/tasks.jsonl").read_text())["task"] == new_prompt
    assert json.loads((retagged / "meta/episodes.jsonl").read_text())["tasks"] == [new_prompt]
    assert infer_pick_place.training_metadata(retagged)[-1] == new_prompt
