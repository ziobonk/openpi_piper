#!/usr/bin/env python3
"""Prepare next-aligned lightUMI TCP data for same-frame chunk-relative training.

An available encoder angle stays in radians. Datasets without an encoder use a
zero placeholder for the required one-dimensional state and seventh action.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.piper.chunk_relative.prepare_chunk_relative_eef import _fixed_list  # noqa: E402
from examples.piper.chunk_relative.prepare_chunk_relative_eef import _stats  # noqa: E402
from examples.piper.data_tools.replay_lightumi_tcp_relative import ROTATION_NAMES  # noqa: E402
from examples.piper.data_tools.replay_lightumi_tcp_relative import decode_tcp_actions  # noqa: E402


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _load_rows(path: Path) -> dict[int, dict]:
    return {int(row["episode_index"]): row for row in map(json.loads, path.read_text().splitlines())}


def convert(source: Path, output: Path, *, robot_open_width_mm: float = 20.0, task: str | None = None) -> None:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output must be separate directories")
    if not np.isfinite(robot_open_width_mm) or robot_open_width_mm <= 0:
        raise ValueError("robot_open_width_mm must be positive and finite")
    if task is not None and not task.strip():
        raise ValueError("task must not be empty")

    info = json.loads((source / "meta/info.json").read_text())
    features = info.get("features", {})
    action_shape = features.get("action", {}).get("shape")
    has_encoder = action_shape == [10]
    expected_actions = ["left_tcp_x_m", "left_tcp_y_m", "left_tcp_z_m", *ROTATION_NAMES]
    if has_encoder:
        expected_actions.append("gripper_angle_rad")
    if info.get("robot_type") != "lightumi" or action_shape not in ([9], [10]):
        raise ValueError("expected lightUMI 9D/10D action dataset")
    if features["action"].get("names") != expected_actions:
        raise ValueError("unexpected lightUMI action column names")
    if has_encoder:
        if features.get("observation.state", {}).get("shape") != [1]:
            raise ValueError("expected one source encoder angle in observation.state")
    elif "observation.state" in features:
        raise ValueError("9D dataset must not declare an encoder state")
    conversion = info.get("lightumi_conversion", {})
    if (
        "next aligned absolute source-tracking-world TCP" not in conversion.get("action", "")
        or conversion.get("rotation_6d") != "columns_grouped:[r00,r10,r20,r01,r11,r21]"
    ):
        raise ValueError("expected next-aligned source-tracking-world TCP actions")
    if not np.isfinite(info.get("fps", np.nan)) or info["fps"] <= 0:
        raise ValueError("source FPS must be positive")

    episodes = _load_rows(source / "meta/episodes.jsonl")
    sources = _load_rows(source / "meta/lightumi_sources.jsonl")
    episode_stats = _load_rows(source / "meta/episodes_stats.jsonl")
    if len(episodes) != info["total_episodes"] or episodes.keys() != sources.keys() or episodes.keys() != episode_stats.keys():
        raise ValueError("episode metadata indices disagree")
    if task is not None:
        tasks = [json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines()]
        if len(tasks) != 1 or tasks[0]["task_index"] != 0 or any(row.get("task_index", 0) != 0 for row in episodes.values()):
            raise ValueError("task override requires a single-task dataset")

    source_stats = json.loads((source / "meta/stats.json").read_text())
    all_states, all_actions = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        shutil.copytree(source / "meta", staging / "meta")
        if task is not None:
            (staging / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
            with (staging / "meta/episodes.jsonl").open("w") as stream:
                for episode in sorted(episodes):
                    stream.write(json.dumps({**episodes[episode], "tasks": [task]}, ensure_ascii=False) + "\n")
        if (source / "videos").exists():
            shutil.copytree(source / "videos", staging / "videos", copy_function=_link_or_copy)
        if len(list((staging / "videos").rglob("*.mp4"))) != int(info.get("total_videos", 0)):
            raise ValueError("video count does not match source metadata")

        for episode in sorted(episodes):
            row = sources[episode]
            if row.get("action_is_next_aligned_tcp") is not True or row.get("last_action_repeated") is not True:
                raise ValueError(f"episode {episode} lacks required action alignment markers")
            path = source / info["data_path"].format(
                episode_chunk=episode // int(info["chunks_size"]), episode_index=episode
            )
            table = pq.read_table(path)
            count = int(episodes[episode]["length"])
            if len(table) != count or count < 3 or row.get("frames") != count:
                raise ValueError(f"episode {episode} length mismatch or too short")
            if not np.array_equal(np.asarray(table["frame_index"].to_pylist()), np.arange(count)):
                raise ValueError(f"episode {episode} frame indices are not contiguous")
            raw_actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            encoder = (
                np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
                if has_encoder else np.zeros((count, 1), dtype=np.float64)
            )
            if raw_actions.shape != (count, action_shape[0]) or encoder.shape != (count, 1):
                raise ValueError(f"episode {episode} has invalid action/state shape")
            if not np.isfinite(raw_actions).all() or not np.isfinite(encoder).all():
                raise ValueError(f"episode {episode} has nonfinite action/state")
            if not np.allclose(raw_actions[-1], raw_actions[-2], rtol=0, atol=1e-6):
                raise ValueError(f"episode {episode} last action must repeat")
            # action[t] is the next frame's pose. The first frame has no source
            # pose; use the first available pose once, then shift all others back.
            next_pose = decode_tcp_actions(raw_actions[:, :9])
            same_frame_pose = np.vstack((next_pose[:1], next_pose[:-1]))
            actions = np.column_stack((same_frame_pose, encoder[:, 0])).astype(np.float32)
            state = encoder.astype(np.float32)
            table = table.append_column("state", _fixed_list(state))
            table = table.append_column("actions", _fixed_list(actions))
            destination = staging / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            episode_stats[episode]["stats"]["state"] = _stats(state)
            episode_stats[episode]["stats"]["actions"] = _stats(actions)
            all_states.append(state)
            all_actions.append(actions)

        states = np.concatenate(all_states)
        actions = np.concatenate(all_actions)
        if len(states) != int(info["total_frames"]):
            raise ValueError("total frame count mismatch")
        grip_min, grip_max = float(states.min()), float(states.max())
        if has_encoder and grip_max <= grip_min:
            raise ValueError("source encoder angle has no variation")
        info["robot_type"] = "piper_chunk_relative_eef"
        info["source_kind"] = "lightumi"
        info["pose_coordinate_frame"] = "source_tracking_world_tcp"
        info["coordinate_transform_applied"] = False
        info["source_action_semantics"] = (
            "source action[t] is next-frame TCP; prepared actions[t] is same-frame TCP; "
            "episode first pose is filled from the first available next-frame action"
        )
        info["first_frame_tcp_fill"] = "next_frame_pose"
        info["action_pose_semantics"] = "same-frame absolute TCP; convert each sampled chunk by inv(T_actions[0]) @ T_actions[k]"
        info["state_pose_semantics"] = (
            "same-frame source encoder angle only, in radians" if has_encoder
            else "constant zero placeholder; source encoder was unavailable"
        )
        info["gripper_mapping"] = (
            {
                "applied_to_dataset": False,
                "source_unit": "rad",
                "source_closed_rad": grip_min,
                "source_open_rad": grip_max,
                "larger_angle_means_open": True,
                "robot_open_width_mm": robot_open_width_mm,
                "mapping_stage": "inference",
            }
            if has_encoder else {
                "available": False,
                "applied_to_dataset": False,
                "mapping_stage": "none",
                "placeholder_value": 0.0,
            }
        )
        state_name = "gripper_angle_rad" if has_encoder else "unobserved_gripper_placeholder"
        info["features"]["state"] = {"dtype": "float32", "shape": [1], "names": [state_name]}
        info["features"]["actions"] = {
            "dtype": "float32", "shape": [7],
            "names": ["tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_rx_rad", "tcp_ry_rad", "tcp_rz_rad", state_name],
        }
        (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        (staging / "meta/stats.json").write_text(
            json.dumps({**source_stats, "state": _stats(states), "actions": _stats(actions)}, indent=2) + "\n"
        )
        with (staging / "meta/episodes_stats.jsonl").open("w") as stream:
            for episode in sorted(episode_stats):
                stream.write(json.dumps(episode_stats[episode]) + "\n")
        staging.rename(output)
        print(f"Wrote {len(states)} frames from {len(episodes)} episodes to {output}")
        if has_encoder:
            print(f"Source encoder angle stays in radians: {grip_min:.6f}..{grip_max:.6f}; mapping is deferred to inference")
        else:
            print("No source encoder: state and action[6] are zero placeholders; gripper control cannot be learned")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("local/datasets/pick_place"))
    parser.add_argument("--output", type=Path, default=Path("local/datasets/pick_place_chunk_relative"))
    parser.add_argument("--robot-open-width-mm", type=float, default=20.0)
    parser.add_argument("--task", help="replace the single dataset task prompt in task and episode metadata")
    args = parser.parse_args()
    convert(args.source, args.output, robot_open_width_mm=args.robot_open_width_mm, task=args.task)


if __name__ == "__main__":
    main()
