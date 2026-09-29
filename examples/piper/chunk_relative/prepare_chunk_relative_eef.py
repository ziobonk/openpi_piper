#!/usr/bin/env python3
"""Create a gripper-only-state dataset from raw absolute PICO World TCP data.

The stored actions remain absolute PICO World TCP poses. At training time,
ChunkRelativeEEFActions converts each sampled action window using its own first
pose as the base. The gripper column is mapped to the robot's physical width
range, so inference needs no source dataset or gripper calibration statistics.
"""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite value in converted dataset")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def convert(source: Path, output: Path, *, robot_open_width_mm: float = 20.0) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not np.isfinite(robot_open_width_mm) or robot_open_width_mm <= 0:
        raise ValueError("robot_open_width_mm must be positive and finite")
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("pose_coordinate_frame") != "pico_world_tcp" or info.get("coordinate_transform_applied") is not False:
        raise ValueError("source must contain untransformed PICO World absolute TCP poses")
    if info["features"]["state"]["shape"] != [7] or info["features"]["actions"]["shape"] != [7]:
        raise ValueError("source state/actions must both be 7D")
    source_stats = json.loads((source / "meta/stats.json").read_text())
    grip_min = float(source_stats["state"]["min"][6])
    grip_max = float(source_stats["state"]["max"][6])
    if not np.isfinite([grip_min, grip_max]).all() or grip_max <= grip_min:
        raise ValueError("source gripper range is invalid")
    parquet_files = sorted((source / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {source / 'data'}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        shutil.copytree(source / "meta", staging / "meta")
        episode_stats = {
            row["episode_index"]: row
            for row in map(json.loads, (source / "meta/episodes_stats.jsonl").read_text().splitlines())
        }
        all_states = []
        all_actions = []
        total_frames = 0
        for path in parquet_files:
            table = pq.read_table(path)
            states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 7:
                raise ValueError(f"invalid state/action shapes in {path}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"nonfinite state/action in {path}")
            if not np.allclose(states, actions, rtol=0, atol=1e-6):
                raise ValueError(f"source must have same-frame state/actions: {path}")
            episode_ids = set(table["episode_index"].to_pylist())
            if len(episode_ids) != 1:
                raise ValueError(f"expected one episode in {path}")
            episode_id = int(episode_ids.pop())
            grip_mm = np.clip((states[:, 6] - grip_min) / (grip_max - grip_min), 0, 1) * robot_open_width_mm
            new_states = grip_mm[:, None].astype(np.float32)
            new_actions = actions.copy()
            new_actions[:, 6] = grip_mm
            table = table.set_column(table.schema.get_field_index("state"), "state", _fixed_list(new_states))
            table = table.set_column(table.schema.get_field_index("actions"), "actions", _fixed_list(new_actions))
            destination = staging / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            episode_stats[episode_id]["stats"]["state"] = _stats(new_states)
            episode_stats[episode_id]["stats"]["actions"] = _stats(new_actions)
            all_states.append(new_states)
            all_actions.append(new_actions)
            total_frames += len(new_states)
        if total_frames != int(info["total_frames"]):
            raise ValueError(f"frame count mismatch: {total_frames} != {info['total_frames']}")

        info["robot_type"] = "piper_chunk_relative_eef"
        info["features"]["state"] = {"dtype": "float32", "shape": [1], "names": ["gripper_width_mm"]}
        info["features"]["actions"]["names"] = [
            "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz", "gripper_width_mm"
        ]
        info["state_pose_semantics"] = "gripper width only; no EEF pose is provided to the model"
        info["action_pose_semantics"] = (
            "same-frame absolute TCP pose stored; each sampled chunk is converted to "
            "inv(T_actions[0]) @ T_actions[k] during training"
        )
        info["gripper_mapping"] = {
            "source_min": grip_min, "source_max": grip_max, "robot_open_width_mm": robot_open_width_mm
        }
        (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        (staging / "meta/stats.json").write_text(json.dumps({
            **source_stats,
            "state": _stats(np.concatenate(all_states)),
            "actions": _stats(np.concatenate(all_actions)),
        }, indent=2) + "\n")
        with (staging / "meta/episodes_stats.jsonl").open("w") as stream:
            for episode_id in sorted(episode_stats):
                stream.write(json.dumps(episode_stats[episode_id]) + "\n")
        staging.rename(output)
        print(f"Wrote {total_frames} frames to {output}")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("local/datasets/pick_cube_raw_action"))
    parser.add_argument("--output", type=Path, default=Path("local/datasets/pick_cube_chunk_relative"))
    parser.add_argument("--robot-open-width-mm", type=float, default=20.0)
    args = parser.parse_args()
    convert(args.source, args.output, robot_open_width_mm=args.robot_open_width_mm)


if __name__ == "__main__":
    main()
