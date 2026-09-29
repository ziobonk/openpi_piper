#!/usr/bin/env python3
"""Compute valid-window pi05 statistics for pick_cube_0928_chunk_relative.

Only the state/actions Parquet columns are read; camera images are not decoded.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from openpi.shared import normalize  # noqa: E402

CONFIG_NAME = "pi05_piper_pick_cube_0928_chunk_relative"
DATASET = ROOT / "pick_cube_0928_chunk_relative"
HORIZON = 50
# Keep the pooled quantile scale well-defined for almost-static dimensions.
MIN_Q_RANGE = np.array([0.001] * 3 + [0.01] * 3 + [0.1], dtype=np.float64)


def _stabilize(stats: normalize.NormStats, minimum: np.ndarray) -> normalize.NormStats:
    q01 = np.asarray(stats.q01, dtype=np.float64).copy()
    q99 = np.asarray(stats.q99, dtype=np.float64).copy()
    span = q99 - q01
    center = (q01 + q99) / 2
    width = np.maximum(span, minimum)
    return normalize.NormStats(
        mean=np.asarray(stats.mean),
        std=np.maximum(np.asarray(stats.std), width / 4.6527),
        q01=center - width / 2,
        q99=center + width / 2,
    )


def main() -> None:
    info_path = DATASET / "meta/info.json"
    info_bytes = info_path.read_bytes()
    info = json.loads(info_bytes)
    expected = {
        "robot_type": "piper_chunk_relative_eef",
        "pose_coordinate_frame": "pico_world_tcp",
        "coordinate_transform_applied": False,
        "fps": 20.0,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"{key}: expected {value!r}, got {info.get(key)!r}")
    for key, shape in (("state", [1]), ("actions", [7])):
        if info["features"][key]["shape"] != shape:
            raise ValueError(f"{key} must have shape {shape}")
    if info.get("gripper_mapping", {}).get("robot_open_width_mm") != 20.0:
        raise ValueError("dataset must map gripper width to 0..20 mm")

    files = sorted((DATASET / "data").rglob("*.parquet"))
    if len(files) != info["total_episodes"]:
        raise ValueError(f"expected {info['total_episodes']} episode Parquet files, got {len(files)}")

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    frame_count = valid_actions = 0
    for index, path in enumerate(files):
        table = pq.read_table(path, columns=["state", "actions"])
        states = np.asarray(table.column("state").to_pylist(), dtype=np.float64)
        actions = np.asarray(table.column("actions").to_pylist(), dtype=np.float64)
        if states.shape != (len(actions), 1) or actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"invalid state/action shape in {path}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"nonfinite state/action in {path}")
        if len(actions) < 2:
            raise ValueError(f"episode is shorter than two frames: {path}")

        state_stats.update(states)
        rotations = Rotation.from_rotvec(actions[:, 3:6])
        for offset in range(min(HORIZON, len(actions))):
            count = len(actions) - offset
            base = rotations[:count]
            rows = np.empty((count, 7), dtype=np.float32)
            rows[:, :3] = base.inv().apply(actions[offset:, :3] - actions[:count, :3])
            rows[:, 3:6] = (base.inv() * rotations[offset:]).as_rotvec()
            if offset == 0:
                rows[:, :6] = 0.0
            rows[:, 6] = actions[offset:, 6]
            action_stats.update(rows)
            valid_actions += count
        frame_count += len(actions)
        print(f"[{index + 1}/{len(files)}] {path.name}: {len(actions)} frames", flush=True)

    if frame_count != info["total_frames"]:
        raise ValueError(f"frame count mismatch: {frame_count} != {info['total_frames']}")
    output = ROOT / "assets" / CONFIG_NAME / DATASET.name
    stats = {
        "state": _stabilize(state_stats.get_statistics(), MIN_Q_RANGE[6:]),
        "actions": _stabilize(action_stats.get_statistics(), MIN_Q_RANGE),
    }
    normalize.save(output, stats)
    (output / "norm_stats_manifest.json").write_text(json.dumps({
        "schema": "piper.chunk_relative_norm.v1",
        "dataset": DATASET.name,
        "config": CONFIG_NAME,
        "dataset_info_sha256": hashlib.sha256(info_bytes).hexdigest(),
        "horizon": HORIZON,
        "frame_count": frame_count,
        "valid_action_rows": valid_actions,
        "action_reference": "first_action_in_chunk",
        "padded_action_rows_included": False,
        "min_quantile_range": MIN_Q_RANGE.tolist(),
    }, indent=2) + "\n")
    print(f"Wrote {output / 'norm_stats.json'} ({valid_actions} valid action rows)")


if __name__ == "__main__":
    main()
