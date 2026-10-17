#!/usr/bin/env python3
"""Compute valid-window pi05 statistics for a Piper chunk-relative dataset.

Only the state/actions Parquet columns are read; camera images are not decoded.
"""

import argparse
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

HORIZON = 50
# Keep the pooled quantile scale well-defined for almost-static dimensions.
MIN_TCP_Q_RANGE = [0.001] * 3 + [0.01] * 3


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "local/datasets/pick_cube_0928_chunk_relative")
    parser.add_argument("--config-name", default="pi05_piper_pick_cube_0928_chunk_relative")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    info_path = dataset / "meta/info.json"
    info_bytes = info_path.read_bytes()
    info = json.loads(info_bytes)
    expected = {"robot_type": "piper_chunk_relative_eef", "coordinate_transform_applied": False}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"{key}: expected {value!r}, got {info.get(key)!r}")
    if info.get("pose_coordinate_frame") not in {"pico_world_tcp", "source_tracking_world_tcp", "robot_base_tcp"}:
        raise ValueError("unsupported absolute TCP coordinate frame")
    if not np.isfinite(info.get("fps", np.nan)) or info["fps"] <= 0:
        raise ValueError("dataset FPS must be positive")
    for key, shape in (("state", [1]), ("actions", [7])):
        if info["features"][key]["shape"] != shape:
            raise ValueError(f"{key} must have shape {shape}")
    gripper_mapping = info.get("gripper_mapping", {})
    unobserved_gripper = gripper_mapping.get("available") is False
    if unobserved_gripper:
        if gripper_mapping.get("placeholder_value") != 0.0 or gripper_mapping.get("mapping_stage") != "none":
            raise ValueError("invalid unobserved gripper metadata")
        gripper_q_range = 0.1
    elif gripper_mapping.get("applied_to_dataset") is False:
        if (
            gripper_mapping.get("source_unit") != "rad"
            or gripper_mapping.get("larger_angle_means_open") is not True
            or gripper_mapping.get("source_open_rad", 0) <= gripper_mapping.get("source_closed_rad", 0)
        ):
            raise ValueError("invalid unmapped source gripper angle metadata")
        gripper_q_range = 0.01
    else:
        width = float(gripper_mapping.get("robot_open_width_mm", 0))
        if not np.isfinite(width) or width <= 0:
            raise ValueError("dataset must declare a positive gripper width")
        gripper_q_range = 0.1
    min_q_range = np.array([*MIN_TCP_Q_RANGE, gripper_q_range], dtype=np.float64)

    files = sorted((dataset / "data").rglob("*.parquet"))
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
        if unobserved_gripper and (np.any(states != 0) or np.any(actions[:, 6] != 0)):
            raise ValueError(f"unobserved gripper placeholders must be zero in {path}")
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
    output = ROOT / "assets" / args.config_name / dataset.name
    stats = {
        "state": _stabilize(state_stats.get_statistics(), min_q_range[6:]),
        "actions": _stabilize(action_stats.get_statistics(), min_q_range),
    }
    normalize.save(output, stats)
    (output / "norm_stats_manifest.json").write_text(json.dumps({
        "schema": "piper.chunk_relative_norm.v1",
        "dataset": dataset.name,
        "config": args.config_name,
        "dataset_info_sha256": hashlib.sha256(info_bytes).hexdigest(),
        "horizon": HORIZON,
        "frame_count": frame_count,
        "valid_action_rows": valid_actions,
        "action_reference": "first_action_in_chunk",
        "padded_action_rows_included": False,
        "min_quantile_range": min_q_range.tolist(),
    }, indent=2) + "\n")
    print(f"Wrote {output / 'norm_stats.json'} ({valid_actions} valid action rows)")


if __name__ == "__main__":
    main()
