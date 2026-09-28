#!/usr/bin/env python3
"""Write Zarr ``data/tcp_action`` in the pick_cube LeRobot layout.

State and actions contain the same-frame TCP pose in PICO World. Missing poses hold the
previous valid pose within each episode (or the first valid pose for leading
frames). NaN gripper values become 0; finite source values remain unchanged. No pose
or temporal conversion is applied; aligned images come from a matching template or the source Zarr.

Use ``--task`` to set one prompt for all episodes during export. To change only
an existing dataset's prompt, use ``--prompt_only --task TEXT --output DATASET``;
this updates tasks.jsonl and episodes.jsonl without rewriting parquet data.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_picodual_to_lerobot import main as convert_main
from convert_picodual_to_lerobot import read_numeric_array


def fill_missing_pose(values: np.ndarray) -> np.ndarray:
    """Hold missing poses and replace NaN gripper values with zero."""
    if np.isinf(values[:, 6]).any():
        raise ValueError("gripper column contains infinite values")
    valid = np.isfinite(values[:, :6]).all(axis=1)
    if not valid.any():
        raise ValueError("episode has no valid pose")
    first = int(np.flatnonzero(valid)[0])
    source_rows = np.maximum.accumulate(np.where(valid, np.arange(len(values)), first))
    result = values.copy()
    result[:, :6] = values[source_rows, :6]
    result[np.isnan(result[:, 6]), 6] = 0
    return result


def feature_stats(values: np.ndarray) -> dict:
    """Describe the stored, finite action rows."""
    if not np.isfinite(values).all():
        raise ValueError("action statistics require finite values")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [len(values)],
    }



def update_single_task_metadata(dataset: Path, task: str) -> None:
    """Change the sole task prompt without rewriting image/action parquet files."""
    task = task.strip()
    if not task:
        raise ValueError("--task must contain non-whitespace text")
    meta = dataset / "meta"
    info = json.loads((meta / "info.json").read_text())
    if int(info.get("total_tasks", -1)) != 1:
        raise ValueError("prompt update requires a single-task dataset")
    episodes_path = meta / "episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("episode metadata count does not match info.json")
    for episode in episodes:
        if len(episode.get("tasks", [])) != 1:
            raise ValueError("prompt update requires exactly one task per episode")
        episode["tasks"] = [task]
    new_files = {
        "tasks.jsonl": json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n",
        "episodes.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes),
    }
    for filename, content in new_files.items():
        temporary = meta / f".{filename}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, meta / filename)
        finally:
            temporary.unlink(missing_ok=True)


def export_from_source(args: argparse.Namespace, raw: np.ndarray, ends: np.ndarray, task: str) -> None:
    """Build aligned images and metadata directly from the source recording."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.staging-", dir=args.output.parent))
    try:
        convert_main([
            "--input", str(args.input), "--output", str(out), "--task", task,
            "--raw-tcp-action",
        ])
        info_path = out / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info.update({
            "missing_pose_policy": "hold previous valid pose within episode; leading frames use first valid pose",
            "missing_pose_frames_filled": int((~np.isfinite(raw[:, :6]).all(axis=1)).sum()),
            "missing_gripper_policy": "NaN gripper values become 0; finite values are unchanged",
            "missing_gripper_frames_filled": int(np.isnan(raw[:, 6]).sum()),
        })
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        backup = None
        if args.output.exists():
            backup = args.output.with_name(f".{args.output.name}.backup-{uuid.uuid4().hex}")
            os.replace(args.output, backup)
        try:
            os.replace(out, args.output)
        except Exception:
            if backup is not None:
                os.replace(backup, args.output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        print(
            f"Wrote {len(raw)} frames to {args.output}; filled "
            f"{info['missing_pose_frames_filled']} missing poses and "
            f"{info['missing_gripper_frames_filled']} missing gripper values"
        )
    finally:
        if out.exists():
            shutil.rmtree(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/combined_20260924_210134.zarr"))
    parser.add_argument("--template", type=Path, default=Path("pick_cube"),
                        help="reuse aligned images if episodes match; otherwise build from source")
    parser.add_argument("--output", type=Path, default=Path("pick_cube_raw_action"))
    parser.add_argument("--overwrite", action="store_true", help="replace a prior export from this script")
    parser.add_argument("--task", type=str, default=None, help="task prompt for all episodes")
    parser.add_argument("--prompt_only", action="store_true",
                        help="update --output metadata prompt only; keep image/action data unchanged")
    args = parser.parse_args()
    if args.task is not None:
        args.task = args.task.strip()
        if not args.task:
            parser.error("--task must contain non-whitespace text")
    if args.prompt_only:
        if args.task is None:
            parser.error("--prompt_only requires --task")
        update_single_task_metadata(args.output, args.task)
        print(f"Updated task prompt in {args.output}/meta; parquet data unchanged")
        return

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}; pass --overwrite to replace it")
        existing_info = json.loads((args.output / "meta" / "info.json").read_text())
        if existing_info.get("source_action_key") not in {"data/action", "data/tcp_action"}:
            raise ValueError("refusing to overwrite a dataset from another action source")
    raw = read_numeric_array(args.input / "data" / "tcp_action")
    ends = read_numeric_array(args.input / "meta" / "episode_ends")
    if raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"expected float32 (N, 7) action, got {raw.dtype} {raw.shape}")
    if not len(ends) or int(ends[-1]) != len(raw):
        raise ValueError("episode_ends does not match action length")
    if np.isinf(raw[:, 6]).any():
        raise ValueError("tcp_action gripper column contains infinite values")
    for ep, (start, end) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
        if not np.isfinite(raw[int(start):int(end), :6]).all(axis=1).any():
            raise ValueError(f"episode {ep} has no finite tcp_action pose to fill missing frames")

    info = json.loads((args.template / "meta" / "info.json").read_text())
    episodes = [json.loads(line) for line in (args.template / "meta" / "episodes.jsonl").read_text().splitlines()]
    episode_stats = [
        json.loads(line) for line in (args.template / "meta" / "episodes_stats.jsonl").read_text().splitlines()
    ]
    if len(episodes) != len(ends) or len(episode_stats) != len(ends):
        task = args.task or (episodes[0]["tasks"][0] if episodes else "pick up the black block and place it into the cup.")
        print(
            f"Template has {len(episodes)} episodes; source has {len(ends)}. "
            "Building aligned images and metadata from source.",
            flush=True,
        )
        export_from_source(args, raw, ends, task)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.staging-", dir=args.output.parent))
    try:
        shutil.copytree(args.template / "meta", out / "meta")
        if args.task is not None:
            update_single_task_metadata(out, args.task)
        (out / "data" / "chunk-000").mkdir(parents=True)
        start = 0
        filled_parts = []
        filled_count = 0
        for ep, end in enumerate(ends):
            end = int(end)
            if episodes[ep]["episode_index"] != ep or episodes[ep]["length"] != end - start:
                raise ValueError(f"template episode {ep} does not match source boundaries")
            path = args.template / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
            table = pq.read_table(path)
            values = fill_missing_pose(raw[start:end])
            filled_parts.append(values)
            filled_count += int((~np.isfinite(raw[start:end, :6]).all(axis=1)).sum())
            if table.num_rows != len(values):
                raise ValueError(f"template episode {ep} has wrong row count")
            if table.column("index").to_pylist() != list(range(start, end)):
                raise ValueError(f"template episode {ep} is not frame aligned")
            action_column = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), 7)
            for name in ("state", "actions"):
                table = table.set_column(table.schema.get_field_index(name), name, action_column)
            pq.write_table(table, out / "data" / "chunk-000" / path.name)
            stats = feature_stats(values)
            episode_stats[ep]["stats"]["state"] = stats
            episode_stats[ep]["stats"]["actions"] = stats
            start = end
            print(f"episode {ep:06d}: {len(values)} frames", flush=True)

        info.update({
            "pose_coordinate_frame": "pico_world_tcp",
            "controller_to_tcp": None,
            "source_action_key": "data/tcp_action",
            "coordinate_transform_applied": False,
            "missing_pose_policy": "hold previous valid pose within episode; leading frames use first valid pose",
            "missing_pose_frames_filled": filled_count,
            "missing_gripper_policy": "NaN gripper values become 0; finite values are unchanged",
            "missing_gripper_frames_filled": int(np.isnan(raw[:, 6]).sum()),
            "state_pose_semantics": "data/tcp_action at the same frame; missing poses held within each episode",
            "action_pose_semantics": (
                "data/tcp_action at the same frame; no temporal shift; missing poses held within each episode"
            ),
        })
        source_names = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz", "left_gripper_width"]
        for name in ("state", "actions"):
            info["features"][name]["names"] = source_names
        (out / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        with (out / "meta" / "episodes_stats.jsonl").open("w") as stream:
            for row in episode_stats:
                stream.write(json.dumps(row) + "\n")
        stats = json.loads((args.template / "meta" / "stats.json").read_text())
        action_stats = feature_stats(np.concatenate(filled_parts))
        stats["state"] = action_stats
        stats["actions"] = action_stats
        (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

        backup = None
        if args.output.exists():
            backup = args.output.with_name(f".{args.output.name}.backup-{uuid.uuid4().hex}")
            os.replace(args.output, backup)
        try:
            os.replace(out, args.output)
        except Exception:
            if backup is not None:
                os.replace(backup, args.output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        print(
            f"Wrote {len(raw)} frames to {args.output}; filled {filled_count} missing poses "
            f"and {info['missing_gripper_frames_filled']} missing gripper values"
        )
    finally:
        if out.exists():
            shutil.rmtree(out)


if __name__ == "__main__":
    main()
