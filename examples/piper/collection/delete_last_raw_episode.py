#!/usr/bin/env python3
"""Preview or delete the last episode of a Piper SDK raw EEF dataset.

Only the tail episode is supported, so remaining episode and global frame
indices stay contiguous. Without --apply this script changes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.piper.collection.collect_eef_udp import _aggregate_stats  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_atomic(path: Path, contents: str) -> None:
    temporary = path.with_name(path.name + ".delete-tmp")
    temporary.write_text(contents)
    os.replace(temporary, path)


def delete_last_episode(dataset: Path, episode: int, *, apply: bool = False) -> Path:
    dataset = dataset.resolve()
    meta = dataset / "meta"
    info_path = meta / "info.json"
    episodes_path = meta / "episodes.jsonl"
    episode_stats_path = meta / "episodes_stats.jsonl"
    stats_path = meta / "stats.json"
    original = {path: path.read_text() for path in (info_path, episodes_path, episode_stats_path, stats_path)}
    info = json.loads(original[info_path])
    episodes = _rows(episodes_path)
    episode_stats = _rows(episode_stats_path)
    if info.get("source_kind") != "piper_sdk_raw" or info.get("video_path") is not None:
        raise ValueError("this tool only supports image-in-Parquet Piper SDK raw datasets")
    if (not episodes or len(episodes) != len(episode_stats)
            or len(episodes) != info["total_episodes"]
            or episode != len(episodes) - 1
            or episodes[-1]["episode_index"] != episode
            or episode_stats[-1]["episode_index"] != episode):
        raise ValueError("only the existing final episode may be deleted")
    count = int(episodes[-1]["length"])
    if count <= 0 or sum(int(row["length"]) for row in episodes) != info["total_frames"]:
        raise ValueError("episode lengths do not match dataset metadata")
    data_path = dataset / info["data_path"].format(
        episode_chunk=episode // int(info["chunks_size"]), episode_index=episode,
    )
    if not data_path.is_file() or pq.read_metadata(data_path).num_rows != count:
        raise ValueError(f"episode Parquet missing or row count differs: {data_path}")
    print(f"Episode {episode}: {count} frames, {data_path}")
    print(f"After deletion: {len(episodes) - 1} episodes, {info['total_frames'] - count} frames")
    if not apply:
        print("Dry run; pass --apply to delete")
        return data_path

    remaining = episodes[:-1]
    remaining_stats = episode_stats[:-1]
    info["total_episodes"] = len(remaining)
    info["total_frames"] -= count
    info["total_chunks"] = (len(remaining) + info["chunks_size"] - 1) // info["chunks_size"]
    info["splits"] = {"train": f"0:{len(remaining)}"}
    updates = {
        episodes_path: "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in remaining),
        episode_stats_path: "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in remaining_stats),
        stats_path: json.dumps(_aggregate_stats(remaining_stats) if remaining_stats else {}, indent=2) + "\n",
        info_path: json.dumps(info, indent=2, ensure_ascii=False) + "\n",
    }
    pending = data_path.with_suffix(".parquet.delete-pending")
    if pending.exists():
        raise FileExistsError(f"pending deletion file already exists: {pending}")
    data_path.rename(pending)
    try:
        for path, contents in updates.items():
            _write_atomic(path, contents)
    except Exception:
        for path, contents in original.items():
            _write_atomic(path, contents)
        pending.rename(data_path)
        raise
    pending.unlink()
    print(f"Deleted episode {episode}")
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="perform the deletion; default is preview only")
    args = parser.parse_args()
    delete_last_episode(args.dataset, args.episode, apply=args.apply)


if __name__ == "__main__":
    main()
