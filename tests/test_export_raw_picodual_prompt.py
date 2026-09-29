"""Prompt edits leave the exported action and image data unchanged."""
# ruff: noqa: E402

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from examples.piper.data_tools import export_raw_picodual_action as exporter


def test_prompt_only_changes_both_task_metadata_files(tmp_path, monkeypatch):
    root = tmp_path / "pick_cube_raw_action"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (root / "data").mkdir()
    data_file = root / "data/episode_000000.parquet"
    data_file.write_bytes(b"existing image and action data")
    (meta / "info.json").write_text(json.dumps({"total_tasks": 1, "total_episodes": 2}))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "old"}) + "\n")
    episodes = [{"episode_index": i, "tasks": ["old"], "length": 3} for i in range(2)]
    (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))

    monkeypatch.setattr(sys, "argv", ["export_raw_picodual_action.py", "--output", str(root),
                                   "--prompt_only", "--task", "  pick up the blue cube  "])
    exporter.main()

    task = json.loads((meta / "tasks.jsonl").read_text().strip())
    updated_episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines()]
    assert task == {"task_index": 0, "task": "pick up the blue cube"}
    assert all(row["tasks"] == ["pick up the blue cube"] for row in updated_episodes)
    assert data_file.read_bytes() == b"existing image and action data"


def test_missing_template_builds_directly_from_source_with_task(tmp_path, monkeypatch):
    import numpy as np

    raw = np.zeros((2, 7), dtype=np.float32)
    ends = np.array([2], dtype=np.int64)
    monkeypatch.setattr(exporter, "read_numeric_array", lambda path: ends if path.name == "episode_ends" else raw)
    calls = []
    monkeypatch.setattr(exporter, "export_from_source", lambda args, actions, episode_ends, task: calls.append(
        (task, actions.copy(), episode_ends.copy())
    ))
    monkeypatch.setattr(sys, "argv", [
        "export_raw_picodual_action.py", "--input", str(tmp_path / "pick.zarr"),
        "--template", str(tmp_path / "missing_template"),
        "--output", str(tmp_path / "new_dataset"), "--task", "pick up the blue cube",
    ])
    exporter.main()
    assert len(calls) == 1
    assert calls[0][0] == "pick up the blue cube"
    np.testing.assert_array_equal(calls[0][1], raw)
    np.testing.assert_array_equal(calls[0][2], ends)
