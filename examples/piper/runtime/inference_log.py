"""Append-only, query-aligned log for live Piper policy inference."""

import json
import os
from pathlib import Path
import time

import cv2
import numpy as np


class InferenceLog:
    """Write model inputs/outputs and every successful EndPoseCtrl command."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        if self.directory.exists():
            raise FileExistsError(f"inference log directory already exists: {self.directory}")
        self.directory.mkdir(parents=True)
        self._events = (self.directory / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        self._start = time.monotonic()
        self._query_id = 0
        self._command_id = 0
        (self.directory / "format.json").write_text(json.dumps({
            "format_version": 1,
            "image_format": "PNG, exact 224x224 RGB model input (file channels decoded as RGB)",
            "model_action_layout": "[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad, gripper]",
            "tcp_command_layout": "[x_m, y_m, z_m, rx_rotvec_rad, ry_rotvec_rad, rz_rotvec_rad]",
            "end_pose_ctrl_layout": "[flange_x_0.001mm, flange_y_0.001mm, flange_z_0.001mm, "
                                    "flange_rx_0.001deg, flange_ry_0.001deg, flange_rz_0.001deg]",
            "events": "events.jsonl; query events carry full model output, command events carry actual sent commands",
        }, indent=2) + "\n", encoding="utf-8")

    def _write(self, event: dict) -> None:
        event["wall_time_ns"] = time.time_ns()
        event["session_time_s"] = time.monotonic() - self._start
        self._events.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def record_episode_start(self, tcp: np.ndarray) -> None:
        self._write({"type": "episode_start", "tcp": np.asarray(tcp, dtype=np.float64).tolist()})

    def record_query(
        self,
        observation: dict,
        actions: np.ndarray,
        *,
        tcp_anchor: np.ndarray | None,
        execute_steps: int,
        first_action_skipped: bool,
        observation_time_ns: int,
    ) -> int:
        query_id = self._query_id
        self._query_id += 1
        images = {}
        for name, key in (("global", "observation/image"), ("wrist", "observation/wrist_image")):
            rgb = np.asarray(observation[key])
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"{key} must be an HWC uint8 RGB image")
            filename = f"query_{query_id:06d}_{name}.png"
            if not cv2.imwrite(str(self.directory / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"failed to save inference image: {filename}")
            images[name] = filename
        self._write({
            "type": "query",
            "query_id": query_id,
            "observation_time_ns": observation_time_ns,
            "state": np.asarray(observation["observation/state"]).tolist(),
            "prompt": observation["prompt"],
            "tcp_anchor": None if tcp_anchor is None else np.asarray(tcp_anchor, dtype=np.float64).tolist(),
            "images": images,
            "model_actions": np.asarray(actions, dtype=np.float64).tolist(),
            "first_action_skipped": first_action_skipped,
            "execute_steps": execute_steps,
        })
        return query_id

    def record_command(
        self,
        *,
        query_id: int,
        execution_index: int,
        model_action_index: int,
        interpolation_index: int,
        interpolation_count: int,
        tcp_pose: np.ndarray,
        end_pose_ctrl: tuple[int, int, int, int, int, int],
    ) -> None:
        self._write({
            "type": "command",
            "command_id": self._command_id,
            "query_id": query_id,
            "execution_index": execution_index,
            "model_action_index": model_action_index,
            "interpolation_index": interpolation_index,
            "interpolation_count": interpolation_count,
            "tcp_pose": np.asarray(tcp_pose, dtype=np.float64).tolist(),
            "end_pose_ctrl": list(end_pose_ctrl),
        })
        self._command_id += 1

    def close(self) -> None:
        if not self._events.closed:
            self._events.flush()
            os.fsync(self._events.fileno())
            self._events.close()
            print(f"[InferenceLog] {self._query_id} queries, {self._command_id} commands → {self.directory}")
