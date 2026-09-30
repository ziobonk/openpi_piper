"""Thread-safe buffers for asynchronous Piper TCP action chunks."""

from collections import deque
import threading

import numpy as np
from scipy.spatial.transform import Rotation


def _valid_chunk(actions: np.ndarray) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0 or not np.isfinite(chunk).all():
        raise ValueError(f"expected finite TCP actions with shape (H, 7), got {chunk.shape}")
    return chunk


class NaiveActionBuffer:
    """Replace pending commands with the latest chunk after dropping expired steps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._actions: deque[np.ndarray] = deque()
        self._executed = 0

    def clear(self) -> None:
        with self._lock:
            self._actions.clear()
            self._executed = 0

    def executed_steps(self) -> int:
        with self._lock:
            return self._executed

    def pop_next_action(self) -> np.ndarray | None:
        with self._lock:
            return self._actions.popleft().copy() if self._actions else None

    def mark_executed(self, action: np.ndarray) -> None:
        del action
        with self._lock:
            self._executed += 1

    def integrate_new_chunk(
        self, actions: np.ndarray, observed_step: int, max_latency_steps: int, min_smooth_steps: int
    ) -> tuple[int, int]:
        del min_smooth_steps
        chunk = _valid_chunk(actions)
        with self._lock:
            elapsed = max(0, self._executed - observed_step)
            if elapsed > max_latency_steps or elapsed >= len(chunk):
                return elapsed, len(self._actions)
            self._actions = deque(row.copy() for row in chunk[elapsed:])
            return elapsed, len(self._actions)


class TemporalEnsemblingActionBuffer:
    """Fuse predictions for the same future control tick in Robot Base coordinates."""

    def __init__(self, exp_weight_m: float = 0.01) -> None:
        if exp_weight_m < 0:
            raise ValueError("exp_weight_m must be nonnegative")
        self._weight = exp_weight_m
        self._lock = threading.Lock()
        self._predictions: dict[int, list[tuple[int, np.ndarray]]] = {}
        self._executed = 0
        self._inference_index = 0

    def clear(self) -> None:
        with self._lock:
            self._predictions.clear()
            self._executed = 0
            self._inference_index = 0

    def executed_steps(self) -> int:
        with self._lock:
            return self._executed

    def pop_next_action(self) -> np.ndarray | None:
        with self._lock:
            predictions = self._predictions.get(self._executed)
            if not predictions:
                return None
            ordered = sorted(predictions, key=lambda item: item[0])
            actions = np.stack([row for _, row in ordered])
            weights = np.exp(-self._weight * np.arange(len(actions), dtype=np.float64))
            weights /= weights.sum()
            pose = np.empty(7, dtype=np.float64)
            pose[:3] = np.average(actions[:, :3], axis=0, weights=weights)
            pose[3:6] = Rotation.from_rotvec(actions[:, 3:6]).mean(weights=weights).as_rotvec()
            pose[6] = np.average(actions[:, 6], weights=weights)
            return pose

    def mark_executed(self, action: np.ndarray) -> None:
        del action
        with self._lock:
            self._predictions.pop(self._executed, None)
            self._executed += 1

    def integrate_new_chunk(
        self, actions: np.ndarray, observed_step: int, max_latency_steps: int, min_smooth_steps: int
    ) -> tuple[int, int]:
        del min_smooth_steps
        chunk = _valid_chunk(actions)
        with self._lock:
            elapsed = max(0, self._executed - observed_step)
            if elapsed > max_latency_steps or elapsed >= len(chunk):
                return elapsed, len(self._predictions)
            request_id = self._inference_index
            self._inference_index += 1
            for offset in range(elapsed, len(chunk)):
                tick = observed_step + offset
                self._predictions.setdefault(tick, []).append((request_id, chunk[offset].copy()))
            return elapsed, len(self._predictions)
