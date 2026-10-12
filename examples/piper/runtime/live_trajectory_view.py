"""OpenCV preview of the TCP targets selected from a live policy chunk."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def _project(point_mm: np.ndarray, center: np.ndarray, scale: float, left: int, top: int) -> tuple[int, int]:
    return (
        round(left + 185 + (point_mm[0] - center[0]) * scale),
        round(top + 172.5 - (point_mm[1] - center[1]) * scale),
    )


class LiveTrajectoryView:
    """Render an observed TCP anchor and the next commands in Robot Base."""

    WINDOW = "Piper Planned TCP"
    _PAIRS = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))

    def __init__(self) -> None:
        self.anchor: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        self._blank = np.full((430, 1170, 3), 25, dtype=np.uint8)
        cv2.putText(
            self._blank,
            "Paused / waiting for policy chunk",
            (300, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    def clear(self) -> None:
        self.anchor = None
        self.targets = None

    def set_chunk(self, anchor: np.ndarray, actions: np.ndarray, execute_steps: int) -> None:
        """Apply ``T_observed @ D[k]`` to exactly the actions that will execute."""
        anchor = np.asarray(anchor, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if anchor.shape != (6,) or actions.ndim != 2 or actions.shape[1] < 6:
            raise ValueError("expected TCP anchor (6,) and chunk actions (N,>=6)")
        if not 1 <= execute_steps <= len(actions) or not np.isfinite(anchor).all() or not np.isfinite(actions).all():
            raise ValueError("invalid planned TCP chunk")
        # SciPy can reject read-only views from a WebSocket receive buffer.
        anchor_rotation = Rotation.from_rotvec(np.array(anchor[3:6], copy=True))
        local_rotations = Rotation.from_rotvec(np.array(actions[:execute_steps, 3:6], copy=True))
        targets = np.empty((execute_steps, 6), dtype=np.float64)
        targets[:, :3] = anchor[:3] + anchor_rotation.apply(actions[:execute_steps, :3])
        targets[:, 3:6] = (anchor_rotation * local_rotations).as_rotvec()
        self.anchor = anchor.copy()
        self.targets = targets

    def render(self, next_step: int, gripper_mm: float | None = None) -> np.ndarray:
        if self.targets is None or self.anchor is None:
            return self._blank.copy()
        if not 0 <= next_step < len(self.targets):
            raise ValueError("next_step is outside the planned chunk")

        canvas = np.full_like(self._blank, 25)
        points = np.vstack((self.anchor[:3], self.targets[:, :3]))
        current = self.targets[next_step]
        rpy_deg = Rotation.from_rotvec(current[3:6].copy()).as_euler("xyz", degrees=True)
        header = f"Robot Base TCP | next {next_step + 1}/{len(self.targets)} | origin {self.anchor[:3]} m"
        cv2.putText(canvas, header, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)
        grip = "?" if gripper_mm is None else f"{gripper_mm:.1f} mm"
        detail = f"target xyz {current[:3].round(3)} m | RPY {rpy_deg.round(1)} deg | grip {grip}"
        cv2.putText(canvas, detail, (16, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)

        for panel, (x_axis, y_axis, label) in enumerate(self._PAIRS):
            left = 10 + panel * 390
            top = 70
            width = 370
            height = 345
            cv2.rectangle(canvas, (left, top), (left + width, top + height), (70, 70, 70), 1)
            cv2.putText(
                canvas,
                f"{label} (Robot Base, mm)",
                (left + 10, top + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            xy_mm = (points[:, [x_axis, y_axis]] - self.anchor[[x_axis, y_axis]]) * 1000
            center = (xy_mm.min(axis=0) + xy_mm.max(axis=0)) / 2
            radius = max(float(np.max(np.abs(xy_mm - center))), 25.0) * 1.25
            scale = 132.0 / radius

            origin = _project(xy_mm[0], center, scale, left, top)
            cv2.line(canvas, (left + 5, origin[1]), (left + width - 5, origin[1]), (48, 48, 48), 1)
            cv2.line(canvas, (origin[0], top + 30), (origin[0], top + height - 5), (48, 48, 48), 1)
            pixels = [_project(point, center, scale, left, top) for point in xy_mm]
            for index in range(1, len(pixels)):
                color = (100, 100, 100) if index <= next_step else (255, 170, 40)
                cv2.line(canvas, pixels[index - 1], pixels[index], color, 2, cv2.LINE_AA)
            cv2.circle(canvas, origin, 5, (70, 210, 70), -1, cv2.LINE_AA)
            cv2.circle(canvas, pixels[next_step + 1], 7, (50, 60, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"range +/-{radius:.0f} mm",
                (left + 10, top + height - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )
        return canvas
