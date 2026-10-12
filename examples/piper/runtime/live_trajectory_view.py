"""Nonblocking previews of TCP targets selected from a live policy chunk."""

from __future__ import annotations

from contextlib import suppress
import multiprocessing as mp
import queue

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
        self._history: list[list[np.ndarray]] = []
        self._active_history: list[np.ndarray] | None = None
        self._history_xyz_cache: np.ndarray | None = None
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
        """Discard the pending chunk while retaining successfully sent targets."""
        self.anchor = None
        self.targets = None
        self._active_history = None

    @property
    def history_xyz(self) -> np.ndarray:
        """Robot Base command history; NaN rows separate independently anchored chunks."""
        if self._history_xyz_cache is None:
            segments = [np.stack(segment) for segment in self._history if len(segment) > 1]
            if segments:
                separator = np.full((1, 3), np.nan)
                rows = [segments[0]]
                for segment in segments[1:]:
                    rows.extend((separator, segment))
                self._history_xyz_cache = np.vstack(rows)
            else:
                self._history_xyz_cache = np.empty((0, 3), dtype=np.float64)
        return self._history_xyz_cache

    def record_sent(self, target_pose: np.ndarray) -> None:
        """Append a target only after its robot command call has completed."""
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (6,) or not np.isfinite(target_pose).all() or self.anchor is None:
            raise ValueError("a finite target and active TCP anchor are required")
        if self._active_history is None:
            self._active_history = [self.anchor[:3].copy()]
            self._history.append(self._active_history)
        self._active_history.append(target_pose[:3].copy())
        self._history_xyz_cache = None

    def start(self) -> None:
        """The OpenCV window runs in the inference process."""

    def close(self) -> None:
        """The inference process already closes all OpenCV windows."""

    def show(self, next_step: int = 0, gripper_mm: float | None = None) -> None:
        cv2.imshow(self.WINDOW, self.render(next_step, gripper_mm))
        cv2.waitKey(1)

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
        self._active_history = None

    def render(self, next_step: int, gripper_mm: float | None = None) -> np.ndarray:
        history_xyz = self.history_xyz
        if (self.targets is None or self.anchor is None) and len(history_xyz) == 0:
            return self._blank.copy()
        if self.targets is not None and not 0 <= next_step <= len(self.targets):
            raise ValueError("next_step is outside the planned chunk")

        canvas = np.full_like(self._blank, 25)
        active = self.targets is not None and self.anchor is not None
        if active:
            points = np.vstack((self.anchor[:3], self.targets[:, :3]))
            current = self.targets[next_step] if next_step < len(self.targets) else None
            reference = self.anchor[:3]
            phase = f"next {next_step + 1}/{len(self.targets)}" if current is not None else "chunk complete"
        else:
            points = np.empty((0, 3), dtype=np.float64)
            current = None
            reference = history_xyz[0]
            phase = "paused"
        header = f"Robot Base TCP | {phase} | sent {sum(len(s) - 1 for s in self._history)} targets"
        cv2.putText(canvas, header, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)
        if current is not None:
            rpy_deg = Rotation.from_rotvec(current[3:6].copy()).as_euler("xyz", degrees=True)
            grip = "?" if gripper_mm is None else f"{gripper_mm:.1f} mm"
            detail = f"target xyz {current[:3].round(3)} m | RPY {rpy_deg.round(1)} deg | grip {grip}"
            cv2.putText(canvas, detail, (16, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)

        finite_history = history_xyz[np.isfinite(history_xyz).all(axis=1)]
        all_points = np.vstack((finite_history, points))

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
            all_xy_mm = (all_points[:, [x_axis, y_axis]] - reference[[x_axis, y_axis]]) * 1000
            center = (all_xy_mm.min(axis=0) + all_xy_mm.max(axis=0)) / 2
            radius = max(float(np.max(np.abs(all_xy_mm - center))), 25.0) * 1.25
            scale = 132.0 / radius

            origin = _project(np.zeros(2), center, scale, left, top)
            cv2.line(canvas, (left + 5, origin[1]), (left + width - 5, origin[1]), (48, 48, 48), 1)
            cv2.line(canvas, (origin[0], top + 30), (origin[0], top + height - 5), (48, 48, 48), 1)
            for segment in self._history:
                coords = (np.stack(segment)[:, [x_axis, y_axis]] - reference[[x_axis, y_axis]]) * 1000
                pixels = [_project(point, center, scale, left, top) for point in coords]
                for index in range(1, len(pixels)):
                    cv2.line(canvas, pixels[index - 1], pixels[index], (100, 100, 100), 2, cv2.LINE_AA)
            if active:
                xy_mm = (points[:, [x_axis, y_axis]] - reference[[x_axis, y_axis]]) * 1000
                pixels = [_project(point, center, scale, left, top) for point in xy_mm]
                for index in range(next_step + 1, len(pixels)):
                    cv2.line(canvas, pixels[index - 1], pixels[index], (255, 170, 40), 2, cv2.LINE_AA)
                cv2.circle(canvas, origin, 5, (70, 210, 70), -1, cv2.LINE_AA)
                if current is not None:
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


def draw_3d_snapshot(
    ax,
    anchor: np.ndarray | None,
    targets: np.ndarray | None,
    next_step: int,
    gripper_mm: float | None,
    history_xyz: np.ndarray | None = None,
) -> None:
    """Draw cumulative sent targets and the active chunk without resetting the camera."""
    elevation, azimuth = ax.elev, ax.azim
    ax.clear()
    history_xyz = np.empty((0, 3)) if history_xyz is None else np.asarray(history_xyz)
    finite_history = history_xyz[np.isfinite(history_xyz).all(axis=1)]
    if len(history_xyz):
        ax.plot(*history_xyz.T, color="0.55", linewidth=2, label="sent target history")

    if anchor is None or targets is None:
        ax.set_title("Paused / waiting for policy chunk")
        all_points = finite_history
    else:
        points = np.vstack((anchor[:3], targets[:, :3]))
        all_points = np.vstack((finite_history, points))
        if next_step < len(targets):
            current = targets[next_step]
            ax.plot(
                *points[next_step:].T,
                color="tab:blue",
                linewidth=2.5,
                marker="o",
                markersize=3,
                label="upcoming targets",
            )
            ax.scatter(*current[:3], color="tab:red", s=80, label="next target")
            extent = max(float(np.ptp(points, axis=0).max()), 0.06)
            axis_length = max(0.015, extent * 0.15)
            rotation = Rotation.from_rotvec(current[3:6].copy()).as_matrix()
            for axis, color in enumerate(("red", "green", "blue")):
                direction = rotation[:, axis] * axis_length
                ax.quiver(*current[:3], *direction, color=color, linewidth=2, arrow_length_ratio=0.22)
            grip = "?" if gripper_mm is None else f"{gripper_mm:.1f} mm"
            ax.set_title(
                f"Robot Base TCP | next {next_step + 1}/{len(targets)} | gripper {grip}\n"
                f"target xyz {np.round(current[:3], 3)} m"
            )
        else:
            ax.set_title(f"Robot Base TCP | chunk complete ({len(targets)} sent)")
        ax.scatter(*anchor[:3], color="tab:green", s=65, label="observed TCP")

    if len(all_points):
        center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
        radius = max(float(np.ptp(all_points, axis=0).max()) / 2, 0.03) * 1.35
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=elevation, azim=azimuth)


def _window_3d_main(updates: mp.Queue) -> None:
    """Run Matplotlib's interactive event loop away from the robot control loop."""
    try:
        import matplotlib as mpl

        mpl.use("TkAgg")
        import matplotlib.pyplot as plt

        fig = plt.figure("Piper Planned TCP 3D", figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        draw_3d_snapshot(ax, None, None, 0, None)
        plt.show(block=False)
        while plt.fignum_exists(fig.number):
            latest = None
            try:
                latest = updates.get(timeout=0.03)
                while True:
                    latest = updates.get_nowait()
            except queue.Empty:
                pass
            if latest is not None:
                if isinstance(latest, str) and latest == "close":
                    break
                draw_3d_snapshot(ax, *latest)
                fig.canvas.draw_idle()
            plt.pause(0.03)
        plt.close(fig)
    except Exception as exc:
        print(f"[Trajectory3D] window failed: {exc}", flush=True)


class LiveTrajectory3DView:
    """Send only the latest planned chunk to a separate, rotatable 3D window."""

    def __init__(self) -> None:
        self._geometry = LiveTrajectoryView()
        self._context = mp.get_context("spawn")
        self._updates = self._context.Queue(maxsize=2)
        self._updates.cancel_join_thread()
        self._process: mp.Process | None = None
        self._generation = 0
        self._last_display: tuple[int, int, float | None] | None = None
        self._warned_closed = False

    @property
    def anchor(self) -> np.ndarray | None:
        return self._geometry.anchor

    @property
    def targets(self) -> np.ndarray | None:
        return self._geometry.targets

    @property
    def history_xyz(self) -> np.ndarray:
        return self._geometry.history_xyz

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = self._context.Process(target=_window_3d_main, args=(self._updates,), daemon=True)
        self._process.start()

    def clear(self) -> None:
        self._geometry.clear()
        self._generation += 1
        self._last_display = None

    def set_chunk(self, anchor: np.ndarray, actions: np.ndarray, execute_steps: int) -> None:
        self._geometry.set_chunk(anchor, actions, execute_steps)
        self._generation += 1
        self._last_display = None

    def record_sent(self, target_pose: np.ndarray) -> None:
        self._geometry.record_sent(target_pose)
        self._generation += 1
        self._last_display = None

    def show(self, next_step: int = 0, gripper_mm: float | None = None) -> None:
        if self._process is None:
            return
        if not self._process.is_alive():
            if not self._warned_closed:
                print("[Trajectory3D] window closed; inference continues")
                self._warned_closed = True
            return
        key = (self._generation, next_step, gripper_mm)
        if key == self._last_display:
            return
        snapshot = (self.anchor, self.targets, next_step, gripper_mm, self.history_xyz)
        try:
            self._updates.put_nowait(snapshot)
        except queue.Full:
            with suppress(queue.Empty):
                self._updates.get_nowait()
            try:
                self._updates.put_nowait(snapshot)
            except queue.Full:
                return
        self._last_display = key

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            try:
                self._updates.put_nowait("close")
            except queue.Full:
                self._process.terminate()
            self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.5)
        self._updates.close()
        self._process = None
