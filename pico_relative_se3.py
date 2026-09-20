"""Episode-relative PICO controller pose to robot TCP targets.

Coordinate convention
---------------------
``T_A_B`` is ``^A T_B`` and maps column vectors from frame B to frame A::

    x_A = T_A_B @ x_B
    T_A_C = T_A_B @ T_B_C

PICO supplies ``T_W_C``.  The fixed controller-to-virtual-TCP mounting
transform is therefore applied on the right: ``T_W_E = T_W_C @ T_C_E``.
For every episode this module stores the first ``T_W_E`` and uses
``delta = inv(T_W_E0) @ T_W_E``.  No PICO-World-to-robot-base extrinsic is
used or required.

Input/output quaternions are explicitly ordered by the ``quaternion_order``
argument (``"xyzw"`` by default).  Rotation matrices map local-frame vectors
to parent-frame vectors, i.e. ``v_parent = R_parent_local @ v_local``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TrackingOriginJumpError(RuntimeError):
    """Raised when PICO changes tracking origin during an active episode."""


def _as_transform(transform: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-7
    ):
        raise ValueError(f"{name} rotation must be in SO(3)")
    return value.copy()


def quaternion_to_matrix(quaternion: np.ndarray, *, order: str = "xyzw") -> np.ndarray:
    """Return the local-to-parent active rotation represented by a quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be a finite length-4 vector")
    if order == "wxyz":
        q = q[[1, 2, 3, 0]]
    elif order != "xyzw":
        raise ValueError('quaternion order must be "xyzw" or "wxyz"')
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(
    position: np.ndarray,
    quaternion: np.ndarray,
    *,
    quaternion_order: str = "xyzw",
) -> np.ndarray:
    """Build a 4x4 local-to-parent pose from position and quaternion."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position must be a finite length-3 vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(quaternion, order=quaternion_order)
    transform[:3, 3] = position
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert an SE(3) transform using ``R.T`` and ``-R.T @ p``."""
    transform = _as_transform(transform, name="transform")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def relative_transform(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return ``inv(reference) @ current`` (current expressed at reference)."""
    return invert_transform(reference) @ _as_transform(current, name="current")


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


@dataclass(frozen=True)
class RelativePoseResult:
    """Raw pose, episode-relative motion, and absolute robot target."""

    world_tcp: np.ndarray
    delta_tcp: np.ndarray
    robot_target: np.ndarray


class PicoRelativeTcpMapper:
    """Stateful, one-instance-per-stream episode-relative SE(3) mapper.

    ``tracking_origin_id`` should be supplied when the PICO runtime exposes an
    origin generation/session identifier.  A changed identifier is a definite
    recenter and aborts the episode.  The step limits provide a fallback jump
    detector; tune or disable them (with ``None``) for the device sample rate
    and expected hand speed because pose samples alone cannot perfectly
    distinguish a recenter from a genuinely fast controller motion.
    """

    def __init__(
        self,
        controller_to_tcp: np.ndarray,
        *,
        quaternion_order: str = "xyzw",
        max_translation_step_m: float | None = 0.25,
        max_rotation_step_rad: float | None = np.deg2rad(60.0),
    ) -> None:
        self.controller_to_tcp = _as_transform(controller_to_tcp, name="controller_to_tcp")
        if quaternion_order not in ("xyzw", "wxyz"):
            raise ValueError('quaternion_order must be "xyzw" or "wxyz"')
        for name, value in (
            ("max_translation_step_m", max_translation_step_m),
            ("max_rotation_step_rad", max_rotation_step_rad),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive or None")
        self.quaternion_order = quaternion_order
        self.max_translation_step_m = max_translation_step_m
        self.max_rotation_step_rad = max_rotation_step_rad
        self.reset()

    def reset(self) -> None:
        """End the current episode and discard its references."""
        self._world_tcp0: np.ndarray | None = None
        self._robot_tcp0: np.ndarray | None = None
        self._previous_world_tcp: np.ndarray | None = None
        self._tracking_origin_id: object | None = None

    def _world_tcp(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        world_controller = pose_to_matrix(position, quaternion, quaternion_order=self.quaternion_order)
        # ^W T_E = ^W T_C @ ^C T_E: the fixed mount is a RIGHT multiply.
        return world_controller @ self.controller_to_tcp

    def begin_episode(
        self,
        controller_position: np.ndarray,
        controller_quaternion: np.ndarray,
        robot_tcp0: np.ndarray,
        *,
        tracking_origin_id: object | None = None,
    ) -> RelativePoseResult:
        """Capture both PICO and robot first-frame poses; return identity motion."""
        world_tcp0 = self._world_tcp(controller_position, controller_quaternion)
        self._world_tcp0 = world_tcp0
        self._previous_world_tcp = world_tcp0
        self._robot_tcp0 = _as_transform(robot_tcp0, name="robot_tcp0")
        self._tracking_origin_id = tracking_origin_id
        return RelativePoseResult(
            world_tcp=world_tcp0.copy(),
            delta_tcp=np.eye(4, dtype=np.float64),
            robot_target=self._robot_tcp0.copy(),
        )

    def update(
        self,
        controller_position: np.ndarray,
        controller_quaternion: np.ndarray,
        *,
        tracking_origin_id: object | None = None,
    ) -> RelativePoseResult:
        """Map one frame; raise on a detected mid-episode tracking-origin jump."""
        if self._world_tcp0 is None or self._robot_tcp0 is None:
            raise RuntimeError("begin_episode must be called before update")
        if self._tracking_origin_id is not None and tracking_origin_id != self._tracking_origin_id:
            raise TrackingOriginJumpError("PICO tracking_origin_id changed during the episode")

        world_tcp = self._world_tcp(controller_position, controller_quaternion)
        assert self._previous_world_tcp is not None
        step = relative_transform(self._previous_world_tcp, world_tcp)
        translation_step = float(np.linalg.norm(step[:3, 3]))
        rotation_step = _rotation_angle(step[:3, :3])
        if (self.max_translation_step_m is not None and translation_step > self.max_translation_step_m) or (
            self.max_rotation_step_rad is not None and rotation_step > self.max_rotation_step_rad
        ):
            raise TrackingOriginJumpError(
                "possible PICO recenter/tracking-origin jump: "
                f"translation step={translation_step:.3f} m, "
                f"rotation step={np.rad2deg(rotation_step):.1f} deg"
            )

        delta_tcp = relative_transform(self._world_tcp0, world_tcp)
        robot_target = self._robot_tcp0 @ delta_tcp
        self._previous_world_tcp = world_tcp
        return RelativePoseResult(
            world_tcp=world_tcp.copy(),
            delta_tcp=delta_tcp,
            robot_target=robot_target,
        )
