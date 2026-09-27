"""Piper EEF action chunks relative to each chunk's first TCP pose."""

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class ChunkRelativeEEFActions(transforms.DataTransformFn):
    """Convert absolute TCP targets to ``inv(T_actions[0]) @ T_actions[k]``.

    The model state contains only the current gripper width. The first action
    is aligned to that observation and becomes the identity pose. Gripper
    targets remain absolute widths in millimetres. During inference there are
    no input actions, so this transform only validates the state shape.
    """

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape != (1,) or not np.isfinite(state).all():
            raise ValueError(f"chunk-relative Piper state must be one finite gripper value, got {state.shape}")
        if "actions" not in data:
            return data

        actions = np.asarray(data["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 2 or not np.isfinite(actions).all():
            raise ValueError(f"chunk-relative Piper actions must be finite (H>=2, 7), got {actions.shape}")

        base_rotation = Rotation.from_rotvec(actions[0, 3:6])
        rotation = Rotation.from_rotvec(actions[:, 3:6])
        relative = actions.copy()
        relative[:, :3] = base_rotation.inv().apply(actions[:, :3] - actions[0, :3])
        relative[:, 3:6] = (base_rotation.inv() * rotation).as_rotvec()
        relative[0, :6] = 0.0
        data["actions"] = relative.astype(np.float32)
        return data
