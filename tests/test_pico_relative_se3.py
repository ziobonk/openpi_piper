from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pico_relative_se3 import PicoRelativeTcpMapper
from pico_relative_se3 import TrackingOriginJumpError
from pico_relative_se3 import invert_transform
from pico_relative_se3 import pose_to_matrix
from pico_relative_se3 import relative_transform


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def test_relative_transform_has_expected_translation_formula():
    reference = _axis_angle([0, 0, 1], np.deg2rad(90))
    reference[:3, 3] = [1.0, 2.0, 3.0]
    current = _axis_angle([1, 0, 0], np.deg2rad(20))
    current[:3, 3] = [2.0, 4.0, 6.0]

    delta = relative_transform(reference, current)

    np.testing.assert_allclose(delta[:3, :3], reference[:3, :3].T @ current[:3, :3])
    np.testing.assert_allclose(delta[:3, 3], reference[:3, :3].T @ (current[:3, 3] - reference[:3, 3]))
    # In this example the world subtraction [1, 2, 3] is not the answer.
    np.testing.assert_allclose(delta[:3, 3], [2.0, -1.0, 3.0], atol=1e-12)


def test_arbitrary_fixed_world_transform_cancels_numerically():
    t0 = _axis_angle([1, 2, -1], 0.73)
    t0[:3, 3] = [0.31, -1.2, 2.4]
    tt = _axis_angle([-2, 1, 3], -1.11)
    tt[:3, 3] = [-0.7, 0.2, 4.1]
    g = _axis_angle([0.2, -0.8, 0.5], 2.17)
    g[:3, 3] = [10.0, -3.0, 0.4]

    expected = invert_transform(t0) @ tt
    after_restart = invert_transform(g @ t0) @ (g @ tt)

    np.testing.assert_allclose(after_restart, expected, atol=2e-15)


def test_mapper_right_multiplies_mount_and_left_multiplies_robot_start():
    mount = _axis_angle([1, 0, 0], np.deg2rad(90))
    mount[:3, 3] = [0.0, 0.0, 0.15]
    robot0 = _axis_angle([0, 1, 0], np.deg2rad(-30))
    robot0[:3, 3] = [0.5, -0.2, 0.8]
    mapper = PicoRelativeTcpMapper(mount, max_translation_step_m=None, max_rotation_step_rad=None)

    first = mapper.begin_episode([1, 2, 3], [0, 0, 0, 1], robot0)
    current_controller = pose_to_matrix([1.1, 1.9, 3.2], [0, 0, np.sin(0.2), np.cos(0.2)])
    result = mapper.update(current_controller[:3, 3], [0, 0, np.sin(0.2), np.cos(0.2)])

    expected_world_tcp = current_controller @ mount
    expected_delta = invert_transform(first.world_tcp) @ expected_world_tcp
    np.testing.assert_allclose(result.world_tcp, expected_world_tcp)
    np.testing.assert_allclose(result.delta_tcp, expected_delta)
    np.testing.assert_allclose(result.robot_target, robot0 @ expected_delta)


def test_quaternion_order_is_explicit():
    xyzw = pose_to_matrix([0, 0, 0], [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    wxyz = pose_to_matrix([0, 0, 0], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], quaternion_order="wxyz")
    np.testing.assert_allclose(xyzw, wxyz)


def test_tracking_origin_change_or_large_jump_aborts_episode():
    mapper = PicoRelativeTcpMapper(np.eye(4), max_translation_step_m=0.2)
    mapper.begin_episode([0, 0, 0], [0, 0, 0, 1], np.eye(4), tracking_origin_id="origin-a")
    with pytest.raises(TrackingOriginJumpError, match="tracking_origin_id"):
        mapper.update([0, 0, 0], [0, 0, 0, 1], tracking_origin_id="origin-b")

    mapper.begin_episode([0, 0, 0], [0, 0, 0, 1], np.eye(4))
    with pytest.raises(TrackingOriginJumpError, match="possible PICO recenter"):
        mapper.update([1, 0, 0], [0, 0, 0, 1])
