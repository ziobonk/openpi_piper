"""Hardware-free checks for Piper asynchronous chunk alignment and fusion."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.piper.runtime.inference_eef_async import PiperEEFAsyncInference
from examples.piper.runtime.temporal_buffers import NaiveActionBuffer, TemporalEnsemblingActionBuffer
from openpi.policies.piper_chunk_relative import ChunkRelativeEEFActions


def _chunk(x_values, rot=0.0):
    rows = np.zeros((len(x_values), 7), dtype=np.float64)
    rows[:, 0] = x_values
    rows[:, 5] = rot
    rows[:, 6] = x_values
    return rows


def test_naive_replaces_pending_actions_and_rejects_stale_result():
    buffer = NaiveActionBuffer()
    buffer.integrate_new_chunk(_chunk([1, 2, 3]), 0, 8, 8)
    first = buffer.pop_next_action()
    buffer.mark_executed(first)
    assert buffer.integrate_new_chunk(_chunk([10, 11, 12]), 0, 8, 8) == (1, 2)
    assert buffer.pop_next_action()[0] == 11
    buffer.mark_executed(_chunk([11])[0])
    assert buffer.integrate_new_chunk(_chunk([20, 21, 22]), 0, 1, 8) == (2, 1)
    assert buffer.pop_next_action()[0] == 12


def test_ensemble_fuses_at_same_tick_and_uses_so3():
    buffer = TemporalEnsemblingActionBuffer(exp_weight_m=0)
    buffer.integrate_new_chunk(_chunk([0, 1], np.deg2rad(179)), 0, 8, 8)
    buffer.integrate_new_chunk(_chunk([2, 3], np.deg2rad(-179)), 0, 8, 8)
    action = buffer.pop_next_action()
    assert action[0] == pytest.approx(1)
    assert abs(action[5]) == pytest.approx(np.pi, abs=1e-6)
    buffer.mark_executed(action)
    assert buffer.pop_next_action()[0] == pytest.approx(2)


def test_rtc_request_shifts_full_chunk_and_preserves_tcp_frame():
    client = PiperEEFAsyncInference.__new__(PiperEEFAsyncInference)
    client._action_horizon = 50
    client._exec_horizon = 3
    client._rtc_previous = _chunk(np.arange(50))
    client._rtc_previous_step = 4
    client._rtc_delays = [0.1, 0.2]
    request = client._rtc_request({"observation/state": np.array([5.0])}, 6)
    assert request["enable_rtc"] is True
    assert request["execute_horizon"] == 4
    assert request["inference_delay"] == 3
    assert request["prev_action_chunk"].shape == (50, 7)
    np.testing.assert_array_equal(request["prev_action_chunk"][:2, 0], [2, 3])
    np.testing.assert_array_equal(request["prev_action_chunk"][-2:, 0], [49, 49])
    client._rtc_previous = None
    assert "prev_action_chunk" not in client._rtc_request({}, 6)


def test_previous_absolute_chunk_uses_training_relative_transform():
    previous = _chunk([0.4, 0.5, 0.6])
    previous[:, 3:6] = Rotation.from_euler("z", 90, degrees=True).as_rotvec()
    result = ChunkRelativeEEFActions()({"state": np.array([5.0]), "actions": previous})
    np.testing.assert_allclose(result["actions"][0, :6], 0, atol=1e-6)
    np.testing.assert_allclose(result["actions"][:, 1], [0, -0.1, -0.2], atol=1e-6)
    np.testing.assert_allclose(result["actions"][:, 6], [0.4, 0.5, 0.6])
