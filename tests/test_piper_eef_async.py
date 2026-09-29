"""异步 EEF 动作缓冲区的无硬件测试。"""

from pathlib import Path
import queue
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threading
from types import SimpleNamespace

import numpy as np
import pytest


from examples.piper.runtime.inference_eef_async import EEFActionBuffer
from examples.piper.runtime.inference_eef_async import PiperEEFAsyncInference


def _chunk(xs, angle=0.0):
    chunk = np.zeros((len(xs), 7), dtype=np.float64)
    chunk[:, 0] = xs
    chunk[:, 5] = angle
    chunk[:, 6] = xs
    return chunk


def test_skips_actions_executed_after_observation_and_smooths_overlap():
    buffer = EEFActionBuffer()
    buffer.integrate_new_chunk(_chunk([0, 1, 2]), 0, 8, 2)
    first = buffer.pop_next_action()
    buffer.mark_executed(first)
    observed_step = buffer.executed_steps()
    second = buffer.pop_next_action()
    buffer.mark_executed(second)

    dropped, size = buffer.integrate_new_chunk(_chunk([10, 11, 12, 13]), observed_step, 8, 2)
    assert (dropped, size) == (1, 3)
    assert buffer.pop_next_action()[0] == pytest.approx(5.0)
    assert buffer.pop_next_action()[0] == pytest.approx(2 + (12 - 2) * 2 / 3)
    assert buffer.pop_next_action()[0] == pytest.approx(13.0)


def test_orientation_blend_uses_short_rotation_arc():
    buffer = EEFActionBuffer()
    buffer.integrate_new_chunk(_chunk([0], np.deg2rad(179)), 0, 8, 2)
    old = buffer.pop_next_action()
    buffer.mark_executed(old)
    buffer.integrate_new_chunk(_chunk([0], np.deg2rad(-179)), 1, 8, 2)
    blended = buffer.pop_next_action()
    assert abs(blended[5]) == pytest.approx(np.pi, abs=1e-6)


def test_pause_discards_in_flight_inference_result():
    started = threading.Event()
    release = threading.Event()

    def infer(_observation):
        started.set()
        assert release.wait(timeout=2)
        return {"actions": _chunk([1, 2])}

    client = PiperEEFAsyncInference.__new__(PiperEEFAsyncInference)
    client._state_lock = threading.Lock()
    client._robot_lock = threading.Lock()
    client._shutdown = threading.Event()
    client._wake_inference = threading.Event()
    client._buffer = EEFActionBuffer()
    client._inferring = True
    client._generation = 1
    client._inference_rate = 3.0
    client._model_action_frame = "robot_absolute"
    client._action_horizon = 2
    client._exec_horizon = 2
    client._latency_k = 8
    client._min_smooth_steps = 2
    client._policy_client = SimpleNamespace(infer=infer)
    client._build_observation = dict

    thread = threading.Thread(target=client._inference_loop)
    thread.start()
    try:
        assert started.wait(timeout=2)
        with client._state_lock:
            client._inferring = False
            client._generation += 1
            client._buffer.clear()
        release.set()
    finally:
        client._shutdown.set()
        release.set()
        client._wake_inference.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert client._buffer.pop_next_action() is None


def test_control_consumes_buffer_while_network_inference_is_blocked():
    infer_started = threading.Event()
    release_infer = threading.Event()
    published = threading.Event()

    def infer(_observation):
        infer_started.set()
        assert release_infer.wait(timeout=2)
        return {"actions": _chunk([2])}

    def publish_and_stop(*_args):
        published.set()
        client._shutdown.set()
        return True

    client = PiperEEFAsyncInference.__new__(PiperEEFAsyncInference)
    client._state_lock = threading.Lock()
    client._robot_lock = threading.Lock()
    client._shutdown = threading.Event()
    client._wake_inference = threading.Event()
    client._input_queue = queue.Queue()
    client._buffer = EEFActionBuffer()
    client._buffer.integrate_new_chunk(_chunk([1]), 0, 8, 1)
    client._running = True
    client._inferring = True
    client._generation = 1
    client._period = 0.05
    client._inference_rate = 3.0
    client._model_action_frame = "robot_absolute"
    client._action_horizon = 1
    client._exec_horizon = 1
    client._latency_k = 8
    client._min_smooth_steps = 1
    client._policy_client = SimpleNamespace(infer=infer)
    client._build_observation = dict
    client._robot = SimpleNamespace(get_eef_pose=lambda: np.zeros(6))
    client._publish_action = publish_and_stop

    worker = threading.Thread(target=client._inference_loop)
    worker.start()
    try:
        assert infer_started.wait(timeout=2)
        PiperEEFAsyncInference._control_loop(client)
        assert published.is_set()
        assert not release_infer.is_set()
    finally:
        with client._state_lock:
            client._inferring = False
            client._generation += 1
        release_infer.set()
        client._shutdown.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
