#!/usr/bin/env python3
"""Piper EEF 异步推理客户端。

沿用 inference_eef.py 的观测格式、相机及夹爪控制：默认直接使用
Robot Base 坐标系下的绝对 TCP 位姿，不执行 PICO/机械臂坐标转换。推理线程按 --inference_rate 查询策略服务器；主线程按
CONTROL_FREQ 执行动作。新动作块按观测之后已经执行的步数跳过过期前缀，并与
缓冲区的剩余动作平滑衔接。

用法:
    python examples/piper/runtime/inference_eef_async.py --host localhost --port 6006 \\
        --rs2_base 231122071797 --usb_wrist 0

按 Enter 开始、s 暂停、r 重置、q 退出。--interactive 在每次开始前输入指令。
法兰与爪尖之间的 GRIPPER_TCP_OFFSET_M 补偿仍由底层控制器负责。
"""

import argparse
from collections import deque
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import threading
import time
from typing import Optional

import cv2
from examples.piper.runtime import inference_eef as eef
from examples.piper.runtime.temporal_buffers import NaiveActionBuffer, TemporalEnsemblingActionBuffer
import numpy as np
from scipy.spatial.transform import Rotation


class EEFActionBuffer:
    """保存尚未执行的 EEF 目标，跨线程替换并平滑动作块。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._actions: deque[np.ndarray] = deque()
        self._last_action: np.ndarray | None = None
        self._executed = 0

    def clear(self):
        with self._lock:
            self._actions.clear()
            self._last_action = None
            self._executed = 0

    def executed_steps(self) -> int:
        with self._lock:
            return self._executed

    def pop_next_action(self) -> np.ndarray | None:
        with self._lock:
            if not self._actions:
                return None
            action = self._actions.popleft()
            return action.copy()

    def mark_executed(self, action: np.ndarray):
        with self._lock:
            self._last_action = action.copy()
            self._executed += 1

    def integrate_new_chunk(
        self,
        actions: np.ndarray,
        observed_step: int,
        max_latency_steps: int,
        min_smooth_steps: int,
    ) -> tuple[int, int]:
        """返回 (跳过的过期步数, 新缓冲区长度)。"""
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.all(np.isfinite(chunk)):
            raise ValueError(f"策略动作必须是有限的 (H, 7) 数组，收到 {chunk.shape}")
        with self._lock:
            elapsed = max(0, self._executed - observed_step)
            if elapsed > max_latency_steps or elapsed >= len(chunk):
                return elapsed, len(self._actions)
            drop_n = elapsed
            new = [a.copy() for a in chunk[drop_n:]]

            old = [a.copy() for a in self._actions]
            if not old and self._last_action is not None:
                old = [self._last_action.copy()]
            if not old:
                self._actions = deque(new)
                return drop_n, len(self._actions)

            min_smooth_steps = max(1, min_smooth_steps)
            if len(old) < min_smooth_steps:
                old.extend(old[-1].copy() for _ in range(min_smooth_steps - len(old)))

            overlap = min(len(old), len(new))
            blended = []
            for i in range(overlap):
                alpha = (i + 1) / (overlap + 1)
                pose = eef._interp_pose(old[i][:6], new[i][:6], alpha)
                grip = old[i][6] + (new[i][6] - old[i][6]) * alpha
                blended.append(np.concatenate([pose, [grip]]))
            self._actions = deque(blended + new[overlap:])
            return drop_n, len(self._actions)


class PiperEEFAsyncInference(eef.PiperEEFInference):
    """独立推理线程 + 固定频率 EEF 控制循环。"""

    def __init__(
        self,
        *,
        inference_rate: float = 3.0,
        latency_k: int = 8,
        min_smooth_steps: int = 8,
        mode: str = "temporal_smoothing",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if inference_rate <= 0:
            raise ValueError("--inference_rate 必须大于 0")
        if latency_k < 0 or min_smooth_steps < 1:
            raise ValueError("--latency_k 必须非负，--min_smooth_steps 必须大于 0")
        if mode not in {"async", "temporal_smoothing", "temporal_ensembling", "rtc"}:
            raise ValueError(f"未知异步推理模式: {mode}")
        if mode == "rtc" and self._model_action_frame != "chunk_relative":
            raise ValueError("RTC 模式需要 chunk_relative 动作坐标系")
        self._mode = mode
        self._inference_rate = inference_rate
        self._latency_k = latency_k
        self._min_smooth_steps = min_smooth_steps
        self._buffer = (
            TemporalEnsemblingActionBuffer() if mode == "temporal_ensembling"
            else EEFActionBuffer() if mode == "temporal_smoothing"
            else NaiveActionBuffer()
        )
        self._rtc_previous: np.ndarray | None = None
        self._rtc_previous_step = 0
        self._rtc_delays: deque[float] = deque(maxlen=10)
        self._state_lock = threading.Lock()
        self._robot_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._wake_inference = threading.Event()
        self._generation = 0
        # 提示词由键盘线程在开始时读取，推理线程只使用已确定的提示词。
        self._prompt_each_start = self._interactive
        self._interactive = False

    def run(self):
        print("=" * 60)
        print("Piper EEF 异步推理客户端")
        print(f"模式:           {self._mode}")
        print(f"服务器:         {self._policy_host}:{self._policy_port}")
        print(f"模型动作坐标系: {self._model_action_frame.upper()}")
        print(f"控制频率:       {eef.CONTROL_FREQ} Hz")
        print(f"推理频率上限:   {self._inference_rate:g} Hz")
        print(f"动作块上限:     {min(self._action_horizon, self._exec_horizon)} 步")
        if self._use_interp:
            print(
                f"位姿插值:       启用 (max_pos={self._max_pos_speed:.2f} m/s, "
                f"max_rot={self._max_rot_speed:.2f} rad/s, freq={self._interp_freq} Hz)"
            )
        print("=" * 60)

        camera_started = False
        worker = None
        try:
            if not self._robot.enable():
                return
            if self._camera:
                camera_started = True
                self._camera.start()
                print("[Camera] 已启动")
            self._connect_policy()
            self._running = True
            worker = threading.Thread(target=self._inference_loop, name="piper-eef-inference", daemon=True)
            worker.start()
            threading.Thread(target=self._keyboard_listener, name="piper-eef-keyboard", daemon=True).start()
            self._control_loop()
        except KeyboardInterrupt:
            print("\n[INFO] 中断信号，退出中...")
        finally:
            with self._state_lock:
                self._running = False
                self._inferring = False
                self._generation += 1
                self._buffer.clear()
                self._clear_rtc_history()
            self._shutdown.set()
            self._wake_inference.set()
            if worker is not None:
                worker.join(timeout=2.0)
            try:
                if self._camera and camera_started:
                    self._camera.stop()
            finally:
                cv2.destroyAllWindows()
                print("[INFO] 已退出 (机械臂和夹爪保持使能)")

    def _connect_policy(self):
        super()._connect_policy()
        if self._mode == "rtc":
            meta = self._policy_client.get_server_metadata()
            if not (meta.get("rtc_supported") and meta.get("rtc_action_dim") == 7):
                raise RuntimeError("RTC 需要使用 pi05_piper_pick_cube_0928_chunk_relative_rtc 启动策略服务")

    def _clear_rtc_history(self):
        if getattr(self, "_mode", None) == "rtc":
            self._rtc_previous = None
            self._rtc_previous_step = 0
            self._rtc_delays.clear()

    def _rtc_request(self, observation: dict, observed_step: int) -> dict:
        """Align the previous full chunk with the current observation tick."""
        request = dict(observation)
        request["enable_rtc"] = True
        request["execute_horizon"] = min(self._action_horizon, self._exec_horizon + 1)
        if self._rtc_previous is not None:
            shift = min(max(0, observed_step - self._rtc_previous_step), len(self._rtc_previous) - 1)
            previous = self._rtc_previous[shift:]
            if shift:
                previous = np.concatenate(
                    [previous, np.repeat(previous[-1:], shift, axis=0)], axis=0
                )
            request["prev_action_chunk"] = previous.copy()
            if self._rtc_delays:
                request["inference_delay"] = min(
                    self._action_horizon - 1,
                    max(0, round(float(np.median(self._rtc_delays)) * eef.CONTROL_FREQ)),
                )
        return request

    def _inference_loop(self):
        period = 1.0 / self._inference_rate
        while not self._shutdown.is_set():
            with self._state_lock:
                active = self._inferring
                generation = self._generation
            if not active:
                self._wake_inference.wait(timeout=0.1)
                self._wake_inference.clear()
                continue

            started = time.monotonic()
            try:
                # 与控制指令共用机械臂锁，保证状态采样不会穿插在下发指令中。
                with self._robot_lock:
                    with self._state_lock:
                        if not self._inferring or generation != self._generation:
                            continue
                    observed_step = self._buffer.executed_steps()
                    observation = self._build_observation()
                    chunk_tcp_base = (
                        self._chunk_tcp_base.copy() if self._model_action_frame == "chunk_relative" else None
                    )
                    if getattr(self, "_mode", "temporal_smoothing") == "rtc":
                        observation = self._rtc_request(observation, observed_step)
                with self._state_lock:
                    if not self._inferring or generation != self._generation:
                        continue
                observed_at = time.monotonic()
                result = self._policy_client.infer(observation)
                infer_seconds = time.monotonic() - observed_at
                full_chunk = np.asarray(result["actions"], dtype=np.float64)
                if full_chunk.ndim != 2 or full_chunk.shape[1] < 7 or not np.isfinite(full_chunk).all():
                    raise ValueError(f"无效的策略动作块: {full_chunk.shape}")
                full_chunk = full_chunk[:self._action_horizon, :7].copy()
                if getattr(self, "_mode", "temporal_smoothing") == "rtc" and len(full_chunk) != self._action_horizon:
                    raise ValueError(f"RTC 需要完整的 {self._action_horizon} 步动作块")
                if chunk_tcp_base is not None:
                    # Fusion and RTC history both use Robot Base TCP coordinates.
                    for action in full_chunk:
                        action[:6] = eef._compose_pose6(chunk_tcp_base, action[:6])
                actions = eef._execution_actions(full_chunk, self._model_action_frame)
                actions = actions[:self._exec_horizon]
                with self._state_lock:
                    if self._inferring and generation == self._generation:
                        dropped, buffered = self._buffer.integrate_new_chunk(
                            actions,
                            observed_step,
                            self._latency_k,
                            self._min_smooth_steps,
                        )
                        if (
                            getattr(self, "_mode", "temporal_smoothing") == "rtc"
                            and dropped <= self._latency_k
                            and dropped < len(actions)
                        ):
                            self._rtc_previous = full_chunk
                            self._rtc_previous_step = observed_step
                            self._rtc_delays.append(infer_seconds)
                    else:
                        continue  # 暂停/重置后丢弃旧请求的结果
                print(
                    f"[Policy] obs={observed_at - started:.3f}s "
                    f"infer={time.monotonic() - observed_at:.3f}s "
                    f"chunk={len(actions)} drop={dropped} buffer={buffered}"
                )
            except Exception as exc:
                print(f"[Policy] 查询失败: {exc}")
                if self._shutdown.wait(timeout=1.0):
                    break
                try:
                    self._connect_policy()
                    print("[Policy] 重连成功")
                except Exception as reconnect_exc:
                    print(f"[Policy] 重连失败: {reconnect_exc}")
            self._shutdown.wait(timeout=max(0.0, period - (time.monotonic() - started)))

    def _process_commands(self):
        while True:
            try:
                command = self._input_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_command(command)

    def _handle_command(self, command):
        cmd, prompt = command if isinstance(command, tuple) else (command, None)
        if cmd == "start":
            with self._state_lock:
                should_start = not self._inferring
            if not should_start:
                return
            with self._robot_lock:
                robot_tcp0 = self._robot.get_eef_pose().astype(np.float64)
            with self._state_lock:
                if not self._inferring:
                    if prompt is not None:
                        self._default_prompt = prompt
                    self._robot_tcp0 = robot_tcp0
                    self._generation += 1
                    self._buffer.clear()
                    self._clear_rtc_history()
                    self._inferring = True
                    self._wake_inference.set()
                    print("[Control] 开始推理...")
        elif cmd in ("stop", "quit", "reset"):
            with self._state_lock:
                was_inferring = self._inferring
                self._inferring = False
                self._generation += 1
                self._buffer.clear()
                self._clear_rtc_history()
            if cmd == "stop":
                print("[Control] 暂停推理")
            elif cmd == "quit":
                self._running = False
                self._shutdown.set()
            else:
                with self._robot_lock:
                    self._robot.go_to_init_pose()
                    self._robot_tcp0 = self._robot.get_eef_pose().astype(np.float64)
                if was_inferring:
                    with self._state_lock:
                        self._generation += 1
                        self._inferring = True
                        self._wake_inference.set()

    def _keyboard_listener(self):
        print("\n操作提示: [Enter] 开始 | [s] 暂停 | [r] 重置 | [q] 退出\n")
        while self._running:
            try:
                line = sys.stdin.readline()
                if line == "":
                    break
                cmd = line.strip().lower()
                if not cmd:
                    if self._prompt_each_start:
                        prompt = input("指令: ").strip() or self._default_prompt
                        self._input_queue.put(("start", prompt))
                    else:
                        self._input_queue.put("start")
                elif cmd == "s":
                    self._input_queue.put("stop")
                elif cmd == "r":
                    self._input_queue.put("reset")
                elif cmd == "q":
                    self._input_queue.put("quit")
                    break
            except (EOFError, OSError):
                break

    def _publish_action(
        self, action: np.ndarray, current_pose: np.ndarray, generation: int
    ) -> Optional[np.ndarray]:
        if self._model_action_frame in {"robot_absolute", "chunk_relative"}:
            target_pose = np.asarray(action[:6], dtype=np.float64)
        else:
            if self._robot_tcp0 is None:
                raise RuntimeError("episode robot TCP reference is not initialized")
            runtime_target = self._model_pose_to_runtime_pose(action[:6])
            target_pose = eef._compose_pose6(self._robot_tcp0, runtime_target)
        gripper_mm = self._gripper_command(
            self._model_gripper_to_runtime(float(action[6])) * self._gripper_prediction_scale
        )

        if not self._use_interp:
            with self._robot_lock:
                self._robot.send_eef_command(target_pose)
                self._robot.send_gripper_command(gripper_mm)
                self._buffer.mark_executed(action)
            return target_pose

        dpos = float(np.linalg.norm(target_pose[:3] - current_pose[:3]))
        r_cur = Rotation.from_rotvec(current_pose[3:6])
        r_tgt = Rotation.from_rotvec(target_pose[3:6])
        drot = float(np.linalg.norm((r_tgt * r_cur.inv()).as_rotvec()))
        duration = max(dpos / self._max_pos_speed, drot / self._max_rot_speed, self._period)
        inner_dt = 1.0 / self._interp_freq
        steps = max(1, round(duration / inner_dt))
        for i in range(1, steps + 1):
            self._process_commands()
            with self._state_lock:
                if not self._inferring or generation != self._generation:
                    return None
            with self._robot_lock:
                self._robot.send_eef_command(eef._interp_pose(current_pose, target_pose, i / steps))
                if i == steps:
                    self._robot.send_gripper_command(gripper_mm)
                    self._buffer.mark_executed(action)
            if self._shutdown.wait(timeout=inner_dt):
                return None
        return target_pose

    def _control_loop(self):
        fps_counter = eef._FPSCounter("control")
        current_pose = None
        previous_generation = None
        total_steps = 0
        while self._running and not self._shutdown.is_set():
            started = time.monotonic()
            self._process_commands()
            with self._state_lock:
                active = self._inferring
                generation = self._generation
                action = self._buffer.pop_next_action() if active else None

            if generation != previous_generation:
                current_pose = None
                previous_generation = generation
            if not active or action is None:
                self._display_cameras()
                self._shutdown.wait(timeout=max(0.0, self._period - (time.monotonic() - started)))
                continue

            if current_pose is None:
                with self._robot_lock:
                    current_pose = self._robot.get_eef_pose()
            published_target = self._publish_action(action, current_pose, generation)
            if published_target is not None:
                current_pose = published_target
                fps_counter.tick()
                total_steps += 1
                if total_steps % 5 == 0:
                    self._display_cameras()
                if total_steps % 100 == 0:
                    with self._robot_lock:
                        state = self._robot.get_state()
                    pos = ", ".join(f"{v:.4f}" for v in state[:3])
                    rot = ", ".join(f"{v:.3f}" for v in state[3:6])
                    print(
                        f"[{total_steps:5d}] fps={fps_counter.fps:.1f} | "
                        f"pos=[{pos}] rot=[{rot}] grip={state[6]:.1f}mm | "
                        f"cmd_grip={action[6]:.1f}mm"
                    )
            else:
                current_pose = None
            self._shutdown.wait(timeout=max(0.0, self._period - (time.monotonic() - started)))

    def _build_observation(self) -> dict:
        """构建与训练数据同坐标系的当前观测。"""
        state = self._robot.get_state().copy()
        if self._model_action_frame == "chunk_relative":
            self._chunk_tcp_base = state[:6].astype(np.float64).copy()
            state = state[6:7]
        elif self._model_action_frame != "robot_absolute":
            if self._robot_tcp0 is None:
                raise RuntimeError("episode robot TCP reference is not initialized")
            state[:6] = eef._relative_pose6(self._robot_tcp0, state[:6])
            state[:6] = self._runtime_pose_to_model_pose(state[:6])
            state[6] = self._runtime_gripper_to_model(float(state[6]))

        base_image = np.zeros((224, 224, 3), dtype=np.uint8)
        wrist_image = base_image.copy()
        if self._camera:
            if self._has_base_cam:
                base_image = self._camera.get_base()
            if self._has_wrist_cam:
                wrist_image = self._camera.get_wrist()

        return {
            "observation/state": state.astype(np.float32),
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "prompt": self._default_prompt,
        }


def main(args=None, *, mode: str = "temporal_smoothing"):
    parser = argparse.ArgumentParser(description="Piper EEF 异步策略服务器推理客户端")
    parser.add_argument("--inference_rate", type=float, default=3.0, help="推理频率上限 Hz (默认: 3)")
    parser.add_argument("--latency_k", type=int, default=8, help="新动作块最多跳过的过期步数 (默认: 8)")
    parser.add_argument("--min_smooth_steps", type=int, default=8, help="新旧动作块的最少平滑步数 (默认: 8)")
    if args is None:
        args = eef._parse_args(parser)
    if args.dataset:
        eef.main(args)
        return
    if args.action_horizon < 1 or args.exec_horizon < 1 or args.max_pos_speed <= 0 or args.max_rot_speed <= 0:
        parser.error("--action_horizon / --exec_horizon 和最大 EEF 速度必须大于 0")
    if args.inference_rate <= 0 or args.latency_k < 0 or args.min_smooth_steps < 1:
        parser.error("--inference_rate / --min_smooth_steps 必须大于 0，--latency_k 必须非负")

    if args.model_action_frame in {"pico_absolute", "tcp_absolute"} and not args.raw_pico_dataset:
        parser.error("真机绝对位姿模式需要 --absolute_pose_dataset")
    if args.model_action_frame in {"pico", "pico_absolute", "tcp_absolute"} and args.legacy_pick_cube_tcp:
        parser.error("PICO 模式不能与 --legacy_pick_cube_tcp 同时使用")
    pico_reference_pose = None
    model_gripper_range = None
    if args.model_action_frame in {"pico_absolute", "tcp_absolute"}:
        pico_reference_pose, model_gripper_range = eef._load_raw_pico_reference(
            args.raw_pico_dataset, args.raw_pico_episode,
            pose_frame="pico_world_tcp" if args.model_action_frame == "tcp_absolute" else "pico_teleop",
        )

    inference = PiperEEFAsyncInference(
        host=args.host,
        port=args.port,
        can_name=args.can_name,
        rs2_base_serial=args.rs2_base,
        cv_base_id=args.cam_ids[0] if args.cam_ids else None,
        cv_wrist_id=args.usb_wrist
        if args.usb_wrist is not None
        else (args.cam_ids[1] if len(args.cam_ids) > 1 else None),
        wrist_exposure=args.wrist_exposure,
        action_horizon=args.action_horizon,
        exec_horizon=args.exec_horizon,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
        interp_freq=args.interp_freq if args.interp_freq and args.interp_freq > 0 else None,
        gripper_port=args.gripper_port or None,
        gripper_close_rad=args.gripper_close_rad,
        gripper_open_width_mm=args.gripper_open_width_mm,
        gripper_binary_threshold_mm=(
            args.gripper_binary_threshold_mm if args.gripper_binary_threshold_mm is not None
            else args.gripper_open_width_mm / 2 if args.model_action_frame in {"pico_absolute", "tcp_absolute"}
            else None
        ),
        gripper_prediction_scale=(
            args.gripper_prediction_scale if args.gripper_prediction_scale is not None
            else 1.0 if args.model_action_frame in {"chunk_relative", "pico_absolute", "tcp_absolute"}
            else eef.GRIPPER_PREDICTION_SCALE
        ),
        binary_gripper=(
            args.binary_gripper if args.binary_gripper is not None
            else args.model_action_frame != "chunk_relative"
        ),
        gripper_current=args.gripper_current,
        gripper_speed=args.gripper_speed,
        default_prompt=args.prompt or eef.DEFAULT_PROMPT,
        interactive=args.interactive,
        model_action_frame=args.model_action_frame,
        legacy_pick_cube_tcp=args.legacy_pick_cube_tcp,
        pico_reference_pose=pico_reference_pose,
        model_gripper_range=model_gripper_range,
        inference_rate=args.inference_rate,
        latency_k=args.latency_k,
        min_smooth_steps=args.min_smooth_steps,
        mode=mode,
    )
    inference.run()


if __name__ == "__main__":
    main()
