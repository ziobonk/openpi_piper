#!/usr/bin/env python3
"""Read-only Piper SDK demonstrations with absolute TCP and raw flange poses.

Move the arm externally or by hand. This process sends no arm or gripper commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "piper_sdk"))

from examples.piper.collection.collect_eef_udp import BackgroundEpisodeSaver  # noqa: E402
from examples.piper.collection.collect_eef_udp import LatestCamera  # noqa: E402
from examples.piper.collection.collect_eef_udp import PickPlaceWriter  # noqa: E402
from examples.piper.collection.collect_eef_udp import TerminalKeys  # noqa: E402
from piper_sdk import C_PiperInterface_V2  # noqa: E402


class PiperFeedback:
    """SDK feedback only; the TCP offset is expressed in the flange frame."""

    def __init__(self, can_name: str, offset_m: np.ndarray, open_width_mm: float, timeout_s: float):
        self.arm = C_PiperInterface_V2(can_name=can_name, judge_flag=False, can_auto_init=True, dh_is_offset=1)
        self.arm.ConnectPort(piper_init=False)
        self.offset_m = offset_m
        self.open_width_mm = open_width_mm
        self.timeout_s = timeout_s
        self._last_pose_stamp = self._last_grip_stamp = None
        self._last_pose_at = self._last_grip_at = 0.0
        self.last_flange_pose: np.ndarray | None = None

    def close(self):
        self.arm.DisconnectPort()

    def feedback_status(self) -> str:
        """Describe both SDK feedback streams without issuing a CAN command."""
        now = time.monotonic()
        pose = self.arm.GetArmEndPoseMsgs()
        grip = self.arm.GetArmGripperMsgs()

        def describe(message, last_at: float) -> str:
            stamp = float(message.time_stamp)
            if stamp <= 0:
                return "never received (timestamp=0)"
            age = "first frame" if last_at == 0 else f"last update {now - last_at:.2f}s ago"
            return f"{age}, stamp={stamp:.3f}, Hz={message.Hz:.1f}"

        return (
            f"TCP: {describe(pose, self._last_pose_at)}; "
            f"gripper: {describe(grip, self._last_grip_at)}; "
            f"CAN: {self.arm.GetCanFps():.1f} fps"
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        pose_msg = self.arm.GetArmEndPoseMsgs()
        grip_msg = self.arm.GetArmGripperMsgs()
        now = time.monotonic()
        pose_stamp = float(pose_msg.time_stamp)
        grip_stamp = float(grip_msg.time_stamp)
        if pose_stamp > 0 and pose_stamp != self._last_pose_stamp:
            self._last_pose_stamp, self._last_pose_at = pose_stamp, now
        if grip_stamp > 0 and grip_stamp != self._last_grip_stamp:
            self._last_grip_stamp, self._last_grip_at = grip_stamp, now
        if now - self._last_pose_at > self.timeout_s or now - self._last_grip_at > self.timeout_s:
            raise RuntimeError(self.feedback_status())
        ep = pose_msg.end_pose
        flange_m = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=np.float64) * 1e-6
        euler_rad = np.deg2rad([ep.RX_axis, ep.RY_axis, ep.RZ_axis]) * 1e-3
        rotation = Rotation.from_euler("xyz", euler_rad)
        tcp_m = flange_m + rotation.apply(self.offset_m)
        raw_grip = float(grip_msg.gripper_state.grippers_angle)
        if not np.isfinite(raw_grip) or raw_grip < -1000 or raw_grip > 71000:
            raise RuntimeError(f"Piper gripper feedback out of range: {raw_grip}")
        width_mm = np.clip(raw_grip / 70000.0, 0.0, 1.0) * self.open_width_mm
        action = np.concatenate([tcp_m, rotation.as_rotvec(), [width_mm]]).astype(np.float32)
        if not np.isfinite(action).all():
            raise RuntimeError("Piper TCP feedback is non-finite")
        self.last_flange_pose = np.concatenate([flange_m, rotation.as_rotvec()]).astype(np.float32)
        return np.array([width_mm], dtype=np.float32), action


def run(args: argparse.Namespace) -> None:
    base_shape = (args.base_height, args.base_width, 3)
    wrist_shape = (args.wrist_height, args.wrist_width, 3)
    offset = np.asarray(args.tcp_offset_m, dtype=np.float64)
    writer = PickPlaceWriter(
        args.output, args.task, fps=args.fps, base_shape=base_shape, wrist_shape=wrist_shape,
        state_dim=7, robot_type="piper_eef", tcp_offset_m=tuple(offset),
        gripper_open_width_mm=args.gripper_open_width_mm, raw_piper_sdk=True,
    )
    saver = BackgroundEpisodeSaver(writer)
    base = LatestCamera("base", base_shape, serial=args.base_camera)
    wrist = LatestCamera("wrist", wrist_shape, serial=args.wrist_rs_serial, cv_id=args.wrist_usb)
    robot = None
    recording = False
    last_error_at = 0.0
    try:
        robot = PiperFeedback(args.can_name, offset, args.gripper_open_width_mm, args.feedback_timeout)
        base.start()
        wrist.start()
        print("[Ready] recording absolute TCP + raw flange; c=start, s=save, d=discard, q=quit; hardware is read only")
        deadline = time.monotonic()
        with TerminalKeys() as keys:
            running = True
            while running:
                saver.check_error()
                events = []
                while True:
                    try:
                        events.append(keys.events.get_nowait())
                    except queue.Empty:
                        break
                base_frame = base.latest()
                wrist_frame = wrist.latest()
                if args.preview and base_frame is not None and wrist_frame is not None:
                    canvas = np.concatenate([
                        cv2.resize(base_frame[0], (640, 360)),
                        cv2.resize(wrist_frame[0], (480, 360)),
                    ], axis=1)
                    cv2.putText(canvas, "REC" if recording else "READY", (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (0, 255, 0), 2)
                    cv2.imshow("Piper SDK read-only collect", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if 0 < key < 128:
                        events.append(chr(key).lower())
                for key in events:
                    if key == "q":
                        running = False
                    elif key == "c":
                        try:
                            robot.read()
                            now = time.monotonic()
                            if base_frame is None or wrist_frame is None or now - min(base_frame[1], wrist_frame[1]) > args.camera_timeout:
                                raise RuntimeError("camera frame missing/stale")
                        except RuntimeError as exc:
                            print(f"[Recording] cannot start: {exc}")
                        else:
                            recording = True
                            print(f"[Recording] episode {saver.next_episode} started")
                    elif key == "s" and recording:
                        saver.save_episode()
                        recording = False
                    elif key == "d":
                        writer.discard()
                        recording = False
                if not running:
                    break
                try:
                    state, action = robot.read()
                    now = time.monotonic()
                    if base_frame is None or wrist_frame is None or now - min(base_frame[1], wrist_frame[1]) > args.camera_timeout:
                        raise RuntimeError("camera frame missing/stale")
                except RuntimeError as exc:
                    if time.monotonic() - last_error_at >= 2.0:
                        print(f"[Feedback] {exc}")
                        last_error_at = time.monotonic()
                    if recording:
                        if writer.recording:
                            saver.save_episode()
                        recording = False
                        print(f"[Recording] paused: {exc}")
                    time.sleep(0.05)
                    continue
                last_error_at = 0.0
                if recording:
                    writer.add_frame(action.copy(), action, base_frame[2], wrist_frame[2],
                                     flange_pose=robot.last_flange_pose)
                deadline += 1.0 / args.fps
                if deadline < time.monotonic() - 1.0 / args.fps:
                    deadline = time.monotonic()
                time.sleep(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        if recording and writer.recording:
            saver.save_episode()
        saver.close()
        base.stop()
        wrist.stop()
        if args.preview:
            cv2.destroyAllWindows()
        if robot is not None:
            robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local/datasets/piper_sdk_raw_eef"))
    parser.add_argument("--task")
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--base-camera", help="global D435i serial")
    wrist = parser.add_mutually_exclusive_group()
    wrist.add_argument("--wrist-rs-serial", help="wrist D435i serial")
    wrist.add_argument("--wrist-usb", type=int, help="wrist USB/OpenCV device index")
    parser.add_argument("--tcp-offset-m", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--probe-feedback", action="store_true", help="read TCP/gripper/CAN status for 5 seconds; no cameras or recording")
    parser.add_argument("--gripper-open-width-mm", type=float, default=70.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--base-width", type=int, default=1280)
    parser.add_argument("--base-height", type=int, default=720)
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    parser.add_argument("--camera-timeout", type=float, default=0.5)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not np.isfinite(args.tcp_offset_m).all() or np.linalg.norm(args.tcp_offset_m) > 1.0:
        parser.error("--tcp-offset-m must be finite and at most 1 metre")
    if not np.isfinite([args.fps, args.gripper_open_width_mm, args.feedback_timeout, args.camera_timeout]).all() or min(
        args.fps, args.gripper_open_width_mm, args.feedback_timeout, args.camera_timeout
    ) <= 0:
        parser.error("fps, gripper width and timeouts must be positive and finite")
    if min(args.base_width, args.base_height, args.wrist_width, args.wrist_height) <= 0:
        parser.error("camera dimensions must be positive")
    if args.probe_feedback:
        feedback = PiperFeedback(args.can_name, np.asarray(args.tcp_offset_m), args.gripper_open_width_mm, args.feedback_timeout)
        try:
            for _ in range(10):
                try:
                    state, action = feedback.read()
                    print(f"[OK] {feedback.feedback_status()}; TCP={action[:6].round(4)}; gripper={state[0]:.2f}mm")
                except RuntimeError as exc:
                    print(f"[WAIT] {exc}")
                time.sleep(0.5)
        finally:
            feedback.close()
        return
    if not args.task or not args.base_camera or (args.wrist_rs_serial is None) == (args.wrist_usb is None):
        parser.error("recording requires --task, --base-camera and one wrist camera")
    run(args)


if __name__ == "__main__":
    main()
