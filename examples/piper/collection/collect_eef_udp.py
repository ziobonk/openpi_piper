#!/usr/bin/env python3
"""单臂 UDP 遥操采集，写入与 pick_place 相同的 LeRobot v2.1 EEF 格式。

UDP 数据格式沿用 demo_dual_udp_teleop.py 的 lightumi.teleop_sender v1。
默认取右手 arms[index=1]，用 RIGHT_T265_TO_ROBOT 将 T265 相对运动映射到 Piper TCP。数据集中的
state 是下发前读取的实际 TCP 位姿，actions 是这一帧下发的绝对 TCP 目标；
两者均为机械臂基坐标系下的 TCP 位姿；PiperEEFController 已对法兰到爪尖
的 GRIPPER_TCP_OFFSET_M 做补偿。夹爪宽度单位为 mm，不做 PICO 坐标转换。

示例:
    python examples/piper/collection/collect_eef_udp.py --output ./piper_udp_eef \\
        --rs2_base 231122071797 --usb_wrist 0 --can_name can0

    # 只检查 UDP，不连接机械臂或相机
    python examples/piper/collection/collect_eef_udp.py --udp_test --print_udp

    # 无 UDP 测试采集：读取机械臂 EEF 和相机，不下发 EEF 移动指令
    python examples/piper/collection/collect_eef_udp.py --no_udp \\
        --rs2_base 231122071797 --usb_wrist 0

按键: e 遥操开关, c 开始录制, s 后台保存本集, d 丢弃本集, r 回初始位姿,
      z 重置遥操参考, t 只旋转开关, g/o/p 手动夹爪, q 退出。
终端支持单键操作；相机预览窗口获得焦点时也可使用相同按键。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import queue
import select
import socket
import sys
import termios
import threading
import time
import tty

import cv2
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from examples.piper.runtime import inference_eef as eef
import numpy as np
from scipy.spatial.transform import Rotation

DATASET_FPS = 20.0
BASE_SHAPE = (720, 1280, 3)
WRIST_SHAPE = (480, 640, 3)
STATE_NAMES = ["left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper_width"]
# 与 demo_dual_udp_teleop.py 的右臂映射一致，仅用于遥操增量。
RIGHT_T265_TO_ROBOT = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
DEFAULT_TASK = "<control mode> end effector <control mode>pick up the red bottle cap and place it into the cup."


@dataclass(frozen=True)
class ArmSample:
    sequence: int
    received_at: float
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    gripper_angle_deg: float | None


def parse_arm_sample(data: bytes, arm_index: int, received_at: float) -> ArmSample | None:
    """从 lightumi v1 包中取单臂有效位姿；坏包直接丢弃。"""
    try:
        packet = json.loads(data.decode("utf-8"))
        if not isinstance(packet, dict) or packet.get("type") != "lightumi.teleop_sender":
            return None
        if int(packet.get("version", -1)) != 1:
            return None
        for arm in packet.get("arms", []):
            if int(arm.get("index", -1)) != arm_index:
                continue
            pose = arm.get("pose") or {}
            if not pose.get("valid"):
                return None
            pos = np.asarray(pose["position_m"], dtype=np.float64)
            quat = np.asarray(pose["quaternion_xyzw"], dtype=np.float64)
            if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all():
                return None
            norm = np.linalg.norm(quat)
            if norm < 1e-6:
                return None
            grip = arm.get("gripper") or {}
            angle = float(grip["angle_deg"]) if grip.get("valid") else None
            if angle is not None and not np.isfinite(angle):
                angle = None
            return ArmSample(
                sequence=int(packet.get("sequence", 0)),
                received_at=received_at,
                position_m=pos,
                quaternion_xyzw=quat / norm,
                gripper_angle_deg=angle,
            )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return None


class UDPPoseReceiver:
    def __init__(self, host: str, port: int, arm_index: int, *, print_packets: bool = False):
        self._arm_index = arm_index
        self._print_queue: queue.Queue[tuple[tuple[str, int], bytes]] | None = (
            queue.Queue(maxsize=512) if print_packets else None
        )
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        try:
            self._socket.bind((host, port))
        except OSError as exc:
            self._socket.close()
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError(f"UDP {host}:{port} 已被占用，请关闭其他 UDP 测试或遥操进程") from exc
            raise
        self._socket.settimeout(0.2)
        self._lock = threading.Lock()
        self._latest: ArmSample | None = None
        self._last_packet_at: float | None = None
        self._packet_count = 0
        self._valid_count = 0
        self._dropped_print_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="piper-udp", daemon=True)
        self._print_thread = (
            threading.Thread(target=self._print_loop, name="piper-udp-print", daemon=True) if print_packets else None
        )

    def start(self):
        if self._print_thread is not None:
            self._print_thread.start()
        self._thread.start()

    def latest(self) -> ArmSample | None:
        with self._lock:
            return self._latest

    def stats(self) -> tuple[float | None, int, int, int]:
        with self._lock:
            return self._last_packet_at, self._packet_count, self._valid_count, self._dropped_print_count

    def stop(self):
        self._stop.set()
        self._socket.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._print_thread is not None and self._print_thread.is_alive():
            self._print_thread.join(timeout=1.0)

    def _print_loop(self):
        assert self._print_queue is not None
        while not self._stop.is_set() or not self._print_queue.empty():
            try:
                source, data = self._print_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            print(f"[UDP] {source[0]}:{source[1]} raw={data.decode('utf-8', errors='replace')!r}", flush=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                data, source = self._socket.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            received_at = time.monotonic()
            sample = parse_arm_sample(data, self._arm_index, received_at)
            with self._lock:
                self._last_packet_at = received_at
                self._packet_count += 1
                if sample is not None:
                    self._latest = sample
                    self._valid_count += 1
            if self._print_queue is not None:
                try:
                    self._print_queue.put_nowait((source, data))
                except queue.Full:
                    with self._lock:
                        self._dropped_print_count += 1


class HamiltonFilter:
    """参考双臂脚本，对位置和 SO(3) 旋转使用球面死区滤波。"""

    def __init__(self):
        self._pos: np.ndarray | None = None
        self._rot: Rotation | None = None

    def reset(self, pos: np.ndarray, quat: np.ndarray):
        self._pos = pos.copy()
        self._rot = Rotation.from_quat(quat)

    @staticmethod
    def _alpha(distance: float, deadzone: float, max_value: float, power: float = 1.2) -> float:
        alpha = np.clip((distance - deadzone) / (max_value + deadzone), 0.0, 1.0)
        if alpha > 0:
            alpha = alpha**power * (distance - deadzone) / max(distance, 1e-12)
        return float(alpha)

    def apply(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._pos is None or self._rot is None:
            self.reset(pos, quat)
            return pos.copy(), quat.copy()
        dp = pos - self._pos
        self._pos += dp * self._alpha(float(np.linalg.norm(dp)), 0.003, 0.05)
        next_rot = Rotation.from_quat(quat)
        dr = (next_rot * self._rot.inv()).as_rotvec()
        self._rot = Rotation.from_rotvec(dr * self._alpha(float(np.linalg.norm(dr)), 0.005, 0.1)) * self._rot
        return self._pos.copy(), self._rot.as_quat()


def tracker_delta(
    prev_pos: np.ndarray,
    prev_quat: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    *,
    pos_scale: float,
    rot_scale: float,
    max_pos_step: float,
    max_rot_step: float,
    rotation_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dp = RIGHT_T265_TO_ROBOT @ ((pos - prev_pos) * pos_scale)
    dr_tracker = (Rotation.from_quat(quat) * Rotation.from_quat(prev_quat).inv()).as_rotvec()
    dr = RIGHT_T265_TO_ROBOT @ (dr_tracker * rot_scale)
    for delta, limit in ((dp, max_pos_step), (dr, max_rot_step)):
        norm = np.linalg.norm(delta)
        if norm > limit:
            delta[:] = delta * (limit / norm)
    if rotation_only:
        dp[:] = 0
    return dp, dr


def apply_delta(target: np.ndarray, dp: np.ndarray, dr: np.ndarray) -> np.ndarray:
    pose = target.copy()
    pose[:3] += dp
    pose[3:6] = (Rotation.from_rotvec(dr) * Rotation.from_rotvec(pose[3:6])).as_rotvec()
    return pose


class LatestCamera:
    """后台采集原始 BGR 帧，避免相机阻塞 20 Hz 控制循环。"""

    def __init__(self, name: str, shape: tuple[int, int, int], *, serial: str | None = None, cv_id: int | None = None):
        if (serial is None) == (cv_id is None):
            raise ValueError(f"{name}: 必须且只能指定 RealSense 序列号或 OpenCV 设备 ID")
        self.name = name
        self.shape = shape
        self._serial = serial
        self._cv_id = cv_id
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._frame_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{name}", daemon=True)
        self._pipeline = None
        self._capture = None

    def start(self):
        height, width, _ = self.shape
        if self._serial is not None:
            import pyrealsense2 as rs

            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self._serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
            try:
                self._pipeline.start(config)
            except Exception:
                self._pipeline = None
                raise
        else:
            self._capture = cv2.VideoCapture(self._cv_id)
            if not self._capture.isOpened():
                self._capture.release()
                self._capture = None
                raise RuntimeError(f"{self.name}: 无法打开 OpenCV 设备 {self._cv_id}")
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._capture.set(cv2.CAP_PROP_FPS, 30)
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0 = 手动 (关闭自动曝光)
            self._capture.set(cv2.CAP_PROP_EXPOSURE, 200)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.latest() is not None:
                print(f"[Camera] {self.name} 已就绪 ({width}x{height})")
                return
            time.sleep(0.05)
        self.stop()
        raise RuntimeError(f"{self.name}: 5 秒内没有收到图像")

    def latest(self) -> tuple[np.ndarray, float, bytes] | None:
        with self._lock:
            if self._frame is None or self._jpeg is None:
                return None
            return self._frame.copy(), self._frame_at, self._jpeg

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _run(self):
        height, width, _ = self.shape
        while not self._stop.is_set():
            try:
                if self._pipeline is not None:
                    frames = self._pipeline.wait_for_frames(timeout_ms=200)
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    frame = np.asanyarray(color.get_data())
                else:
                    ok, frame = self._capture.read()
                    if not ok:
                        time.sleep(0.02)
                        continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    continue
                with self._lock:
                    self._frame = frame.copy()
                    self._jpeg = encoded.tobytes()
                    self._frame_at = time.monotonic()
            except Exception as exc:
                print(f"[Camera] {self.name}: {exc}")
                time.sleep(0.1)


def _feature_stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _image_stats(jpegs: list[bytes], expected_shape: tuple[int, int, int]) -> dict:
    low = np.full(3, 255.0)
    high = np.zeros(3)
    pixel_sum = np.zeros(3)
    pixel_sq = np.zeros(3)
    pixels = 0
    for data in jpegs:
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("无法解码刚录制的 JPEG 图像")
        if bgr.shape != expected_shape:
            raise ValueError(f"JPEG 图像尺寸错误: {bgr.shape}，期望 {expected_shape}")
        for i, channel in enumerate(reversed(cv2.split(bgr))):
            minimum, maximum, _, _ = cv2.minMaxLoc(channel)
            low[i] = min(low[i], minimum)
            high[i] = max(high[i], maximum)
        mean_bgr, std_bgr = cv2.meanStdDev(bgr)
        mean_rgb = mean_bgr.ravel()[::-1]
        std_rgb = std_bgr.ravel()[::-1]
        frame_pixels = bgr.shape[0] * bgr.shape[1]
        pixel_sum += mean_rgb * frame_pixels
        pixel_sq += (np.square(std_rgb) + np.square(mean_rgb)) * frame_pixels
        pixels += frame_pixels
    mean = pixel_sum / pixels
    std = np.sqrt(np.maximum(pixel_sq / pixels - np.square(mean), 0.0))
    return {
        "min": low.reshape(3, 1, 1).tolist(),
        "max": high.reshape(3, 1, 1).tolist(),
        "mean": mean.reshape(3, 1, 1).tolist(),
        "std": std.reshape(3, 1, 1).tolist(),
        "count": [len(jpegs)],
    }


def _aggregate_stats(episodes: list[dict]) -> dict:
    out = {}
    for key in ("state", "actions", "image", "wrist_image"):
        parts = [episode["stats"][key] for episode in episodes]
        counts = np.asarray([part["count"][0] for part in parts], dtype=np.float64)
        total = counts.sum()
        mean = sum(np.asarray(part["mean"]) * count for part, count in zip(parts, counts, strict=True)) / total
        second = (
            sum(
                (np.square(np.asarray(part["std"])) + np.square(np.asarray(part["mean"]))) * count
                for part, count in zip(parts, counts, strict=True)
            )
            / total
        )
        out[key] = {
            "min": np.minimum.reduce([np.asarray(part["min"]) for part in parts]).tolist(),
            "max": np.maximum.reduce([np.asarray(part["max"]) for part in parts]).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(second - np.square(mean), 0.0)).tolist(),
            "count": [int(total)],
        }
    return out


def _jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _atomic_text(path: Path, contents: str):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(contents, encoding="utf-8")
    os.replace(temp, path)


class PickPlaceWriter:
    """写入 pick_place 的字段、Parquet 路径、任务与统计元数据。"""

    def __init__(self, root: Path, task: str):
        self.root = root
        self.task = task
        self._frames: list[dict] = []
        self._episode_stats: list[dict] = []
        self._episodes: list[dict] = []
        self._tasks: list[dict] = []
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info.get("codebase_version") != "v2.1" or info.get("fps") != DATASET_FPS:
                raise ValueError("现有数据集版本或帧率与 pick_place 不兼容")
            if info.get("robot_type") != "piper_eef":
                raise ValueError("现有数据集不是机械臂基坐标系 EEF 数据，不能混合追加")
            if info.get("features") != self._features():
                raise ValueError("现有数据集的 EEF/图像特征与 pick_place 不兼容")
            self.info = info
            self._episodes = self._read_jsonl(root / "meta" / "episodes.jsonl")
            self._episode_stats = self._read_jsonl(root / "meta" / "episodes_stats.jsonl")
            self._tasks = self._read_jsonl(root / "meta" / "tasks.jsonl")
            if len(self._episodes) != info["total_episodes"] or len(self._episode_stats) != len(self._episodes):
                raise ValueError("现有数据集元数据不完整，请先修复再追加")
            if self._episodes and sum(e["length"] for e in self._episodes) != info["total_frames"]:
                raise ValueError("现有数据集的帧数元数据不一致")
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError(f"输出目录已有文件但没有 LeRobot 元数据: {root}")
            self.info = self._new_info()
        existing = next((entry for entry in self._tasks if entry["task"] == task), None)
        self.task_index = existing["task_index"] if existing else len(self._tasks)
        if existing is None:
            self._tasks.append({"task_index": self.task_index, "task": task})

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _features() -> dict:
        features = {name: {"dtype": "float32", "shape": [7], "names": STATE_NAMES} for name in ("state", "actions")}
        for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            features[name] = {
                "dtype": "float32" if name == "timestamp" else "int64",
                "shape": [1],
                "names": None,
            }
        for name, shape in (("image", BASE_SHAPE), ("wrist_image", WRIST_SHAPE)):
            features[name] = {"dtype": "image", "shape": list(shape), "names": ["height", "width", "channel"]}
        return features

    @classmethod
    def _new_info(cls) -> dict:
        return {
            "codebase_version": "v2.1",
            "robot_type": "piper_eef",
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 0,
            "chunks_size": 1000,
            "fps": DATASET_FPS,
            "splits": {"train": "0:0"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": cls._features(),
        }

    @property
    def recording(self) -> bool:
        return bool(self._frames)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def next_episode(self) -> int:
        return self.info["total_episodes"]

    def add_frame(self, state: np.ndarray, action: np.ndarray, base_jpeg: bytes, wrist_jpeg: bytes):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.shape != (7,) or action.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("state/actions 必须是有限的 7 维 EEF 位姿")
        if not base_jpeg.startswith(b"\xff\xd8") or not wrist_jpeg.startswith(b"\xff\xd8"):
            raise ValueError("相机图像必须是 JPEG 字节")
        self._frames.append({"state": state, "actions": action, "image": base_jpeg, "wrist_image": wrist_jpeg})

    def discard(self):
        count = len(self._frames)
        self._frames.clear()
        print(f"[Recording] 已丢弃 {count} 帧")

    def save_episode(self) -> Path | None:
        if not self._frames:
            print("[Recording] 没有可保存的帧")
            return None
        path = self.write_episode(self._frames)
        self._frames.clear()
        return path

    def take_frames(self) -> list[dict]:
        frames = self._frames
        self._frames = []
        return frames

    def write_episode(self, source_frames: list[dict]) -> Path:
        """将已结束的 episode 写盘；调用方保证按 episode 顺序串行执行。"""
        from datasets import Dataset
        from datasets import Features
        from datasets import Image
        from datasets import Sequence
        from datasets import Value

        episode_index = self.next_episode
        count = len(source_frames)
        index_offset = self.info["total_frames"]
        frames = []
        for i, frame in enumerate(source_frames):
            frames.append(
                {
                    "state": frame["state"],
                    "actions": frame["actions"],
                    "timestamp": np.float32(i / DATASET_FPS),
                    "frame_index": i,
                    "episode_index": episode_index,
                    "index": index_offset + i,
                    "task_index": self.task_index,
                    "image": {"bytes": frame["image"], "path": f"frame_{i:06d}.jpg"},
                    "wrist_image": {"bytes": frame["wrist_image"], "path": f"frame_{i:06d}.jpg"},
                }
            )
        features = Features(
            {
                "state": Sequence(length=7, feature=Value(dtype="float32")),
                "actions": Sequence(length=7, feature=Value(dtype="float32")),
                "timestamp": Value(dtype="float32"),
                "frame_index": Value(dtype="int64"),
                "episode_index": Value(dtype="int64"),
                "index": Value(dtype="int64"),
                "task_index": Value(dtype="int64"),
                "image": Image(),
                "wrist_image": Image(),
            }
        )
        stats = {
            "state": _feature_stats(np.stack([f["state"] for f in source_frames])),
            "actions": _feature_stats(np.stack([f["actions"] for f in source_frames])),
            "image": _image_stats([f["image"] for f in source_frames], BASE_SHAPE),
            "wrist_image": _image_stats([f["wrist_image"] for f in source_frames], WRIST_SHAPE),
        }
        parquet_path = (
            self.root
            / "data"
            / f"chunk-{episode_index // self.info['chunks_size']:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if parquet_path.exists():
            raise FileExistsError(f"Episode 文件已存在: {parquet_path}")
        temp_path = parquet_path.with_suffix(".parquet.tmp")
        try:
            Dataset.from_list(frames, features=features).to_parquet(str(temp_path))
            os.replace(temp_path, parquet_path)
        finally:
            temp_path.unlink(missing_ok=True)

        self._episodes.append({"episode_index": episode_index, "tasks": [self.task], "length": count})
        self._episode_stats.append({"episode_index": episode_index, "stats": stats})
        self.info["total_episodes"] += 1
        self.info["total_frames"] += count
        self.info["total_tasks"] = len(self._tasks)
        self.info["total_chunks"] = (self.info["total_episodes"] + self.info["chunks_size"] - 1) // self.info[
            "chunks_size"
        ]
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        _atomic_text(meta / "tasks.jsonl", _jsonl(self._tasks))
        _atomic_text(meta / "episodes_stats.jsonl", _jsonl(self._episode_stats))
        _atomic_text(meta / "episodes.jsonl", _jsonl(self._episodes))
        _atomic_text(meta / "stats.json", json.dumps(_aggregate_stats(self._episode_stats), indent=2))
        _atomic_text(meta / "info.json", json.dumps(self.info, indent=2, ensure_ascii=False))
        print(f"[Recording] Episode {episode_index} 已保存: {count} 帧 -> {parquet_path}")
        return parquet_path


class BackgroundEpisodeSaver:
    """把已结束的录制交给单个后台线程，避免写盘阻塞采集控制。"""

    def __init__(self, writer: PickPlaceWriter):
        self._writer = writer
        self._next_episode = writer.next_episode
        self._queue: queue.Queue[list[dict] | None] = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._failed_batches: list[list[dict]] = []
        self._closed = False

    @property
    def next_episode(self) -> int:
        return self._next_episode

    def check_error(self):
        if self._error is not None:
            raise RuntimeError("后台保存 Episode 失败") from self._error

    def save_episode(self) -> int | None:
        self.check_error()
        if self._closed:
            raise RuntimeError("后台保存线程已关闭")
        frames = self._writer.take_frames()
        if not frames:
            print("[Recording] 没有可保存的帧")
            return None
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="piper-save", daemon=True)
            self._thread.start()
        episode_index = self._next_episode
        self._next_episode += 1
        self._queue.put(frames)
        print(f"[Recording] Episode {episode_index} 已排队保存: {len(frames)} 帧")
        return episode_index

    def close(self):
        if not self._closed:
            self._closed = True
            if self._thread is not None:
                self._queue.put(None)
                self._thread.join()
        self.check_error()

    def _run(self):
        while True:
            frames = self._queue.get()
            if frames is None:
                return
            if self._error is not None:
                self._failed_batches.append(frames)
                continue
            try:
                self._writer.write_episode(frames)
            except Exception as exc:
                self._failed_batches.append(frames)
                self._error = exc
                print(f"[Recording] 后台保存失败: {exc}")


class TerminalKeys:
    """终端单键读取；预览窗口按键由主循环处理。"""

    def __init__(self):
        self.events: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._read, name="piper-keys", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _read(self):
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = os.read(sys.stdin.fileno(), 1)
            if not ch:
                return
            key = ch.decode("utf-8", errors="ignore").lower()
            if key:
                self.events.put(key)


def _gripper_from_angle(angle_deg: float, open_width_mm: float) -> float:
    return float(np.clip((angle_deg - 7.0) / 13.0, 0.0, 1.0) * open_width_mm)


def _record_pose(arm_pose: np.ndarray, gripper_mm: float) -> np.ndarray:
    """机械臂 TCP 位姿 + 夹爪宽度；位姿补偿已由 PiperEEFController 完成。"""
    return np.concatenate([arm_pose, [gripper_mm]]).astype(np.float32)


def _preview(base: np.ndarray, wrist: np.ndarray, *, recording: bool) -> str | None:
    base_small = cv2.resize(base, (640, 360), interpolation=cv2.INTER_AREA)
    wrist_small = cv2.resize(wrist, (480, 360), interpolation=cv2.INTER_AREA)
    canvas = np.concatenate([base_small, wrist_small], axis=1)
    cv2.putText(
        canvas,
        "REC" if recording else "READY",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if recording else (0, 255, 0),
        2,
    )
    cv2.imshow("Piper UDP EEF collect", canvas)
    key = cv2.waitKey(1) & 0xFF
    return chr(key).lower() if 0 < key < 128 else None


def _run_udp_test(receiver: UDPPoseReceiver):
    print("[UDP] 测试模式，按 Ctrl+C 退出")
    previous = None
    try:
        while True:
            sample = receiver.latest()
            if sample is not None and sample.sequence != previous:
                previous = sample.sequence
                print(
                    f"seq={sample.sequence} pos={sample.position_m.round(4)} "
                    f"quat={sample.quaternion_xyzw.round(4)} grip={sample.gripper_angle_deg}"
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def run(args):
    if args.udp_test:
        receiver = UDPPoseReceiver(args.udp_host, args.udp_port, args.arm_index, print_packets=args.print_udp)
        try:
            receiver.start()
            _run_udp_test(receiver)
        finally:
            receiver.stop()
        return

    writer = PickPlaceWriter(Path(args.output), args.task)
    saver = BackgroundEpisodeSaver(writer)
    receiver = (
        None
        if args.no_udp
        else UDPPoseReceiver(args.udp_host, args.udp_port, args.arm_index, print_packets=args.print_udp)
    )
    gripper = None
    robot = None
    try:
        if receiver is not None:
            receiver.start()
        if args.gripper_port and eef._HAS_DM_GRIPPER:
            gripper = eef.DmGripperController(
                port=args.gripper_port,
                close_rad=args.gripper_close_rad,
                open_width_mm=args.gripper_open_width_mm,
                current_limit=args.gripper_current,
                speed_rad_s=args.gripper_speed,
            )
        robot = eef.PiperEEFController(args.can_name, gripper=gripper)
        base_camera = LatestCamera("base", BASE_SHAPE, serial=args.rs2_base, cv_id=args.base_cv_id)
        wrist_camera = LatestCamera("wrist", WRIST_SHAPE, serial=args.rs2_wrist, cv_id=args.usb_wrist)
    except Exception:
        if receiver is not None:
            receiver.stop()
        if robot is not None:
            robot.close_gripper_transport()
        elif gripper is not None:
            gripper.shutdown()
        raise
    teleop = False
    recording_active = False
    rotation_only = False
    target = None
    prev_pos = None
    prev_quat = None
    filtered = HamiltonFilter()
    gripper_target = 0.0
    running = True
    preview_enabled = not args.no_preview
    period = 1.0 / DATASET_FPS
    deadline = time.monotonic()

    def current_sample() -> ArmSample | None:
        if receiver is None:
            return None
        sample = receiver.latest()
        if sample is None or time.monotonic() - sample.received_at > args.udp_timeout:
            return None
        return sample

    def zero_reference(sample: ArmSample | None) -> ArmSample | None:
        nonlocal target, prev_pos, prev_quat
        target = robot.get_eef_pose()
        if receiver is not None:
            sample = current_sample()
        if sample is not None:
            prev_pos = sample.position_m.copy()
            prev_quat = sample.quaternion_xyzw.copy()
            filtered.reset(prev_pos, prev_quat)
        return sample

    def handle_key(key: str):
        nonlocal teleop, recording_active, rotation_only, gripper_target, running, deadline
        key = key.lower()
        if key == "q":
            running = False
        elif key in ("e", "c"):
            if key == "e" and teleop:
                teleop = False
                if recording_active:
                    saver.save_episode()
                    recording_active = False
                print("[Teleop] 暂停")
                return
            sample = current_sample()
            if sample is None and not args.no_udp:
                print("[Teleop] 等待有效 UDP 位姿")
                return
            if not teleop:
                sample = zero_reference(sample)
                if sample is None and not args.no_udp:
                    print("[Teleop] 读取机械臂位姿期间 UDP 位姿过期，等待新包")
                    return
                teleop = True
                if args.no_udp:
                    print("[Test] 开始无 UDP 采集")
                else:
                    print(f"[Teleop] 开始，已重置遥操参考；UDP seq={sample.sequence}")
            if key == "c":
                if recording_active:
                    print("[Recording] 已在录制")
                else:
                    recording_active = True
                    print(f"[Recording] 开始 Episode {saver.next_episode}: {writer.task}")
        elif key == "s":
            if recording_active:
                saver.save_episode()
                recording_active = False
            teleop = False
        elif key == "d":
            writer.discard()
            recording_active = False
            teleop = False
        elif key == "r":
            if recording_active:
                saver.save_episode()
                recording_active = False
            teleop = False
            robot.go_to_init_pose()
            deadline = time.monotonic()
            print("[Teleop] 已回初始位姿")
        elif key == "z":
            sample = current_sample()
            if sample is not None or args.no_udp:
                sample = zero_reference(sample)
                if sample is not None or args.no_udp:
                    print("[Test] 已更新 EEF 参考" if args.no_udp else "[Teleop] 已重置遥操参考")
        elif key == "t":
            rotation_only = not rotation_only
            print(f"[Teleop] 只旋转: {'开' if rotation_only else '关'}")
        elif key == "g":
            gripper_target = args.gripper_open_width_mm if gripper_target < args.gripper_open_width_mm / 2 else 0.0
        elif key == "o":
            gripper_target = min(args.gripper_open_width_mm, gripper_target + args.gripper_step_mm)
        elif key == "p":
            gripper_target = max(0.0, gripper_target - args.gripper_step_mm)

    try:
        if not robot.enable(speed_pct=args.speed_pct):
            raise RuntimeError("Piper 使能失败")
        base_camera.start()
        wrist_camera.start()
        gripper_target = float(np.clip(robot.get_gripper_mm(), 0, args.gripper_open_width_mm))
        if args.no_udp:
            print(f"[Ready] 无 UDP 测试采集，输出 {args.output}；不下发 EEF 移动指令")
        else:
            print(f"[Ready] UDP {args.udp_host}:{args.udp_port}, arm_index={args.arm_index}, 输出 {args.output}")
        print(f"[Ready] TCP 夹爪偏移 {eef.GRIPPER_TCP_OFFSET_M.tolist()} m；记录机械臂基坐标系位姿")
        print("按 e 遥操, c 录制, s 后台保存, d 丢弃, r 重置, z 重新对齐, t 只旋转, q 退出")

        with TerminalKeys() as keys:
            while running:
                saver.check_error()
                while True:
                    try:
                        handle_key(keys.events.get_nowait())
                    except queue.Empty:
                        break
                if not running:
                    break
                sample = current_sample()
                if teleop and sample is None and not args.no_udp:
                    teleop = False
                    if recording_active:
                        saver.save_episode()
                        recording_active = False
                    now = time.monotonic()
                    last_valid = receiver.latest()
                    last_packet_at, packet_count, valid_count, dropped_print_count = receiver.stats()
                    valid_age = "无" if last_valid is None else f"{now - last_valid.received_at:.2f}s"
                    packet_age = "无" if last_packet_at is None else f"{now - last_packet_at:.2f}s"
                    seq = "无" if last_valid is None else str(last_valid.sequence)
                    reason = (
                        "仍收到原始包，但选中 arm 无有效位姿"
                        if last_packet_at is not None and now - last_packet_at <= args.udp_timeout
                        else "没有收到新 UDP 包"
                    )
                    print_drop_info = f", 打印队列丢弃={dropped_print_count}" if args.print_udp else ""
                    print(
                        f"[UDP] 位姿超时 ({reason})：arm_index={args.arm_index}, seq={seq}, "
                        f"有效包距今={valid_age}, 原始包距今={packet_age}, "
                        f"总包={packet_count}, 有效包={valid_count}{print_drop_info}；"
                        "已停止遥操和录制"
                    )

                base_latest = base_camera.latest()
                wrist_latest = wrist_camera.latest()
                if base_latest is None or wrist_latest is None:
                    raise RuntimeError("相机未提供图像")
                base, base_at, base_jpeg = base_latest
                wrist, wrist_at, wrist_jpeg = wrist_latest
                camera_ok = time.monotonic() - min(base_at, wrist_at) < args.camera_timeout
                if teleop and not camera_ok:
                    if recording_active:
                        saver.save_episode()
                        recording_active = False
                    teleop = False
                    print("[Camera] 图像超时，已停止遥操和录制")

                if teleop:
                    arm_state = robot.get_state().astype(np.float64)
                    if args.no_udp:
                        # 采集链路测试：动作位姿取本帧实际反馈，不向机械臂下发 EEF 命令。
                        target = arm_state[:6].copy()
                    else:
                        if sample is None or target is None or prev_pos is None or prev_quat is None:
                            raise RuntimeError("遥操参考位姿尚未初始化")
                        if args.no_filter:
                            pos, quat = sample.position_m, sample.quaternion_xyzw
                        else:
                            pos, quat = filtered.apply(sample.position_m, sample.quaternion_xyzw)
                        dp, dr = tracker_delta(
                            prev_pos,
                            prev_quat,
                            pos,
                            quat,
                            pos_scale=args.pos_scale,
                            rot_scale=args.rot_scale,
                            max_pos_step=args.max_pos_speed * period,
                            max_rot_step=args.max_rot_speed * period,
                            rotation_only=rotation_only,
                        )
                        prev_pos, prev_quat = pos.copy(), quat.copy()
                        target = apply_delta(target, dp, dr)
                        if args.gripper_source == "udp" and sample.gripper_angle_deg is not None:
                            gripper_target = _gripper_from_angle(sample.gripper_angle_deg, args.gripper_open_width_mm)
                        robot.send_eef_command(target)
                    robot.send_gripper_command(gripper_target)
                    if recording_active and camera_ok:
                        writer.add_frame(
                            _record_pose(arm_state[:6], float(arm_state[6])),
                            _record_pose(target, gripper_target),
                            base_jpeg,
                            wrist_jpeg,
                        )

                if preview_enabled:
                    try:
                        key = _preview(base, wrist, recording=recording_active)
                        if key:
                            handle_key(key)
                    except cv2.error as exc:
                        print(f"[Preview] 无法显示窗口: {exc}")
                        preview_enabled = False

                deadline += period
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline = time.monotonic()
    except KeyboardInterrupt:
        print("\n[Info] Ctrl+C，结束采集")
    finally:
        try:
            if recording_active:
                saver.save_episode()
        finally:
            cleanup_steps = [
                ("base camera", base_camera.stop),
                ("wrist camera", wrist_camera.stop),
                ("preview", cv2.destroyAllWindows),
                ("gripper", robot.close_gripper_transport),
            ]
            if receiver is not None:
                cleanup_steps.insert(0, ("UDP", receiver.stop))
            for label, cleanup in cleanup_steps:
                try:
                    cleanup()
                except Exception as exc:
                    print(f"[Cleanup] {label}: {exc}")
            saver.close()
            print("[Info] 已退出，机械臂保持使能")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="单臂 UDP 遥操采集 pick_place 格式的 EEF 数据集")
    parser.add_argument("--output", default="./piper_udp_eef", help="LeRobot 数据集输出目录，存在时追加 episode")
    parser.add_argument("--task", default=DEFAULT_TASK, help="本次录制的语言指令")
    parser.add_argument("--can_name", default="can0")
    parser.add_argument("--udp_host", default="0.0.0.0")
    parser.add_argument("--udp_port", type=int, default=5005)
    parser.add_argument("--arm_index", type=int, default=1, help="UDP arms 中选择的单臂 index (默认右手 1)")
    parser.add_argument("--udp_timeout", type=float, default=0.3, help="UDP 位姿超时时间 (秒)")
    parser.add_argument("--rs2_base", default=None, help="基座 RealSense 序列号 (1280x720)")
    parser.add_argument("--base_cv_id", type=int, default=None, help="基座 OpenCV 设备 ID，替代 --rs2_base")
    parser.add_argument("--rs2_wrist", default=None, help="腕部 RealSense 序列号 (640x480)")
    parser.add_argument("--usb_wrist", type=int, default=None, help="腕部 USB/OpenCV 设备 ID，替代 --rs2_wrist")
    parser.add_argument("--camera_timeout", type=float, default=0.5, help="图像超时时间 (秒)")
    parser.add_argument("--no_preview", action="store_true")
    parser.add_argument("--udp_test", action="store_true", help="仅检查 UDP，不连接机械臂/相机")
    parser.add_argument("--print_udp", action="store_true", help="逐包打印 UDP 来源和原始内容 (调试用)")
    parser.add_argument(
        "--no_udp", action="store_true", help="不创建 UDP 接收器，记录机械臂反馈与相机；不下发 EEF 移动指令"
    )
    parser.add_argument("--speed_pct", type=int, default=40, help="Piper EEF 控制速度百分比")
    parser.add_argument("--max_pos_speed", type=float, default=0.25, help="最大 TCP 平移速度 (m/s)")
    parser.add_argument("--max_rot_speed", type=float, default=0.6, help="最大 TCP 旋转速度 (rad/s)")
    parser.add_argument("--pos_scale", type=float, default=1.2)
    parser.add_argument("--rot_scale", type=float, default=1.0)
    parser.add_argument("--no_filter", action="store_true", help="关闭 Hamilton 滤波")
    parser.add_argument("--gripper_source", choices=("udp", "keys"), default="udp")
    parser.add_argument("--gripper_port", default="/dev/ttyACM0", help="DM 夹爪串口；传空字符串使用原生夹爪")
    parser.add_argument("--gripper_open_width_mm", type=float, default=eef.GRIPPER_OPEN_WIDTH_MM)
    parser.add_argument("--gripper_close_rad", type=float, default=eef.GRIPPER_CLOSE_RAD)
    parser.add_argument("--gripper_current", type=float, default=eef.GRIPPER_CURRENT_LIMIT)
    parser.add_argument("--gripper_speed", type=float, default=eef.GRIPPER_SPEED_RAD_S)
    parser.add_argument("--gripper_step_mm", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.no_udp and args.udp_test:
        parser.error("--no_udp 和 --udp_test 不能同时使用")
    if args.no_udp:
        args.gripper_source = "keys"
    if not args.udp_test and (
        (args.rs2_base is None) == (args.base_cv_id is None) or (args.rs2_wrist is None) == (args.usb_wrist is None)
    ):
        parser.error("基座和腕部相机各需指定一个 RealSense 序列号或 OpenCV 设备 ID")
    if (
        min(
            args.udp_timeout,
            args.camera_timeout,
            args.max_pos_speed,
            args.max_rot_speed,
            args.pos_scale,
            args.rot_scale,
            args.gripper_open_width_mm,
            args.gripper_close_rad,
            args.gripper_speed,
            args.gripper_step_mm,
        )
        <= 0
    ):
        parser.error("超时、速度、比例及夹爪宽度必须大于 0")
    if not 1 <= args.udp_port <= 65535 or args.arm_index < 0:
        parser.error("--udp_port 必须在 1..65535，--arm_index 必须非负")
    if not 20 <= args.speed_pct <= 100 or not 0 <= args.gripper_current <= 1:
        parser.error("--speed_pct 必须在 20..100，--gripper_current 必须在 0..1")
    if not args.task.strip():
        parser.error("--task 不能为空")
    return args


if __name__ == "__main__":
    run(parse_args())
