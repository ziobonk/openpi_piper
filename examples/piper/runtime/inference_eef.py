#!/usr/bin/env python3
"""
Piper 末端位姿 (EEF) 推理脚本 — 用于 pi05_piper_eef / pi05_piper_eef_lora 配置。

默认恢复 2026-09-19 的机械臂绝对 TCP 推理方式：当前机械臂 TCP 位姿
直接作为 state。策略服务器把相对当前 state 的动作增量还原为 Robot Base 下
的绝对 TCP 目标后，客户端直接下发。法兰与爪尖之间的 TCP 偏移由控制器补偿。
使用后来训练的其他坐标系模型时，需显式选择对应的 --model_action_frame。

默认模式下服务器返回的 ``actions`` 是 Robot Base 下的绝对目标：

    [x, y, z, rx, ry, rz(旋转向量/axis-angle, rad), gripper_command]

夹爪物理命令约定: 0 = 全闭，20 = 全开。策略预测值约 0.1 表示开，约 0 表示关，
执行前会经过二值映射。

与 ``examples/piper/runtime/inference.py``（关节空间）的区别:
    - state 读取 ``GetArmEndPoseMsgs()`` (0.001mm / 0.001deg RPY)，转成
      [x, y, z(m), rotvec(rad)]；夹爪读当前宽度转 mm。
    - 执行用 ``EndPoseCtrl``（末端位姿控制模式 ``MotionCtrl_2 0x00``），把
      rotvec 转回 RPY(deg) 下发；夹爪默认用达妙 DM-J4310-2EC (力位模式,
      USB2CAN ``/dev/ttyACM1``)，mm ↔ rad 反比映射，传 ``--gripper_port ""``
      可回退到原生 ``GripperCtrl``。

旋转约定: RPY 顺序为 ``xyz``（外旋 XYZ / 内旋 ZYX），与 diffusion_policy_piper
的 ``PiperInterpolationController`` 及训练数据采集时保持一致。

用法:
    # 1. 在 GPU 服务器上启动与旧版 Robot Base 绝对 TCP 数据匹配的 checkpoint
    uv run scripts/serve_policy.py --port 6006 policy:checkpoint \
        --policy.config=pi05_piper_eef \
        --policy.dir=<旧版绝对 TCP 模型 checkpoint>

    # 2. 在机械臂端运行推理 (默认 robot_absolute；仅用于旧版 Robot Base 模型)
    python examples/piper/runtime/inference_eef.py --host localhost --port 6006 \
        --rs2_base 231122071797 --usb_wrist 0

    # 交互模式 (每次推理前输入指令)
    python examples/piper/runtime/inference_eef.py --host localhost --interactive

    # PICO World 下绝对 TCP 位姿数据训练的模型：在机器人端提供数据集用于坐标锚点
    python examples/piper/runtime/inference_eef.py --host localhost --port 6006 \
        --model_action_frame tcp_absolute --absolute_pose_dataset ./local/datasets/pick_cube_raw_action \
        --reference_episode 0 --no-binary_gripper

    # 数据集离线评测 (可视化末端位姿，可导出 GIF，不连接真实机械臂)
    python examples/piper/runtime/inference_eef.py --dataset ./pick_place --episode 0 --gif eval.gif --no_show
    python examples/piper/runtime/inference_eef.py --dataset ./pick_place --init_pose 0.0 -0.7 -0.4 0 0 0

前置条件:
    1. CAN 模块已激活:  bash can_activate.sh can0 1000000
    2. 机械臂已上电、处于从臂模式
    3. piper_sdk 已安装:  cd piper_sdk && pip install .
    4. openpi-client 已安装: cd packages/openpi-client && pip install -e .
    5. scipy 已安装: uv sync (用于 RPY ↔ rotvec 转换)
    6. (达妙夹爪) USB2CAN 已插好 (/dev/ttyACM1)、DMTool 已关闭、电机 24V 已上电；
       pyserial 已安装 (uv pip install pyserial)。夹爪 CTRL_MODE 需为 4 (力位)，
       否则先用 examples/piper/dm_gripper/test_dm_gripper.py mode 4 --save 设好。

键盘控制:
    Enter   — 开始/继续推理
    s       — 暂停推理
    q       — 退出
    r       — 重置机械臂到初始关节位姿
"""

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ---- 可选依赖 (数据集离线评测 + 可视化) ----
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import matplotlib

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from PIL import Image as _PILImage

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ===========================================================================
# 导入
# ===========================================================================

# --- Piper SDK ---
OPENPI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PIPER_SDK_PATH = os.path.join(OPENPI_ROOT, "piper_sdk")
PIPER_EXAMPLES_PATH = os.path.dirname(__file__)
if OPENPI_ROOT not in sys.path:
    sys.path.insert(0, OPENPI_ROOT)
if PIPER_SDK_PATH not in sys.path:
    sys.path.insert(0, PIPER_SDK_PATH)
if PIPER_EXAMPLES_PATH not in sys.path:
    sys.path.insert(0, PIPER_EXAMPLES_PATH)

try:
    from piper_sdk import C_PiperInterface_V2  # type: ignore[import-untyped]
except ImportError:
    print("[ERROR] 无法导入 piper_sdk，请先安装:")
    print("  cd piper_sdk && pip install .")
    sys.exit(1)

# --- openpi-client (策略服务器客户端) ---
try:
    from openpi_client import websocket_client_policy
except ImportError:
    print("[ERROR] 无法导入 openpi_client，请先安装:")
    print("  cd packages/openpi-client && pip install -e .")
    sys.exit(1)

# --- 相机工具 (支持 RealSense D435i/D405 和 OpenCV) ---
from camera_utils import create_cameras

from examples.piper.transforms.inference_action_transform import pico_relative_actions_to_tcp_relative
from examples.piper.transforms.inference_action_transform import tcp_relative_actions_to_pico_relative

# --- DM-J4310-2EC 夹爪驱动 (力位模式, USB2CAN /dev/ttyACM1) ---
# 与机械臂的 can0 相互独立；驱动代码在 examples/piper/dm_gripper/ 下。
_DM_GRIPPER_PATH = os.path.join(os.path.dirname(__file__), "..", "dm_gripper")
if _DM_GRIPPER_PATH not in sys.path:
    sys.path.insert(0, _DM_GRIPPER_PATH)
try:
    from usb2can import Usb2Can
    from dm_j4310 import DmJ4310
    from dm_gripper import DmGripper

    _HAS_DM_GRIPPER = True
except ImportError as _e:
    Usb2Can = DmJ4310 = DmGripper = None
    _HAS_DM_GRIPPER = False
    print(f"[WARN] 无法导入 DM 夹爪驱动 ({_e})，将回退到原生夹爪。")


# ===========================================================================
# 常量 — 根据你的模型和任务调整
# ===========================================================================

# 默认动作块长度 (与 pi05_piper_eef 的 action_horizon=10 一致)。实时模式下
# 服务器返回的实际长度才是最终依据，不能假定服务器一定返回该长度。
DEFAULT_ACTION_HORIZON = 50

# 每次执行多少步后再重新查询模型 (≤ action_horizon)
DEFAULT_EXEC_HORIZON = 1
# 训练数据频率 (20 fps)
CONTROL_FREQ = 20
# 默认速度百分比
DEFAULT_SPEED_PCT = 40
# 默认最大末端线速度 / 角速度 (插值模式下的速度上限)
DEFAULT_MAX_POS_SPEED = 0.25  # m/s
DEFAULT_MAX_ROT_SPEED = 0.6  # rad/s
# 默认插值频率 (Hz), None=不插值(直接 EndPoseCtrl)
DEFAULT_INTERP_FREQ = None
# 夹爪控制力矩
GRIPPER_EFFORT = 1000

# ---- 单位换算 ----
M_TO_001MM = 1_000_000.0  # m  → 0.001 mm (piper 内部位置单位)
_001MM_TO_M = 1e-6  # 0.001 mm → m
RAD_TO_001DEG = 1000.0 * 180.0 / np.pi  # rad → 0.001 deg
_001DEG_TO_RAD = np.pi / (1000.0 * 180.0)  # 0.001 deg → rad
PIPER_GRIPPER_OPEN_RAW = 70_000  # 原生 Piper 夹爪约 70mm 全开

# ---- DM 夹爪 (达妙 DM-J4310-2EC, 力位模式) ----
GRIPPER_PORT = "/dev/ttyACM0"  # USB2CAN 串口
GRIPPER_CAN_ID = 1  # 电机 CAN ID (ESC_ID)
GRIPPER_OPEN_RAD = 0.0  # 全开位置 (rad)
GRIPPER_CLOSE_RAD = 15.0  # 全闭位置 (rad)
GRIPPER_OPEN_WIDTH_MM = 20.0  # 二值控制的全开命令；0=全闭
GRIPPER_PREDICTION_SCALE = 200.0  # 预测 0.1 → 20mm
GRIPPER_BINARY_THRESHOLD_MM = 0.2  # 缩放后 >0.2 时开，否则关
GRIPPER_CURRENT_LIMIT = 0.3  # 力矩电流上限 (i_des, 0..1.0)
GRIPPER_SPEED_RAD_S = 6.0  # 移动速度上限 (rad/s)

# 法兰(J6) → 爪尖(TCP) 的固定偏移，法兰坐标系内，单位米 (纯平移，姿态不随夹爪变化)。
# Piper SDK 的 EndPoseCtrl / GetArmEndPoseMsgs 都是法兰位姿，推理状态和动作
# 使用爪尖位姿，所以在收发两端都要补偿这一步:
#   读: 爪尖 = 法兰 + R @ offset      写: 法兰 = 爪尖 - R @ offset
# ⚠ 请按实际测量值填写 (夹爪本体长度 + 指尖到法兰的距离)。默认全 0 = 不补偿。
GRIPPER_TCP_OFFSET_M = np.array([0.0, 0.0, 0.22], dtype=np.float64)

# pick_cube was converted with the previous controller-to-virtual-TCP mount.
# The current robot TCP convention follows the updated pico_arm_transform.py.
# H = inv(^C T_E_old) @ ^C T_E_current = ^E_old T_E_current.
# It converts episode-relative poses by conjugation, never by subtracting xyz
# or Euler angles:
#   D_old     = H @ D_current @ inv(H)
#   D_current = inv(H) @ D_old @ H
PICK_CUBE_OLD_TO_CURRENT_TCP = np.array(
    [
        [0.8658017045821326, -0.01954129381682107, 0.5000055461478783, 0.06966768097352391],
        [0.03803783504616464, 0.998916159233334, -0.02682591895024177, -0.001363372983792826],
        [-0.4989394065892914, 0.04224505484064194, 0.8656066219096504, -0.05992131303582916],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# 初始关节位姿 (重置用, 弧度)
INIT_JOINTS_RAD = np.array([-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# 默认语言指令 (推理模式回退用；评测模式默认读数据集 tasks.jsonl)
DEFAULT_PROMPT = "<control mode> end effector <control mode>pick up the red bottle cap and place it into the cup."

# ===========================================================================
# EEF 位姿插值工具
# ===========================================================================


def _interp_pose(p0: np.ndarray, p1: np.ndarray, alpha: float) -> np.ndarray:
    """位置线性插值 + 旋转球面插值 (SLERP, rotvec)。

    p0/p1: (6,) [x, y, z, rotvec(3)]。alpha ∈ [0, 1]。
    旋转插值: 相对旋转 rotvec 按 alpha 缩放再复合，等价于 SLERP。
    """
    pos = p0[:3] + (p1[:3] - p0[:3]) * alpha
    r0 = Rotation.from_rotvec(p0[3:6])
    r1 = Rotation.from_rotvec(p1[3:6])
    r_rel = r1 * r0.inv()
    r = Rotation.from_rotvec(r_rel.as_rotvec() * alpha) * r0
    return np.concatenate([pos, r.as_rotvec()])


def _pose6_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert ``[xyz, rotvec]`` to a local-to-parent 4x4 transform."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"pose must have shape (6,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _matrix_to_pose6(transform: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transform to ``[xyz, rotvec]``."""
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        [transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]
    )


def _relative_pose6(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return ``inv(T_reference) @ T_current`` as ``[xyz, rotvec]``."""
    reference_matrix = _pose6_to_matrix(reference)
    current_matrix = _pose6_to_matrix(current)
    relative = np.eye(4, dtype=np.float64)
    relative[:3, :3] = reference_matrix[:3, :3].T @ current_matrix[:3, :3]
    relative[:3, 3] = reference_matrix[:3, :3].T @ (
        current_matrix[:3, 3] - reference_matrix[:3, 3]
    )
    return _matrix_to_pose6(relative)


def _compose_pose6(parent: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Return ``T_parent @ T_local`` as ``[xyz, rotvec]``."""
    return _matrix_to_pose6(_pose6_to_matrix(parent) @ _pose6_to_matrix(local))


def _conjugate_pose6(frame_change: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Return ``frame_change @ pose @ inv(frame_change)`` as ``[xyz, rotvec]``."""
    frame_change = np.asarray(frame_change, dtype=np.float64)
    if frame_change.shape != (4, 4):
        raise ValueError(f"frame_change must have shape (4, 4), got {frame_change.shape}")
    return _matrix_to_pose6(
        frame_change @ _pose6_to_matrix(pose) @ np.linalg.inv(frame_change)
    )


def _episode_poses_to_runtime(
    poses: np.ndarray,
    *,
    model_action_frame: str = "tcp",
    legacy_pick_cube_tcp: bool = False,
    tcp_start_pose: Optional[np.ndarray] = None,
    pico_reference_pose: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert model-frame episode poses for offline execution visualization.

    ``poses`` keeps the model/dataset convention. For an old ``pick_cube``
    model, each pose is first conjugated into the current robot TCP frame. If
    ``tcp_start_pose`` is provided, the result is then anchored in Robot Base:
    ``^B T_target(t) = ^B T_start @ D_current(t)``.
    """
    if model_action_frame not in {"robot_absolute", "chunk_relative", "tcp", "pico", "pico_absolute", "tcp_absolute"}:
        raise ValueError(f"invalid model_action_frame: {model_action_frame!r}")
    if model_action_frame in {"chunk_relative", "pico", "pico_absolute", "tcp_absolute"} and legacy_pick_cube_tcp:
        raise ValueError("model_action_frame='pico' and legacy_pick_cube_tcp are mutually exclusive")

    converted = np.asarray(poses, dtype=np.float64).copy()
    if converted.ndim != 2 or converted.shape[1] < 6:
        raise ValueError(f"poses must have shape (N, >=6), got {converted.shape}")
    start = None
    if tcp_start_pose is not None:
        start = np.asarray(tcp_start_pose, dtype=np.float64)
        if start.shape != (6,):
            raise ValueError(f"tcp_start_pose must have shape (6,), got {start.shape}")

    if model_action_frame in {"robot_absolute", "chunk_relative"}:
        return converted
    if model_action_frame in {"pico_absolute", "tcp_absolute"}:
        if pico_reference_pose is None:
            raise ValueError("absolute pose mode requires pico_reference_pose")
        reference = np.asarray(pico_reference_pose, dtype=np.float64)
        for index, pose in enumerate(converted):
            converted[index, :6] = _relative_pose6(reference, pose[:6])
    if model_action_frame in {"pico", "pico_absolute"}:
        converted = pico_relative_actions_to_tcp_relative(converted)

    old_to_current_inverse = np.linalg.inv(PICK_CUBE_OLD_TO_CURRENT_TCP)
    for index, pose in enumerate(converted):
        runtime_pose = (
            _conjugate_pose6(old_to_current_inverse, pose[:6])
            if legacy_pick_cube_tcp
            else pose[:6].copy()
        )
        if start is not None:
            runtime_pose = _compose_pose6(start, runtime_pose)
        converted[index, :6] = runtime_pose
    return converted


def _execution_actions(actions: np.ndarray, model_action_frame: str) -> np.ndarray:
    """Drop the same-frame action in raw PICO chunks before execution."""
    actions = np.asarray(actions)
    if model_action_frame in {"chunk_relative", "pico_absolute", "tcp_absolute"}:
        if actions.ndim != 2 or len(actions) < 2:
            raise ValueError("同帧 action 模式至少需要 2 步动作，才能跳过第 0 步")
        return actions[1:]
    return actions


def _load_raw_pico_reference(
    data_dir: str, episode: int = 0, *, pose_frame: str = "pico_teleop"
) -> tuple[np.ndarray, tuple[float, float]]:
    """Load one training episode's absolute PICO start and gripper limits."""
    import pyarrow.parquet as pq

    root = Path(data_dir)
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("pose_coordinate_frame") != pose_frame or info.get("coordinate_transform_applied") is not False:
        raise ValueError(f"数据集必须是未经坐标转换的 {pose_frame} 绝对位姿")
    chunk = episode // int(info["chunks_size"])
    path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    state = np.asarray(pq.read_table(path, columns=["state"])["state"][0].as_py(), dtype=np.float64)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError(f"episode {episode} 的初始 state 无效")
    stats = json.loads((root / "meta" / "stats.json").read_text())["state"]
    closed, opened = float(stats["min"][6]), float(stats["max"][6])
    if not np.isfinite([closed, opened]).all() or opened <= closed:
        raise ValueError("数据集夹爪范围无效")
    return state[:6], (closed, opened)


# ===========================================================================
# 数据集加载 / 图像解码 (离线评测用)
# ===========================================================================


def _load_eef_dataset(data_dir: str) -> tuple[dict, dict]:
    """加载 LeRobot 格式的 EEF 数据集 (如 piper_picodual_eef / pick_place)。

    返回 (meta, frames_per_episode)，其中 frames_per_episode: {episode_index: DataFrame}。
    state/actions 均为 7 维 [x, y, z(m), rotvec(rad), gripper_width(mm)]，
    与模型训练的 repack 一致。
    """
    if not _HAS_PANDAS:
        raise ImportError("需要 pandas: uv pip install pandas")

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    meta: dict = {}
    for name in ["info.json", "tasks.jsonl", "episodes.jsonl"]:
        p = root / "meta" / name
        if not p.exists():
            continue
        key = name.replace(".jsonl", "").replace(".json", "")
        if name.endswith(".jsonl"):
            meta[key] = [json.loads(line) for line in p.read_text().strip().split("\n") if line]
        else:
            meta[key] = json.loads(p.read_text())

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {root / 'data'}")

    frames: dict[int, "pd.DataFrame"] = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for ep in df["episode_index"].unique():
            frames[int(ep)] = df[df["episode_index"] == ep].sort_values("frame_index")

    return meta, frames


def _decode_image(row, key: str) -> np.ndarray:
    """从 parquet row 解码图像为 (H, W, 3) uint8 RGB。"""
    val = row[key]
    if isinstance(val, dict):
        data = val.get("bytes", val.get("path", None))
    elif isinstance(val, bytes):
        data = val
    elif isinstance(val, np.ndarray):
        return val
    else:
        data = None

    if isinstance(data, bytes):
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _resize_pad_rgb(img: np.ndarray, target: int = 224) -> np.ndarray:
    """等比缩放 + 居中填充到 (target, target, 3) RGB，与训练 ResizeImages 一致。"""
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((target, target, 3), dtype=np.uint8)
    ph = (target - nh) // 2
    pw = (target - nw) // 2
    out[ph : ph + nh, pw : pw + nw] = resized
    return out


# ===========================================================================
# Piper 末端位姿控制封装
# ===========================================================================


class PiperEEFController:
    """封装 Piper SDK 的末端位姿读取 / 控制接口。

    对外统一使用训练数据单位:
        EEF 位姿  [x, y, z(m), rx, ry, rz(rotvec rad)]  shape=(6,)
        夹爪      gripper_width(mm)，0=全闭，20=全开
    """

    def __init__(
        self,
        can_name: str = "can0",
        gripper=None,
        control_gripper: bool = True,
        gripper_command_max_mm: float = GRIPPER_OPEN_WIDTH_MM,
    ):
        self._piper = C_PiperInterface_V2(
            can_name=can_name,
            judge_flag=False,
            can_auto_init=True,
            dh_is_offset=1,
        )
        self._piper.ConnectPort()
        time.sleep(0.1)
        self._enabled = False
        # gripper: DmGripperController (达妙夹爪) 或 None (回退原生夹爪)
        self._gripper = gripper
        # control_gripper=False 时完全不触碰夹爪 (不下发任何夹爪指令)
        self._control_gripper = control_gripper
        self._gripper_command_max_mm = float(gripper_command_max_mm)

    # ---- 使能 / 去使能 ----

    def enable(self, speed_pct: int = DEFAULT_SPEED_PCT, max_acc: int = 200) -> bool:
        print("[Piper] 正在使能...")
        deadline = time.monotonic() + 10.0
        while not self._piper.EnablePiper():
            if time.monotonic() > deadline:
                print("[Piper] 使能超时! 检查机械臂状态。")
                return False
            time.sleep(0.01)
        self._enabled = True
        self.set_end_pose_mode(speed_pct)
        # 夹爪初始化
        if self._control_gripper:
            if self._gripper is not None:
                if not self._gripper.enable():
                    print("[Piper] 警告: DM 夹爪使能失败，仍继续 (夹爪可能不受控)")
            else:
                self._piper.GripperCtrl(0, GRIPPER_EFFORT, 0x02, 0)
                time.sleep(0.05)
                self._piper.GripperCtrl(0, GRIPPER_EFFORT, 0x01, 0)
        print("[Piper] 使能成功 (末端位姿控制模式)")
        return True

    def disable(self):
        print("[Piper] 正在去使能...")
        self._piper.DisableArm()
        if self._control_gripper and self._gripper is not None:
            self._gripper.disable()
        self._enabled = False
        time.sleep(0.1)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ---- 读取 ----

    def get_eef_pose(self) -> np.ndarray:
        """读取末端位姿 [x, y, z(m), rotvec(rad)], shape=(6,), float64。

        SDK 返回法兰 (J6) 位姿，这里叠加 GRIPPER_TCP_OFFSET_M 得到爪尖 (TCP) 位姿。
        """
        ep = self._piper.GetArmEndPoseMsgs().end_pose
        pos = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=np.float64) * _001MM_TO_M
        euler = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], dtype=np.float64) * _001DEG_TO_RAD
        R = Rotation.from_euler("xyz", euler).as_matrix()
        rotvec = Rotation.from_matrix(R).as_rotvec()
        pos = pos + R @ GRIPPER_TCP_OFFSET_M
        return np.concatenate([pos, rotvec])

    def get_gripper_mm(self) -> float:
        """读取夹爪开度：0=全闭，最大值=全开。"""
        if self._gripper is not None:
            return self._gripper.read_mm()
        raw = self._piper.GetArmGripperMsgs().gripper_state.grippers_angle
        open_fraction = np.clip(float(raw) / PIPER_GRIPPER_OPEN_RAW, 0.0, 1.0)
        return self._gripper_command_max_mm * open_fraction

    def get_state(self) -> np.ndarray:
        """完整状态 [x, y, z(m), rotvec(rad), gripper(mm)], shape=(7,), float32。"""
        return np.concatenate([self.get_eef_pose(), [self.get_gripper_mm()]]).astype(np.float32)

    # ---- 控制 ----

    def set_end_pose_mode(self, speed_pct: int = DEFAULT_SPEED_PCT):
        """切换到末端位姿控制模式 (0x00)。"""
        self._piper.MotionCtrl_2(0x01, 0x00, max(20, min(100, speed_pct)), 0x00)

    def set_speed(self, speed_pct: int):
        """动态调整速度百分比 (20-100)。"""
        self._piper.MotionCtrl_2(0x01, 0x00, max(20, min(100, speed_pct)), 0x00)

    def send_eef_command(self, pose: np.ndarray):
        """发送末端位姿指令。pose=(6,) [x,y,z(m), rotvec(rad)]，是爪尖 (TCP) 位姿。

        SDK EndPoseCtrl 接收法兰 (J6) 位姿，这里扣掉 GRIPPER_TCP_OFFSET_M。
        """
        R = Rotation.from_rotvec(pose[3:6])
        pos = pose[:3] - R.apply(GRIPPER_TCP_OFFSET_M)
        euler = R.as_euler("xyz")
        x = int(round(pos[0] * M_TO_001MM))
        y = int(round(pos[1] * M_TO_001MM))
        z = int(round(pos[2] * M_TO_001MM))
        rx = int(round(euler[0] * RAD_TO_001DEG))
        ry = int(round(euler[1] * RAD_TO_001DEG))
        rz = int(round(euler[2] * RAD_TO_001DEG))
        self._piper.EndPoseCtrl(x, y, z, rx, ry, rz)

    def send_gripper_command(self, gripper_mm: float, effort: int = GRIPPER_EFFORT):
        """发送二值夹爪指令：0=全闭，最大值=全开。"""
        if self._gripper is not None:
            self._gripper.send(gripper_mm)
            return
        open_fraction = np.clip(float(gripper_mm) / self._gripper_command_max_mm, 0.0, 1.0)
        raw = int(round(PIPER_GRIPPER_OPEN_RAW * open_fraction))
        self._piper.GripperCtrl(raw, effort, 0x01, 0)

    def execute_action(self, action: np.ndarray, speed_pct: int = DEFAULT_SPEED_PCT):
        """执行单个动作。action=(7,) [x,y,z,rotvec(3), gripper_mm]。"""
        self.send_eef_command(action[:6])
        if self._control_gripper:
            self.send_gripper_command(action[6])

    def go_to_init_pose(self, wait: float = 3.0):
        """重置: 切关节模式回到初始关节位姿，再切回末端位姿模式。"""
        print("[Piper] 回到初始位姿 (关节空间) ...")
        # 关节空间定位 (JointCtrl), 与 piper_interpolation_controller 的 homing 一致
        raw = [int(round(float(j) * RAD_TO_001DEG)) for j in INIT_JOINTS_RAD]
        self._piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)  # 关节模式
        self._piper.JointCtrl(*raw)
        time.sleep(wait)
        self._piper.MotionCtrl_2(0x01, 0x00, DEFAULT_SPEED_PCT, 0x00)  # 切回 EEF 模式
        if self._control_gripper:
            if self._gripper is not None:
                self._gripper.open()  # 达妙夹爪: 0 rad = 全开 (安全默认)
            else:
                self._piper.GripperCtrl(PIPER_GRIPPER_OPEN_RAW, GRIPPER_EFFORT, 0x01, 0)
        print("[Piper] 已到初始位姿")

    def close_gripper_transport(self):
        """关闭 DM 夹爪电机并释放 USB2CAN 串口 (原生夹爪为 no-op)。"""
        if self._gripper is not None:
            self._gripper.shutdown()


# ===========================================================================
# DM 夹爪控制封装 (达妙 DM-J4310-2EC, 力位模式)
# ===========================================================================


class DmGripperController:
    """达妙 DM-J4310-2EC 夹爪控制器 (力位模式)。

    通过 USB2CAN 模块 (``/dev/ttyACM1``) 控制，与机械臂的 can0 相互独立。
    二值物理命令约定: 0 = 全闭，open_width_mm = 全开；
    电机约定 0 rad = 全开，close_rad = 全闭，故二者为反比映射。
    """

    def __init__(
        self,
        port: str = GRIPPER_PORT,
        can_id: int = GRIPPER_CAN_ID,
        open_rad: float = GRIPPER_OPEN_RAD,
        close_rad: float = GRIPPER_CLOSE_RAD,
        open_width_mm: float = GRIPPER_OPEN_WIDTH_MM,
        current_limit: float = GRIPPER_CURRENT_LIMIT,
        speed_rad_s: float = GRIPPER_SPEED_RAD_S,
    ):
        self._bus = Usb2Can(port=port)
        self._motor = DmJ4310(self._bus, can_id=can_id)
        self._gripper = DmGripper(
            self._motor, open_rad=open_rad, close_rad=close_rad, speed_rad_s=speed_rad_s, current_limit=current_limit
        )
        self._open_width_mm = float(open_width_mm)
        self._close_rad = float(close_rad)

    # ---- 生命周期 ----
    def enable(self) -> bool:
        return self._gripper.enable()

    def disable(self):
        self._gripper.disable()

    def open(self):
        """全开 (0 rad)。阻塞直到到位。"""
        self._gripper.open()

    def shutdown(self):
        """去使能电机并关闭串口。"""
        self._gripper.shutdown()

    # ---- mm <-> rad 映射 ----
    def _mm_to_rad(self, mm: float) -> float:
        frac = max(0.0, min(1.0, float(mm) / self._open_width_mm))
        return self._close_rad * (1.0 - frac)

    def _rad_to_mm(self, rad: float) -> float:
        frac = max(0.0, min(1.0, float(rad) / self._close_rad))
        return self._open_width_mm * (1.0 - frac)

    # ---- 控制 / 读取 ----
    def send(self, gripper_mm: float):
        """发送夹爪宽度指令 (mm)。非阻塞，20Hz 循环里每次下发即可。"""
        self._gripper.set_position(self._mm_to_rad(gripper_mm))

    def read_mm(self) -> float:
        """读取当前夹爪宽度 (mm)。"""
        pos = self._gripper.position()
        if pos is None:
            return 0.0
        return self._rad_to_mm(pos)


# ===========================================================================
# 推理引擎
# ===========================================================================


class _RealRobotTrajectoryRecorder:
    """Collect and atomically save real-robot inference samples as NPZ."""

    FORMAT_VERSION = 1

    def __init__(self, output_path: str):
        path = Path(output_path).expanduser()
        if path.suffix == "":
            path = path.with_suffix(".npz")
        if path.suffix.lower() != ".npz":
            raise ValueError("trajectory_log 必须使用 .npz 扩展名")
        if path.exists():
            raise FileExistsError(f"轨迹日志已存在，为避免覆盖请换一个路径: {path}")
        self.path = path
        self._session_start = time.monotonic()
        self._episode_id = -1
        self._episode_start_tcp: list[np.ndarray] = []
        self._episode_index: list[int] = []
        self._action_index: list[int] = []
        self._command_time_s: list[float] = []
        self._measurement_time_s: list[float] = []
        self._wall_time_ns: list[int] = []
        self._model_action: list[np.ndarray] = []
        self._runtime_relative_target: list[np.ndarray] = []
        self._base_target_tcp: list[np.ndarray] = []
        self._measured_tcp: list[np.ndarray] = []
        self._scaled_gripper_mm: list[float] = []
        self._command_gripper_mm: list[float] = []
        self._last_saved_count = -1
        self._owns_output = False

    def begin_episode(self, robot_tcp0: np.ndarray) -> None:
        tcp0 = np.asarray(robot_tcp0, dtype=np.float64)
        if tcp0.shape != (6,):
            raise ValueError(f"robot_tcp0 must have shape (6,), got {tcp0.shape}")
        self._episode_id += 1
        self._episode_start_tcp.append(tcp0.copy())

    def record(
        self,
        *,
        action_index: int,
        command_time: float,
        model_action: np.ndarray,
        runtime_relative_target: np.ndarray,
        base_target_tcp: np.ndarray,
        scaled_gripper_mm: float,
        command_gripper_mm: float,
        measured_tcp: np.ndarray,
        measurement_time: float,
    ) -> None:
        if self._episode_id < 0:
            raise RuntimeError("begin_episode must be called before record")
        self._episode_index.append(self._episode_id)
        self._action_index.append(int(action_index))
        self._command_time_s.append(command_time - self._session_start)
        self._measurement_time_s.append(measurement_time - self._session_start)
        self._wall_time_ns.append(time.time_ns())
        self._model_action.append(np.asarray(model_action, dtype=np.float64).copy())
        self._runtime_relative_target.append(
            np.asarray(runtime_relative_target, dtype=np.float64).copy()
        )
        self._base_target_tcp.append(np.asarray(base_target_tcp, dtype=np.float64).copy())
        self._measured_tcp.append(np.asarray(measured_tcp, dtype=np.float64).copy())
        self._scaled_gripper_mm.append(float(scaled_gripper_mm))
        self._command_gripper_mm.append(float(command_gripper_mm))

    @staticmethod
    def _stack(rows: list[np.ndarray], width: int) -> np.ndarray:
        return np.stack(rows).astype(np.float64) if rows else np.empty((0, width), dtype=np.float64)

    def save(
        self, *, legacy_pick_cube_tcp: bool, prompt: str, model_action_frame: str = "robot_absolute"
    ) -> None:
        count = len(self._episode_index)
        if count == self._last_saved_count:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self._owns_output:
            raise FileExistsError(f"轨迹日志已存在，拒绝覆盖: {self.path}")
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "pose_layout": "[x_m, y_m, z_m, rx_rotvec_rad, ry_rotvec_rad, rz_rotvec_rad]",
            "model_action_layout": "pose6 + predicted_gripper",
            "model_action_frame": model_action_frame,
            "legacy_pick_cube_tcp": bool(legacy_pick_cube_tcp),
            "prompt": prompt,
        }
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        with temporary_path.open("wb") as file:
            np.savez_compressed(
                file,
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                episode_start_tcp=self._stack(self._episode_start_tcp, 6),
                episode_index=np.asarray(self._episode_index, dtype=np.int32),
                action_index=np.asarray(self._action_index, dtype=np.int32),
                command_time_s=np.asarray(self._command_time_s, dtype=np.float64),
                measurement_time_s=np.asarray(self._measurement_time_s, dtype=np.float64),
                wall_time_ns=np.asarray(self._wall_time_ns, dtype=np.int64),
                model_action=self._stack(self._model_action, 7),
                runtime_relative_target=self._stack(self._runtime_relative_target, 6),
                base_target_tcp=self._stack(self._base_target_tcp, 6),
                measured_tcp=self._stack(self._measured_tcp, 6),
                scaled_gripper_mm=np.asarray(self._scaled_gripper_mm, dtype=np.float64),
                command_gripper_mm=np.asarray(self._command_gripper_mm, dtype=np.float64),
            )
        os.replace(temporary_path, self.path)
        self._owns_output = True
        self._last_saved_count = count
        print(f"[Trajectory] 已保存 {count} 个控制样本 → {self.path}")


class PiperEEFInference:
    """Piper 末端位姿推理主循环 (receding horizon control)。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        can_name: str = "can0",
        rs2_base_serial: Optional[str] = None,
        cv_base_id: Optional[int] = None,
        cv_wrist_id: Optional[int] = None,
        wrist_exposure: Optional[float] = None,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        exec_horizon: int = DEFAULT_EXEC_HORIZON,
        max_pos_speed: float = DEFAULT_MAX_POS_SPEED,
        max_rot_speed: float = DEFAULT_MAX_ROT_SPEED,
        interp_freq: Optional[float] = DEFAULT_INTERP_FREQ,
        gripper_port: Optional[str] = GRIPPER_PORT,
        gripper_close_rad: float = GRIPPER_CLOSE_RAD,
        gripper_open_width_mm: float = GRIPPER_OPEN_WIDTH_MM,
        gripper_binary_threshold_mm: Optional[float] = None,
        gripper_prediction_scale: float = GRIPPER_PREDICTION_SCALE,
        binary_gripper: bool = True,
        gripper_current: float = GRIPPER_CURRENT_LIMIT,
        gripper_speed: float = GRIPPER_SPEED_RAD_S,
        default_prompt: str = (
            "<control mode> end effector <control mode>pick up the black block and place it into the cup."
        ),
        interactive: bool = False,
        model_action_frame: str = "robot_absolute",
        legacy_pick_cube_tcp: bool = False,
        pico_reference_pose: Optional[np.ndarray] = None,
        model_gripper_range: Optional[tuple[float, float]] = None,
        trajectory_log: Optional[str] = None,
    ):
        self._action_horizon = action_horizon
        self._exec_horizon = exec_horizon
        self._max_pos_speed = max_pos_speed
        self._max_rot_speed = max_rot_speed
        self._interp_freq = interp_freq
        self._use_interp = interp_freq is not None
        self._default_prompt = default_prompt
        self._interactive = interactive
        if model_action_frame not in {"robot_absolute", "chunk_relative", "tcp", "pico", "pico_absolute", "tcp_absolute"}:
            raise ValueError(f"无效 model_action_frame: {model_action_frame!r}")
        self._model_action_frame = model_action_frame
        self._legacy_pick_cube_tcp = bool(legacy_pick_cube_tcp)
        if self._model_action_frame in {"chunk_relative", "pico", "pico_absolute", "tcp_absolute"} and self._legacy_pick_cube_tcp:
            raise ValueError(
                "model_action_frame='pico' 与 legacy_pick_cube_tcp 不能同时启用"
            )
        self._pico_reference_pose = (
            None if pico_reference_pose is None else np.asarray(pico_reference_pose, dtype=np.float64)
        )
        if self._model_action_frame in {"pico_absolute", "tcp_absolute"}:
            if self._pico_reference_pose is None or self._pico_reference_pose.shape != (6,):
                raise ValueError("绝对位姿模式需要 6 维初始位姿")
            if model_gripper_range is None or model_gripper_range[1] <= model_gripper_range[0]:
                raise ValueError("绝对位姿模式需要有效的训练数据夹爪范围")
        self._model_gripper_range = model_gripper_range
        self._trajectory_recorder = (
            None if trajectory_log is None else _RealRobotTrajectoryRecorder(trajectory_log)
        )
        self._period = 1.0 / CONTROL_FREQ
        self._gripper_open_width_mm = float(gripper_open_width_mm)
        self._gripper_binary_threshold_mm = (
            GRIPPER_BINARY_THRESHOLD_MM
            if gripper_binary_threshold_mm is None
            else float(gripper_binary_threshold_mm)
        )
        self._gripper_prediction_scale = float(gripper_prediction_scale)
        self._binary_gripper = bool(binary_gripper)
        if self._gripper_prediction_scale <= 0.0:
            raise ValueError(
                f"gripper_prediction_scale 必须大于 0，实际为 {self._gripper_prediction_scale}"
            )
        if not 0.0 <= self._gripper_binary_threshold_mm <= self._gripper_open_width_mm:
            raise ValueError(
                "gripper_binary_threshold_mm 必须在 0 和 gripper_open_width_mm 之间，"
                f"实际为 {self._gripper_binary_threshold_mm}"
            )

        # 连接 (机械臂 + 夹爪)
        if _HAS_DM_GRIPPER and gripper_port:
            print(f"[Gripper] 使用达妙 DM-J4310-2EC 夹爪 ({gripper_port})")
            gripper = DmGripperController(
                port=gripper_port,
                close_rad=gripper_close_rad,
                open_width_mm=gripper_open_width_mm,
                current_limit=gripper_current,
                speed_rad_s=gripper_speed,
            )
        else:
            if not _HAS_DM_GRIPPER:
                print("[Gripper] 未加载达妙夹爪驱动，回退原生夹爪")
            else:
                print("[Gripper] 未指定夹爪串口，回退原生夹爪")
            gripper = None
        self._robot = PiperEEFController(
            can_name,
            gripper=gripper,
            gripper_command_max_mm=gripper_open_width_mm,
        )
        self._policy_client: Optional[websocket_client_policy.WebsocketClientPolicy] = None
        self._policy_host = host
        self._policy_port = port

        # 相机: 基座 = RealSense (D435i)，腕部 = USB 摄像头 (OpenCV)
        self._camera = create_cameras(
            base_serial=rs2_base_serial,
            base_rs_size=(1280, 720),  # 与 pick_cube 数据集的 base 图像分辨率一致
            base_cv_id=cv_base_id,
            wrist_cv_id=cv_wrist_id,
            wrist_cv_exposure=wrist_exposure,
        )
        self._has_base_cam = bool(rs2_base_serial or cv_base_id is not None)
        self._has_wrist_cam = cv_wrist_id is not None

        # 状态
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._inferring = False
        self._action_cache: Optional[np.ndarray] = None  # (action_horizon, 7)
        self._cache_step = 0
        self._robot_tcp0: Optional[np.ndarray] = None
        self._chunk_tcp_base: Optional[np.ndarray] = None

    def _runtime_pose_to_model_pose(self, pose: np.ndarray) -> np.ndarray:
        """Map a robot-TCP relative pose into the training pose frame."""
        converted = np.asarray(pose, dtype=np.float64).copy()
        if getattr(self, "_model_action_frame", "tcp") == "tcp_absolute":
            return _compose_pose6(self._pico_reference_pose, converted)
        if getattr(self, "_model_action_frame", "tcp") in {"pico", "pico_absolute"}:
            converted = tcp_relative_actions_to_pico_relative(converted)
            if self._model_action_frame == "pico_absolute":
                converted[:6] = _compose_pose6(self._pico_reference_pose, converted[:6])
            return converted
        if self._legacy_pick_cube_tcp:
            return _conjugate_pose6(PICK_CUBE_OLD_TO_CURRENT_TCP, converted)
        return converted

    def _model_pose_to_runtime_pose(self, pose: np.ndarray) -> np.ndarray:
        """Map a model target into the current robot-TCP relative frame."""
        converted = np.asarray(pose, dtype=np.float64).copy()
        if getattr(self, "_model_action_frame", "tcp") == "tcp_absolute":
            return _relative_pose6(self._pico_reference_pose, converted)
        if getattr(self, "_model_action_frame", "tcp") in {"pico", "pico_absolute"}:
            if self._model_action_frame == "pico_absolute":
                converted[:6] = _relative_pose6(self._pico_reference_pose, converted[:6])
            return pico_relative_actions_to_tcp_relative(converted)
        if self._legacy_pick_cube_tcp:
            return _conjugate_pose6(np.linalg.inv(PICK_CUBE_OLD_TO_CURRENT_TCP), converted)
        return converted

    def _runtime_gripper_to_model(self, width_mm: float) -> float:
        if self._model_gripper_range is None:
            return width_mm
        closed, opened = self._model_gripper_range
        return closed + np.clip(width_mm / self._gripper_open_width_mm, 0.0, 1.0) * (opened - closed)

    def _model_gripper_to_runtime(self, width: float) -> float:
        if self._model_gripper_range is None:
            return width
        closed, opened = self._model_gripper_range
        return (width - closed) / (opened - closed) * self._gripper_open_width_mm

    # ======================== 运行 ========================

    def run(self):
        print("=" * 60)
        print("Piper EEF 推理客户端 (策略服务器模式)")
        print(f"服务器:         {self._policy_host}:{self._policy_port}")
        print(f"模型动作坐标系: {self._model_action_frame.upper()}")
        print(
            "TCP 兼容模式:    "
            + ("旧 pick_cube → 当前 TCP (SE(3) 共轭)" if self._legacy_pick_cube_tcp else "关闭")
        )
        print(
            "轨迹记录:       "
            + (str(self._trajectory_recorder.path) if self._trajectory_recorder is not None else "关闭")
        )
        print(
            f"控制频率:       {CONTROL_FREQ} Hz "
            f"(期望动作块 {self._action_horizon} 步, 每块最多执行 {self._exec_horizon} 步)"
        )
        print(f"夹爪预测缩放:   pred × {self._gripper_prediction_scale:g}")
        if self._binary_gripper:
            print(
                f"夹爪控制模式:   二值；<= {self._gripper_binary_threshold_mm:.3f} 关(0mm), "
                f"> {self._gripper_binary_threshold_mm:.3f} 开({self._gripper_open_width_mm:.1f}mm)"
            )
        else:
            print(f"夹爪控制模式:   连续；范围 0–{self._gripper_open_width_mm:.1f}mm")
        if self._use_interp:
            print(
                f"位姿插值:       启用 (max_pos={self._max_pos_speed:.2f} m/s, "
                f"max_rot={self._max_rot_speed:.2f} rad/s, freq={self._interp_freq} Hz)"
            )
        else:
            print("位姿插值:       关闭 (直接 EndPoseCtrl)")
        print("=" * 60)

        # 1. 使能机械臂
        if not self._robot.enable():
            return

        # 2. 启动相机
        if self._camera:
            self._camera.start()
            print("[Camera] 已启动")

        # 3. 连接策略服务器
        self._connect_policy()

        self._running = True

        # 4. 键盘监听
        input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        input_thread.start()

        try:
            self._control_loop()
        except KeyboardInterrupt:
            print("\n[INFO] 中断信号，退出中...")
        finally:
            self._save_trajectory_log()
            if self._camera:
                self._camera.stop()
            cv2.destroyAllWindows()
            print("[INFO] 已退出 (机械臂和夹爪保持使能)")

    def _save_trajectory_log(self) -> None:
        if self._trajectory_recorder is None:
            return
        try:
            self._trajectory_recorder.save(
                legacy_pick_cube_tcp=self._legacy_pick_cube_tcp,
                prompt=self._default_prompt,
                model_action_frame=self._model_action_frame,
            )
        except Exception as exc:
            print(f"[Trajectory] 保存失败: {exc}")

    def _connect_policy(self):
        print(f"[Policy] 正在连接 ws://{self._policy_host}:{self._policy_port} ...")
        self._policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=self._policy_host,
            port=self._policy_port,
        )
        meta = self._policy_client.get_server_metadata()
        print(f"[Policy] 已连接. 服务器元数据: {meta}")

    # ======================== 控制循环 ========================

    def _control_loop(self):
        fps_counter = _FPSCounter("control")
        cur_pose: Optional[np.ndarray] = None
        data_dt = 1.0 / CONTROL_FREQ

        while self._running:
            loop_start = time.monotonic()

            # 处理键盘事件
            try:
                cmd = self._input_queue.get_nowait()
                self._handle_command(cmd)
            except queue.Empty:
                pass

            if not self._inferring:
                self._display_cameras()
                time.sleep(0.01)
                continue

            # --- 推理 + 动作执行 ---
            # 服务器的 action_horizon 由其模型配置决定。执行步数必须同时受
            # exec_horizon 和实际返回长度约束，避免把动作块最后一步重复执行。
            cache_exec_steps = (
                0
                if self._action_cache is None
                else min(self._exec_horizon, self._action_cache.shape[0])
            )
            if self._action_cache is None or self._cache_step >= cache_exec_steps:
                self._query_policy()
                self._cache_step = 0
                cur_pose = self._robot.get_eef_pose()

            if self._action_cache is not None:
                idx = min(self._cache_step, self._action_cache.shape[0] - 1)
                action = self._action_cache[idx]  # (7,) [x,y,z,rotvec,gripper_mm]

                if self._model_action_frame == "chunk_relative":
                    if self._chunk_tcp_base is None:
                        raise RuntimeError("chunk TCP base was not captured with the observation")
                    runtime_target = action[:6].astype(np.float64)
                    target_pose = _compose_pose6(self._chunk_tcp_base, runtime_target)
                elif self._model_action_frame == "robot_absolute":
                    target_pose = action[:6].astype(np.float64)
                    runtime_target = (
                        _relative_pose6(self._robot_tcp0, target_pose)
                        if self._robot_tcp0 is not None else target_pose.copy()
                    )
                else:
                    if self._robot_tcp0 is None:
                        raise RuntimeError("episode robot TCP reference is not initialized")
                    runtime_target = self._model_pose_to_runtime_pose(action[:6])
                    target_pose = _compose_pose6(self._robot_tcp0, runtime_target)
                predicted_gripper_mm = float(action[6])
                scaled_gripper_mm = (
                    self._model_gripper_to_runtime(predicted_gripper_mm) * self._gripper_prediction_scale
                )
                gripper_mm = self._gripper_command(scaled_gripper_mm)

                if cur_pose is None:
                    cur_pose = self._robot.get_eef_pose()
                command_time = time.monotonic()
                self._send_eef_cmd(cur_pose, target_pose, gripper_mm, data_dt)
                if self._trajectory_recorder is not None:
                    try:
                        measured_tcp = self._robot.get_eef_pose().astype(np.float64)
                    except Exception as exc:
                        print(f"[Trajectory] TCP 读回失败，当前样本记为 NaN: {exc}")
                        measured_tcp = np.full(6, np.nan, dtype=np.float64)
                    measurement_time = time.monotonic()
                    self._trajectory_recorder.record(
                        action_index=idx,
                        command_time=command_time,
                        model_action=action,
                        runtime_relative_target=runtime_target,
                        base_target_tcp=target_pose,
                        scaled_gripper_mm=scaled_gripper_mm,
                        command_gripper_mm=gripper_mm,
                        measured_tcp=measured_tcp,
                        measurement_time=measurement_time,
                    )
                cur_pose = target_pose.copy()

                self._cache_step += 1
                fps_counter.tick()

                if fps_counter.count % 5 == 0:
                    self._display_cameras()

                if fps_counter.count % 100 == 0:
                    state = self._robot.get_state()
                    p_str = ", ".join(f"{v:.4f}" for v in state[:3])
                    r_str = ", ".join(f"{v:.3f}" for v in state[3:6])
                    print(
                        f"[{fps_counter.count:5d}] fps={fps_counter.fps:.1f} | "
                        f"pos=[{p_str}] rot=[{r_str}] grip={state[6]:.1f}mm | "
                        f"pred_grip={predicted_gripper_mm:.3f} | "
                        f"scaled_grip={scaled_gripper_mm:.3f} | cmd_grip={gripper_mm:.1f}mm"
                    )

            # 控制频率
            elapsed = time.monotonic() - loop_start
            if elapsed < self._period:
                time.sleep(self._period - elapsed)

    # ======================== EEF 控制 ========================

    def _gripper_command(self, scaled_mm: float) -> float:
        """按配置生成二值命令，或生成限幅后的连续命令。"""
        if self._binary_gripper:
            return (
                self._gripper_open_width_mm
                if float(scaled_mm) > self._gripper_binary_threshold_mm
                else 0.0
            )
        return float(np.clip(scaled_mm, 0.0, self._gripper_open_width_mm))

    def _send_eef_cmd(
        self,
        cur_pose: np.ndarray,
        target_pose: np.ndarray,
        gripper_mm: float,
        data_dt: float,
    ):
        """发送一个 EEF 位姿目标，可选带插值。

        cur_pose/target_pose: (6,) [x,y,z(m), rotvec(rad)]。
        插值模式: 位置线性 + 旋转 SLERP，线速度/角速度受 max_*_speed 限制。
        """
        if not self._use_interp:
            self._robot.send_eef_command(target_pose)
            self._robot.send_gripper_command(gripper_mm)
            time.sleep(data_dt)
            return

        # 相对运动量
        dpos = float(np.linalg.norm(target_pose[:3] - cur_pose[:3]))
        r_cur = Rotation.from_rotvec(cur_pose[3:6])
        r_tgt = Rotation.from_rotvec(target_pose[3:6])
        drot = float(np.linalg.norm((r_tgt * r_cur.inv()).as_rotvec()))

        dur_pos = dpos / self._max_pos_speed
        dur_rot = drot / self._max_rot_speed
        dur = max(dur_pos, dur_rot, data_dt)
        inner_dt = 1.0 / self._interp_freq
        n = max(1, int(round(dur / inner_dt)))

        for i in range(1, n + 1):
            pose = _interp_pose(cur_pose, target_pose, i / n)
            self._robot.send_eef_command(pose)
            time.sleep(inner_dt)

        self._robot.send_gripper_command(gripper_mm)

    def _display_cameras(self):
        """将所有相机画面拼接显示。"""
        if not self._camera:
            return

        frames = []
        if self._has_base_cam:
            img = self._camera.get_base()
            cv2.putText(img, "base", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(img)
        if self._has_wrist_cam:
            img = self._camera.get_wrist()
            cv2.putText(img, "wrist", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frames.append(img)

        if frames:
            vis = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
            cv2.imshow("Piper EEF Inference", vis[..., ::-1])  # RGB → BGR
            cv2.waitKey(1)

    def _query_policy(self):
        try:
            obs = self._build_observation()
            result = self._policy_client.infer(obs)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < 7:
                raise ValueError(f"策略 actions 形状应为 (N, >=7)，实际为 {actions.shape}")
            self._action_cache = _execution_actions(actions, self._model_action_frame)[:, :7]
            grip = self._action_cache[:, 6]
            scaled_grip = np.array(
                [self._model_gripper_to_runtime(float(value)) for value in grip]
            ) * self._gripper_prediction_scale
            execute_steps = min(self._exec_horizon, self._action_cache.shape[0])
            executed_scaled_grip = scaled_grip[:execute_steps]
            mode_info = ""
            if self._binary_gripper:
                open_indices = np.flatnonzero(scaled_grip > self._gripper_binary_threshold_mm)
                first_open = str(int(open_indices[0])) if open_indices.size else "无"
                if open_indices.size and open_indices[0] >= execute_steps:
                    first_open += " (本块不执行)"
                mode_info = f" | first_grip>{self._gripper_binary_threshold_mm:.3f}={first_open}"
            print(
                f"  [Policy] actions={self._action_cache.shape[0]} | execute={execute_steps} | "
                f"grip_range=[{float(grip.min()):.3f}, {float(grip.max()):.3f}] | "
                f"scaled_executed=[{float(executed_scaled_grip.min()):.3f}, "
                f"{float(executed_scaled_grip.max()):.3f}]"
                f"{mode_info}"
            )
            timings = result.get("policy_timing", {})
            server_timings = result.get("server_timing", {})
            if timings or server_timings:
                parts = []
                if timings:
                    parts.append(f"infer={timings.get('infer_ms', 0):.0f}ms")
                if server_timings:
                    parts.append(f"server={server_timings.get('total_ms', 0):.0f}ms")
                print(f"  [Policy] {' | '.join(parts)}")
        except Exception as e:
            print(f"[Policy] 查询失败: {e}")
            time.sleep(1)
            try:
                self._connect_policy()
                print("[Policy] 重连成功")
            except Exception:
                print("[Policy] 重连失败，暂停推理")
                self._inferring = False

    def _build_observation(self) -> dict:
        """构建与训练数据同坐标系的当前观测。"""
        state = self._robot.get_state().copy()  # Robot Base TCP pose + gripper
        if self._model_action_frame == "chunk_relative":
            self._chunk_tcp_base = state[:6].astype(np.float64).copy()
            state = state[6:7]
        elif self._model_action_frame != "robot_absolute":
            if self._robot_tcp0 is None:
                raise RuntimeError("episode robot TCP reference is not initialized")
            state[:6] = _relative_pose6(self._robot_tcp0, state[:6])
            state[:6] = self._runtime_pose_to_model_pose(state[:6])
            state[6] = self._runtime_gripper_to_model(float(state[6]))

        base_image = np.zeros((224, 224, 3), dtype=np.uint8)
        wrist_image = base_image.copy()
        if self._camera:
            if self._has_base_cam:
                base_image = self._camera.get_base()
            if self._has_wrist_cam:
                wrist_image = self._camera.get_wrist()

        if self._interactive:
            prompt = input("指令: ").strip() or self._default_prompt
        else:
            prompt = self._default_prompt

        # 格式必须匹配策略服务器端配置的 transforms (与 training repack 一致)。
        return {
            "observation/state": state.astype(np.float32),
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "prompt": prompt,
        }

    # ======================== 键盘交互 ========================

    def _handle_command(self, cmd: str):
        if cmd == "start":
            if not self._inferring:
                print("[Control] 开始推理...")
                self._robot_tcp0 = self._robot.get_eef_pose().astype(np.float64)
                if self._trajectory_recorder is not None:
                    self._trajectory_recorder.begin_episode(self._robot_tcp0)
                self._action_cache = None
                self._cache_step = 0
                self._inferring = True
        elif cmd == "stop":
            print("[Control] 暂停推理")
            self._inferring = False
            self._action_cache = None
            self._cache_step = 0
            self._save_trajectory_log()
        elif cmd == "quit":
            self._running = False
        elif cmd == "open_gripper":
            # 手动测试会暂停策略，防止下一控制周期立刻被策略的闭爪命令覆盖。
            self._inferring = False
            self._action_cache = None
            self._cache_step = 0
            self._save_trajectory_log()
            print(f"[Control] 暂停推理并手动开爪 (command={self._gripper_open_width_mm:.1f}mm)")
            self._robot.send_gripper_command(self._gripper_open_width_mm)
        elif cmd == "reset":
            was_inferring = self._inferring
            self._inferring = False
            if was_inferring:
                self._save_trajectory_log()
            self._robot.go_to_init_pose()
            self._robot_tcp0 = self._robot.get_eef_pose().astype(np.float64)
            if was_inferring and self._trajectory_recorder is not None:
                self._trajectory_recorder.begin_episode(self._robot_tcp0)
            self._action_cache = None
            self._cache_step = 0
            self._inferring = was_inferring

    def _keyboard_listener(self):
        print("\n操作提示:")
        print("  [Enter]  开始推理")
        print("  [s]      暂停")
        print("  [o]      暂停并手动开爪 (驱动测试)")
        print("  [r]      重置到初始位姿")
        print("  [q]      退出\n")

        while self._running:
            try:
                ch = sys.stdin.readline().strip().lower()
                if ch == "":
                    self._input_queue.put("start")
                elif ch == "s":
                    self._input_queue.put("stop")
                elif ch == "q":
                    self._input_queue.put("quit")
                elif ch == "o":
                    self._input_queue.put("open_gripper")
                elif ch == "r":
                    self._input_queue.put("reset")
            except (EOFError, OSError):
                break


# ===========================================================================
# 数据集离线评测 (可视化末端位姿 + GIF 导出)
# ===========================================================================


class PiperEEFDatasetEval:
    """用 LeRobot EEF 数据集回放观测、查询策略服务器，评测并可视化末端位姿。

    与真实推理 (PiperEEFInference) 的区别:
        - 不连接真实机械臂 / 相机，state 与图像直接取自数据集。
        - 支持闭环 (预测位姿反馈，默认) / 开环 (每步用真值位姿) 两种 rollout。
        - 可视化末端位姿 (3D 位置轨迹 + 朝向 triad + 夹爪 + 误差)，可导出 GIF。
        - ``--init_pose`` 覆盖模型的 episode-relative 初始 state。
        - ``--tcp_start_pose`` 把输出轨迹锚定到 Robot Base 下的真实 TCP 起点，
          不改变送入模型的相对状态。

    用法:
        python examples/piper/runtime/inference_eef.py --dataset ./pick_place --episode 0 --gif eval.gif
        python examples/piper/runtime/inference_eef.py --dataset ./pick_place --init_pose 0.0 -0.7 -0.4 0 0 0
        python examples/piper/runtime/inference_eef.py --dataset ./local/datasets/pick_cube --episode 0 \
            --legacy_pick_cube_tcp --tcp_start_pose 0.35 0.0 0.25 0.0 1.57 0.0
    """

    def __init__(
        self,
        data_dir: str,
        host: str = "localhost",
        port: int = 8000,
        episode: Optional[int] = None,
        init_pose: Optional[np.ndarray] = None,
        tcp_start_pose: Optional[np.ndarray] = None,
        model_action_frame: str = "robot_absolute",
        legacy_pick_cube_tcp: bool = False,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        exec_horizon: int = DEFAULT_EXEC_HORIZON,
        closed_loop: bool = True,
        max_frames: int = 0,
        prompt: Optional[str] = None,
        gif_path: Optional[str] = None,
        gif_fps: int = 15,
        show: bool = True,
    ):
        if not _HAS_PANDAS:
            raise ImportError("需要 pandas: uv pip install pandas")
        if not _HAS_MPL:
            raise ImportError("需要 matplotlib: uv pip install matplotlib")

        self._data_dir = data_dir
        self._host = host
        self._port = port
        self._episode = episode
        self._init_pose = None if init_pose is None else np.asarray(init_pose, dtype=np.float64)
        self._tcp_start_pose = (
            None if tcp_start_pose is None else np.asarray(tcp_start_pose, dtype=np.float64)
        )
        if self._tcp_start_pose is not None and self._tcp_start_pose.shape != (6,):
            raise ValueError(
                "tcp_start_pose 必须是 6 维 [x y z rx ry rz]，"
                f"实际为 {self._tcp_start_pose.shape}"
            )
        if model_action_frame not in {"robot_absolute", "chunk_relative", "tcp", "pico", "pico_absolute", "tcp_absolute"}:
            raise ValueError(
                f"model_action_frame 必须为 'tcp' 或 'pico'，实际为 {model_action_frame!r}"
            )
        self._model_action_frame = model_action_frame
        self._legacy_pick_cube_tcp = bool(legacy_pick_cube_tcp)
        if self._model_action_frame in {"chunk_relative", "pico", "pico_absolute", "tcp_absolute"} and self._legacy_pick_cube_tcp:
            raise ValueError(
                "model_action_frame='pico' 与 legacy_pick_cube_tcp 不能同时启用"
            )
        self._action_horizon = action_horizon
        self._exec_horizon = exec_horizon
        self._closed_loop = closed_loop
        self._max_frames = max_frames
        self._prompt = prompt
        self._gif_path = gif_path
        self._gif_fps = gif_fps
        self._show = show

        # matplotlib 后端: 纯 GIF 导出用 Agg (无头)，交互显示用 TkAgg。
        if self._gif_path and not self._show:
            matplotlib.use("Agg")
        elif self._show:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass

        self._meta, self._frames = _load_eef_dataset(data_dir)
        self._episode_indices = sorted(self._frames.keys())
        if not self._episode_indices:
            raise ValueError("数据集中没有 episode")

        self._policy_client: Optional[websocket_client_policy.WebsocketClientPolicy] = None

    # ======================== 运行入口 ========================

    def run(self):
        ep = self._episode if self._episode is not None else self._episode_indices[0]
        if ep not in self._frames:
            print(f"[ERROR] Episode {ep} 不存在。可用: {self._episode_indices}")
            return

        df = self._frames[ep]
        if self._max_frames > 0 and len(df) > self._max_frames:
            df = df.iloc[: self._max_frames]
        task = self._resolve_task(ep)
        self._prompt = task or self._prompt

        print("=" * 60)
        print("Piper EEF 数据集离线评测")
        print(f"数据集:   {self._data_dir}")
        print(f"Episode:  {ep} ({len(df)} 帧)")
        print(f"任务:     {self._prompt or '(无)'}")
        print(f"Rollout:  {'闭环 (预测位姿反馈)' if self._closed_loop else '开环 (真值位姿)'}")
        print(f"模型动作坐标系: {self._model_action_frame.upper()}")
        print(
            "TCP frame: "
            + ("旧 pick_cube 模型 → 当前机器人 TCP" if self._legacy_pick_cube_tcp else "当前机器人 TCP")
        )
        if self._tcp_start_pose is not None:
            print(f"TCP 起点: Robot Base {self._tcp_start_pose}")
            print("目标关系: ^B T_target(t) = ^B T_start @ D_current(t)")
        else:
            print("TCP 起点: 未设置；显示 episode-relative 轨迹")
        print(f"服务器:   ws://{self._host}:{self._port}")
        print("=" * 60)

        self._connect_policy()

        model_pred, model_states, model_gt = self._rollout(df)
        transform_kwargs = {
            "pico_reference_pose": (
                np.asarray(df["state"].iloc[0], dtype=np.float64)[:6]
                if self._model_action_frame in {"pico_absolute", "tcp_absolute"} else None
            ),
            "model_action_frame": self._model_action_frame,
            "legacy_pick_cube_tcp": self._legacy_pick_cube_tcp,
            "tcp_start_pose": self._tcp_start_pose,
        }
        pred = _episode_poses_to_runtime(model_pred, **transform_kwargs)
        states = _episode_poses_to_runtime(model_states, **transform_kwargs)
        gt = _episode_poses_to_runtime(model_gt, **transform_kwargs)
        pos_err, rot_err, grip_err = self._compute_metrics(pred, gt)

        print("-" * 60)
        print("评测误差 (predicted vs ground-truth 目标位姿):")
        print(f"  位置:   mean={pos_err.mean() * 1000:.1f}mm  max={pos_err.max() * 1000:.1f}mm")
        print(f"  朝向:   mean={np.rad2deg(rot_err.mean()):.2f}deg  max={np.rad2deg(rot_err.max()):.2f}deg")
        print(f"  夹爪:   mean={grip_err.mean():.2f}mm  max={grip_err.max():.2f}mm")
        print("-" * 60)

        if self._gif_path:
            self._export_gif(pred, states, gt, pos_err, rot_err, task, self._gif_path)
        if self._show:
            self._show_interactive(pred, states, gt, pos_err, rot_err, task)

    # ======================== 策略服务器连接 ========================

    def _connect_policy(self):
        print(f"[Policy] 正在连接 ws://{self._host}:{self._port} ...")
        self._policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=self._host,
            port=self._port,
        )
        meta = self._policy_client.get_server_metadata()
        print(f"[Policy] 已连接. 服务器元数据: {meta}")

    # ======================== 数据访问 ========================

    def _resolve_task(self, ep: int) -> str:
        if self._prompt:
            return self._prompt
        episodes_meta = self._meta.get("episodes", [])
        if ep < len(episodes_meta):
            tasks = episodes_meta[ep].get("tasks", [])
            if tasks:
                return ", ".join(tasks)
        tasks_meta = self._meta.get("tasks", [])
        if tasks_meta:
            return tasks_meta[0].get("task", "")
        return ""

    def _build_observation(self, state: np.ndarray, base_img: np.ndarray, wrist_img: np.ndarray) -> dict:
        return {
            "observation/state": state.astype(np.float32),
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "prompt": self._prompt or "",
        }

    # ======================== Rollout ========================

    def _rollout(self, df: "pd.DataFrame") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """回放数据集并逐帧查询策略，返回 (predicted, states, gt)。

        - states: (n,7) 数据集当前位姿 (真值输入 state)。
        - gt:     (n,7) 数据集目标位姿 (actions，即下一帧真值位姿)。
        - predicted: (n,7) 模型预测的绝对目标位姿。
        """
        n = len(df)
        states = np.stack(df["state"].values).astype(np.float64)
        gt = np.stack(df["actions"].values).astype(np.float64)
        if self._model_action_frame == "chunk_relative":
            if states.shape != (n, 1) or gt.shape != (n, 7):
                raise ValueError("chunk_relative dataset needs state (1,) and absolute action (7,)")
            states = gt.copy()  # Same-frame TCP for offline visualization only.
            gt = np.concatenate([gt[1:], gt[-1:]], axis=0)
        elif self._model_action_frame in {"pico_absolute", "tcp_absolute"}:
            gt = np.concatenate([gt[1:], gt[-1:]], axis=0)

        if self._init_pose is not None:
            pose = self._init_pose.copy()
            if pose.shape[0] == 6:
                pose = np.concatenate([pose, [states[0, 6]]])
            print(f"[Eval] 使用人工模型相对初始 state: {pose}")
        else:
            pose = states[0].copy()

        predicted: list[np.ndarray] = []
        cache: Optional[np.ndarray] = None
        cache_step = 0

        for t in range(n):
            if not self._closed_loop:
                pose = states[t].copy()

            horizon = self._exec_horizon if cache is None else min(self._exec_horizon, cache.shape[0])
            if cache is None or cache_step >= horizon:
                base = _resize_pad_rgb(_decode_image(df.iloc[t], "image"))
                wrist = _resize_pad_rgb(_decode_image(df.iloc[t], "wrist_image"))
                try:
                    policy_state = pose[6:7] if self._model_action_frame == "chunk_relative" else pose
                    result = self._policy_client.infer(self._build_observation(policy_state, base, wrist))
                except Exception as e:
                    print(f"[Policy] 查询失败 (frame {t}): {e}，重连...")
                    self._connect_policy()
                    result = self._policy_client.infer(self._build_observation(policy_state, base, wrist))
                cache = np.asarray(result["actions"], dtype=np.float64)  # (H,7)
                cache = _execution_actions(cache, self._model_action_frame)
                if self._model_action_frame == "chunk_relative":
                    cache = cache.copy()
                    for action in cache:
                        action[:6] = _compose_pose6(pose[:6], action[:6])
                cache_step = 0
                if t % 50 == 0 or t == n - 1:
                    print(f"  [Policy] frame {t}: 查询完成 (chunk={cache.shape[0]} 步)")

            action = cache[min(cache_step, cache.shape[0] - 1)]  # (7,) 绝对位姿
            predicted.append(action)
            if self._closed_loop:
                pose = action.copy()  # 预测位姿反馈作为下一帧 state
            cache_step += 1

        return np.asarray(predicted, dtype=np.float64), states, gt

    @staticmethod
    def _compute_metrics(pred: np.ndarray, gt: np.ndarray):
        pos_err = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)
        rot_err = np.array(
            [
                (Rotation.from_rotvec(pred[i, 3:6]) * Rotation.from_rotvec(gt[i, 3:6]).inv()).magnitude()
                for i in range(pred.shape[0])
            ]
        )
        grip_err = np.abs(pred[:, 6] - gt[:, 6])
        return pos_err, rot_err, grip_err

    # ======================== 可视化 ========================

    @staticmethod
    def _init_figure():
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 8), dpi=120)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.32, wspace=0.22)
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax_grip = fig.add_subplot(gs[0, 1])
        ax_perr = fig.add_subplot(gs[1, 0])
        ax_rerr = fig.add_subplot(gs[1, 1])
        return fig, ax3d, ax_grip, ax_perr, ax_rerr

    @staticmethod
    def _draw_orientation(ax, pos: np.ndarray, rotvec: np.ndarray, scale: float):
        """在当前 EEF 位姿处绘制 x/y/z 朝向 triad (红/绿/蓝)。"""
        r = Rotation.from_rotvec(rotvec).as_matrix()
        for i, c in enumerate(["r", "g", "b"]):
            axis = r[:, i] * scale
            ax.plot(
                [pos[0], pos[0] + axis[0]],
                [pos[1], pos[1] + axis[1]],
                [pos[2], pos[2] + axis[2]],
                color=c,
                linewidth=2.0,
            )

    def _draw_frame(self, ax3d, ax_grip, ax_perr, ax_rerr, pred, states, gt, pos_err, rot_err, idx):
        n = pred.shape[0]
        for ax in (ax3d, ax_grip, ax_perr, ax_rerr):
            ax.clear()

        # --- 3D 位置轨迹 (predicted vs gt) ---
        ax3d.plot(
            pred[: idx + 1, 0],
            pred[: idx + 1, 1],
            pred[: idx + 1, 2],
            color="#1f77b4",
            linestyle="--",
            linewidth=1.0,
            label="predicted",
        )
        ax3d.plot(
            gt[: idx + 1, 0], gt[: idx + 1, 1], gt[: idx + 1, 2], color="#d62728", linewidth=1.0, label="ground truth"
        )
        ax3d.scatter(*pred[idx, :3], color="#1f77b4", s=40)
        ax3d.scatter(*gt[idx, :3], color="#d62728", s=40)
        ax3d.scatter(*states[0, :3], color="k", marker="o", s=60, label="start")
        self._draw_orientation(ax3d, pred[idx, :3], pred[idx, 3:6], scale=self._orientation_scale(pred, gt))
        ax3d.set_xlabel("x (m)")
        ax3d.set_ylabel("y (m)")
        ax3d.set_zlabel("z (m)")
        ax3d.set_title(f"EEF position (frame {idx}/{n - 1})", fontsize=10)
        ax3d.legend(loc="upper right", fontsize=7)
        self._set_3d_bounds(ax3d, pred, gt)

        t = np.arange(n)
        # --- 夹爪 ---
        ax_grip.plot(t[: idx + 1], pred[: idx + 1, 6], color="#1f77b4", linewidth=1.2, label="predicted")
        ax_grip.plot(t[: idx + 1], gt[: idx + 1, 6], color="#d62728", linewidth=1.2, label="ground truth")
        ax_grip.set_title("Gripper width (mm)")
        ax_grip.set_xlabel("frame")
        ax_grip.set_ylabel("mm")
        ax_grip.set_xlim(0, n - 1)
        ax_grip.legend(loc="upper right", fontsize=7)

        # --- 位置误差 ---
        ax_perr.plot(t[: idx + 1], pos_err[: idx + 1] * 1000.0, color="#1f77b4", linewidth=1.2)
        ax_perr.set_title("Position error (mm)")
        ax_perr.set_xlabel("frame")
        ax_perr.set_ylabel("mm")
        ax_perr.set_xlim(0, n - 1)
        ax_perr.axhline(pos_err.mean() * 1000.0, color="gray", linestyle=":", linewidth=1.0)

        # --- 朝向误差 ---
        ax_rerr.plot(t[: idx + 1], np.rad2deg(rot_err[: idx + 1]), color="#1f77b4", linewidth=1.2)
        ax_rerr.set_title("Rotation error (deg)")
        ax_rerr.set_xlabel("frame")
        ax_rerr.set_ylabel("deg")
        ax_rerr.set_xlim(0, n - 1)
        ax_rerr.axhline(np.rad2deg(rot_err.mean()), color="gray", linestyle=":", linewidth=1.0)

    @staticmethod
    def _orientation_scale(pred: np.ndarray, gt: np.ndarray) -> float:
        """朝向 triad 的显示长度，取轨迹跨度的一定比例。"""
        span = np.concatenate([pred[:, :3], gt[:, :3]], axis=0).ptp(axis=0).max()
        return max(0.02, 0.08 * span)

    @staticmethod
    def _set_3d_bounds(ax3d, pred: np.ndarray, gt: np.ndarray):
        pts = np.concatenate([pred[:, :3], gt[:, :3]], axis=0)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        c = (lo + hi) / 2
        r = max((hi - lo).max() / 2, 0.01)
        ax3d.set_xlim(c[0] - r, c[0] + r)
        ax3d.set_ylim(c[1] - r, c[1] + r)
        ax3d.set_zlim(c[2] - r, c[2] + r)

    def _export_gif(self, pred, states, gt, pos_err, rot_err, task, out_path: str):
        if not _HAS_PIL:
            print("[ERROR] 需要 Pillow: uv pip install pillow")
            return

        import matplotlib.pyplot as plt

        fig, ax3d, ax_grip, ax_perr, ax_rerr = self._init_figure()
        fig.suptitle(f"EEF pose rollout — {task or '(无)'}", fontsize=12, fontweight="bold")

        n = pred.shape[0]
        imgs = []
        for idx in range(n):
            self._draw_frame(ax3d, ax_grip, ax_perr, ax_rerr, pred, states, gt, pos_err, rot_err, idx)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            h, w = fig.canvas.get_width_height()[::-1]
            data = data.reshape((h, w, 4))
            imgs.append(data[..., :3].copy())  # .copy() 防止 canvas buffer 被复写
            if idx % 50 == 0:
                print(f"  GIF frame {idx}/{n}")

        print(f"[Eval] 导出 GIF → {out_path} ({len(imgs)} 帧)")
        pil_frames = [_PILImage.fromarray(img) for img in imgs]
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / self._gif_fps),
            loop=0,
        )
        plt.close("all")
        print(f"[Eval] 完成: {out_path}")

    def _show_interactive(self, pred, states, gt, pos_err, rot_err, task):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        fig, ax3d, ax_grip, ax_perr, ax_rerr = self._init_figure()
        fig.suptitle(f"EEF pose rollout — {task or '(无)'}", fontsize=12, fontweight="bold")
        n = pred.shape[0]

        ax_slider = fig.add_axes([0.15, 0.02, 0.7, 0.03])
        slider = Slider(ax_slider, "frame", 0, n - 1, valinit=0, valfmt="%d")

        def update(idx):
            idx = int(idx)
            self._draw_frame(ax3d, ax_grip, ax_perr, ax_rerr, pred, states, gt, pos_err, rot_err, idx)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        update(0)
        plt.show()


# ===========================================================================
# FPS 计数器
# ===========================================================================


class _FPSCounter:
    def __init__(self, label: str = ""):
        self.label = label
        self.count = 0
        self._last = time.monotonic()
        self.fps = 0.0

    def tick(self):
        self.count += 1
        now = time.monotonic()
        if now - self._last >= 1.0:
            self.fps = self.count / (now - self._last)
            self.count = 0
            self._last = now


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args(parser=None, argv=None):
    import argparse

    p = parser or argparse.ArgumentParser(description="Piper EEF 策略服务器推理客户端")
    p.add_argument("--host", default="localhost", help="策略服务器地址 (默认: localhost)")
    p.add_argument("--port", type=int, default=6006, help="策略服务器端口 (默认: 6006)")
    p.add_argument("--can_name", default="can0", help="CAN 端口名称 (默认: can0)")
    p.add_argument("--rs2_base", default=None, help="D435i 基座相机序列号")
    p.add_argument("--usb_wrist", type=int, default=None, help="USB 腕部相机 OpenCV 设备 ID (默认: None)")
    p.add_argument(
        "--wrist_exposure",
        type=float,
        default=None,
        help="USB 腕部相机手动曝光 (ms)。默认 None=自动曝光；如 15.0 表示固定 15ms 曝光",
    )
    p.add_argument("--cam_ids", type=int, nargs="*", default=[], help="OpenCV 设备 ID 回退。第一个=基座，第二个=腕部")
    p.add_argument(
        "--action_horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help=f"动作块长度 (默认: {DEFAULT_ACTION_HORIZON})",
    )
    p.add_argument(
        "--exec_horizon",
        type=int,
        default=DEFAULT_EXEC_HORIZON,
        help=(
            f"每次执行步数再重新推理 (默认: {DEFAULT_EXEC_HORIZON})"
            if parser is None
            else f"每个异步动作块最多保留的步数 (默认: {DEFAULT_EXEC_HORIZON})"
        ),
    )
    p.add_argument(
        "--prompt",
        default="<control mode> end effector <control mode>pick up the blue cube and place it on the yellow cube.",
        help="语言指令。评测模式默认读数据集 tasks.jsonl；推理模式默认用内置指令。",
    )
    p.add_argument("--interactive", action="store_true", help="交互模式: 每次推理前手动输入指令")
    p.add_argument(
        "--model_action_frame",
        choices=("robot_absolute", "chunk_relative", "tcp", "pico", "pico_absolute", "tcp_absolute"),
        default="robot_absolute",
        help=(
            "模型 state/actions 的坐标系（默认: Robot Base 下的绝对 TCP）。"
            "每个动作块首帧为基准、state 只有夹爪时选 chunk_relative；"
            "使用 data/tcp_relative_action 训练时保持 tcp；"
            "PICO-relative 标签选 pico；绝对 PICO 控制器位姿选 pico_absolute；"
            "data/tcp_action 导出的绝对 TCP 位姿选 tcp_absolute"
        ),
    )
    p.add_argument(
        "--absolute_pose_dataset", "--raw_pico_dataset",
        dest="raw_pico_dataset",
        metavar="DATASET",
        default=None,
        help="真机绝对位姿模式的训练数据集路径；读取首帧位姿和夹爪范围",
    )
    p.add_argument(
        "--reference_episode", "--raw_pico_episode", dest="raw_pico_episode",
        type=int, default=0, metavar="EPISODE",
        help="真机绝对位姿模式的参考 episode 编号 (默认: 0)",
    )
    p.add_argument(
        "--legacy_pick_cube_tcp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "兼容由旧 pico_arm_transform.py 转换并训练的 pick_cube 模型："
            "在当前机器人 TCP 与旧模型 TCP 之间做完整 SE(3) 共轭变换"
        ),
    )
    p.add_argument(
        "--trajectory_log",
        default=None,
        help=(
            "仅真机推理：保存每步模型目标、Base 目标和机械臂实际 TCP 到 .npz；"
            "为避免误覆盖，目标文件必须不存在"
        ),
    )
    p.add_argument(
        "--max_pos_speed",
        type=float,
        default=DEFAULT_MAX_POS_SPEED,
        help=f"最大末端线速度 m/s (默认: {DEFAULT_MAX_POS_SPEED})",
    )
    p.add_argument(
        "--max_rot_speed",
        type=float,
        default=DEFAULT_MAX_ROT_SPEED,
        help=f"最大末端角速度 rad/s (默认: {DEFAULT_MAX_ROT_SPEED})",
    )
    p.add_argument(
        "--interp_freq",
        type=float,
        default=DEFAULT_INTERP_FREQ,
        help=f"插值频率 Hz (默认: {DEFAULT_INTERP_FREQ}, 0/None=不插值)",
    )
    p.add_argument(
        "--gripper_port", default=GRIPPER_PORT, help=f"DM 夹爪 USB2CAN 串口 (默认: {GRIPPER_PORT}；传空串回退原生夹爪)"
    )
    p.add_argument(
        "--gripper_close_rad",
        type=float,
        default=GRIPPER_CLOSE_RAD,
        help=f"DM 夹爪全闭位置 rad (默认: {GRIPPER_CLOSE_RAD})",
    )
    p.add_argument(
        "--gripper_open_width_mm",
        type=float,
        default=GRIPPER_OPEN_WIDTH_MM,
        help=f"夹爪全开命令 mm (默认: {GRIPPER_OPEN_WIDTH_MM}；0 对应全闭)",
    )
    p.add_argument(
        "--gripper_binary_threshold_mm",
        type=float,
        default=None,
        help=(
            "夹爪二值阈值：策略输出大于该值时全开，否则全闭。"
            f"默认: {GRIPPER_BINARY_THRESHOLD_MM}"
        ),
    )
    p.add_argument(
        "--gripper_prediction_scale",
        type=float,
        default=None,
        help=f"策略夹爪预测值的额外缩放；绝对位姿模式默认 1，其他模式默认 {GRIPPER_PREDICTION_SCALE}",
    )
    p.add_argument(
        "--binary_gripper",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用夹爪二值控制；chunk_relative 默认连续，其他模式默认二值",
    )
    p.add_argument(
        "--gripper_current",
        type=float,
        default=GRIPPER_CURRENT_LIMIT,
        help=f"DM 夹爪力矩电流上限 i_des 0..1.0 (默认: {GRIPPER_CURRENT_LIMIT})",
    )
    p.add_argument(
        "--gripper_speed",
        type=float,
        default=GRIPPER_SPEED_RAD_S,
        help=f"DM 夹爪移动速度 rad/s (默认: {GRIPPER_SPEED_RAD_S})",
    )

    # ---- 数据集离线评测 (可视化末端位姿 + GIF) ----
    p.add_argument(
        "--dataset",
        default=None,
        help="LeRobot EEF 数据集目录 (如 ./pick_place)。设置后进入离线评测模式，不连接真实机械臂/相机。",
    )
    p.add_argument("--episode", type=int, default=None, help="数据集 episode 索引 (默认: 第一个)")
    p.add_argument("--gif", default=None, help="导出评测可视化动图为 GIF (如 eval.gif)")
    p.add_argument(
        "--init_pose",
        type=float,
        nargs="+",
        default=None,
        metavar="P",
        help=(
            "人工设置模型初始 state [x y z rx ry rz (gripper)]；坐标系与训练数据一致。 "
            "(6 或 7 个值；仅闭环 rollout 生效)。这不是 Robot Base 绝对位姿"
        ),
    )
    p.add_argument(
        "--tcp_start_pose",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help=(
            "仅离线评测：Robot Base 下的 episode 初始 TCP "
            "[x y z(m) rx ry rz(rotvec rad)]；输出用 T_start @ D(t) 显示/评估"
        ),
    )
    p.add_argument("--open_loop", action="store_true", help="开环评测 (每步用数据集真值位姿作为 state，默认闭环)")
    p.add_argument("--max_frames", type=int, default=0, help="最多评测多少帧 (默认 0=全部)")
    p.add_argument("--gif_fps", type=int, default=15, help=f"GIF 帧率 (默认: 15)")
    p.add_argument("--no_show", action="store_true", help="不弹出交互可视化窗口 (配合 --gif 无头导出)")
    return p.parse_args(argv)


def main(args=None):
    if args is None:
        args = _parse_args()

    if args.model_action_frame in {"chunk_relative", "pico", "pico_absolute", "tcp_absolute"} and args.legacy_pick_cube_tcp:
        print("[ERROR] PICO 模式与 --legacy_pick_cube_tcp 不能同时启用")
        return

    if (
        args.dataset is None
        and args.model_action_frame in {"pico_absolute", "tcp_absolute"}
        and not args.raw_pico_dataset
    ):
        print("[ERROR] 真机绝对位姿模式需要 --absolute_pose_dataset")
        return
    if args.dataset and args.raw_pico_dataset:
        print("[ERROR] 离线评测用 --dataset 即可，不需要 --absolute_pose_dataset")
        return

    # 数据集离线评测模式 (不连接真实机械臂 / 相机)
    if args.dataset:
        if args.trajectory_log is not None:
            print("[ERROR] --trajectory_log 仅用于真机推理，离线模式请使用 --gif")
            return
        init_pose = None
        if args.init_pose:
            init_pose = np.asarray(args.init_pose, dtype=np.float64)
            if init_pose.shape[0] not in (6, 7):
                print(f"[ERROR] --init_pose 需要 6 或 7 个值，收到 {init_pose.shape[0]} 个")
                return
        evaluator = PiperEEFDatasetEval(
            data_dir=args.dataset,
            host=args.host,
            port=args.port,
            episode=args.episode,
            init_pose=init_pose,
            tcp_start_pose=args.tcp_start_pose,
            model_action_frame=args.model_action_frame,
            legacy_pick_cube_tcp=args.legacy_pick_cube_tcp,
            action_horizon=args.action_horizon,
            exec_horizon=args.exec_horizon,
            closed_loop=not args.open_loop,
            max_frames=args.max_frames,
            prompt=args.prompt,
            gif_path=args.gif,
            gif_fps=args.gif_fps,
            show=not args.no_show,
        )
        evaluator.run()
        return

    if args.tcp_start_pose is not None:
        print("[ERROR] --tcp_start_pose 仅用于带 --dataset 的离线评测模式")
        return

    pico_reference_pose = None
    model_gripper_range = None
    if args.model_action_frame in {"pico_absolute", "tcp_absolute"}:
        pico_reference_pose, model_gripper_range = _load_raw_pico_reference(
            args.raw_pico_dataset, args.raw_pico_episode,
            pose_frame="pico_world_tcp" if args.model_action_frame == "tcp_absolute" else "pico_teleop",
        )
        print(f"[Pose] 参考 episode {args.raw_pico_episode} 首帧位姿: {pico_reference_pose}")
        print(f"[Pose] 训练夹爪范围: {model_gripper_range} -> 机器人 0..{args.gripper_open_width_mm:g} mm")

    inference = PiperEEFInference(
        host=args.host,
        port=args.port,
        can_name=args.can_name,
        rs2_base_serial=args.rs2_base,
        cv_base_id=args.cam_ids[0] if len(args.cam_ids) > 0 else None,
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
            else GRIPPER_PREDICTION_SCALE
        ),
        binary_gripper=(
            args.binary_gripper if args.binary_gripper is not None
            else args.model_action_frame != "chunk_relative"
        ),
        gripper_current=args.gripper_current,
        gripper_speed=args.gripper_speed,
        default_prompt=args.prompt or DEFAULT_PROMPT,
        interactive=args.interactive,
        model_action_frame=args.model_action_frame,
        legacy_pick_cube_tcp=args.legacy_pick_cube_tcp,
        pico_reference_pose=pico_reference_pose,
        model_gripper_range=model_gripper_range,
        trajectory_log=args.trajectory_log,
    )
    inference.run()


if __name__ == "__main__":
    main()
