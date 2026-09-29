#!/usr/bin/env python3
"""
Piper 末端位姿 (EEF) 数据集回放脚本 — 增量模式 (delta pose)。

从 LeRobot EEF 数据集 (如 pick_place) 读取 state/action，以**增量位姿**方式回放到
Piper 机械臂上。增量语义与 ``src/openpi/training/config.py`` 中
``LeRobotPiperEEFDataConfig`` 的 ``use_delta_pose_actions=True`` 一致
(即 ``transforms.DeltaEEFPoseActions`` / ``AbsoluteEEFPoseActions``):

    delta = action ⊖ state
        位姿:  delta = inv(T_state) @ T_action                   (完整 SE(3) 相对变换)
               平移和旋转均表达在当前 TCP 坐标系中
        夹爪:  保持绝对 (mm)

回放时每帧把 delta 复合到「当前末端位姿」上:

    target = delta ⊕ cur_pose
        位姿:  T_target = T_cur @ T_delta
        夹爪:  透传 delta 中的绝对宽度

增量回放的意义: 轨迹由**相对运动**定义，与绝对坐标系无关。机械臂可从任意起始位姿
(默认读取其当前末端位姿) 复现记录轨迹的相对运动，对采集/回放两次标定之间的绝对
偏移鲁棒。对比 ``--absolute`` 模式则是直接下发数据集中的绝对末端位姿。

与 ``examples/piper/data_tools/replay_dataset.py`` (关节空间) 的区别:
    - state/action 为 7 维 EEF 位姿 [x, y, z(m), rotvec(rad), gripper_width(mm)]。
    - 用 ``EndPoseCtrl`` 末端位姿模式控制 (``MotionCtrl_2 0x00``)。
    - 复用 ``inference_eef.py`` 的 PiperEEFController / DmGripperController。

用法:
    # 增量模式回放 (默认，从机械臂当前末端位姿出发；未给 --episode 时启动交互选择)
    python examples/piper/data_tools/replay_dataset_eef.py --data_dir ./pick_place
    python examples/piper/data_tools/replay_dataset_eef.py --data_dir ./pick_place --episode 2
    python examples/piper/data_tools/replay_dataset_eef.py --data_dir ./pick_place --speed 0.5 --loop

    # 不控制夹爪 (仅末端位姿，夹爪保持不动/手动控制)
    python examples/piper/data_tools/replay_dataset_eef.py --data_dir ./pick_place --no_gripper

    # 绝对模式对比 (直接下发数据集绝对位姿，需 --no_align 时不做坐标系对齐)
    python examples/piper/data_tools/replay_dataset_eef.py --data_dir ./pick_place --absolute

    # 仅查看数据集信息 (不回放，不连接机械臂)
    python examples/piper/data_tools/replay_dataset_eef.py --data_dir ./pick_place --info_only

前置条件:
    1. CAN 模块已激活:  bash can_activate.sh can0 1000000
    2. 机械臂已上电、处于从臂模式
    3. piper_sdk 已安装:  cd piper_sdk && pip install .
    4. scipy 已安装 (用于 rotvec ↔ SO(3) 组合)
    5. (达妙夹爪，除非 --no_gripper) USB2CAN 已插好 (/dev/ttyACM1)、DMTool 已关闭、
       电机 24V 已上电；否则传 --gripper_port "" 回退到原生 GripperCtrl。

键盘控制:
    Enter   — 开始回放当前 episode
    Space   — 暂停/继续
    r       — 重置到当前 episode 开头 (增量模式下重新读取当前末端位姿作为起点)
    n       — 下一个 episode
    p       — 上一个 episode
    +/-     — 加速/减速 (0.25x ~ 4x)
    q       — 退出（机械臂保持使能）
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "./pick_place"

# 增量位姿的维度划分 (与 LeRobotPiperEEFDataConfig 的 position_dims/rotation_dims 一致)
POSITION_DIMS = 3
ROTATION_DIMS = 3

# 回放速度百分比 (机器人末端位姿模式的速度档，较低速保安全)
DEFAULT_REPLAY_SPEED_PCT = 30

# 达妙 DM-J4310-2EC 夹爪 (力位模式, USB2CAN)。与 inference_eef.py 一致，但端口
# 默认取实际设备 /dev/ttyACM1 (inference_eef 常量里写的是 ttyACM0，见 piper-dm-gripper 备忘)。
DEFAULT_GRIPPER_PORT = "/dev/ttyACM1"
DEFAULT_GRIPPER_CLOSE_RAD = 10.0  # 全闭位置 (rad)；与物理满程一致，可覆盖
DEFAULT_GRIPPER_OPEN_WIDTH_MM = 23.0  # 全开宽度 (mm)，模型 gripper_width 最大值
DEFAULT_GRIPPER_CURRENT_LIMIT = 0.3  # 力矩电流上限 (i_des, 0..1.0)
DEFAULT_GRIPPER_SPEED_RAD_S = 2.0  # 移动速度上限 (rad/s)


# ===========================================================================
# 增量位姿 (delta pose)
# ===========================================================================


def delta_pose(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """``action ⊖ state`` → 增量位姿。

    与 ``src/openpi/transforms.py`` 的 ``DeltaEEFPoseActions`` 一致，计算
    完整右相对变换 ``inv(T_state) @ T_action``：平移和旋转均表达在当前 TCP
    坐标系中。
      - 夹爪 (第 7 维): 保持绝对。
    """
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    out = action.copy()
    r0, r1 = POSITION_DIMS, POSITION_DIMS + ROTATION_DIMS
    r_state = Rotation.from_rotvec(state[r0:r1])
    r_action = Rotation.from_rotvec(action[r0:r1])
    out[:POSITION_DIMS] = r_state.inv().apply(action[:POSITION_DIMS] - state[:POSITION_DIMS])
    out[r0:r1] = (r_state.inv() * r_action).as_rotvec()

    return out


def absolute_pose(state: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """``delta ⊕ state`` → 绝对位姿 (增量转绝对)。

    与 ``src/openpi/transforms.py`` 的 ``AbsoluteEEFPoseActions`` 一致，计算
    ``T_target = T_state @ T_delta``。
      - 夹爪: 透传 delta 中的绝对宽度。
    """
    state = np.asarray(state, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    out = delta.copy()
    r0, r1 = POSITION_DIMS, POSITION_DIMS + ROTATION_DIMS
    r_state = Rotation.from_rotvec(state[r0:r1])
    r_delta = Rotation.from_rotvec(delta[r0:r1])
    out[:POSITION_DIMS] = state[:POSITION_DIMS] + r_state.apply(delta[:POSITION_DIMS])
    out[r0:r1] = (r_state * r_delta).as_rotvec()

    return out


# ===========================================================================
# 数据集加载
# ===========================================================================


def load_eef_dataset(data_dir: str) -> tuple[dict, dict]:
    """加载 LeRobot EEF 数据集的元信息与按 episode 分组的 DataFrame。

    Returns:
        (meta, frames_per_episode)
        meta: 元信息 dict (info, tasks, episodes, episodes_stats)
        frames_per_episode: {episode_index: DataFrame}
    """
    if not HAS_PANDAS:
        raise ImportError("需要 pandas: uv pip install pandas")

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    meta: dict = {}
    for name in ["info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]:
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


def get_dataset_fps(meta: dict) -> float:
    return float(meta.get("info", {}).get("fps", 20))


def get_episode_task(meta: dict, ep: int) -> str:
    tasks = meta.get("tasks", [])
    if tasks:
        return tasks[0].get("task", "(无)")
    return "(无)"


# ===========================================================================
# 回放引擎
# ===========================================================================


class EEFDatasetReplayer:
    """从 EEF 数据集读取 state/action，以增量 (delta) 方式回放到机械臂。"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        can_name: str = "can0",
        speed: float = 1.0,
        loop: bool = False,
        absolute: bool = False,
        readback: bool = False,
        align: bool = True,
        control_gripper: bool = True,
        speed_pct: int = DEFAULT_REPLAY_SPEED_PCT,
        gripper_port: Optional[str] = DEFAULT_GRIPPER_PORT,
        gripper_close_rad: float = DEFAULT_GRIPPER_CLOSE_RAD,
        gripper_open_width_mm: float = DEFAULT_GRIPPER_OPEN_WIDTH_MM,
        gripper_current: float = DEFAULT_GRIPPER_CURRENT_LIMIT,
        gripper_speed: float = DEFAULT_GRIPPER_SPEED_RAD_S,
    ):
        # 延迟导入: 复用 inference_eef.py 的 EEF 控制器与夹爪驱动，--info_only 不依赖 piper_sdk。
        from examples.piper.runtime import inference_eef as eef

        self._data_dir = data_dir
        self._speed = speed
        self._loop = loop
        self._absolute = absolute
        self._readback = readback
        self._align = align
        self._control_gripper = control_gripper
        self._speed_pct = speed_pct

        # 数据
        self._meta, self._frames = load_eef_dataset(data_dir)
        self._fps = get_dataset_fps(self._meta)
        self._episode_indices = sorted(self._frames.keys())
        if not self._episode_indices:
            raise ValueError("数据集中没有 episode")

        # 达妙夹爪 (可选，--no_gripper 时不控制夹爪)
        gripper = None
        if control_gripper and eef._HAS_DM_GRIPPER and gripper_port:
            print(f"[Gripper] 使用达妙 DM-J4310-2EC 夹爪 ({gripper_port})")
            gripper = eef.DmGripperController(
                port=gripper_port,
                close_rad=gripper_close_rad,
                open_width_mm=gripper_open_width_mm,
                current_limit=gripper_current,
                speed_rad_s=gripper_speed,
            )
        elif control_gripper and eef._HAS_DM_GRIPPER:
            print("[Gripper] 未指定夹爪串口，回退原生夹爪")
        elif control_gripper:
            print("[Gripper] 未加载达妙夹爪驱动，回退原生夹爪")
        else:
            print("[Gripper] 不控制夹爪 (--no_gripper)")

        # 机械臂 + 坐标系对齐（仅回放脚本的绝对模式可选使用）
        self._robot = eef.PiperEEFController(can_name, gripper=gripper, control_gripper=control_gripper)
        self._pico_to_arm = None
        if self._absolute and self._align:
            from examples.piper.transforms.pico_arm_transform import PicoToArmConverter

            self._pico_to_arm = PicoToArmConverter(
                pico_reference_position=np.zeros(3, dtype=np.float64),
                arm_reference_position=np.zeros(3, dtype=np.float64),
            )

        # 状态
        self._current_ep = self._episode_indices[0]
        self._playing = False
        self._running = True
        self._frame_idx = 0
        self._n = 0
        self._cur_pose: Optional[np.ndarray] = None  # 增量模式跟踪的当前末端位姿 (6,)
        self._states = None
        self._actions = None
        self._deltas = None
        self._precompute(self._current_ep)

    # ---- 数据预处理 ----

    def _precompute(self, ep: int):
        """预计算当前 episode 的 state/action/delta 序列。"""
        df = self._frames[ep]
        self._n = len(df)
        self._states = np.stack(df["state"].values).astype(np.float64)  # (n, 7)
        self._actions = np.stack(df["actions"].values).astype(np.float64)  # (n, 7)
        self._deltas = np.stack([delta_pose(self._states[t], self._actions[t]) for t in range(self._n)])  # (n, 7)
        self._frame_idx = 0
        self._cur_pose = None

    # ---- 运行入口 ----

    def run(self):
        mode = (
            "绝对 (absolute)"
            if self._absolute
            else ("增量 (delta, 读回反馈)" if self._readback else "增量 (delta, 跟踪下发)")
        )
        print("=" * 60)
        print("Piper EEF 数据集回放")
        print(f"数据集: {self._data_dir}")
        print(f"Episodes: {len(self._episode_indices)} 个")
        print(f"FPS:      {self._fps}")
        print(f"模式:     {mode}")
        print("=" * 60)

        if not self._robot.enable(speed_pct=self._speed_pct):
            print("[ERROR] 机械臂使能失败，退出。")
            return

        self._show_episode_info(self._current_ep)
        self._show_help()

        import threading

        input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        input_thread.start()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[INFO] 中断信号，退出中...")
        finally:
            print("[INFO] 已退出 (机械臂保持使能)")

    # ---- 主循环 ----

    def _main_loop(self):
        period = 1.0 / self._fps / self._speed

        while self._running:
            if not self._playing:
                time.sleep(0.05)
                continue

            loop_start = time.monotonic()

            if self._frame_idx >= self._n:
                self._on_episode_end()
                if not self._playing:
                    continue

            try:
                self._step()
            except Exception as e:
                print(f"\n[ERROR] 执行 action 失败: {e}")
                self._playing = False

            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    def _step(self):
        """发送一帧动作并推进帧序号。"""
        frame = self._frame_idx

        if self._absolute:
            target = self._actions[frame].copy()
            if self._pico_to_arm is not None:
                quat = Rotation.from_rotvec(target[3:6]).as_quat()
                arm_pos, arm_quat = self._pico_to_arm.convert_pose(target[:3], quat)
                target[:6] = np.concatenate([arm_pos, Rotation.from_quat(arm_quat).as_rotvec()])
        else:
            if self._readback:
                cur = self._robot.get_eef_pose()
            else:
                if self._cur_pose is None:
                    self._cur_pose = self._robot.get_eef_pose()
                cur = self._cur_pose
            target = absolute_pose(cur, self._deltas[frame])
            if not self._readback:
                self._cur_pose = target[:6].copy()

        self._robot.execute_action(target)

        if frame % 50 == 0 or frame == self._n - 1:
            p_str = ", ".join(f"{target[i]:.4f}" for i in range(3))
            r_str = ", ".join(f"{target[i]:.3f}" for i in range(3, 6))
            bar = self._progress_bar(frame + 1, self._n)
            print(
                f"\r[Ep{self._current_ep}] {bar} {frame + 1}/{self._n} | "
                f"pos=[{p_str}] rot=[{r_str}] grip={target[6]:.1f}mm | {self._speed:.2f}x",
                end="",
                flush=True,
            )

        self._frame_idx += 1

    def _on_episode_end(self):
        print(f"\n[Replay] Episode {self._current_ep} 回放完成 ({self._n} 帧)")

        if self._loop:
            self._precompute(self._current_ep)
            print(f"[Replay] 循环回放 Episode {self._current_ep}...")
            return

        self._playing = False

        current_pos = self._episode_indices.index(self._current_ep)
        if current_pos + 1 < len(self._episode_indices):
            self._current_ep = self._episode_indices[current_pos + 1]
            self._precompute(self._current_ep)
            self._show_episode_info(self._current_ep)
            print("按 Enter 继续回放")
        else:
            print("[Replay] 所有 episode 已播完。按 Enter 重新开始或按 q 退出。")

    # ---- 显示 ----

    def _show_episode_info(self, ep: int):
        task = get_episode_task(self._meta, ep)

        pos_delta = np.linalg.norm(self._deltas[:, :3], axis=1)
        rot_delta = np.array([Rotation.from_rotvec(self._deltas[t, 3:6]).magnitude() for t in range(self._n)])

        print(f"\n{'─' * 40}")
        print(f"  Episode:     {ep}")
        print(f"  任务指令:    {task}")
        print(f"  帧数:        {self._n}")
        print("  位置范围 (m):")
        for i, axis in enumerate("xyz"):
            print(f"    {axis}: {self._states[:, i].min():7.3f} ~ {self._states[:, i].max():7.3f}")
        print(f"  夹爪范围:     {self._states[:, 6].min():6.1f} ~ {self._states[:, 6].max():6.1f} mm")
        print("  增量位姿 (delta):")
        print(f"    位置: mean={pos_delta.mean() * 1000:.1f}mm  max={pos_delta.max() * 1000:.1f}mm")
        print(f"    旋转: mean={np.rad2deg(rot_delta.mean()):.2f}deg  max={np.rad2deg(rot_delta.max()):.2f}deg")
        print(f"  当前速度:     {self._speed:.2f}x (机器人速度档 {self._speed_pct}%)")
        print(f"{'─' * 40}")

    def _show_help(self):
        print("""
操作提示:
  [Enter]  开始/继续回放
  [Space]  暂停
  [r]      重置到当前 episode 开头
  [n]      下一个 episode
  [p]      上一个 episode
  [=/-]    加速/减速 (0.25x ~ 4x)
  [q]      退出
""")

    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 40) -> str:
        ratio = min(current / max(total, 1), 1.0)
        filled = int(width * ratio)
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    # ---- 键盘交互 ----

    def _keyboard_listener(self):
        while self._running:
            try:
                ch = sys.stdin.readline().strip().lower()
                self._handle_key(ch)
            except (EOFError, OSError):
                break

    def _handle_key(self, ch: str):
        if ch == "":  # Enter
            if not self._playing:
                if self._frame_idx >= self._n:
                    self._frame_idx = 0
                print(f"\n[Replay] 开始回放 Episode {self._current_ep}")
                self._playing = True
        elif ch == " ":
            if self._playing:
                print(f"\n[Replay] 暂停 (frame {self._frame_idx})")
                self._playing = False
            else:
                print(f"\n[Replay] 继续 (frame {self._frame_idx})")
                self._playing = True
        elif ch == "r":
            was_playing = self._playing
            self._playing = False
            self._frame_idx = 0
            self._cur_pose = None  # 增量模式重新读取当前末端位姿作为起点
            print(f"\n[Replay] 重置到 Episode {self._current_ep} 开头")
            if was_playing:
                print("按 Enter 重新开始")
        elif ch == "n":
            was_playing = self._playing
            self._playing = False
            current_pos = self._episode_indices.index(self._current_ep)
            if current_pos + 1 < len(self._episode_indices):
                self._current_ep = self._episode_indices[current_pos + 1]
                self._precompute(self._current_ep)
                self._show_episode_info(self._current_ep)
                print("按 Enter 开始回放")
            else:
                print(f"[Replay] 已是最后一个 episode ({self._current_ep})")
        elif ch == "p":
            was_playing = self._playing
            self._playing = False
            current_pos = self._episode_indices.index(self._current_ep)
            if current_pos > 0:
                self._current_ep = self._episode_indices[current_pos - 1]
                self._precompute(self._current_ep)
                self._show_episode_info(self._current_ep)
                print("按 Enter 开始回放")
            else:
                print(f"[Replay] 已是第一个 episode ({self._current_ep})")
        elif ch in ("=", "+"):
            self._speed = min(self._speed * 2, 4.0)
            print(f"[Replay] 速度: {self._speed:.2f}x")
        elif ch == "-":
            self._speed = max(self._speed / 2, 0.25)
            print(f"[Replay] 速度: {self._speed:.2f}x")
        elif ch == "q":
            print("[Replay] 退出...")
            self._running = False
            self._playing = False


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args():
    p = argparse.ArgumentParser(
        description="Piper 末端位姿 (EEF) 数据集回放 — 增量 (delta) 模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python replay_dataset_eef.py --data_dir ./pick_place
  python replay_dataset_eef.py --data_dir ./pick_place --episode 2
  python replay_dataset_eef.py --data_dir ./pick_place --speed 0.5 --loop
  python replay_dataset_eef.py --data_dir ./pick_place --absolute   # 绝对模式对比
        """,
    )
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help=f"数据集目录 (默认: {DEFAULT_DATA_DIR})")
    p.add_argument("--episode", type=int, default=None, help="起始 episode (默认: 启动时交互选择)")
    p.add_argument("--speed", type=float, default=1.0, help="回放速度倍率 (默认: 1.0)")
    p.add_argument("--loop", action="store_true", help="循环回放当前 episode")
    p.add_argument("--can_name", default="can0", help="CAN 端口名称 (默认: can0)")

    p.add_argument("--absolute", action="store_true", help="绝对模式: 直接下发数据集绝对末端位姿 (默认增量模式)")
    p.add_argument(
        "--readback", action="store_true", help="增量模式下每帧读回机械臂实际末端位姿作为基准 (默认跟踪已下发目标)"
    )
    p.add_argument("--no_align", action="store_true", help="绝对模式下不做 PICO→机械臂 坐标系对齐 (数据系→机械臂系)")
    p.add_argument("--no_gripper", action="store_true", help="不控制夹爪 (仅末端位姿，夹爪保持不动)")
    p.add_argument(
        "--speed_pct",
        type=int,
        default=DEFAULT_REPLAY_SPEED_PCT,
        help=f"机器人末端位姿速度档百分比 (默认: {DEFAULT_REPLAY_SPEED_PCT})",
    )

    p.add_argument(
        "--gripper_port",
        default=DEFAULT_GRIPPER_PORT,
        help=f"达妙夹爪 USB2CAN 串口 (默认: {DEFAULT_GRIPPER_PORT}；传空串回退原生夹爪)",
    )
    p.add_argument(
        "--gripper_close_rad",
        type=float,
        default=DEFAULT_GRIPPER_CLOSE_RAD,
        help=f"达妙夹爪全闭位置 rad (默认: {DEFAULT_GRIPPER_CLOSE_RAD})",
    )
    p.add_argument(
        "--gripper_open_width_mm",
        type=float,
        default=DEFAULT_GRIPPER_OPEN_WIDTH_MM,
        help=f"达妙夹爪全开宽度 mm (默认: {DEFAULT_GRIPPER_OPEN_WIDTH_MM})",
    )
    p.add_argument(
        "--gripper_current",
        type=float,
        default=DEFAULT_GRIPPER_CURRENT_LIMIT,
        help=f"达妙夹爪力矩电流上限 i_des 0..1.0 (默认: {DEFAULT_GRIPPER_CURRENT_LIMIT})",
    )
    p.add_argument(
        "--gripper_speed",
        type=float,
        default=DEFAULT_GRIPPER_SPEED_RAD_S,
        help=f"达妙夹爪移动速度 rad/s (默认: {DEFAULT_GRIPPER_SPEED_RAD_S})",
    )

    p.add_argument("--info_only", action="store_true", help="仅显示数据集信息，不回放")
    return p.parse_args()


def _show_info(args):
    meta, frames = load_eef_dataset(args.data_dir)
    print(f"数据集: {args.data_dir}")
    print(f"Episodes: {sorted(frames.keys())}")
    print(f"FPS:      {get_dataset_fps(meta)}")
    print(f"任务:     {get_episode_task(meta, 0)}")
    for ep in sorted(frames.keys()):
        df = frames[ep]
        states = np.stack(df["state"].values).astype(np.float64)
        print(f"\n  Episode {ep}: {len(df)} 帧")
        for i, axis in enumerate("xyz"):
            print(f"    {axis}: {states[:, i].min():.3f} ~ {states[:, i].max():.3f} m")
        print(f"    gripper: {states[:, 6].min():.1f} ~ {states[:, 6].max():.1f} mm")


def _select_episode(episode_indices: list[int], frames: dict) -> int:
    """交互式选择 episode。列出概要，读取用户输入；回车返回第一个 episode。"""
    first = episode_indices[0]
    print("\n可用 episodes:")
    for i in range(0, len(episode_indices), 8):
        chunk = episode_indices[i : i + 8]
        parts = [f"{ep:>2}:{len(frames[ep]):>4}帧" for ep in chunk]
        print("    " + "  ".join(parts))

    while True:
        try:
            s = input(f"选择 episode [{episode_indices[0]}-{episode_indices[-1]}] (回车={first}): ").strip()
            if s == "":
                return first
            ep = int(s)
            if ep in episode_indices:
                return ep
            print(f"  [WARNING] episode {ep} 不存在，请重新输入")
        except (EOFError, ValueError):
            return first


def main():
    args = _parse_args()

    if not HAS_PANDAS:
        print("[ERROR] 需要 pandas: uv pip install pandas")
        sys.exit(1)

    if args.info_only:
        _show_info(args)
        return

    replayer = EEFDatasetReplayer(
        data_dir=args.data_dir,
        can_name=args.can_name,
        speed=args.speed,
        loop=args.loop,
        absolute=args.absolute,
        readback=args.readback,
        align=not args.no_align,
        control_gripper=not args.no_gripper,
        speed_pct=args.speed_pct,
        gripper_port=args.gripper_port or None,
        gripper_close_rad=args.gripper_close_rad,
        gripper_open_width_mm=args.gripper_open_width_mm,
        gripper_current=args.gripper_current,
        gripper_speed=args.gripper_speed,
    )

    # episode 选择: 未显式指定 --episode 时交互选择
    if args.episode is None:
        args.episode = _select_episode(replayer._episode_indices, replayer._frames)

    if args.episode in replayer._episode_indices:
        replayer._current_ep = args.episode
        replayer._precompute(args.episode)
    else:
        print(f"[WARNING] Episode {args.episode} 不存在，使用第一个 episode ({replayer._episode_indices[0]})")

    replayer.run()


if __name__ == "__main__":
    main()
