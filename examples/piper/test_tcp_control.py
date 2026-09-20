#!/usr/bin/env python3
"""
以爪尖 (TCP) 为控制点的 Piper 末端位姿测试脚本。

用途: 验证 inference_eef.py 里的 ``GRIPPER_TCP_OFFSET_M`` (法兰→爪尖) 补偿是否正确。

原理:
    Piper SDK 的 ``EndPoseCtrl`` / ``GetArmEndPoseMsgs`` 收发的是**法兰 (J6)** 位姿，
    而本脚本对外统一用**爪尖 (TCP)** 位姿。二者相差一个固定偏移:
        读: 爪尖 = 法兰 + R @ offset
        写: 法兰 = 爪尖 - R @ offset
    本脚本复用 ``inference_eef.PiperEEFController`` 的 ``get_eef_pose`` /
    ``send_eef_command``，直接以爪尖为控制点做微调 (jog)，并同时打印法兰与爪尖位姿。

键盘:
    w / s       爪尖 X 轴 + / -
    a / d       爪尖 Y 轴 + / -
    q / e       爪尖 Z 轴 + / -
    i / k       爪尖 RX + / -   (绕世界 X 轴)
    j / l       爪尖 RY + / -   (绕世界 Y 轴)
    u / o       爪尖 RZ + / -   (绕世界 Z 轴)
    r           回到初始关节位姿
    h / Enter   打印帮助
    x           退出 (或 Ctrl+C)

用法:
    python examples/piper/test_tcp_control.py
    python examples/piper/test_tcp_control.py --offset 0 0 0.22 --step 0.005
    python examples/piper/test_tcp_control.py --offset 0 0 0     # 关闭补偿，对照差异

验证建议:
    1. 先让爪尖指一个固定参考点，记下爪尖位姿。
    2. 按 u/o 让爪尖绕 Z 轴转 ±90°，若 offset 正确，物理爪尖应**原地不动** (只转姿态)。
    3. 若爪尖明显画圈/漂移，说明 offset 值不准，需重新测量法兰→爪尖。

⚠ 真实机械臂会动，请保证周围有安全空间，随时按 x / Ctrl+C 停止。
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

OPENPI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (OPENPI_ROOT, os.path.join(OPENPI_ROOT, "piper_sdk"), os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference_eef import (  # noqa: E402
    DEFAULT_SPEED_PCT,
    GRIPPER_TCP_OFFSET_M,
    PiperEEFController,
    _001DEG_TO_RAD,
    _001MM_TO_M,
)

_HELP = """\
键盘 (以爪尖为控制点):
  位置  w/s=±X   a/d=±Y   q/e=±Z
  姿态  i/k=±RX  j/l=±RY  u/o=±RZ
  r=回初始位姿   h=帮助   x=退出
"""


def _getch() -> str:
    """读取单个按键 (Linux termios，阻塞等待一个字符)。"""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _pose_str(pose: np.ndarray) -> str:
    x, y, z = pose[:3]
    rx, ry, rz = np.rad2deg(pose[3:6])
    return f"x={x:+.4f}m y={y:+.4f}m z={z:+.4f}m | RX={rx:+7.2f}° RY={ry:+7.2f}° RZ={rz:+7.2f}°"


def _read_flange(ctrl: PiperEEFController) -> np.ndarray:
    """直接读 SDK 原始法兰位姿 [x, y, z(m), euler(rad)] (不补偿)。"""
    ep = ctrl._piper.GetArmEndPoseMsgs().end_pose
    pos = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis], dtype=np.float64) * _001MM_TO_M
    euler = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], dtype=np.float64) * _001DEG_TO_RAD
    return np.concatenate([pos, euler])


def main() -> None:
    p = argparse.ArgumentParser(description="以爪尖为控制点的 Piper 末端位姿测试")
    p.add_argument("--can", default="can0", help="CAN 接口名")
    p.add_argument(
        "--offset", type=float, nargs=3, default=None,
        help="法兰→爪尖偏移 (米, x y z)。默认用 inference_eef.GRIPPER_TCP_OFFSET_M",
    )
    p.add_argument("--step", type=float, default=0.005, help="位置微调步长 (米)")
    p.add_argument("--rot-step", type=float, default=2.0, help="姿态微调步长 (度)")
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED_PCT, help="速度百分比 20-100")
    args = p.parse_args()

    if args.offset is not None:
        # 原地修改共享数组，inference_eef 里的补偿函数立即生效
        GRIPPER_TCP_OFFSET_M[:] = np.asarray(args.offset, dtype=np.float64)

    print("=" * 72)
    print("爪尖 (TCP) 控制点测试")
    print(f"  CAN:                 {args.can}")
    print(f"  GRIPPER_TCP_OFFSET_M: {GRIPPER_TCP_OFFSET_M.tolist()} m")
    print(f"  位置步长:            {args.step * 1000:.1f} mm")
    print(f"  姿态步长:            {args.rot_step:.1f}°")
    print(f"  速度:                {args.speed}%")
    print("=" * 72)

    ctrl = PiperEEFController(can_name=args.can, gripper=None, control_gripper=False)
    if not ctrl.enable(speed_pct=args.speed):
        print("[ERROR] 使能失败，退出")
        sys.exit(1)

    print(_HELP)

    step_pos = args.step
    step_rot = np.deg2rad(args.rot_step)

    try:
        while True:
            flange = _read_flange(ctrl)
            tcp = ctrl.get_eef_pose()
            print("─" * 72)
            print(f"法兰: {_pose_str(flange)}")
            print(f"爪尖: {_pose_str(tcp)}")
            print("─" * 72)

            ch = _getch()
            if ch in ("x", "X", "\x03"):  # x / Ctrl+C
                print("[INFO] 退出")
                break
            if ch in ("\r", "\n", "h", "H", "?"):
                print(_HELP)
                continue
            if ch == "\x1b":  # 忽略方向键转义序列首字节
                continue

            dpos = np.zeros(3)
            drot = np.zeros(3)
            if ch in ("w", "W"):
                dpos[0] = +step_pos
            elif ch in ("s", "S"):
                dpos[0] = -step_pos
            elif ch in ("a", "A"):
                dpos[1] = +step_pos
            elif ch in ("d", "D"):
                dpos[1] = -step_pos
            elif ch in ("q", "Q"):
                dpos[2] = +step_pos
            elif ch in ("e", "E"):
                dpos[2] = -step_pos
            elif ch in ("i", "I"):
                drot[0] = +step_rot
            elif ch in ("k", "K"):
                drot[0] = -step_rot
            elif ch in ("j", "J"):
                drot[1] = +step_rot
            elif ch in ("l", "L"):
                drot[1] = -step_rot
            elif ch in ("u", "U"):
                drot[2] = +step_rot
            elif ch in ("o", "O"):
                drot[2] = -step_rot
            elif ch in ("r", "R"):
                ctrl.go_to_init_pose()
                continue
            else:
                print(f"[WARN] 未知按键 '{ch}'，按 h 查看帮助")
                continue

            # 在爪尖空间累加增量: 位置按世界系平移，姿态按世界系左乘 (与 RPY 约定一致)
            target = tcp.copy()
            target[:3] = target[:3] + dpos
            if np.any(drot):
                R = Rotation.from_euler("xyz", drot) * Rotation.from_rotvec(tcp[3:6])
                target[3:6] = R.as_rotvec()
            ctrl.send_eef_command(target)
    finally:
        # 退出时保持使能 (不调用 DisableArm)，便于后续手动或其它脚本继续控制。
        # 如需失能，可用 python -c "..." 或重新跑带 disable 的脚本。
        print("[INFO] 已退出，机械臂保持使能状态")


if __name__ == "__main__":
    main()
