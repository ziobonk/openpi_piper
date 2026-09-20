#!/usr/bin/env python3
"""
Test / bring-up script for the DM-J4310-2EC gripper motor (force-position
mode) on the Piper arm, over the Damiao USB2CAN module (``/dev/ttyACM1``).

Position convention: 0 rad = OPEN, 10 rad = CLOSED.

Before running: close DMTool (it holds the serial port), power the motor (24V).

Usage (from examples/piper/dm_gripper):

    python test_dm_gripper.py info                    # read-only: registers
    python test_dm_gripper.py read                    # read-only: one feedback frame
    python test_dm_gripper.py mode 4 --save           # switch to force-position (once)
    python test_dm_gripper.py enable                  # enable only
    python test_dm_gripper.py open                    # move to OPEN  (0 rad)
    python test_dm_gripper.py close                   # move to CLOSED (10 rad)
    python test_dm_gripper.py selftest                # full enable -> open -> close -> open -> disable

Options:
    --port /dev/ttyACM1      serial device of the USB2CAN module
    --id 1                   motor CAN ID (ESC_ID)
    --speed 1.0              move speed, rad/s
    --open 0.0               OPEN position, rad
    --close 10.0             CLOSED position, rad
    --current 0.3            torque-current limit (i_des, 0..1.0)
"""

import argparse
import time

from usb2can import Usb2Can
from dm_j4310 import DmJ4310, CTRL_MODE_FORCE_POS
from dm_gripper import DmGripper


def _build(args):
    bus = Usb2Can(port=args.port)
    motor = DmJ4310(bus, can_id=args.id)
    gripper = DmGripper(motor, open_rad=args.open, close_rad=args.close,
                        speed_rad_s=args.speed, current_limit=args.current)
    return bus, motor, gripper


def cmd_info(args):
    bus, motor, _ = _build(args)
    try:
        info = motor.read_info()
        print("\n=== DM-J4310-2EC info ===")
        for k, v in info.items():
            print(f"  {k:10s} = {v}")
        mode = info.get("CTRL_MODE")
        esc = info.get("ESC_ID")
        print()
        if esc != args.id:
            print(f"[warn] ESC_ID={esc} != requested --id {args.id} — pass --id {esc}")
        if mode == CTRL_MODE_FORCE_POS:
            print("[ok] control mode = force-position (4)")
        else:
            print(f"[warn] CTRL_MODE={mode} (force-position is 4). "
                  f"Switch while disabled: `mode 4 --save`.")
    finally:
        bus.close()


def cmd_read(args):
    bus, _, gripper = _build(args)
    try:
        for _ in range(5):
            fb = gripper.read_feedback()  # pokes the poll-based motor first
            if fb is None:
                print("  (no feedback)")
            else:
                print(f"  err={fb['err']} id={fb['id']} pos={fb['pos']:8.3f} rad "
                      f"vel={fb['vel']:6.2f} rad/s torque={fb['torque']:5.2f} Nm "
                      f"t_mos={fb['t_mos']}C t_rotor={fb['t_rotor']}C")
            time.sleep(0.3)
    finally:
        bus.close()


def cmd_enable(args):
    bus, motor, gripper = _build(args)
    try:
        ok = gripper.enable()
        print(f"[{'ok' if ok else 'FAIL'}] enable")
    finally:
        gripper.disable()
        bus.close()


def cmd_disable(args):
    bus, motor, gripper = _build(args)
    try:
        gripper.disable()
        print("[ok] disabled")
    finally:
        bus.close()


def cmd_open(args):
    _run_move(args, open_)


def cmd_close(args):
    _run_move(args, close_)


def _run_move(args, fn):
    bus, motor, gripper = _build(args)
    try:
        ok = fn(gripper, speed_rad_s=args.speed)
        print(f"[{'ok' if ok else 'FAIL'}] move complete")
    finally:
        gripper.shutdown()


def open_(g, speed_rad_s=None):
    return g.open(speed_rad_s=speed_rad_s)


def close_(g, speed_rad_s=None):
    return g.close(speed_rad_s=speed_rad_s)


def cmd_mode(args):
    """Switch the control mode (default: 4 = force-position)."""
    mode = args.mode_value if args.mode_value is not None else args.mode
    bus, motor, _ = _build(args)
    try:
        motor.disable()
        time.sleep(0.1)
        ok = motor.set_control_mode(mode)
        if ok and args.save:
            time.sleep(0.05)
            motor.save_params()
        time.sleep(0.1)
        cur = motor.read_reg(0x0A, as_float=False)
        print(f"CTRL_MODE now = {cur}  "
              f"({'ok' if cur == mode else 'FAIL — expected ' + str(mode)})"
              + ("  (saved to flash)" if args.save else ""))
    finally:
        bus.close()


def cmd_selftest(args):
    bus, motor, gripper = _build(args)
    try:
        print("=== reading registers ===")
        info = motor.read_info()
        for k in ("ESC_ID", "MST_ID", "CTRL_MODE", "PMAX", "VMAX", "TMAX", "Gr"):
            print(f"  {k:10s} = {info.get(k)}")
        if info.get("CTRL_MODE") != CTRL_MODE_FORCE_POS:
            print("[warn] CTRL_MODE is not force-position (4) — commands may be ignored")
        if info.get("ESC_ID") != args.id:
            print(f"[warn] ESC_ID={info.get('ESC_ID')} != --id {args.id}")

        print("\n=== enabling ===")
        if not gripper.enable():
            print("[FAIL] could not enable")
            return 1

        fb = gripper.read_feedback()
        print(f"  initial pos = {fb['pos'] if fb else '?'} rad")

        print("\n=== OPEN (0 rad) ===")
        if not gripper.open():
            print("[FAIL] open")
            return 1
        print("\n=== CLOSE (10 rad) ===")
        if not gripper.close():
            print("[FAIL] close")
            return 1
        print("\n=== OPEN (0 rad) ===")
        if not gripper.open():
            print("[FAIL] open")
            return 1

        print("\n[ok] selftest passed")
        return 0
    finally:
        gripper.disable()
        bus.close()


def main():
    ap = argparse.ArgumentParser(description="DM-J4310-2EC gripper test")
    ap.add_argument("action", nargs="?", default="info",
                    choices=["info", "read", "enable", "disable", "mode",
                             "open", "close", "selftest"])
    ap.add_argument("mode_value", nargs="?", type=int, default=None,
                    help="mode number for the `mode` action (e.g. `mode 4`)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--open", type=float, default=0.0)
    ap.add_argument("--close", type=float, default=15.0)
    ap.add_argument("--current", type=float, default=0.3,
                    help="torque-current limit (i_des, 0..1.0) for force-position")
    ap.add_argument("--mode", type=int, default=4,
                    help="control mode to switch to (4 = force-position)")
    ap.add_argument("--save", action="store_true",
                    help="persist the mode to flash (with `mode`)")
    args = ap.parse_args()

    handlers = {
        "info": cmd_info, "read": cmd_read, "enable": cmd_enable,
        "disable": cmd_disable, "mode": cmd_mode,
        "open": cmd_open, "close": cmd_close,
        "selftest": cmd_selftest,
    }
    rc = handlers[args.action](args)
    raise SystemExit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
