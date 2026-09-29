#!/usr/bin/env python3
"""夹爪控制测试 —— 排查闭合失败的原因。"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "piper_sdk"))
from piper_sdk import C_PiperInterface_V2

CAN = sys.argv[1] if len(sys.argv) > 1 else "can0"

p = C_PiperInterface_V2(can_name=CAN, judge_flag=False, can_auto_init=True, dh_is_offset=1)
p.ConnectPort()
time.sleep(0.1)

print(f"[{CAN}] 使能...")
p.EnablePiper()
p.MotionCtrl_2(0x01, 0x01, 40, 0x00)
time.sleep(0.3)

def read(label=""):
    raw = p.GetArmGripperMsgs().gripper_state.grippers_angle
    print(f"  {label} raw={raw} ({raw/1000:.1f}mm, {raw/1_000_000:.4f}m)")
    return raw

def cmd(pos_raw, label):
    """发送夹爪指令并等待。"""
    print(f"\n--- {label}: GripperCtrl({pos_raw}) ---")
    p.GripperCtrl(pos_raw, 1000, 0x02, 0); time.sleep(0.03)
    p.GripperCtrl(pos_raw, 1000, 0x01, 0); time.sleep(1.5)
    return read(label)

# 读取初始状态
print("\n=== 初始 ===")
r0 = read("初始")
print(f"  get_joints[0]: {p.GetArmJointMsgs().joint_state.joint_1}")

# 尝试不同方向：数字越小夹爪越闭合
# 典型的 Piper 夹爪: 0 = 闭合, ~70000 = 全开 (~70mm)

cmd(50000, "张开 50000")
cmd(20000, "半开 20000")
cmd(0,     "闭合 0")
cmd(100,   "闭合 100")     # 模型输出 0.0001m
cmd(1000,  "闭合 1000")    # 模型输出 0.001m
cmd(3000,  "闭合 3000")    # 模型输出 0.003m
cmd(5000,  "闭合 5000")    # 模型输出 0.005m

print("\n=== DONE ===")
print(f"如果所有 raw 值都没有明显变化，夹爪可能没有使能或硬件有问题。")
print(f"尝试手动闭合：GripperCtrl(0, 1000, 0x01, 0)")
p.GripperCtrl(0, 1000, 0x01, 0)
time.sleep(1.0)
read("最终")
print("机械臂保持使能状态，未去使能。按 Ctrl+C 退出。")
while True:
    time.sleep(1)
