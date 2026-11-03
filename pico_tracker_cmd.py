#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PICO 体感追踪器遥操作指令发送工具（PC 端，键盘按键版）

向运行 PicoDualTrackerUdpSender 的 PICO 头显发送 UDP 控制指令，
替代原手柄按键承担的功能：

  - 修改目标 IP   ：原 VR 内 IP 编辑器   -> 按键 i / 命令 "set_ip"
  - 坐标系复位    ：原右手 B 键          -> 按键 r / 命令 "reset_origin"
  - 发送数据开关  ：原右手 A 键          -> 按键 s、t / 命令 "start"、"stop"

另提供：

  - 夹爪开合      ：原扳机，现由 PC 驱动 -> 按键 g、c、1~9、0 / 命令 "set_gripper"
  - 左右手重新绑定：追踪器版特有         -> 按键 b / 命令 "rebind"
  - 单双臂切换    ：追踪器版特有         -> 按键 m / 命令 "set_arm_mode"

数据链路（Pico 代替 T265 后）：
  PICO 头显 --无线UDP--> 发送端电脑(udp_sender.py 中继) --UDP--> 接收端电脑
                            编码器(UART)------^
  "set_ip" 应把目标设成 发送端电脑 的 IP（换电脑只改这一个 IP）。
  中继模式下夹爪角度由发送端电脑的编码器给出，头显的 set_gripper 值会被覆盖，
  本工具的夹爪按键仅当"头显直连接收端"时才有效。

协议（头显监听 UDP 端口，默认 5006）：
  {"type": "lightumi.teleop_cmd", "command": "<命令名>", <参数...>}

用法：
  python pico_tracker_cmd.py                     # 键盘控制模式（按下按键立即发送，无需回车）
  python pico_tracker_cmd.py --host 192.168.3.10 # 指定头显 IP（默认全网广播）
  python pico_tracker_cmd.py origin              # 单条命令模式（脚本调用用）
  python pico_tracker_cmd.py ip 192.168.3.74:5005
  python pico_tracker_cmd.py arm left            # 单左臂；dual/left/right
  python pico_tracker_cmd.py gripper 0.3 right
  python pico_tracker_cmd.py --raw '{"type":"lightumi.teleop_cmd","command":"start"}'

键盘模式按键表：
  s / t     开始 / 停止发送数据
  r         坐标系复位
  g / c     夹爪全开 / 全闭（value 0 / 1，默认左右手同时）
  1~9, 0    夹爪开到 10%~90%；0 = 100%（全闭）
  i         目标 IP 设为本机（自动发现；显式指定请用单条命令模式 ip <地址>）
  b         重新绑定左右手腕追踪器
  m         切换双臂/单左臂/单右臂
  h         显示帮助
  q / Esc   退出

注意：指令走 UDP，不保证送达；若头显状态未变化，再按一次即可。
"""

import argparse
import json
import os
import select
import socket
import sys
import time

DEFAULT_COMMAND_PORT = 5006
BROADCAST_HOST = "255.255.255.255"

if os.name == "nt":
    import msvcrt

KEY_HELP = """\
=== PICO 体感追踪器按键控制（按下即发，无需回车） ===
发送目标: {host}:{port}

  s / t     开始 / 停止发送数据
  r         坐标系复位
  g / c     夹爪全开 / 全闭（value 0 / 1，默认左右手同时）
  1~9, 0    夹爪开到 10%~90%；0 = 100%（全闭）
  i         目标 IP 设为本机（自动发现）
  b         重新绑定左右手腕追踪器
  m         切换双臂/单左臂/单右臂
  h         显示本帮助
  q / Esc   退出
"""

CLI_HELP = """\
单条命令用法:
  python pico_tracker_cmd.py [--host <ip>] [--port <n>] <命令> [参数]

命令:
  ip [addr[:port]]      修改遥测目标 IP（应设为发送端电脑/中继的 IP；缺省 addr = 用本机地址自动发现）
  origin                坐标系复位
  start / stop          开始 / 停止数据发送
  arm <mode>            设置单双臂模式（mode: dual/left/right; left/right 为单臂）
  gripper <0~1> [side]  夹爪开合（0=全开, 1=全闭; side: left/right/both, 默认 both）
  rebind                重新绑定左右手腕追踪器
  host <addr[:port]>    修改指令发送目标
  send <json>           发送原始 JSON
  help                  显示本帮助
无参数启动则进入键盘控制模式。
"""


def build_cmd(name, **params):
    """构造指令 JSON。"""
    cmd = {"type": "lightumi.teleop_cmd", "command": name}
    cmd.update(params)
    return cmd


def normalize_arm_mode(text):
    """把单双臂模式别名规整为 dual/left/right；非法值返回 None。"""
    mode = str(text).strip().lower()
    if mode in ("dual", "both", "double", "two", "2", "双", "双臂"):
        return "dual"
    if mode in ("left", "l", "0", "左", "左臂", "单左", "单左臂"):
        return "left"
    if mode in ("right", "r", "1", "右", "右臂", "单右", "单右臂"):
        return "right"
    return None


def parse_addr(text):
    """解析 "ip[:port]" 形式的地址，返回 (ip, port)，port 缺省为 None。"""
    if ":" in text:
        host, _, port_str = text.rpartition(":")
        if port_str.isdigit():
            return host, int(port_str)
        return host, None
    return text, None


class Commander:
    """UDP 指令发送器。"""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send(self, cmd):
        data = json.dumps(cmd).encode("utf-8")
        self.send_raw(data)

    def send_raw(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        try:
            self.sock.sendto(data, (self.host, self.port))
            print("-> {}:{}  {}".format(self.host, self.port, data.decode("utf-8")))
        except OSError as e:
            print("发送失败: {}".format(e))

    def set_dest(self, addr):
        ip, port = parse_addr(addr)
        if port:
            self.port = port
        self.host = ip
        print("指令目标已改为 {}:{}".format(self.host, self.port))


# ----------------------------------------------------------------------
# 键盘读取（Windows 用 msvcrt，类 Unix 用 termios raw 模式）
# ----------------------------------------------------------------------

def get_key():
    """读取单个按键（无需回车）。返回小写字符；无效键返回空串。"""
    if os.name == "nt":
        try:
            ch = msvcrt.getch()
        except OSError:
            return ""  # 非控制台环境（如管道重定向）
        if ch in (b"\x00", b"\xe0"):  # 方向键等功能键：吞掉第二字节后忽略
            try:
                msvcrt.getch()
            except OSError:
                pass
            return ""
        if ch == b"\x03":  # Ctrl+C
            raise KeyboardInterrupt
        return ch.decode("ascii", errors="replace").lower()

    # 类 Unix 平台分支（IDE 在本机 Windows 上会误报"不可达"，Linux/macOS 上实际可达）
    import termios  # pyright: ignore[reportUnreachable]
    import tty  # pyright: ignore[reportUnreachable]

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    if ch == "\x03":  # Ctrl+C
        raise KeyboardInterrupt
    if ch == "\x1b":  # Esc 或方向键等转义序列开头
        r, _, _ = select.select([sys.stdin], [], [], 0.05)
        if r:
            sys.stdin.read(2)  # 吞掉 "[A" 之类的剩余字节
            return ""
    return ch.lower()


def keyboard_mode(commander):
    print(KEY_HELP.format(host=commander.host, port=commander.port))
    empty_strikes = 0
    arm_modes = ["dual", "left", "right"]
    arm_mode_index = 0

    while True:
        try:
            k = get_key()
        except KeyboardInterrupt:
            print("\n退出")
            break

        if not k:
            empty_strikes += 1
            if empty_strikes >= 10:
                print("无法读取键盘输入（请在真实终端/控制台里运行本脚本）")
                break
            time.sleep(0.05)
            continue
        empty_strikes = 0

        if k in ("q", "\x1b"):
            print("退出")
            break
        elif k == "s":
            commander.send(build_cmd("start"))
        elif k == "t":
            commander.send(build_cmd("stop"))
        elif k == "r":
            commander.send(build_cmd("reset_origin"))
        elif k == "g":
            commander.send(build_cmd("set_gripper", side="both", value=0.0))
        elif k == "c":
            commander.send(build_cmd("set_gripper", side="both", value=1.0))
        elif k in "123456789":
            commander.send(build_cmd("set_gripper", side="both", value=int(k) / 10.0))
        elif k == "0":
            commander.send(build_cmd("set_gripper", side="both", value=1.0))
        elif k == "b":
            commander.send(build_cmd("rebind"))
        elif k == "m":
            arm_mode_index = (arm_mode_index + 1) % len(arm_modes)
            mode = arm_modes[arm_mode_index]
            commander.send(build_cmd("set_arm_mode", mode=mode))
            print("臂模式: {}".format(mode))
        elif k == "i":
            commander.send(build_cmd("set_ip"))
            print("（未指定 IP，头显将把指令来源地址设为目标）")
        elif k == "h":
            print(KEY_HELP.format(host=commander.host, port=commander.port))
        else:
            print("未绑定按键: {}（按 h 查看帮助）".format(repr(k)))


# ----------------------------------------------------------------------
# 单条命令模式（脚本调用用）
# ----------------------------------------------------------------------

def run_command(commander, name, args):
    """执行一条命令。返回 False 表示命令名无法识别。"""
    name = name.lower()

    if name in ("ip", "set_ip", "setip"):
        ip, port = None, None
        if args:
            ip, port = parse_addr(args[0])
        cmd = build_cmd("set_ip")
        if ip:
            cmd["ip"] = ip
        if port:
            cmd["port"] = port
        commander.send(cmd)
        if ip is None:
            print("（未指定 IP，头显将把指令来源地址设为目标）")
        return True

    if name in ("origin", "reset", "reset_origin"):
        commander.send(build_cmd("reset_origin"))
        return True

    if name == "start":
        commander.send(build_cmd("start"))
        return True

    if name == "stop":
        commander.send(build_cmd("stop"))
        return True

    if name in ("arm", "arms", "mode", "arm_mode", "set_arm_mode"):
        if not args:
            print("用法: arm <dual|left|right>")
            return True
        mode = normalize_arm_mode(args[0])
        if mode is None:
            print("未知臂模式: {}（可用: dual/left/right）".format(args[0]))
            return True
        commander.send(build_cmd("set_arm_mode", mode=mode))
        return True

    if name in ("dual", "both"):
        commander.send(build_cmd("set_arm_mode", mode="dual"))
        return True

    if name in ("left", "left_only"):
        commander.send(build_cmd("set_arm_mode", mode="left"))
        return True

    if name in ("right", "right_only"):
        commander.send(build_cmd("set_arm_mode", mode="right"))
        return True

    if name in ("gripper", "grip"):
        if not args:
            print("用法: gripper <0~1> [left|right|both]")
            return True
        try:
            value = float(args[0])
        except ValueError:
            print("夹爪值必须是 0~1 的数字: {}".format(args[0]))
            return True
        value = max(0.0, min(1.0, value))
        side = args[1] if len(args) > 1 else "both"
        commander.send(build_cmd("set_gripper", side=side, value=value))
        return True

    if name == "rebind":
        commander.send(build_cmd("rebind"))
        return True

    if name == "host":
        if not args:
            print("用法: host <ip[:port]>")
            return True
        commander.set_dest(args[0])
        return True

    if name == "send":
        if not args:
            print("用法: send <json>")
            return True
        commander.send_raw(" ".join(args))
        return True

    if name in ("help", "h", "?"):
        print(CLI_HELP)
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="给 PICO 头显（PicoDualTrackerUdpSender）发送遥操作控制指令。",
        add_help=True,
    )
    parser.add_argument(
        "--host",
        default=BROADCAST_HOST,
        help="指令发送目标 IP（默认 %(default)s，全网广播）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_COMMAND_PORT,
        help="头显指令监听端口（默认 %(default)s）",
    )
    parser.add_argument(
        "--raw",
        metavar="JSON",
        help="直接发送一段原始 JSON 指令后退出",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="要发送的命令（ip/origin/start/stop/gripper/rebind/...），留空进入键盘控制模式",
    )
    args = parser.parse_args()

    commander = Commander(args.host, args.port)

    if args.raw:
        commander.send_raw(args.raw)
        return

    if args.cmd:
        # 单条命令模式：执行后退出。
        if not run_command(commander, args.cmd[0], args.cmd[1:]):
            print("未知命令: {}".format(args.cmd[0]))
            print(CLI_HELP)
            sys.exit(2)
        return

    # 键盘控制模式。
    keyboard_mode(commander)


if __name__ == "__main__":
    main()
