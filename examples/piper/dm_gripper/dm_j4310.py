"""
Damiao DM-J4310-2EC motor driver (transport-agnostic).

Protocol facts, taken from the official manual
"DM-J4310-2EC V1.1 减速电机使用说明书 V1.3" (CAN 通信 chapter):

  * Standard CAN frames, default 1 Mbps.
  * MIT control frame ID = the motor's CAN ID (ESC_ID register 0x08).
  * Feedback is poll-based: whenever the driver receives a frame whose ID
    matches its CAN ID (low 8 bits compared, upper 3 bits ignored) it answers
    with a feedback frame on MST_ID (register 0x07, default 0).
  * All integer fields are linearly mapped floats (MIT mode / feedback):
        p int16 <-> [-PMAX, +PMAX] rad   (PMAX default 12.5)
        v int12 <-> [-VMAX, +VMAX] rad/s (VMAX default 30)
        t int12 <-> [-TMAX, +TMAX] Nm    (TMAX default 10)
        kp uint12 <-> [0, 500]
        kd uint12 <-> [0, 5]
    If PMAX/VMAX/TMAX were changed in the debug assistant, pass the new values
    to the constructor — mismatched mapping silently scales commands/feedback.
  * Command frames (ID = motor CAN ID, first two data bytes matter):
        enable FF FC      disable FF FD      set zero FF FE      clear err FF FB
  * Position-velocity (CTRL_MODE=2): frame ID = 0x100 + CAN ID,
        data = <f p_des> <f v_des>  (little-endian float32 x2).
  * Force-position (CTRL_MODE=4): frame ID = 0x300 + CAN ID,
        data = <f p_des> <H v_des*100> <H i_des*10000>.
  * Parameter read/write on ID 0x7FF:
        read:  [CANID_L, CANID_H, 0x33, RID]      -> reply on MST_ID with
               float32/u32 in D[4:8]
        write: [CANID_L, CANID_H, 0x55, RID, val] -> same reply format
        save:  [CANID_L, CANID_H, 0xAA, 0x01] (disabled state only)
  * On power-up the reported output position lies within [-pi, pi] rad.
"""

import struct
import time

# Register addresses (from manual 寄存器列表及范围)
REG = {
    "OC_Value": 0x03,   # overcurrent protection value, per-unit (0,1.0], float
    "ACC": 0x04,        # acceleration (rotor), Krad/s^2, float
    "DEC": 0x05,        # deceleration (negative), float
    "MAX_SPD": 0x06,    # max speed (rotor), rad/s, float
    "MST_ID": 0x07,     # feedback frame ID, u32
    "ESC_ID": 0x08,     # receive (control) frame ID, u32
    "CTRL_MODE": 0x0A,  # 1=MIT 2=pos-vel 3=vel 4=force-pos, u32
    "Gr": 0x14,         # gear ratio, float, RO
    "PMAX": 0x15,       # position mapping range, float
    "VMAX": 0x16,       # velocity mapping range, float
    "TMAX": 0x17,       # torque mapping range, float
    "GREF": 0x1E,       # gearbox torque efficiency (0,1.0], float
    "Deta": 0x1F,       # speed-loop damping factor, [1.0,30.0], float
    "CAN_BR": 0x23,     # CAN baud code: 4 = 1M, u32
    "Imax": 0x3B,       # driver max phase current, A, float, RO
    "VBus": 0x3C,       # supply voltage, V, float, RO
    "Tpcb": 0x3D,       # driver temperature, C, float, RO
    "Tmtr": 0x3E,       # motor coil temperature, C, float, RO
    "P_M": 0x50,        # motor position folded to output shaft, rad, float, RO
    "XOUT": 0x51,       # output-shaft encoder position, rad, float, RO
}

CTRL_MODE_MIT = 1
CTRL_MODE_POS_VEL = 2
CTRL_MODE_VEL = 3
CTRL_MODE_FORCE_POS = 4

# Feedback ERR nibble -> meaning (manual V1.3)
ERR_MEANING = {
    0: "disabled",
    1: "enabled",
    3: "output-shaft calib error",
    4: "sensor output error",
    5: "motor encoder calib error",
    8: "over-voltage",
    9: "under-voltage",
    0xA: "over-current",
    0xB: "MOS over-temp",
    0xC: "coil over-temp",
    0xD: "communication lost",
    0xE: "overload",
}


def _pack_mit(p_int16, v_int12, kp_u12, kd_u12, t_int12):
    """Pack MIT control fields into 8 bytes (big-endian bit packing)."""
    p = p_int16 & 0xFFFF
    v = v_int12 & 0xFFF
    kp = kp_u12 & 0xFFF
    kd = kd_u12 & 0xFFF
    t = t_int12 & 0xFFF
    return bytes([
        (p >> 8) & 0xFF, p & 0xFF,
        (v >> 4) & 0xFF,
        ((v & 0x0F) << 4) | ((kp >> 8) & 0x0F),
        kp & 0xFF,
        (kd >> 4) & 0xFF,
        ((kd & 0x0F) << 4) | ((t >> 8) & 0x0F),
        t & 0xFF,
    ])


def _unpack_int12_unsigned(b0, b1):
    return ((b0 & 0xFF) << 4) | ((b1 >> 4) & 0x0F)


class DmJ4310:
    """Protocol layer for one DM-J4310-2EC motor.

    ``bus`` is any object with ``.send(can_id, data, extended=...)`` and
    ``.read_frame(timeout) -> (can_id, data, is_ext) | None`` — e.g. the
    ``Usb2Can`` transport in this package (Linux /dev/ttyACM1) or a
    socketcan wrapper.
    """

    def __init__(self, bus, can_id=1, master_id=None,
                 p_max=12.5, v_max=30.0, t_max=10.0):
        self.bus = bus
        self.can_id = can_id
        self.master_id = master_id if master_id is not None else 0
        self.p_max = float(p_max)
        self.v_max = float(v_max)
        self.t_max = float(t_max)

    # ---------- helpers ----------
    @staticmethod
    def _clip(x, lo, hi):
        return max(lo, min(hi, x))

    def _wait_reply(self, timeout=0.5):
        """Read frames until a reply on master_id arrives."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.bus.read_frame(timeout=deadline - time.monotonic())
            if frame is None:
                return None
            can_id, data, _is_ext = frame
            if can_id == self.master_id:
                return data
        return None

    # ---------- basic commands (frame ID = motor CAN ID) ----------
    def _cmd(self, d0, d1):
        # official example sends 0xFF x7 + code (enable FC / disable FD /
        # zero FE / clear errors FB); the driver only checks the last byte
        self.bus.send(self.can_id, bytes([0xFF] * 7 + [d1]))

    def enable(self):
        self._cmd(0xFF, 0xFC)

    def disable(self):
        self._cmd(0xFF, 0xFD)

    def set_zero(self):
        """Save current output position as zero; also zeroes the position command."""
        self._cmd(0xFF, 0xFE)

    def clear_errors(self):
        self._cmd(0xFF, 0xFB)

    # ---------- position-velocity mode (CTRL_MODE=2) ----------
    def set_position_vel(self, p_rad, v_rad_s=1.0):
        """
        Position-velocity command: frame ID = 0x100 + can_id, payload is
        ``<f p_des> <f v_des>`` (little-endian float32 x2).  ``p_des`` is the
        target output-shaft position (rad), ``v_des`` caps the maximum absolute
        speed (rad/s).  The driver runs its own cascaded position loop.
        """
        payload = struct.pack("<ff", float(p_rad), float(v_rad_s))
        self.bus.send(0x100 + self.can_id, payload)

    # ---------- MIT mode (CTRL_MODE=1) ----------
    def set_position(self, p_rad, v_rad_s=0.0, kp=20.0, kd=1.0, t_ff=0.0):
        """MIT position command (kp [0,500], kd [0,5], kd must be > 0)."""
        def to_uint(x, lo, hi, bits):
            x = self._clip(x, lo, hi)
            return int(round((x - lo) / (hi - lo) * ((1 << bits) - 1)))

        p = to_uint(p_rad, -self.p_max, self.p_max, 16)
        v = to_uint(v_rad_s, -self.v_max, self.v_max, 12)
        t = to_uint(t_ff, -self.t_max, self.t_max, 12)
        kp_i = to_uint(kp, 0.0, 500.0, 12)
        kd_i = to_uint(kd, 0.0, 5.0, 12)
        self.bus.send(self.can_id, _pack_mit(p, v, kp_i, kd_i, t))

    # ---------- force-position mode (CTRL_MODE=4) ----------
    def set_force_position(self, p_rad, v_rad_s=1.0, i_limit=0.5):
        """
        Force-position hybrid command: frame ID = 0x300 + can_id.  Position-velocity
        control plus a HARDWARE current (torque) limit — ideal for a gripper.

        Payload (all little-endian):
            p_des  float32, output-shaft rad
            v_des  uint16, round(rad/s * 100), clamped to 0..10000
            i_des  uint16, round(per_unit * 10000), clamped to 0..10000
        """
        v_int = max(0, min(10000, int(round(v_rad_s * 100.0))))
        i_int = max(0, min(10000, int(round(i_limit * 10000.0))))
        payload = struct.pack("<f", float(p_rad)) + struct.pack("<H", v_int) \
            + struct.pack("<H", i_int)
        self.bus.send(0x300 + self.can_id, payload)

    def set_control_mode(self, mode):
        """Switch control mode via register 0x0A (no reset; zeroes commands)."""
        return self.write_reg(REG["CTRL_MODE"], mode, as_float=False)

    # ---------- feedback ----------
    def read_feedback(self, timeout=0.3):
        """
        Poll-based feedback frame on MST_ID:

            D0 = (ERR << 4) | ID_low4
            D1:D2 = POS int16, D3:D4 = VEL int12, D4:D5 = T int12
            D6 = T_MOS (C), D7 = T_Rotor (C)

        Returns dict or None.  ``pos`` is the rotor position folded to the
        output shaft (same frame as p_des in pos-vel mode), rad.
        """
        data = self._wait_reply(timeout)
        if data is None or len(data) < 8:
            return None

        def from_uint(raw, lo, hi, bits):
            return raw / ((1 << bits) - 1) * (hi - lo) + lo

        pos_raw = struct.unpack(">H", bytes(data[1:3]))[0]
        vel_raw = _unpack_int12_unsigned(data[3], data[4])
        t_raw = ((data[4] & 0x0F) << 8) | data[5]   # D4 low nibble = T[11:8], D5 = T[7:0]
        return {
            "err": (data[0] >> 4) & 0x0F,
            "id": data[0] & 0x0F,
            "pos": from_uint(pos_raw, -self.p_max, self.p_max, 16),   # rad, output shaft
            "vel": from_uint(vel_raw, -self.v_max, self.v_max, 12),   # rad/s
            "torque": from_uint(t_raw, -self.t_max, self.t_max, 12),  # Nm
            "t_mos": data[6],
            "t_rotor": data[7],
        }

    # ---------- parameter registers ----------
    def read_reg(self, rid, as_float=True, timeout=0.5):
        """
        Read a parameter register via the 0x7FF broadcast channel.
        Reply: [CANID_L, CANID_H, 0x33, RID, value_le32] on MST_ID.
        """
        self.bus.send(0x7FF, bytes([self.can_id & 0xFF, (self.can_id >> 8) & 0xFF,
                                    0x33, rid]))
        data = self._wait_reply(timeout)
        if data is None or len(data) < 8:
            return None
        if data[0] != (self.can_id & 0xFF) or data[2] != 0x33 or data[3] != rid:
            return None
        raw = bytes(data[4:8])
        return struct.unpack("<f", raw)[0] if as_float else struct.unpack("<I", raw)[0]

    def write_reg(self, rid, value, as_float=True, timeout=0.5):
        payload = bytes([self.can_id & 0xFF, (self.can_id >> 8) & 0xFF, 0x55, rid])
        payload += struct.pack("<f", value) if as_float else struct.pack("<I", value)
        self.bus.send(0x7FF, payload)
        data = self._wait_reply(timeout)
        return data is not None

    def save_params(self):
        """Persist all params to flash.  Only works while disabled; max 30ms."""
        self.bus.send(0x7FF, bytes([self.can_id & 0xFF, (self.can_id >> 8) & 0xFF,
                                    0xAA, 0x01]))

    # ---------- convenience ----------
    def read_info(self):
        """Read identity + mapping registers; returns dict with None on failures."""
        info = {}
        for key in ("ESC_ID", "MST_ID", "CTRL_MODE", "CAN_BR"):
            v = self.read_reg(REG[key], as_float=False)
            info[key] = v
            time.sleep(0.03)
        for key in ("PMAX", "VMAX", "TMAX", "ACC", "DEC", "MAX_SPD", "Gr",
                    "Imax", "VBus", "Tmtr", "XOUT", "P_M"):
            v = self.read_reg(REG[key], as_float=True)
            info[key] = v
            time.sleep(0.03)
        return info

    def sync_mapping(self, timeout=0.5, retries=3):
        """Read PMAX/VMAX/TMAX from the motor and update the mapping ranges.

        The constructor defaults (p_max=12.5, v_max=30, t_max=10) assume factory
        settings.  If PMAX was raised in the debug assistant (this gripper runs
        at 25 rad), leaving the default silently mis-scales feedback position
        decoding and command clamping.  Call once after connecting.
        """
        for rid, attr in ((REG["PMAX"], "p_max"), (REG["VMAX"], "v_max"),
                          (REG["TMAX"], "t_max")):
            v = None
            for _ in range(retries):
                v = self.read_reg(rid, as_float=True, timeout=timeout)
                if v is not None:
                    break
                time.sleep(0.1)
            if v is not None and 0.0 < v < 10000.0:
                setattr(self, attr, float(v))
