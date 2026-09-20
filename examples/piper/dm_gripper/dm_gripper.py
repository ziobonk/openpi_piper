"""
Gripper controller for the Damiao DM-J4310-2EC on the Piper arm.

Drives the motor in FORCE-POSITION hybrid mode (CTRL_MODE = 4) over the USB2CAN
module (``/dev/ttyACM1``): position control with a speed cap, plus a HARDWARE
torque-current limit (``i_des``) so the jaws can never push harder than the
configured grip force — the anti-crush protection a gripper needs.

Position convention (per the installation):

    0 rad  -> OPEN
    10 rad -> CLOSED

``p_des`` is the output-shaft position in radians; ``v_des`` caps the maximum
absolute speed (rad/s); ``i_des`` (``current_limit``, 0..1.0 of max phase
current) saturates the current loop.  The driver runs its own cascaded
position loop, so a move is a smooth, speed-limited rotation.

Safety
------
* ``enable()`` re-sends the enable frame until the motor confirms (feedback is
  poll-based), so a single lost frame cannot leave it half-enabled.
* ``disable()`` is called on shutdown / on error so the motor is never left
  holding a stale target.
* ``current_limit`` caps the hardware torque current, bounding the grip force.
* Targets are clamped to the feedback wrap band (default +/-12.5 rad minus a
  margin) so the command never wraps the absolute encoder.
"""

import time

from dm_j4310 import DmJ4310, CTRL_MODE_FORCE_POS, ERR_MEANING

DEFAULT_OPEN_RAD = 0.0
DEFAULT_CLOSE_RAD = 5.0
DEFAULT_CURRENT_LIMIT = 0.3   # per-unit torque-current limit (i_des 0..1.0)
WRAP_MARGIN_RAD = 0.3      # stay this far inside the +/-PMAX feedback wrap
SETTLE_TOL_RAD = 0.03      # consider arrived within this (rad)


class DmGripper:
    """High-level open/close controller for the DM-J4310-2EC gripper."""

    def __init__(self, motor, open_rad=DEFAULT_OPEN_RAD, close_rad=DEFAULT_CLOSE_RAD,
                 speed_rad_s=2.0, current_limit=DEFAULT_CURRENT_LIMIT):
        """
        motor: a DmJ4310 instance (force-position mode, CTRL_MODE=4).
        open_rad / close_rad: output-shaft positions for OPEN / CLOSED (rad).
        speed_rad_s: max move speed (rad/s) used by set_position / open / close.
        current_limit: per-unit torque-current limit (i_des, 0..1.0).  In
            force-position mode this caps the current loop, so the jaws can
            never push harder than this — the anti-crush protection.
        """
        self.motor = motor
        # Match the motor's real PMAX/VMAX/TMAX so feedback position decoding
        # and command clamping are correct (this motor runs PMAX=25 rad, not 12.5).
        sync = getattr(self.motor, "sync_mapping", None)
        if sync is not None:
            sync()
        self.open_rad = float(open_rad)
        self.close_rad = float(close_rad)
        self.speed_rad_s = float(speed_rad_s)
        self.current_limit = max(0.0, min(1.0, float(current_limit)))
        self._enabled = False
        self._last_cmd_rad = None   # last commanded position, for feedback poking

    def configure_mode(self, mode=CTRL_MODE_FORCE_POS):
        """
        Switch the motor's CTRL_MODE register (default: 4 = force-position).
        The motor must be DISABLED.  Returns True if the register read back
        matches ``mode``.  Call ``motor.save_params()`` afterwards to persist.
        """
        self.motor.disable()
        time.sleep(0.1)
        self.motor.set_control_mode(mode)
        time.sleep(0.1)
        return self.motor.read_reg(0x0A, as_float=False) == mode

    def set_current_limit(self, per_unit):
        """Set the hardware torque/current limit (i_des, 0..1.0)."""
        self.current_limit = max(0.0, min(1.0, float(per_unit)))

    # ---------- limits ----------
    def _p_max(self):
        return getattr(self.motor, "p_max", 12.5)

    def _clamp(self, rad):
        lo = -self._p_max() + WRAP_MARGIN_RAD
        hi = +self._p_max() - WRAP_MARGIN_RAD
        return max(lo, min(hi, rad))

    # ---------- enable / disable ----------
    def enable(self, timeout=2.0):
        """Enable the motor and confirm via feedback (ERR == 1)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.motor.enable()
            fb = self.motor.read_feedback(timeout=0.3)
            if fb is not None and fb["err"] == 1:
                self._enabled = True
                # Latch the current position so the next _poke() sends a
                # "hold position" command instead of a disable (which would
                # turn the motor straight back off — see _poke()).
                self._last_cmd_rad = float(fb["pos"])
                return True
            time.sleep(0.02)
        print("[gripper] enable failed (no ERR==1 feedback)")
        return False

    def disable(self):
        """Disable the motor; sends a few times so the polled driver latches it."""
        self._enabled = False
        for _ in range(3):
            self.motor.disable()
            time.sleep(0.02)

    @property
    def is_enabled(self):
        return self._enabled

    # ---------- feedback ----------
    def _poke(self):
        """Trigger one polled feedback frame (feedback is request-based)."""
        if self._enabled and self._last_cmd_rad is not None:
            self.motor.set_force_position(self._last_cmd_rad, self.speed_rad_s,
                                          self.current_limit)
        else:
            self.motor.disable()   # harmless re-disable pokes the reply

    def read_state(self):
        """Poke, then read one feedback frame.  Returns the raw dict or None."""
        self._poke()
        return self.motor.read_feedback(timeout=0.3)

    def read_feedback(self):
        """Return the raw feedback dict (pos/vel/torque/temps/err) or None.

        Pokes the motor first (feedback is request-based), so it is safe to
        call standalone; move_to() streams commands directly instead.
        """
        return self.read_state()

    def position(self):
        """Current output-shaft position (rad), or None on feedback loss."""
        fb = self.read_state()
        return fb["pos"] if fb is not None else None

    def state(self):
        """
        One polled read.  Returns (pos_rad, err) or None.  Prints a human
        description of the ERR state.
        """
        fb = self.read_state()
        if fb is None:
            return None
        err = fb["err"]
        label = ERR_MEANING.get(err, "unknown")
        return fb, f"err={err}({label})"

    # ---------- motion ----------
    def set_position(self, rad, speed_rad_s=None):
        """Send one force-position command (non-blocking)."""
        v = self.speed_rad_s if speed_rad_s is None else speed_rad_s
        self._last_cmd_rad = self._clamp(rad)
        self.motor.set_force_position(self._last_cmd_rad, v, self.current_limit)

    def move_to(self, rad, speed_rad_s=None, timeout=20.0):
        """
        Move to ``rad`` and wait for arrival (blocking).  Returns True on
        arrival, False on timeout / feedback loss.  The driver keeps receiving
        fresh position commands until the feedback reaches the target, so the
        trapezoidal profile is re-issued continuously.
        """
        if not self._enabled and not self.enable():
            return False
        target = self._clamp(rad)
        v = self.speed_rad_s if speed_rad_s is None else speed_rad_s
        deadline = time.monotonic() + timeout
        last_print = 0.0
        while time.monotonic() < deadline:
            self.motor.set_force_position(target, v, self.current_limit)
            fb = self.motor.read_feedback(timeout=0.1)
            if fb is None:
                print("[gripper] feedback lost during move")
                return False
            now = time.monotonic()
            if now - last_print > 0.5:
                print(f"[gripper] pos={fb['pos']:7.3f} rad  target={target:7.3f} "
                      f"vel={fb['vel']:6.2f}  torque={fb['torque']:5.2f} Nm")
                last_print = now
            if abs(fb["pos"] - target) <= SETTLE_TOL_RAD:
                return True
            time.sleep(0.02)
        print("[gripper] move timeout")
        return False

    def open(self, speed_rad_s=None, timeout=20.0):
        """Move to the OPEN position (0 rad by default)."""
        return self.move_to(self.open_rad, speed_rad_s=speed_rad_s, timeout=timeout)

    def close(self, speed_rad_s=None, timeout=20.0):
        """Move to the CLOSED position (10 rad by default)."""
        return self.move_to(self.close_rad, speed_rad_s=speed_rad_s, timeout=timeout)

    def shutdown(self):
        """Disable and close the transport.  Call from a finally block."""
        try:
            self.disable()
        finally:
            self.motor.bus.close()
