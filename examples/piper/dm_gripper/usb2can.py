"""
Transport for the Damiao USB2CAN module (old generation, HDSC CDC rev).

This module enumerates as a CDC serial port — on Linux it appears as
``/dev/ttyACM*`` (VID 2E88 : PID 4603, "HDSC CDC Device").  It bridges a host
serial port to a CAN bus and speaks Damiao's closed serial protocol.

Protocol (from Damiao's official "USB 转CAN 帧格式" materials and the
"USB转CAN-linux" Python example):

  * Serial: 921600 baud, 8N1 (the firmware ignores the baud value).
  * No handshake / heartbeat — just open the port and send frames.

Send CAN frame (30 bytes, cmd 0x01 = forward with feedback, 0x03 = no feedback)::

    55 AA 1E <cmd> <times u32le> <interval u32le (100us)> <idType> <canid u32le>
    <frameType> <len> <idAcc> <dataAcc> <data 8 bytes, real bytes first> <crc>

  crc: the device does not validate it (official example uses fixed 0x88).

Receive (16-byte frames)::

    AA <CMD> <info> <canid u32le> <data 8 bytes> 55

  CMD: 00 heartbeat, 01 rx fail, 11 rx success, 02 tx fail, 12 tx success,
       03 baud-set fail, 13 baud-set success, EE comm error.
  info: bits0-5 data len, bit6 ide, bit7 rtr.

Only standard 11-bit frames are needed for the DM-J4310-2EC motor.

NOTE: the Damiao debug assistant (DMTool) must be closed while this runs —
the serial port is single-client.
"""

import struct
import time

import serial


class Usb2Can:
    """Minimal USB2CAN serial transport.  Duck-typed CAN API:

    ``send(can_id, data, extended=False)`` and
    ``read_frame(timeout=None) -> (can_id, data, is_ext) | None``
    """

    # CAN baud-rate codes for the ``55 05`` configuration command.
    BAUD_INDEX = {
        1000: 0, 800: 1, 666: 2, 500: 3, 400: 4,
        250: 5, 200: 6, 125: 7, 100: 8,
    }

    def __init__(self, port="/dev/ttyACM1", serial_baud=921600, can_baud_kbps=1000):
        self.ser = serial.Serial(port, serial_baud, timeout=0.5)
        self.ser.reset_input_buffer()
        if can_baud_kbps not in self.BAUD_INDEX:
            raise ValueError(f"unsupported CAN baud {can_baud_kbps} kbps")

        # Stop any previous continuous sending, then set the CAN baud rate.
        self.ser.write(bytes([0x55, 0x03, 0x00, 0x00]))
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.ser.write(bytes([0x55, 0x05, self.BAUD_INDEX[can_baud_kbps], 0xAA, 0x55]))
        time.sleep(0.2)
        print(f"[usb2can] {port} open at {serial_baud} baud, CAN {can_baud_kbps} kbps")

    # ---------- CAN API ----------
    def send(self, can_id, data=(), extended=False, with_feedback=True):
        """Send one CAN frame (30-byte protocol).  ``data``: bytes, max 8."""
        if len(data) > 8:
            raise ValueError("CAN data max 8 bytes")
        frame = bytearray([
            0x55, 0xAA, 0x1E,
            0x01 if with_feedback else 0x03,   # command
        ])
        frame += struct.pack("<I", 1)          # send times
        frame += struct.pack("<I", 10)         # interval, 100us units (1ms)
        frame.append(0x01 if extended else 0x00)          # id type
        frame += struct.pack("<I", can_id & 0x1FFFFFFF)   # CAN ID, little-endian
        frame.append(0x00)                     # frame type: data
        frame.append(len(data))                # data length
        frame.append(0x00)                     # idAcc
        frame.append(0x00)                     # dataAcc
        frame += bytes(data) + bytes(8 - len(data))       # data, real bytes first
        frame.append(0x88)                     # crc (not validated by device)
        self.ser.write(bytes(frame))

    def read_frame(self, timeout=None):
        """
        Read one CAN data frame (``AA 11 ... 55``).  Heartbeats, send-acks and
        baud-set replies are skipped; other 16-byte ``AA..55`` frames are ignored.
        Returns ``(can_id, data, is_ext)`` or ``None`` on timeout.
        """
        prev = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            while True:
                # sync on 0xAA
                b = self.ser.read(1)
                if not b:
                    return None
                if b[0] != 0xAA:
                    continue
                rest = self.ser.read(15)
                if len(rest) < 15:
                    return None
                if rest[14] != 0x55:
                    continue  # false sync, keep scanning
                cmd, info = rest[0], rest[1]
                if cmd != 0x11:        # only "receive success" frames
                    continue
                can_id = struct.unpack("<I", rest[2:6])[0]
                dlc = info & 0x3F
                is_ext = bool(info & 0x40)
                return can_id, rest[6:6 + dlc], is_ext
        finally:
            self.ser.timeout = prev

    def close(self):
        self.ser.close()
