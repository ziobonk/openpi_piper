"""PICO 位姿到机械臂位姿的纯坐标转换模块。

约定
----
1. 默认四元数顺序为 ``(x, y, z, w)``。
2. 旋转矩阵是主动旋转，表示“设备/工具局部坐标 -> 各自基坐标”。
3. 基坐标关系为 ``x_arm=-y_pico, y_arm=x_pico, z_arm=z_pico``。
4. 图示标定姿态下，工具轴满足
   ``x_tool=-z_arm, y_tool=y_arm, z_tool=x_arm``。

该文件不包含绘图或机械臂通信代码，可直接复制到控制工程中使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "C_ARM_FROM_PICO",
    "R_ARM_REFERENCE",
    "Q_PICO_REFERENCE_XYZW",
    "PICO_FROM_TCP_ARM_REFERENCE_METERS",
    "PicoToArmConverter",
    "pico_pose_to_arm_pose",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "rotation_xyz_degrees",
]


# PICO 基坐标中的向量改写成 arm 基坐标分量：
# [x_a, y_a, z_a] = [-y_p, x_p, z_p]
C_ARM_FROM_PICO = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# 图中机械臂末端参考姿态；三列依次是工具 +X、+Y、+Z 轴。
R_ARM_REFERENCE = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
)

# 两设备物理姿态相同时记录的 PICO 参考四元数 (x, y, z, w)。
Q_PICO_REFERENCE_XYZW = np.array([-0.79374, -0.01137, -0.00801, 0.60810])

# 标定参考姿态下，“从 TCP 指向 PICO 跟踪原点”的向量，使用 arm 基坐标表示。
# 测量值：arm X=-160 mm，arm Y=0 mm，arm Z=+77 mm。
PICO_FROM_TCP_ARM_REFERENCE_METERS = np.array([-0.160, 0.0, 0.077])


def _normalize_quaternion(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=float)
    if q.shape != (4,):
        raise ValueError("四元数必须是长度为 4 的 (x, y, z, w)")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("四元数长度不能为 0")
    return q / norm


def quaternion_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """把 ``(x, y, z, w)`` 四元数转换为 3×3 主动旋转矩阵。"""
    x, y, z, w = _normalize_quaternion(q_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """把 3×3 旋转矩阵转换为 ``(x, y, z, w)``，并固定为 ``w>=0``。"""
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("旋转矩阵必须是 3x3")

    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            w = (R[1, 0] - R[0, 1]) / s

    q = _normalize_quaternion(np.array([x, y, z, w]))
    return -q if q[3] < 0.0 else q


@dataclass(frozen=True)
class PicoToArmConverter:
    """保存一次标定参数，并高效转换连续的 PICO 位姿。"""

    pico_reference_position: np.ndarray
    arm_reference_position: np.ndarray
    pico_reference_quaternion_xyzw: np.ndarray = field(
        default_factory=lambda: Q_PICO_REFERENCE_XYZW.copy()
    )
    arm_reference_rotation: np.ndarray = field(
        default_factory=lambda: R_ARM_REFERENCE.copy()
    )
    position_scale: float = 1.0
    pico_from_tcp_arm_reference: np.ndarray | None = None

    def __post_init__(self) -> None:
        p_p = np.asarray(self.pico_reference_position, dtype=float)
        p_a = np.asarray(self.arm_reference_position, dtype=float)
        if p_p.shape != (3,) or p_a.shape != (3,):
            raise ValueError("参考位置必须是长度为 3 的向量")
        if self.position_scale <= 0.0:
            raise ValueError("position_scale 必须大于 0")

        if self.pico_from_tcp_arm_reference is None:
            # 常量以米保存；乘 position_scale 后与 arm 输出位置使用相同单位。
            pico_from_tcp = (
                self.position_scale * PICO_FROM_TCP_ARM_REFERENCE_METERS
            )
        else:
            pico_from_tcp = np.asarray(
                self.pico_from_tcp_arm_reference, dtype=float
            )
        if pico_from_tcp.shape != (3,):
            raise ValueError(
                "pico_from_tcp_arm_reference 必须是长度为 3 的向量"
            )

        q_p_ref = _normalize_quaternion(self.pico_reference_quaternion_xyzw)
        R_p_ref = quaternion_to_matrix(q_p_ref)
        R_a_ref = np.asarray(self.arm_reference_rotation, dtype=float)
        if R_a_ref.shape != (3, 3):
            raise ValueError("arm_reference_rotation 必须是 3x3")

        # 完整关系：R_a = C * R_p * B。
        # 由标定姿态 R_a_ref = C * R_p_ref * B 解出固定工具轴对齐矩阵 B。
        tool_alignment = R_p_ref.T @ C_ARM_FROM_PICO.T @ R_a_ref

        object.__setattr__(self, "pico_reference_position", p_p)
        object.__setattr__(self, "arm_reference_position", p_a)
        object.__setattr__(self, "pico_reference_quaternion_xyzw", q_p_ref)
        object.__setattr__(self, "arm_reference_rotation", R_a_ref)
        object.__setattr__(self, "tool_alignment", tool_alignment)
        object.__setattr__(
            self, "pico_from_tcp_arm_reference", pico_from_tcp
        )

    def _convert_position_with_rotation(
        self, pico_position: np.ndarray, R_arm: np.ndarray
    ) -> np.ndarray:
        position = np.asarray(pico_position, dtype=float)
        if position.shape != (3,):
            raise ValueError("位置必须是长度为 3 的向量")

        delta_pico = position - self.pico_reference_position
        tracker_translation = self.position_scale * (
            C_ARM_FROM_PICO @ delta_pico
        )

        # 安装偏移会随刚体旋转。d_ref/d_current 都是“TCP -> PICO”的向量，
        # 因此 TCP 位移 = PICO 跟踪原点位移 - (d_current - d_ref)。
        arm_rotation_delta = R_arm @ self.arm_reference_rotation.T
        offset_reference = self.pico_from_tcp_arm_reference
        offset_current = arm_rotation_delta @ offset_reference
        lever_arm_correction = offset_current - offset_reference

        return (
            self.arm_reference_position
            + tracker_translation
            - lever_arm_correction
        )

    def convert_position(
        self,
        pico_position: np.ndarray,
        pico_quaternion_xyzw: np.ndarray | None = None,
    ) -> np.ndarray:
        """把 PICO 跟踪原点位置转换为机械臂 TCP 位置。

        若没有提供当前四元数，则按参考姿态计算，此时没有旋转杠杆臂补偿。
        处理实时位姿时应使用 :meth:`convert_pose`，或同时传入当前四元数。
        """
        q = (
            self.pico_reference_quaternion_xyzw
            if pico_quaternion_xyzw is None
            else pico_quaternion_xyzw
        )
        R_arm = self.convert_rotation_matrix(q)
        return self._convert_position_with_rotation(pico_position, R_arm)

    def convert_rotation_matrix(self, pico_quaternion_xyzw: np.ndarray) -> np.ndarray:
        """把 PICO 四元数转换为机械臂末端旋转矩阵。"""
        R_p = quaternion_to_matrix(pico_quaternion_xyzw)
        return C_ARM_FROM_PICO @ R_p @ self.tool_alignment

    def convert_quaternion(self, pico_quaternion_xyzw: np.ndarray) -> np.ndarray:
        """把 PICO 四元数转换为机械臂四元数 ``(x, y, z, w)``。"""
        return matrix_to_quaternion(
            self.convert_rotation_matrix(pico_quaternion_xyzw)
        )

    def convert_pose(
        self, pico_position: np.ndarray, pico_quaternion_xyzw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(arm_position, arm_quaternion_xyzw)``。"""
        R_arm = self.convert_rotation_matrix(pico_quaternion_xyzw)
        return (
            self._convert_position_with_rotation(pico_position, R_arm),
            matrix_to_quaternion(R_arm),
        )


def pico_pose_to_arm_pose(
    pico_position: np.ndarray,
    pico_quaternion: np.ndarray,
    *,
    pico_reference_position: np.ndarray = (0.0, 0.0, 0.0),
    arm_reference_position: np.ndarray = (0.0, 0.0, 0.0),
    position_scale: float = 1.0,
    quaternion_order: str = "xyzw",
    pico_from_tcp_arm_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """一次性把 PICO 位姿转换为机械臂位姿。

    返回 ``(arm_position, arm_quaternion_xyzw)``。实时循环建议预先创建
    :class:`PicoToArmConverter`，避免每帧重复计算标定矩阵。

    ``pico_from_tcp_arm_reference`` 是标定姿态下“TCP -> PICO 跟踪原点”
    的向量，单位必须与机械臂输出位置一致。传入 ``None`` 时自动使用本次
    测量值 ``(-0.160, 0, 0.077) m * position_scale``；传入零向量可关闭
    安装偏移补偿。
    """
    q = np.asarray(pico_quaternion, dtype=float)
    if q.shape != (4,):
        raise ValueError("pico_quaternion 必须是长度为 4 的向量")

    order = quaternion_order.lower()
    if order == "wxyz":
        q = q[[1, 2, 3, 0]]
    elif order != "xyzw":
        raise ValueError('quaternion_order 只能是 "xyzw" 或 "wxyz"')

    converter = PicoToArmConverter(
        pico_reference_position=np.asarray(pico_reference_position, dtype=float),
        arm_reference_position=np.asarray(arm_reference_position, dtype=float),
        position_scale=position_scale,
        pico_from_tcp_arm_reference=pico_from_tcp_arm_reference,
    )
    return converter.convert_pose(np.asarray(pico_position, dtype=float), q)


def rotation_xyz_degrees(rx: float, ry: float, rz: float) -> np.ndarray:
    """构造 ``Rz(rz) @ Ry(ry) @ Rx(rx)``，角度单位为度。"""
    ax, ay, az = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx
