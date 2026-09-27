"""推理动作的 PICO/TCP 相对坐标转换（可单文件复制使用）。

动作格式：``[x, y, z, rx, ry, rz, gripper, ...]``。

* 位置 ``x, y, z`` 的单位是米；
* 姿态 ``rx, ry, rz`` 是旋转向量，单位是弧度；
* 第 7 列及后续数据（例如夹爪宽度）会原样保留；
* 支持 ``(D,)``、``(T, D)``、``(B, T, D)`` 等形状，只要求 ``D >= 6``。

本文件已经内置标定矩阵和安装偏移，不依赖项目中的其他 Python 文件，
只需要安装 NumPy 和 SciPy。

重要：如果模型使用 ``data/tcp_relative_action`` 作为训练标签，那么模型
输出本身已经是 TCP 相对动作，不要再次执行 PICO -> TCP 转换。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# B = ^PICO R_TCP：把 TCP 坐标系中的向量改写成 PICO 坐标分量。
# 该矩阵由 2026-09-25 的 TCP Y/Z 直线轨迹拟合并正交化得到。
PICO_FROM_TCP_ROTATION = np.array(
    [
        [0.010987775592181, 0.999629654664230, -0.024896230706806],
        [0.719339982922249, 0.009392321564406, 0.694594682721549],
        [0.694571276225093, -0.025540904672729, -0.718970377993102],
    ],
    dtype=np.float64,
)

# PICO -> TCP 的直接坐标旋转矩阵。
# 因为旋转矩阵的逆等于转置，所以这里直接使用 B.T。
TCP_FROM_PICO_ROTATION = PICO_FROM_TCP_ROTATION.T.copy()

# 从 TCP 原点指向 PICO 跟踪原点的向量，分量在 TCP 坐标系中表示。
# 这个偏移用于补偿设备旋转时，因为两个原点不重合而产生的杠杆臂位移。
PICO_FROM_TCP_OFFSET_METERS = np.array([-0.106, 0.0, -0.170])


def _prepare(actions: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """检查输入，并临时整理成二维数组以便统一进行批量计算。"""
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] < 6:
        raise ValueError(
            "动作最后一维至少应包含 [x, y, z, rx, ry, rz]"
        )
    if not np.all(np.isfinite(array[..., :6])):
        raise ValueError("动作的位置或姿态中包含 NaN/无穷大")
    return array.reshape(-1, array.shape[-1]).copy(), array.shape


def pico_relative_actions_to_tcp_relative(
    pico_actions: np.ndarray,
) -> np.ndarray:
    """把 PICO 相对动作转换成 TCP 相对动作。

    输入和输出都是相对于各自首帧的位姿，而不是世界坐标中的绝对位姿。
    数学关系为 ``DeltaTcp = X^-1 @ DeltaPico @ X``。

    不能只对位置乘旋转矩阵：当动作包含旋转时，PICO 跟踪原点会绕 TCP
    原点运动，因此还必须加入安装偏移产生的杠杆臂补偿。
    """
    result, original_shape = _prepare(pico_actions)
    B = PICO_FROM_TCP_ROTATION
    d = PICO_FROM_TCP_OFFSET_METERS

    # 将输入的旋转向量转换为旋转矩阵，然后用固定安装旋转做共轭变换：
    # R_tcp = B.T @ R_pico @ B
    R_pico = Rotation.from_rotvec(result[:, 3:6]).as_matrix()
    R_tcp = B.T[None, :, :] @ R_pico @ B[None, :, :]

    # B.T @ p_pico：把相对位移从 PICO 坐标改写到 TCP 坐标。
    # (R_tcp - I) @ d：PICO/TCP 原点不重合导致的旋转杠杆臂位移。
    p_pico = result[:, :3]
    lever_arm = np.einsum("nij,j->ni", R_tcp - np.eye(3), d)
    result[:, :3] = (B.T @ p_pico.T).T - lever_arm
    result[:, 3:6] = Rotation.from_matrix(R_tcp).as_rotvec()

    # 第 7 列及后续列从未被修改，因此夹爪动作会保持原值。
    return result.reshape(original_shape)


def tcp_relative_actions_to_pico_relative(
    tcp_actions: np.ndarray,
) -> np.ndarray:
    """把 TCP 相对动作转换成 PICO 相对动作（上一个函数的逆变换）。

    如果模型使用 ``tcp_relative_action`` 训练，但下游程序要求接收 PICO
    相对动作，可以使用这个函数。若下游直接接收 TCP 动作，则无需转换。
    """
    result, original_shape = _prepare(tcp_actions)
    B = PICO_FROM_TCP_ROTATION
    d = PICO_FROM_TCP_OFFSET_METERS

    # 正向关系为 R_tcp = B.T @ R_pico @ B，因此逆变换为：
    # R_pico = B @ R_tcp @ B.T
    R_tcp = Rotation.from_rotvec(result[:, 3:6]).as_matrix()
    R_pico = B[None, :, :] @ R_tcp @ B.T[None, :, :]

    # 由 p_tcp = B.T @ p_pico - (R_tcp - I) @ d 解出：
    # p_pico = B @ (p_tcp + (R_tcp - I) @ d)
    p_tcp = result[:, :3]
    lever_arm = np.einsum("nij,j->ni", R_tcp - np.eye(3), d)
    result[:, :3] = (B @ (p_tcp + lever_arm).T).T
    result[:, 3:6] = Rotation.from_matrix(R_pico).as_rotvec()

    return result.reshape(original_shape)


def pico_to_tcp_euler_xyz_degrees() -> np.ndarray:
    """返回 PICO -> TCP 标定旋转的固定轴 XYZ 欧拉角，单位为度。

    这里使用 SciPy 的小写 ``"xyz"`` 约定，即依次绕固定 X、Y、Z 轴
    旋转。欧拉角与旋转顺序绑定，换成其他顺序会得到不同的数值。
    """
    return Rotation.from_matrix(TCP_FROM_PICO_ROTATION).as_euler(
        "xyz", degrees=True
    )


if __name__ == "__main__":
    angles = pico_to_tcp_euler_xyz_degrees()
    print("PICO -> TCP 固定轴 XYZ 欧拉角（度）:")
    print(f"X = {angles[0]:.9f}, Y = {angles[1]:.9f}, Z = {angles[2]:.9f}")
