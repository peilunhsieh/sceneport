"""Transform math, FBX time ticks (46186158000 per second), and opt-in axis/unit conversion."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

TICKS_PER_SECOND = 46186158000

# FBX RotationOrder enum -> axes in the order they are applied.
ROTATION_ORDERS = ("XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX")


def translation_matrix(t: ArrayLike) -> NDArray[np.float64]:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def scale_matrix(s: ArrayLike) -> NDArray[np.float64]:
    return np.diag([*np.asarray(s, dtype=float), 1.0])


def _axis_rotation(axis: str, degrees: float) -> NDArray[np.float64]:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    m = np.eye(4)
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def euler_matrix(degrees: ArrayLike, order: int | str = 0) -> NDArray[np.float64]:
    """Rotation matrix for FBX Euler angles (x, y, z) in degrees.

    `order` is the FBX RotationOrder enum (0 = XYZ) or its name. "XYZ" rotates about X
    first, so the matrix is Rz @ Ry @ Rx (column vectors).
    """
    axes = ROTATION_ORDERS[order] if isinstance(order, int) else order
    angles = dict(zip("XYZ", np.asarray(degrees, dtype=float)))
    m = np.eye(4)
    for axis in axes:
        m = _axis_rotation(axis, angles[axis]) @ m
    return m


def decompose_matrix(m: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split a 4x4 matrix into (translation, XYZ Euler degrees, scale). Shear is lost.

    A mirrored matrix (negative determinant) gets a negative X scale.
    """
    m = np.asarray(m, dtype=float)
    translation = m[:3, 3].copy()
    basis = m[:3, :3]
    scale = np.linalg.norm(basis, axis=0)
    if np.linalg.det(basis) < 0:
        scale[0] = -scale[0]
    r = basis / np.where(scale == 0, 1, scale)
    # r = Rz @ Ry @ Rx, so r[2, 0] = -sin(y)
    cy = np.hypot(r[0, 0], r[1, 0])
    y = np.arctan2(-r[2, 0], cy)
    if cy > 1e-9:
        x = np.arctan2(r[2, 1], r[2, 2])
        z = np.arctan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: only x + z (or x - z) is defined; put it all in x
        x = np.arctan2(-r[1, 2], r[1, 1])
        z = 0.0
    return translation, np.degrees([x, y, z]), scale


def seconds_to_ticks(seconds: float) -> int:
    return round(seconds * TICKS_PER_SECOND)


def ticks_to_seconds(ticks: int) -> float:
    return ticks / TICKS_PER_SECOND
