"""Transform math, FBX time ticks (46186158000 per second), and opt-in axis/unit conversion.

Matrices are 4x4 for column vectors. Functions accept batches: leading axes on the
inputs (e.g. (F, 3) Euler angles for F frames) give matrices with the same leading axes.
Quaternions are (w, x, y, z), as in USD and Blender.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

TICKS_PER_SECOND = 46186158000

# FBX RotationOrder enum -> axes in the order they are applied.
ROTATION_ORDERS = ("XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX")


def _identity(batch: tuple[int, ...]) -> NDArray[np.float64]:
    return np.broadcast_to(np.eye(4), batch + (4, 4)).copy()


def translation_matrix(t: ArrayLike) -> NDArray[np.float64]:
    t = np.asarray(t, dtype=np.float64)
    m = _identity(t.shape[:-1])
    m[..., :3, 3] = t
    return m


def scale_matrix(s: ArrayLike) -> NDArray[np.float64]:
    s = np.asarray(s, dtype=np.float64)
    m = _identity(s.shape[:-1])
    m[..., 0, 0], m[..., 1, 1], m[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    return m


def _axis_rotation(axis: str, degrees: NDArray) -> NDArray[np.float64]:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    i, j = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    m = _identity(np.shape(degrees))
    m[..., i, i], m[..., i, j], m[..., j, i], m[..., j, j] = c, -s, s, c
    return m


def _axes(order: int | str) -> str:
    return ROTATION_ORDERS[order] if isinstance(order, (int, np.integer)) else order


def euler_matrix(degrees: ArrayLike, order: int | str = 0) -> NDArray[np.float64]:
    """Rotation matrix for FBX Euler angles (x, y, z) in degrees.

    `order` is the FBX RotationOrder enum (0 = XYZ) or its name. "XYZ" rotates about X
    first, so the matrix is Rz @ Ry @ Rx (column vectors).
    """
    degrees = np.asarray(degrees, dtype=np.float64)
    m = _identity(degrees.shape[:-1])
    for axis in _axes(order):
        m = _axis_rotation(axis, degrees[..., "XYZ".index(axis)]) @ m
    return m


def euler_from_matrix(m: ArrayLike, order: int | str = 0) -> NDArray[np.float64]:
    """Euler angles (x, y, z) in degrees for `order` from a rotation matrix (scale removed).

    At gimbal lock only a sum/difference of two angles is defined; the third
    axis applied gets 0.
    """
    r = np.asarray(m, dtype=np.float64)[..., :3, :3]
    axes = _axes(order)
    i, j, k = ("XYZ".index(a) for a in axes)  # applied first, second, third
    parity = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
    # r = R_k(c) @ R_j(b) @ R_i(a); r[k, i] = -parity * sin(b)
    cos_b = np.hypot(r[..., i, i], r[..., j, i])
    b = np.arctan2(-parity * r[..., k, i], cos_b)
    locked = cos_b < 1e-9
    a = np.where(locked, np.arctan2(-parity * r[..., j, k], r[..., j, j]), np.arctan2(parity * r[..., k, j], r[..., k, k]))
    c = np.where(locked, 0.0, np.arctan2(parity * r[..., j, i], r[..., i, i]))
    out = np.empty(r.shape[:-2] + (3,))
    out[..., i], out[..., j], out[..., k] = a, b, c
    return np.degrees(out)


def quaternion_matrix(q: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrix for unit quaternions (w, x, y, z); they are normalized first."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = _identity(q.shape[:-1])
    m[..., 0, 0], m[..., 0, 1], m[..., 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)
    m[..., 1, 0], m[..., 1, 1], m[..., 1, 2] = 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)
    m[..., 2, 0], m[..., 2, 1], m[..., 2, 2] = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    return m


def quaternion_from_matrix(m: ArrayLike) -> NDArray[np.float64]:
    """Unit quaternion (w, x, y, z), with w >= 0, of a rotation matrix (scale removed)."""
    r = np.asarray(m, dtype=np.float64)[..., :3, :3]
    r00, r11, r22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    candidates = np.stack([  # 4 * each component squared; the largest gives the most accurate formula
        1 + r00 + r11 + r22, 1 + r00 - r11 - r22, 1 - r00 + r11 - r22, 1 - r00 - r11 + r22,
    ], axis=-1)  # fmt: skip
    best = candidates.argmax(axis=-1)
    s = np.sqrt(np.maximum(np.take_along_axis(candidates, best[..., None], -1)[..., 0], 1e-300)) * 2
    d21, d02, d10 = r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]
    s01, s02, s12 = r[..., 0, 1] + r[..., 1, 0], r[..., 0, 2] + r[..., 2, 0], r[..., 1, 2] + r[..., 2, 1]
    options = np.stack([
        np.stack([s / 4, d21 / s, d02 / s, d10 / s], -1),
        np.stack([d21 / s, s / 4, s01 / s, s02 / s], -1),
        np.stack([d02 / s, s01 / s, s / 4, s12 / s], -1),
        np.stack([d10 / s, s02 / s, s12 / s, s / 4], -1),
    ], axis=-2)  # fmt: skip
    q = np.take_along_axis(options, best[..., None, None], -2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def quaternion_continuity(q: ArrayLike) -> NDArray[np.float64]:
    """Flip signs along the first axis so consecutive quaternions stay on the same side
    (q and -q are the same rotation), avoiding long-way-round interpolation."""
    q = np.array(q, dtype=np.float64)
    for f in range(1, len(q)):
        if np.sum(q[f] * q[f - 1]) < 0:
            q[f] = -q[f]
    return q


def decompose_matrix(m: ArrayLike, order: int | str = 0) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split 4x4 matrices into (translation, Euler degrees for `order`, scale). Shear is lost.

    A mirrored matrix (negative determinant) gets a negative X scale.
    """
    m = np.asarray(m, dtype=np.float64)
    translation = m[..., :3, 3].copy()
    basis = m[..., :3, :3]
    scale = np.linalg.norm(basis, axis=-2)
    scale[..., 0] = np.where(np.linalg.det(basis) < 0, -scale[..., 0], scale[..., 0])
    rotation = basis / np.where(scale == 0, 1, scale)[..., None, :]
    return translation, euler_from_matrix(rotation, order), scale


def seconds_to_ticks(seconds: float) -> int:
    return round(seconds * TICKS_PER_SECOND)


def ticks_to_seconds(ticks: int) -> float:
    return ticks / TICKS_PER_SECOND
