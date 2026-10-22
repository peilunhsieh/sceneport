"""Rotation helpers in sceneport.math."""

import numpy as np

from sceneport.math import quaternion_matrix, quaternion_multiply


def test_quaternion_multiply_composes_rotations():
    q = np.random.default_rng(4).normal(size=(2, 5, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    product = quaternion_multiply(q[0], q[1])  # rotate by q[1], then by q[0]
    np.testing.assert_allclose(quaternion_matrix(product), quaternion_matrix(q[0]) @ quaternion_matrix(q[1]), atol=1e-12)
