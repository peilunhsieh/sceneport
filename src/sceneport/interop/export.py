"""Shared helpers for geometry-only formats (OBJ, PLY): meshes baked into world space."""

from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ..scene.types import Mesh, Node, Primvar, Scene


def warn_dropped(scene: Scene, format_name: str) -> None:
    """Warn once about scene data a geometry-only format can't hold."""
    meshes = scene.meshes
    dropped = [
        what
        for what, present in (
            ("blend shapes", any(m.blend_shapes for m in meshes)),
            ("skins", any(m.skin is not None for m in meshes)),
            ("animation", bool(scene.animations)),
        )
        if present
    ]
    if dropped:
        warnings.warn(f"{format_name} holds geometry only; {', '.join(dropped)} are not written (meshes are baked as posed)")


def mesh_in_world(scene: Scene) -> Iterator[tuple[Node, Mesh, NDArray, Primvar | None]]:
    """(node, mesh, world-space vertices, world-space normals) for every mesh node.

    Blend shapes are applied at their default weights and skinned meshes are posed
    by the scene's current joint transforms. Normals are transformed by the node's
    (or, when skinned, the bind) matrix and renormalized.
    """
    world = scene.world_matrices()
    for node, matrix in world.items():
        mesh = node.mesh
        if mesh is None:
            continue
        if mesh.skin is not None:
            vertices = mesh.skinned_vertices(mesh.skin.joint_world(scene))
            matrix = mesh.skin.geom_bind
        else:
            matrix = matrix @ node.transform.geometric_matrix()
            vertices = mesh.deformed_vertices() @ matrix[:3, :3].T + matrix[:3, 3]
        normals = None
        if mesh.normals is not None:
            inverse_transpose = np.linalg.inv(matrix[:3, :3]).T
            values = mesh.normals.values @ inverse_transpose.T
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-300)
            normals = Primvar(values, mesh.normals.interpolation, mesh.normals.indices)
        yield node, mesh, vertices, normals
