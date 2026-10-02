"""Shared export helpers: meshes baked into world space (OBJ, PLY) and texture file paths."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ..scene.material import AssetPath
from ..scene.types import Mesh, Node, Primvar, Scene


def warn_dropped(scene: Scene, format_name: str, *, materials: bool = True) -> None:
    """Warn once about scene data a geometry-only format can't hold (`materials=False`: it holds materials)."""
    meshes = scene.meshes
    dropped = [
        what
        for what, present in (
            ("blend shapes", any(m.blend_shapes for m in meshes)),
            ("skins", any(m.skin is not None for m in meshes)),
            ("animation", bool(scene.animations)),  # includes per-vertex (point sample) animation
            ("environment lights", any(node.light is not None for node, _ in scene.walk())),
            ("materials", materials and (bool(scene.materials) or any(node.materials for node, _ in scene.walk()))),
        )
        if present
    ]
    if dropped:
        warnings.warn(f"{format_name} can't hold {', '.join(dropped)}; they are not written (meshes are baked as posed)")


def unit_scale(scene: Scene, meters_per_unit: float | None) -> float:
    """Factor from the scene's units to `meters_per_unit` (None: keep the scene's units)."""
    if meters_per_unit is None:
        return 1.0
    if meters_per_unit <= 0:
        raise ValueError(f"meters_per_unit must be positive, got {meters_per_unit}")
    return scene.meters_per_unit / meters_per_unit


def mesh_in_world(scene: Scene, scale: float = 1.0) -> Iterator[tuple[Node, Mesh, NDArray, Primvar | None]]:
    """(node, mesh, world-space vertices, world-space normals) for every mesh node.

    Blend shapes are applied at their default weights and skinned meshes are posed
    by the scene's current joint transforms. Normals are transformed by the node's
    (or, when skinned, the bind) matrix and renormalized. Vertices are multiplied
    by `scale` (e.g. unit_scale's factor).
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
        yield node, mesh, vertices * scale if scale != 1.0 else vertices, normals


def texture_path(
    asset: AssetPath,
    directory: str | os.PathLike | None,
    *,
    content: bytes | None = None,
    fallback_name: str = "texture.png",
    drop_missing: bool = False,
) -> str | None:
    """How a file written to `directory` should refer to the texture `asset`.

    Relative to `directory` when the texture is on disk; for an image only
    embedded in the source (FBX `content`), the bytes are written to
    `directory`/textures and that path is returned; otherwise the path as
    authored (None with `drop_missing`). Forward slashes throughout.
    """
    authored = asset.path.replace("\\", "/")
    if directory is None:
        return authored
    if asset.resolved and os.path.isfile(asset.resolved):
        try:
            return os.path.relpath(asset.resolved, directory).replace(os.sep, "/")
        except ValueError:  # another drive
            return asset.resolved.replace(os.sep, "/")
    if content:
        name = os.path.basename(authored) or fallback_name
        target = Path(directory) / "textures" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return f"textures/{name}"
    return None if drop_missing else authored
