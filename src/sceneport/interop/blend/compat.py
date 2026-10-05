"""Scene rewrites applied before building Blender data (pure numpy: no bpy here).

- In-betweens are split into plain shapes (Blender shape keys have none), as for FBX
  (see fbx/compat.py).
- The scene moves to Blender's axes and units: Z up, -Y front, meters
  (reframe.change_basis), so the data is Z-up itself rather than turned by a
  rotation on the root objects.
- Geometric transforms (FBX's offset of a node's own geometry) are baked into the
  mesh: Blender has none. A mesh shared by nodes with different geometric
  transforms is copied per transform.

The input scene is never modified.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from ...scene.types import Inbetween, Mesh, Primvar, Scene
from ..fbx.compat import _split_inbetweens
from ..reframe import change_basis, copy_scene

UP, FRONT, RIGHT = "+Z", "-Y", "+X"  # Blender's axes, in FBX's meaning of front


def prepare(scene: Scene) -> Scene:
    """The scene as Blender holds it (a copy when anything changes)."""
    if any(shape.inbetweens for mesh in scene.meshes for shape in mesh.blend_shapes):
        scene = _split_inbetweens(scene)
    scene = change_basis(scene, up=UP, front=FRONT, right=RIGHT, meters_per_unit=1.0)
    return _bake_geometric(scene)


def _bake_geometric(scene: Scene) -> Scene:
    baked = [node for node, _ in scene.walk()
             if node.mesh is not None and not np.array_equal(node.transform.geometric_matrix(), np.eye(4))]  # fmt: skip
    if not baked:
        return scene
    transforms = {
        node: dataclasses.replace(node.transform, geometric_translation=np.zeros(3), geometric_rotation=np.zeros(3),
                                  geometric_scale=np.ones(3))
        for node in baked
    }  # fmt: skip
    copy = copy_scene(scene, {}, transforms, lambda clip: clip.channels)
    copies: dict[tuple[int, bytes], Mesh] = {}
    for (node, _), (new, _) in zip(scene.walk(), copy.walk()):
        if node in transforms:
            g = node.transform.geometric_matrix()
            key = (id(node.mesh), g.tobytes())
            if key not in copies:
                copies[key] = _transformed_mesh(new.mesh, g)
            new.mesh = copies[key]
    return copy


def _transformed_mesh(mesh: Mesh, matrix: NDArray) -> Mesh:
    """`mesh` with its points moved by `matrix` (normals by its inverse transpose, offsets by its linear part)."""
    linear = matrix[:3, :3]
    normal = np.linalg.inv(linear).T

    def directions(values):
        if values is None:
            return None
        out = values @ normal.T
        return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)

    def offsets(values):
        return None if values is None else values @ linear.T

    normals = mesh.normals
    if normals is not None:
        normals = Primvar(directions(normals.values), normals.interpolation, normals.indices)
    shapes = [
        dataclasses.replace(shape, offsets=offsets(shape.offsets), normal_offsets=offsets(shape.normal_offsets),
                            inbetweens=[Inbetween(i.weight, offsets(i.offsets), offsets(i.normal_offsets)) for i in shape.inbetweens])
        for shape in mesh.blend_shapes
    ]  # fmt: skip
    return dataclasses.replace(
        mesh, vertices=mesh.vertices @ linear.T + matrix[:3, 3], normals=normals, blend_shapes=shapes
    )

