"""Scene data classes: Scene, Node, Mesh, BlendShape, Material, ..."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np
from numpy.typing import NDArray

from ..document.properties import Properties70
from ..document.templates import default_global_settings
from ..math import euler_matrix, scale_matrix, translation_matrix

# USD primvar interpolation names; see scene/layers.py for the FBX mapping modes.
Interpolation = Literal["vertex", "faceVarying", "uniform", "constant"]


def _vec3(default: float) -> NDArray[np.float64]:
    return field(default_factory=lambda: np.full(3, default))


@dataclass
class Primvar:
    """Per-vertex, per-corner, per-face or constant data (normals, UVs, ...).

    `values` has one row per stored value. If `indices` is set, element i uses
    `values[indices[i]]` (FBX IndexToDirect); otherwise element i uses `values[i]`.
    """

    values: NDArray
    interpolation: Interpolation
    indices: NDArray[np.int32] | None = None

    def expanded(self) -> NDArray:
        """One row per element (vertex, corner, face or 1), with indices resolved."""
        return self.values if self.indices is None else self.values[self.indices]

    def __len__(self) -> int:
        """Number of elements this primvar covers."""
        return len(self.values) if self.indices is None else len(self.indices)


@dataclass
class Mesh:
    """Polygon mesh in USD layout: face i uses the next `face_vertex_counts[i]` entries of `face_vertex_indices`."""

    name: str
    vertices: NDArray[np.float64]  # (N, 3)
    face_vertex_counts: NDArray[np.int32]  # (F,)
    face_vertex_indices: NDArray[np.int32]  # (C,), C = sum of counts = number of corners
    normals: Primvar | None = None
    uvs: dict[str, Primvar] = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.face_vertex_counts = np.asarray(self.face_vertex_counts, dtype=np.int32)
        self.face_vertex_indices = np.asarray(self.face_vertex_indices, dtype=np.int32)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.face_vertex_counts)

    @property
    def n_corners(self) -> int:
        return len(self.face_vertex_indices)

    def face_offsets(self) -> NDArray[np.int64]:
        """Index of each face's first corner."""
        return np.cumsum(self.face_vertex_counts) - self.face_vertex_counts

    def element_count(self, interpolation: Interpolation) -> int:
        return {"vertex": self.n_vertices, "faceVarying": self.n_corners, "uniform": self.n_faces, "constant": 1}[
            interpolation
        ]

    def corner_values(self, primvar: Primvar) -> NDArray:
        """`primvar` expanded to one row per face corner."""
        values = primvar.expanded()
        match primvar.interpolation:
            case "vertex":
                return values[self.face_vertex_indices]
            case "faceVarying":
                return values
            case "uniform":
                return np.repeat(values, self.face_vertex_counts, axis=0)
            case "constant":
                return np.repeat(values[:1], self.n_corners, axis=0)
        raise ValueError(f"unknown interpolation {primvar.interpolation!r}")

    def triangulate_corners(self) -> NDArray[np.int64]:
        """(T, 3) corner indices of a fan triangulation (correct for convex faces)."""
        counts = self.face_vertex_counts.astype(np.int64)
        tris_per_face = np.maximum(counts - 2, 0)
        n_tris = int(tris_per_face.sum())
        first_tri = np.cumsum(tris_per_face) - tris_per_face
        start = np.repeat(self.face_offsets(), tris_per_face)
        k = np.arange(n_tris) - np.repeat(first_tri, tris_per_face) + 1
        return np.stack([start, start + k, start + k + 1], axis=1)

    def triangles(self) -> NDArray[np.int32]:
        """(T, 3) vertex indices of a fan triangulation."""
        return self.face_vertex_indices[self.triangulate_corners()]

    def validate(self) -> None:
        if (self.face_vertex_counts < 3).any():
            raise ValueError(f"mesh {self.name!r}: faces need at least 3 vertices")
        if self.face_vertex_counts.sum() != self.n_corners:
            raise ValueError(f"mesh {self.name!r}: face_vertex_counts don't add up to face_vertex_indices")
        if self.n_corners and (self.face_vertex_indices.min() < 0 or self.face_vertex_indices.max() >= self.n_vertices):
            raise ValueError(f"mesh {self.name!r}: vertex index out of range")
        primvars = {"normals": self.normals, **{f"uvs[{k!r}]": v for k, v in self.uvs.items()}}
        for label, pv in primvars.items():
            if pv is None:
                continue
            expected = self.element_count(pv.interpolation)
            if len(pv) != expected:
                raise ValueError(f"mesh {self.name!r}: {label} has {len(pv)} {pv.interpolation} elements, expected {expected}")


# Transform field -> FBX Model property. Rotations, pivots and offsets are Vector3D.
TRANSFORM_PROPERTIES = {
    "translation": "Lcl Translation",
    "rotation": "Lcl Rotation",
    "scale": "Lcl Scaling",
    "pre_rotation": "PreRotation",
    "post_rotation": "PostRotation",
    "rotation_pivot": "RotationPivot",
    "rotation_offset": "RotationOffset",
    "scaling_pivot": "ScalingPivot",
    "scaling_offset": "ScalingOffset",
    "geometric_translation": "GeometricTranslation",
    "geometric_rotation": "GeometricRotation",
    "geometric_scale": "GeometricScaling",
}
# Only honored when the Model has RotationActive = 1 (as in Blender and the FBX SDK).
ROTATION_ACTIVE_FIELDS = ("pre_rotation", "post_rotation", "rotation_order")


@dataclass
class Transform:
    """FBX node transform. Rotations are Euler angles in degrees.

    local = T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1
    """

    translation: NDArray[np.float64] = _vec3(0.0)
    rotation: NDArray[np.float64] = _vec3(0.0)
    scale: NDArray[np.float64] = _vec3(1.0)
    rotation_order: int = 0  # FBX enum, 0 = XYZ; see fbxio.math.ROTATION_ORDERS
    pre_rotation: NDArray[np.float64] = _vec3(0.0)
    post_rotation: NDArray[np.float64] = _vec3(0.0)
    rotation_pivot: NDArray[np.float64] = _vec3(0.0)
    rotation_offset: NDArray[np.float64] = _vec3(0.0)
    scaling_pivot: NDArray[np.float64] = _vec3(0.0)
    scaling_offset: NDArray[np.float64] = _vec3(0.0)
    # Applied to the node's own geometry only, not inherited by children.
    geometric_translation: NDArray[np.float64] = _vec3(0.0)
    geometric_rotation: NDArray[np.float64] = _vec3(0.0)
    geometric_scale: NDArray[np.float64] = _vec3(1.0)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name != "rotation_order":
                setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64).reshape(3))

    def matrix(self) -> NDArray[np.float64]:
        """4x4 local matrix (column vectors)."""
        rp, sp = translation_matrix(self.rotation_pivot), translation_matrix(self.scaling_pivot)
        return (
            translation_matrix(self.translation)
            @ translation_matrix(self.rotation_offset)
            @ rp
            @ euler_matrix(self.pre_rotation)
            @ euler_matrix(self.rotation, self.rotation_order)
            @ np.linalg.inv(euler_matrix(self.post_rotation))
            @ np.linalg.inv(rp)
            @ translation_matrix(self.scaling_offset)
            @ sp
            @ scale_matrix(self.scale)
            @ np.linalg.inv(sp)
        )

    def geometric_matrix(self) -> NDArray[np.float64]:
        return (
            translation_matrix(self.geometric_translation)
            @ euler_matrix(self.geometric_rotation)
            @ scale_matrix(self.geometric_scale)
        )


@dataclass
class Node:
    """A scene node (FBX Model). Several nodes may share one Mesh (instancing)."""

    name: str
    transform: Transform = field(default_factory=Transform)
    mesh: Mesh | None = None
    children: list[Node] = field(default_factory=list)

    def add(self, child: Node) -> Node:
        self.children.append(child)
        return child


@dataclass
class Scene:
    nodes: list[Node] = field(default_factory=list)  # root nodes
    # FBX GlobalSettings (axes, units, frame rate), stored as in the file.
    settings: Properties70 = field(default_factory=default_global_settings)

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def walk(self) -> Iterator[tuple[Node, NDArray[np.float64]]]:
        """Depth-first (node, world matrix) pairs. The world matrix excludes the geometric transform."""
        stack = [(n, np.eye(4)) for n in reversed(self.nodes)]
        while stack:
            node, parent = stack.pop()
            world = parent @ node.transform.matrix()
            yield node, world
            stack.extend((c, world) for c in reversed(node.children))

    def find(self, name: str) -> Node | None:
        return next((n for n, _ in self.walk() if n.name == name), None)

    @property
    def meshes(self) -> list[Mesh]:
        """Distinct meshes in the scene, in traversal order."""
        seen: dict[int, Mesh] = {}
        for node, _ in self.walk():
            if node.mesh is not None:
                seen.setdefault(id(node.mesh), node.mesh)
        return list(seen.values())

    @property
    def unit_scale(self) -> float:
        """Centimeters per file unit (UnitScaleFactor): 1.0 = cm, 100.0 = m."""
        return self.settings.get("UnitScaleFactor", 1.0)

    @property
    def up_axis(self) -> tuple[int, int]:
        """(axis, sign), axis 0/1/2 = X/Y/Z."""
        return self.settings.get("UpAxis", 1), self.settings.get("UpAxisSign", 1)

    @property
    def front_axis(self) -> tuple[int, int]:
        return self.settings.get("FrontAxis", 2), self.settings.get("FrontAxisSign", 1)

    @property
    def coord_axis(self) -> tuple[int, int]:
        return self.settings.get("CoordAxis", 0), self.settings.get("CoordAxisSign", 1)
