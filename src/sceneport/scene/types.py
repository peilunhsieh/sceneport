"""Scene data classes: Scene, Node, Mesh, BlendShape, Material, ..."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Literal

import numpy as np
from numpy.typing import NDArray

from ..math import euler_matrix, quaternion_matrix, scale_matrix, translation_matrix

if TYPE_CHECKING:
    from .animation import Animation
    from .material import Material

# USD primvar interpolation names; see interop/fbx/layers.py for the FBX mapping modes.
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


def _vec3_array(value) -> NDArray[np.float64] | None:
    return None if value is None else np.asarray(value, dtype=np.float64).reshape(-1, 3)


@dataclass
class Inbetween:
    """An intermediate shape reached at `weight` (usually between 0 and 1), same vertices as its BlendShape."""

    weight: float
    offsets: NDArray[np.float64]  # (K, 3)
    normal_offsets: NDArray[np.float64] | None = None

    def __post_init__(self):
        self.offsets = _vec3_array(self.offsets)
        self.normal_offsets = _vec3_array(self.normal_offsets)


@dataclass
class BlendShape:
    """Per-vertex position offsets (shape key / morph target).

    Offsets apply to the vertices in `indices`, or to every vertex (in order) if
    `indices` is None. At weight w the offsets are interpolated linearly through
    (0, zero), the in-betweens, and (1, offsets); weights outside that range
    extrapolate the nearest segment.
    """

    name: str
    offsets: NDArray[np.float64]  # (K, 3)
    indices: NDArray[np.int32] | None = None  # (K,) vertex indices; None = all vertices
    normal_offsets: NDArray[np.float64] | None = None  # (K, 3)
    weight: float = 0.0  # default weight, 1.0 = fully applied
    inbetweens: list[Inbetween] = field(default_factory=list)

    def __post_init__(self):
        self.offsets = _vec3_array(self.offsets)
        self.normal_offsets = _vec3_array(self.normal_offsets)
        if self.indices is not None:
            self.indices = np.asarray(self.indices, dtype=np.int32)

    def offsets_at(self, weight: float | NDArray) -> NDArray[np.float64]:
        """Offsets at `weight`, interpolated through the in-betweens.

        A scalar weight gives (K, 3); an array of F weights gives (F, K, 3).
        """
        weight = np.asarray(weight, dtype=np.float64)
        if not self.inbetweens:
            return weight[..., None, None] * self.offsets
        keys = [(0.0, np.zeros_like(self.offsets)), *((i.weight, i.offsets) for i in self.inbetweens), (1.0, self.offsets)]
        keys.sort(key=lambda k: k[0])
        key_weights = np.array([k[0] for k in keys])
        key_offsets = np.stack([k[1] for k in keys])  # (L, K, 3)
        # Segment containing each weight, or the first/last one to extrapolate.
        i = np.clip(np.searchsorted(key_weights, weight) - 1, 0, len(keys) - 2)
        t = (weight - key_weights[i]) / (key_weights[i + 1] - key_weights[i])
        return key_offsets[i] + (key_offsets[i + 1] - key_offsets[i]) * t[..., None, None]

    def dense_offsets(self, n_vertices: int, weight: float = 1.0) -> NDArray[np.float64]:
        """(n_vertices, 3) offsets at `weight`, zero for unaffected vertices."""
        offsets = self.offsets_at(weight)
        if self.indices is None:
            return offsets.copy()
        out = np.zeros((n_vertices, 3))
        out[self.indices] = offsets
        return out


@dataclass
class Mesh:
    """Polygon mesh in USD layout: face i uses the next `face_vertex_counts[i]` entries of `face_vertex_indices`."""

    name: str
    vertices: NDArray[np.float64]  # (N, 3)
    face_vertex_counts: NDArray[np.int32]  # (F,)
    face_vertex_indices: NDArray[np.int32]  # (C,), C = sum of counts = number of corners
    normals: Primvar | None = None
    uvs: dict[str, Primvar] = field(default_factory=dict)
    blend_shapes: list[BlendShape] = field(default_factory=list)
    skin: Skin | None = None
    # (F,) material slot of each face, an index into the node's `materials`; None = every face uses slot 0.
    material_indices: NDArray[np.int32] | None = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.face_vertex_counts = np.asarray(self.face_vertex_counts, dtype=np.int32)
        self.face_vertex_indices = np.asarray(self.face_vertex_indices, dtype=np.int32)
        if self.material_indices is not None:
            self.material_indices = np.asarray(self.material_indices, dtype=np.int32)

    def deformed_vertices(
        self, weights: dict[str, float] | NDArray | None = None, vertices: NDArray | None = None
    ) -> NDArray[np.float64]:
        """Vertices with blend shapes applied.

        `weights` is one of:
          - None: every shape at its own `weight`
          - a dict {shape name: weight}; shapes left out use their own `weight`
          - an (S,) array, one weight per shape in `blend_shapes` order -> (N, 3)
          - an (F, S) array, one row per frame -> (F, N, 3)
        Weights are not clamped; in-betweens are interpolated through.
        `vertices` replaces the rest positions: (N, 3), or (F, N, 3) per frame (e.g.
        animated points, see PointSamples), which gives (F, N, 3).
        """
        w = self._weight_array(weights)
        frames = np.atleast_2d(w)
        base = self.vertices if vertices is None else np.asarray(vertices, dtype=np.float64)
        if base.shape[-2:] != self.vertices.shape or base.ndim not in (2, 3):
            raise ValueError(f"vertices must be (N, 3) or (F, N, 3) with N = {self.n_vertices}, got {base.shape}")
        if base.ndim == 3:
            if len(frames) not in (1, len(base)):
                raise ValueError(f"frame counts differ: {len(base)} vertex frames, {len(frames)} weight rows")
            frames = np.broadcast_to(frames, (len(base), frames.shape[1]))
            out = base.copy()
        else:
            out = np.repeat(base[None], len(frames), axis=0)
        for s, shape in enumerate(self.blend_shapes):
            column = frames[:, s]
            if column.any():
                target = slice(None) if shape.indices is None else shape.indices
                out[:, target] += shape.offsets_at(column)
        return out[0] if w.ndim == 1 and base.ndim == 2 else out

    def _weight_array(self, weights: dict[str, float] | NDArray | None) -> NDArray[np.float64]:
        defaults = np.array([shape.weight for shape in self.blend_shapes], dtype=np.float64)
        if weights is None:
            return defaults
        if isinstance(weights, dict):
            index = {shape.name: s for s, shape in enumerate(self.blend_shapes)}
            unknown = set(weights) - set(index)
            if unknown:
                raise KeyError(f"mesh {self.name!r} has no blend shapes named {sorted(unknown)}")
            for name, value in weights.items():
                defaults[index[name]] = value
            return defaults
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim not in (1, 2) or w.shape[-1] != len(self.blend_shapes):
            raise ValueError(
                f"weights must have shape (S,) or (F, S) with S = {len(self.blend_shapes)}, got {w.shape}"
            )
        return w

    def skinned_vertices(
        self, joint_world: NDArray, weights: dict[str, float] | NDArray | None = None, vertices: NDArray | None = None
    ) -> NDArray[np.float64]:
        """World-space vertices: blend shapes (see `deformed_vertices`), then linear blend skinning.

        `joint_world` holds the world matrices of `skin.joints`, (J, 4, 4) for one pose
        or (F, J, 4, 4) for F frames; `Skin.joint_world(scene)` gives the current pose.
        Blend shape `weights` and replacement rest `vertices` may also have a frame axis.
        Returns (N, 3) or (F, N, 3).
        """
        if self.skin is None:
            raise ValueError(f"mesh {self.name!r} has no skin")
        skin = self.skin
        joint_world = np.asarray(joint_world, dtype=np.float64)
        if joint_world.shape[-3:] != (len(skin.joints), 4, 4) or joint_world.ndim not in (3, 4):
            raise ValueError(f"joint_world must be (J, 4, 4) or (F, J, 4, 4) with J = {len(skin.joints)}, got {joint_world.shape}")
        base = self.deformed_vertices(weights, vertices)
        single = joint_world.ndim == 3 and base.ndim == 2

        matrices = joint_world if joint_world.ndim == 4 else joint_world[None]
        matrices = matrices @ skin.inverse_bind @ skin.geom_bind  # (F or 1, J, 4, 4)
        points = base if base.ndim == 3 else base[None]  # (F or 1, N, 3)
        frames = max(len(matrices), len(points))
        if len(matrices) not in (1, frames) or len(points) not in (1, frames):
            raise ValueError(f"frame counts differ: {len(matrices)} joint poses, {len(points)} blend shape rows")
        matrices = np.broadcast_to(matrices, (frames,) + matrices.shape[1:])
        homogeneous = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)
        homogeneous = np.broadcast_to(homogeneous, (frames,) + homogeneous.shape[1:])  # (F, N, 4)

        out = np.zeros((frames, self.n_vertices, 3))
        for k in range(skin.joint_indices.shape[1]):  # one influence slot at a time keeps memory at (F, N, 3, 4)
            per_vertex = matrices[:, skin.joint_indices[:, k], :3, :]
            out += skin.joint_weights[None, :, k, None] * np.einsum("fnij,fnj->fni", per_vertex, homogeneous)
        return out[0] if single else out

    def blend_shape_matrix(self, dtype=np.float64) -> NDArray:
        """(S, N, 3) dense basis: row s is blend shape s's offsets at weight 1, zero where it doesn't reach.

        With it, deformed vertices are `vertices + einsum("fs,snk->fnk", weights, matrix)`.
        Shapes with in-betweens aren't linear in their weight, so they are rejected.
        """
        nonlinear = [shape.name for shape in self.blend_shapes if shape.inbetweens]
        if nonlinear:
            raise ValueError(f"blend shapes with in-betweens have no single basis vector: {nonlinear}")
        out = np.zeros((len(self.blend_shapes), self.n_vertices, 3), dtype=dtype)
        for s, shape in enumerate(self.blend_shapes):
            out[s, slice(None) if shape.indices is None else shape.indices] = shape.offsets
        return out

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
        for shape in self.blend_shapes:
            label = f"mesh {self.name!r}: blend shape {shape.name!r}"
            count = self.n_vertices if shape.indices is None else len(shape.indices)
            if shape.indices is not None and len(shape.indices) and (
                shape.indices.min() < 0 or shape.indices.max() >= self.n_vertices
            ):
                raise ValueError(f"{label}: vertex index out of range")
            if shape.indices is not None and len(np.unique(shape.indices)) != len(shape.indices):
                raise ValueError(f"{label}: duplicate vertex indices")
            arrays = [("offsets", shape.offsets), ("normal_offsets", shape.normal_offsets)]
            for inbetween in shape.inbetweens:
                if inbetween.weight in (0.0, 1.0):
                    raise ValueError(f"{label}: in-between weight must not be 0 or 1")
                arrays += [("in-between offsets", inbetween.offsets), ("in-between normal_offsets", inbetween.normal_offsets)]
            for what, array in arrays:
                if array is not None and len(array) != count:
                    raise ValueError(f"{label}: {what} has {len(array)} rows, expected {count}")
        if self.skin is not None:
            skin, label = self.skin, f"mesh {self.name!r}: skin"
            if skin.joint_indices.shape != skin.joint_weights.shape or len(skin.joint_indices) != self.n_vertices:
                raise ValueError(
                    f"{label}: joint_indices {skin.joint_indices.shape} and joint_weights {skin.joint_weights.shape} "
                    f"must both be (N, K) with N = {self.n_vertices}"
                )
            if skin.joint_indices.size and (skin.joint_indices.min() < 0 or skin.joint_indices.max() >= len(skin.joints)):
                raise ValueError(f"{label}: joint index out of range")
            if skin.inverse_bind.shape != (len(skin.joints), 4, 4):
                raise ValueError(f"{label}: inverse_bind must be (J, 4, 4) with J = {len(skin.joints)}")
        if self.material_indices is not None:
            if self.material_indices.shape != (self.n_faces,):
                raise ValueError(f"mesh {self.name!r}: material_indices must be (F,) with F = {self.n_faces}")
            if self.n_faces and self.material_indices.min() < 0:
                raise ValueError(f"mesh {self.name!r}: negative material index")


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
    rotation_order: int = 0  # FBX enum, 0 = XYZ; see sceneport.math.ROTATION_ORDERS
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

    def matrix(self, translation=None, rotation=None, scale=None, orientation=None) -> NDArray[np.float64]:
        """4x4 local matrix (column vectors).

        The keyword arguments replace the stored translation / Euler rotation / scale,
        e.g. with animated values; an (F, 3) array gives (F, 4, 4) matrices. A unit
        quaternion `orientation` (w, x, y, z), possibly (F, 4), replaces the Euler
        rotation term; pre/post rotation still apply.
        """
        translation = self.translation if translation is None else translation
        scale = self.scale if scale is None else scale
        if orientation is not None:
            rotate = quaternion_matrix(orientation)
        else:
            rotate = euler_matrix(self.rotation if rotation is None else rotation, self.rotation_order)
        rp, sp = translation_matrix(self.rotation_pivot), translation_matrix(self.scaling_pivot)
        before = translation_matrix(self.rotation_offset) @ rp @ euler_matrix(self.pre_rotation)
        after = np.linalg.inv(euler_matrix(self.post_rotation)) @ np.linalg.inv(rp) @ translation_matrix(self.scaling_offset) @ sp
        return translation_matrix(translation) @ before @ rotate @ after @ scale_matrix(scale) @ np.linalg.inv(sp)

    def geometric_matrix(self) -> NDArray[np.float64]:
        return (
            translation_matrix(self.geometric_translation)
            @ euler_matrix(self.geometric_rotation)
            @ scale_matrix(self.geometric_scale)
        )


@dataclass(eq=False)
class Node:
    """A scene node (FBX Model). Several nodes may share one Mesh (instancing).

    Joints are ordinary nodes with `joint=True`; other nodes (props, helpers) can be
    their children. Nodes compare by identity, so they can be used as dict keys.
    """

    name: str
    transform: Transform = field(default_factory=Transform)
    mesh: Mesh | None = None
    children: list[Node] = field(default_factory=list)
    joint: bool = False
    # Material slots of the node's mesh (FBX and Blender keep them per object); see Mesh.material_indices.
    materials: list[Material | None] = field(default_factory=list)
    # Format-specific data kept for round trips, keyed by format, e.g. extras["fbx"].
    extras: dict[str, Any] = field(default_factory=dict)

    def add(self, child: Node) -> Node:
        self.children.append(child)
        return child


@dataclass
class Skin:
    """Linear blend skinning of a mesh to joint Nodes (the glTF layout).

    `joint_indices` / `joint_weights` are (N, K): each vertex's K influences, as
    indices into `joints` and their weights (zero weight = unused slot).
    `inverse_bind[j]` is the inverse of joints[j]'s world matrix at bind time and
    `geom_bind` the mesh's world matrix at bind time. A vertex v skins to world as

        sum_k  w_k * joint_world[j_k] @ inverse_bind[j_k] @ geom_bind @ v
    """

    joints: list[Node]
    joint_indices: NDArray[np.int32]  # (N, K)
    joint_weights: NDArray[np.float64]  # (N, K)
    inverse_bind: NDArray[np.float64]  # (J, 4, 4)
    geom_bind: NDArray[np.float64] = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        self.joint_indices = np.asarray(self.joint_indices, dtype=np.int32)
        self.joint_weights = np.asarray(self.joint_weights, dtype=np.float64)
        if self.joint_indices.ndim == 1:  # one influence per vertex
            self.joint_indices = self.joint_indices[:, None]
        if self.joint_weights.ndim == 1:
            self.joint_weights = self.joint_weights[:, None]
        self.inverse_bind = np.asarray(self.inverse_bind, dtype=np.float64).reshape(-1, 4, 4)
        self.geom_bind = np.asarray(self.geom_bind, dtype=np.float64).reshape(4, 4)

    def joint_world(self, scene: Scene) -> NDArray[np.float64]:
        """(J, 4, 4) current world matrices of `joints`, from the node transforms in `scene`."""
        world = scene.world_matrices()
        return np.stack([world[joint] for joint in self.joints])

    def bind_world(self) -> NDArray[np.float64]:
        """(J, 4, 4) world matrices of `joints` at bind time."""
        return np.linalg.inv(self.inverse_bind)

    def dense_weights(self) -> NDArray[np.float64]:
        """(N, J) weight of every joint on every vertex (the SMPL-style layout)."""
        out = np.zeros((len(self.joint_indices), len(self.joints)))
        rows = np.repeat(np.arange(len(self.joint_indices)), self.joint_indices.shape[1])
        np.add.at(out, (rows, self.joint_indices.ravel()), self.joint_weights.ravel())
        return out


AXES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


@dataclass
class Scene:
    """Root nodes plus the coordinate system the data is expressed in.

    Units and axes describe the data; nothing is converted on load or save.
    Axes are signed names such as "+Y", with FBX's meaning of up, front and
    right (FBX UpAxis, FrontAxis and CoordAxis).
    """

    nodes: list[Node] = field(default_factory=list)  # root nodes
    meters_per_unit: float = 1.0
    up_axis: str = "+Y"
    front_axis: str = "+Z"
    right_axis: str = "+X"
    # Format-specific data kept for round trips, keyed by format, e.g. extras["fbx"].
    extras: dict[str, Any] = field(default_factory=dict)
    animations: list[Animation] = field(default_factory=list)  # clips, see scene/animation.py
    materials: list[Material] = field(default_factory=list)  # every material, bound or not; see scene/material.py

    def __post_init__(self):
        for name in ("up_axis", "front_axis", "right_axis"):
            if getattr(self, name) not in AXES:
                raise ValueError(f"{name} must be one of {AXES}, got {getattr(self, name)!r}")

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

    def world_matrices(self) -> dict[Node, NDArray[np.float64]]:
        """World matrix of every node, keyed by node."""
        return dict(self.walk())

    def parents(self) -> dict[Node, Node | None]:
        """Parent of every node (None for root nodes)."""
        out: dict[Node, Node | None] = {}
        stack = [(n, None) for n in self.nodes]
        while stack:
            node, parent = stack.pop()
            out[node] = parent
            stack.extend((c, node) for c in node.children)
        return out

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
