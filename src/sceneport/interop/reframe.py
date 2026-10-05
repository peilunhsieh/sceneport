"""Rewrite a scene without changing what it looks like: shallow copies and node reframing.

Used where a format expresses the same scene differently (see fbx/compat.py, and
USD dome lights whose image pole is Z), and to move a scene to other axes and units
(change_basis, for Blender). The input scene is never modified: changed parts are
copies, unchanged meshes and arrays are shared.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from numpy.typing import NDArray

from ..math import ROTATION_ORDERS, TICKS_PER_SECOND, _axes, decompose_matrix
from ..scene.animation import Animation, Channel
from ..scene.types import Inbetween, Mesh, Node, Primvar, Scene, Transform

FLIP = -np.eye(4)
FLIP[3, 3] = 1.0  # -I on the linear part: a point reflection


def copy_scene(scene: Scene, meshes: dict[int, Mesh], transforms: dict[Node, Transform], channels) -> Scene:
    """A copy with new Nodes (transforms from `transforms` where given), meshes replaced per `meshes`
    (every skinned mesh is copied so its skin points at the new joints) and each clip's channels
    from `channels(clip)`, re-targeted to the copies."""
    node_map: dict[Node, Node] = {}

    def copy_node(node: Node) -> Node:
        new = dataclasses.replace(node, transform=transforms.get(node, node.transform), children=[])
        node_map[node] = new
        new.children = [copy_node(child) for child in node.children]
        return new

    roots = [copy_node(node) for node in scene.nodes]
    mesh_map: dict[int, Mesh] = {}
    for mesh in scene.meshes:
        new = meshes.get(id(mesh), mesh)
        if new.skin is not None:
            new = dataclasses.replace(new, skin=dataclasses.replace(new.skin, joints=[node_map[j] for j in new.skin.joints]))
        mesh_map[id(mesh)] = new
    for new in node_map.values():
        if new.mesh is not None:
            new.mesh = mesh_map.get(id(new.mesh), new.mesh)

    def retarget(channel: Channel) -> Channel:
        target = mesh_map[id(channel.target)] if channel.is_blend_shape else node_map[channel.target]  # nodes: transforms, cameras, lights
        return dataclasses.replace(channel, target=target)

    def retarget_points(samples):
        return dataclasses.replace(samples, mesh=mesh_map.get(id(samples.mesh), samples.mesh))

    animations = [
        dataclasses.replace(clip, channels=[retarget(c) for c in channels(clip)],
                            point_samples=[retarget_points(p) for p in clip.point_samples])
        for clip in scene.animations
    ]  # fmt: skip
    return dataclasses.replace(scene, nodes=roots, animations=animations)


def key_times(clip: Animation, channels: list[Channel]) -> NDArray[np.float64]:
    """Every frame of the clip plus the channels' own keys, merged on whole FBX ticks."""
    times = np.concatenate([clip.frame_times(), *(c.times for c in channels)])
    return np.unique(np.round(times * TICKS_PER_SECOND)) / TICKS_PER_SECOND


def reframe(scene: Scene, frames: dict[Node, NDArray]) -> Scene:
    """Turn the local frame of each node in `frames` by its matrix F without moving anything:
    world' = world @ F, so local' = F_parent^-1 @ local @ F_self, a mesh's geometric transform
    becomes F^-1 @ geometric and a joint's inverse bind F^-1 @ inverse bind. Animated nodes whose
    local matrix changes are baked per frame. Used to unmirror joints, to turn cameras
    between the -Z (USD) and +X (FBX) view axes and to bring a Z-pole dome light to +Y.
    """
    parents = scene.parents()
    inverse = {node: np.linalg.inv(f) for node, f in frames.items()}

    def local_frame(node: Node, local: NDArray) -> NDArray:
        parent = parents[node]
        return inverse.get(parent, np.eye(4)) @ local @ frames.get(node, np.eye(4))

    affected = {node for node in parents if node in frames or parents[node] in frames}
    transforms = {}
    for node in affected:
        geometric = inverse[node] @ node.transform.geometric_matrix() if node in frames and node.mesh is not None else None
        transforms[node] = _transform(local_frame(node, node.transform.matrix()), node.transform, geometric)

    meshes = {}
    for mesh in scene.meshes:
        if mesh.skin is not None and any(j in frames for j in mesh.skin.joints):
            turns = np.stack([inverse.get(j, np.eye(4)) for j in mesh.skin.joints])
            meshes[id(mesh)] = dataclasses.replace(mesh, skin=dataclasses.replace(mesh.skin, inverse_bind=turns @ mesh.skin.inverse_bind))

    def channels(clip: Animation) -> list[Channel]:
        animated = [n for n in affected if any(c.target is n and c.is_transform for c in clip.channels)]
        if not animated:
            return clip.channels
        out = [c for c in clip.channels if not c.is_transform or c.target not in animated]  # camera / light channels stay
        for node in animated:
            times = key_times(clip, [c for c in clip.channels if c.target is node and c.is_transform])
            local = local_frame(node, node.transform.matrix(**clip.node_values(times)[node]))
            translation, rotation, scale = decompose_matrix(local, node.transform.rotation_order)
            rotation = np.degrees(np.unwrap(np.radians(rotation), axis=0))
            for prop, values in (("translation", translation), ("rotation", rotation), ("scale", scale)):
                out += [Channel(node, prop, axis, times, values[:, axis]) for axis in range(3)]
        return out

    return copy_scene(scene, meshes, transforms, channels)


def _transform(local: NDArray, original: Transform, geometric: NDArray | None = None) -> Transform:
    """A plain translation / rotation / scale Transform for `local`, keeping the rotation order and
    the geometric part (or taking it from the `geometric` matrix)."""
    translation, rotation, scale = decompose_matrix(local, original.rotation_order)
    if geometric is None:
        g_translation, g_rotation, g_scale = original.geometric_translation, original.geometric_rotation, original.geometric_scale
    else:
        g_translation, g_rotation, g_scale = decompose_matrix(geometric)
    return Transform(
        translation=translation, rotation=rotation, scale=scale, rotation_order=original.rotation_order,
        geometric_translation=g_translation, geometric_rotation=g_rotation, geometric_scale=g_scale,
    )  # fmt: skip


def mirrored_joints(scene: Scene) -> set[Node]:
    """Joints whose rest world matrix is mirrored (negative determinant), e.g. a rig's right side built by
    mirroring; their children inherit the mirror. Blender can't make bones of them."""
    return {node for node, world in scene.walk() if node.joint and np.linalg.det(world[:3, :3]) < 0}


def unmirror(scene: Scene, joints: set[Node] | None = None) -> Scene:
    """`scene` with mirrored joints (all of them by default) re-expressed without negative scale: each joint's
    frame is turned by -I, so skinning and joint positions don't change; the joints' axes point the other way."""
    joints = mirrored_joints(scene) if joints is None else joints
    for mesh in scene.meshes:  # a skin that mirrors relative to its bind pose can't be expressed without a mirror
        if mesh.skin is None:
            continue
        bind_mirrored = np.linalg.det(mesh.skin.bind_world()[:, :3, :3]) < 0
        odd = [j.name for j, b in zip(mesh.skin.joints, bind_mirrored) if (j in joints) != bool(b)]
        if odd:
            warnings.warn(
                f"mesh {mesh.name!r}: joints {odd[:3]}{'...' if len(odd) > 3 else ''} are mirrored in their rest pose "
                "but not their bind pose (or the reverse); they can't be fully unmirrored, so Blender may still "
                "drop the animation"
            )
    return reframe(scene, {node: FLIP for node in joints}) if joints else scene


# -- axes and units ---------------------------------------------------------------------

CAMERA_DISTANCES = ("near", "far", "focus_distance")  # camera fields in scene units
LIGHT_DISTANCES = ("radius", "width", "height", "length")  # light fields in scene units


def axis_vector(axis: str) -> NDArray[np.float64]:
    """The unit vector of a signed axis name such as "-Y"."""
    v = np.zeros(3)
    v["XYZ".index(axis[1])] = 1.0 if axis[0] == "+" else -1.0
    return v


def basis_matrix(scene: Scene, up: str, front: str, right: str) -> NDArray[np.float64]:
    """The 3x3 signed permutation taking the scene's right / up / front axes to `right` / `up` / `front`."""
    source = np.column_stack([axis_vector(a) for a in (scene.right_axis, scene.up_axis, scene.front_axis)])
    target = np.column_stack([axis_vector(a) for a in (right, up, front)])
    for name, axes in (("scene", source), ("target", target)):
        if abs(np.linalg.det(axes)) != 1.0:
            raise ValueError(f"the {name} axes don't use X, Y and Z once each")
    return target @ source.T


def change_basis(scene: Scene, *, up: str, front: str, right: str, meters_per_unit: float) -> Scene:
    """`scene` expressed with other axes and units, looking the same.

    With M = s * C (C the axis change, s the unit factor), every node's local matrix
    becomes M @ local @ M^-1 and mesh points, offsets, inverse binds and camera / light
    distances follow; the result's world matrices are M @ world @ M^-1. Since C only
    swaps and flips axes, a plain translation / Euler / scale transform stays one (its
    rotation order permuted) and its animation channels map one-to-one, keys unchanged;
    nodes with pivots or pre / post rotations become plain transforms (baked per frame
    when animated). Cameras and lights keep looking down their node's -Z (and a dome's
    pole stays +Y): their frames are turned by C after the change. A mirroring C (one
    that changes handedness) also reverses each face's winding.
    """
    c = basis_matrix(scene, up, front, right)
    s = scene.meters_per_unit / meters_per_unit
    if np.array_equal(c, np.eye(3)) and s == 1.0:
        return dataclasses.replace(scene, up_axis=up, front_axis=front, right_axis=right, meters_per_unit=meters_per_unit)
    m, m_inv = np.eye(4), np.eye(4)
    m[:3, :3], m_inv[:3, :3] = s * c, c.T / s
    perm = np.abs(c).argmax(axis=0)  # source axis i -> target axis perm[i]
    sign = c[perm, np.arange(3)]
    det = round(np.linalg.det(c))

    def mapped(values: NDArray, factor: NDArray | float = 1.0) -> NDArray:
        out = np.empty_like(values)
        out[..., perm] = values * factor
        return out

    def order_of(transform: Transform) -> int:
        return ROTATION_ORDERS.index("".join("XYZ"[perm["XYZ".index(a)]] for a in _axes(transform.rotation_order)))

    def geometric(transform: Transform) -> tuple:
        if np.array_equal(transform.geometric_matrix(), np.eye(4)):
            return transform.geometric_translation, transform.geometric_rotation, transform.geometric_scale
        return decompose_matrix(m @ transform.geometric_matrix() @ m_inv)

    plain_nodes = {node for node, _ in scene.walk() if _plain(node.transform)}
    transforms = {}
    for node, _ in scene.walk():
        t = node.transform
        gt, gr, gs = geometric(t)
        if node in plain_nodes:
            translation, rotation, scale = mapped(t.translation, s * sign), mapped(t.rotation, sign * det), mapped(t.scale)
        else:
            translation, rotation, scale = decompose_matrix(m @ t.matrix() @ m_inv, order_of(t))
        transforms[node] = Transform(
            translation=translation, rotation=rotation, scale=scale, rotation_order=order_of(t),
            geometric_translation=gt, geometric_rotation=gr, geometric_scale=gs,
        )  # fmt: skip

    flip = det < 0
    meshes = {id(mesh): _mesh_in_basis(mesh, s * c, c, m, m_inv, flip) for mesh in scene.meshes}

    def channels(clip: Animation) -> list[Channel]:
        out = []
        for channel in clip.channels:
            node = channel.target
            if channel.is_transform and node not in plain_nodes:
                continue  # baked below
            if channel.property == "orientation" and channel.component > 0:
                axis = channel.component - 1
                out.append(dataclasses.replace(channel, component=1 + perm[axis], values=channel.values * det * sign[axis]))
            elif channel.property in ("translation", "rotation", "scale"):
                factor = {"translation": s * sign, "rotation": sign * det, "scale": np.ones(3)}[channel.property]
                out.append(dataclasses.replace(channel, component=perm[channel.component],
                                               values=channel.values * factor[channel.component]))  # fmt: skip
            elif channel.attribute in {("camera", f) for f in CAMERA_DISTANCES} | {("light", f) for f in LIGHT_DISTANCES}:
                out.append(dataclasses.replace(channel, values=channel.values * s))
            else:  # orientation w, blend shape weights, other camera / light fields
                out.append(channel)
        for node in {c.target for c in clip.channels if c.is_transform and c.target not in plain_nodes}:
            times = key_times(clip, [c for c in clip.channels if c.target is node and c.is_transform])
            local = m @ node.transform.matrix(**clip.node_values(times)[node]) @ m_inv
            translation, rotation, scale = decompose_matrix(local, transforms[node].rotation_order)
            rotation = np.degrees(np.unwrap(np.radians(rotation), axis=0))
            for prop, values in (("translation", translation), ("rotation", rotation), ("scale", scale)):
                out += [Channel(node, prop, axis, times, values[:, axis]) for axis in range(3)]
        return out

    converted = copy_scene(scene, meshes, transforms, channels)
    for clip in converted.animations:
        clip.point_samples = [dataclasses.replace(p, points=p.points @ (s * c).T) for p in clip.point_samples]
    converted = dataclasses.replace(converted, up_axis=up, front_axis=front, right_axis=right, meters_per_unit=meters_per_unit)
    for node, _ in converted.walk():
        if node.camera is not None and s != 1.0:
            node.camera = dataclasses.replace(node.camera, **{f: _scaled(getattr(node.camera, f), s) for f in CAMERA_DISTANCES})
        if node.light is not None and s != 1.0:
            node.light = dataclasses.replace(node.light, **{f: getattr(node.light, f) * s for f in LIGHT_DISTANCES})
    c4 = np.eye(4)
    c4[:3, :3] = c
    frames = {node: c4 for node, _ in converted.walk() if node.camera is not None or node.light is not None}
    if frames and not np.array_equal(c, np.eye(3)):
        converted = reframe(converted, frames)
    return converted


def _plain(transform: Transform) -> bool:
    """No pivots, offsets or pre / post rotations: just translation, Euler rotation and scale."""
    return not any(np.any(getattr(transform, f)) for f in
                   ("pre_rotation", "post_rotation", "rotation_pivot", "rotation_offset", "scaling_pivot", "scaling_offset"))  # fmt: skip


def _scaled(value: float | None, s: float) -> float | None:
    return None if value is None else value * s


def _mesh_in_basis(mesh: Mesh, linear: NDArray, rotation: NDArray, m: NDArray, m_inv: NDArray, flip: bool) -> Mesh:
    """`mesh` with points and offsets times `linear`, normals times `rotation`, skin matrices conjugated by
    m, and (`flip`) each face's corners reversed so faces keep facing outward after a mirroring change."""

    def points(values):
        return None if values is None else values @ linear.T

    def directions(values):
        return None if values is None else values @ rotation.T

    corners = _reversed_corners(mesh.face_vertex_counts) if flip else None

    def per_corner(primvar: Primvar | None, transform=None) -> Primvar | None:
        if primvar is None:
            return None
        values = primvar.values if transform is None else transform(primvar.values)
        indices = primvar.indices
        if corners is not None and primvar.interpolation == "faceVarying":
            if indices is not None:
                indices = indices[corners]
            else:
                values = values[corners]
        return Primvar(values, primvar.interpolation, indices)

    shapes = [
        dataclasses.replace(shape, offsets=points(shape.offsets), normal_offsets=directions(shape.normal_offsets),
                            inbetweens=[Inbetween(i.weight, points(i.offsets), directions(i.normal_offsets)) for i in shape.inbetweens])
        for shape in mesh.blend_shapes
    ]  # fmt: skip
    skin = mesh.skin
    if skin is not None:
        skin = dataclasses.replace(skin, inverse_bind=m @ skin.inverse_bind @ m_inv, geom_bind=m @ skin.geom_bind @ m_inv)
    return dataclasses.replace(
        mesh,
        vertices=points(mesh.vertices),
        face_vertex_indices=mesh.face_vertex_indices if corners is None else mesh.face_vertex_indices[corners],
        normals=per_corner(mesh.normals, directions),
        uvs={name: per_corner(uv) for name, uv in mesh.uvs.items()},
        blend_shapes=shapes,
        skin=skin,
    )


def _reversed_corners(counts: NDArray) -> NDArray[np.int64]:
    """Corner order that reverses every face's corners (each face keeps its first corner)."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    face = np.repeat(np.arange(len(counts)), counts)
    k = np.arange(counts.sum()) - starts[face]
    return starts[face] + np.where(k == 0, 0, counts[face] - k)
