"""Scene rewrites applied before writing FBX, for readers that can't handle everything FBX can hold.

Both keep what the scene looks like at every frame; they change how it's expressed.
(Cameras are also turned from looking down -Z to FBX's +X here, always; see reframe.)

- split_inbetweens: Blender's FBX importer refuses in-between shapes. A shape's
  weight interpolates linearly through its in-betweens, so it equals a sum of
  plain shapes (one per in-between, plus the full shape) whose weights are
  "hat" functions of the original weight. Each becomes a plain BlendShape named
  "<shape>_<in-between weight>" (the full shape keeps its name); weight channels
  are rewritten per frame (exact at every frame, linear in between).
- unmirror_joints: a joint with a mirrored world matrix (negative scale, e.g. a
  rig's right side) can't be a Blender bone. Each such joint's frame is turned by
  F = -I: world' = world @ F, so local' = F_parent @ local @ F_self and inverse
  bind' = F @ inverse bind. Skinning (world @ inverse bind) and joint positions
  are unchanged; the joint's axes point the other way. Animated joints in a
  mirrored chain are baked per frame.

The input scene is never modified: changed parts are copies, unchanged meshes
and arrays are shared.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from ...math import TICKS_PER_SECOND, decompose_matrix
from ...scene.animation import Animation, Channel
from ...scene.types import BlendShape, Mesh, Node, Scene, Skin, Transform

FLIP = -np.eye(4)
FLIP[3, 3] = 1.0  # -I on the linear part: a point reflection
# A camera looking down -Z (sceneport, USD) turned to look down +X (FBX): a 90 degree turn about Y,
# as Blender's FBX exporter does (MAT_CONVERT_CAMERA).
CAMERA_TO_FBX = np.array([[0.0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])


def prepare(scene: Scene, *, unmirror_joints: bool = True, split_inbetweens: bool = True) -> Scene:
    """The scene to write: in-betweens split and mirrored joints unmirrored, as requested (a copy when anything changes)."""
    if split_inbetweens and any(shape.inbetweens for mesh in scene.meshes for shape in mesh.blend_shapes):
        scene = _split_inbetweens(scene)
    if unmirror_joints:
        flipped = _mirrored_joints(scene)
        if flipped:
            scene = _unmirror(scene, flipped)
    cameras = [node for node, _ in scene.walk() if node.camera is not None]
    if cameras:  # always: FBX cameras look down +X
        scene = reframe(scene, {node: CAMERA_TO_FBX for node in cameras})
    return scene


# -- shallow copies ---------------------------------------------------------------


def _copy(scene: Scene, meshes: dict[int, Mesh], transforms: dict[Node, Transform], channels) -> Scene:
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
        target = mesh_map[id(channel.target)] if channel.is_blend_shape else node_map[channel.target]
        return dataclasses.replace(channel, target=target)

    animations = [dataclasses.replace(clip, channels=[retarget(c) for c in channels(clip)]) for clip in scene.animations]
    return dataclasses.replace(scene, nodes=roots, animations=animations)


def _key_times(clip: Animation, channels: list[Channel]) -> NDArray[np.float64]:
    """Every frame of the clip plus the channels' own keys, merged on whole FBX ticks."""
    times = np.concatenate([clip.frame_times(), *(c.times for c in channels)])
    return np.unique(np.round(times * TICKS_PER_SECOND)) / TICKS_PER_SECOND


# -- in-betweens -----------------------------------------------------------------


def hat_weights(weight, key_weights: NDArray) -> NDArray[np.float64]:
    """(..., K) weights of the K key shapes (sorted `key_weights`, none of them 0) that reproduce a shape with
    in-betweens at `weight`: linear interpolation through (0, no offset) and the keys, extrapolating the end
    segments, exactly as BlendShape.offsets_at does."""
    weight = np.asarray(weight, dtype=np.float64)
    nodes = np.sort(np.append(key_weights, 0.0))
    i = np.clip(np.searchsorted(nodes, weight) - 1, 0, len(nodes) - 2)
    t = (weight - nodes[i]) / (nodes[i + 1] - nodes[i])
    out = np.zeros(weight.shape + (len(nodes),))
    np.put_along_axis(out, i[..., None], (1 - t)[..., None], axis=-1)
    np.put_along_axis(out, (i + 1)[..., None], np.take_along_axis(out, (i + 1)[..., None], axis=-1) + t[..., None], axis=-1)
    return np.delete(out, np.searchsorted(nodes, 0.0), axis=-1)  # the zero node carries no offsets


def _split_inbetweens(scene: Scene) -> Scene:
    meshes: dict[int, Mesh] = {}
    plans: dict[int, dict[str, tuple[list[str], NDArray]]] = {}  # id(Mesh) -> shape name -> (part names, key weights)
    for mesh in scene.meshes:
        if not any(shape.inbetweens for shape in mesh.blend_shapes):
            continue
        taken = {shape.name for shape in mesh.blend_shapes}
        shapes, plan = [], {}
        for shape in mesh.blend_shapes:
            if not shape.inbetweens:
                shapes.append(shape)
                continue
            keys = sorted([(i.weight, i.offsets, i.normal_offsets) for i in shape.inbetweens] + [(1.0, shape.offsets, shape.normal_offsets)],
                          key=lambda k: k[0])  # fmt: skip
            key_weights = np.array([k[0] for k in keys])
            defaults = hat_weights(shape.weight, key_weights)
            names = []
            for (weight, offsets, normals), default in zip(keys, defaults):
                name = shape.name
                if weight != 1.0:
                    name = f"{shape.name}_{weight:g}"
                    while name in taken:
                        name += "_"
                    taken.add(name)
                names.append(name)
                shapes.append(BlendShape(name, offsets, indices=shape.indices, normal_offsets=normals, weight=float(default)))
            plan[shape.name] = (names, key_weights)
        meshes[id(mesh)] = dataclasses.replace(mesh, blend_shapes=shapes)
        plans[id(mesh)] = plan

    def channels(clip: Animation) -> list[Channel]:
        out = []
        for channel in clip.channels:
            plan = plans.get(id(channel.target), {}).get(channel.property) if channel.is_blend_shape else None
            if plan is None:
                out.append(channel)
                continue
            names, key_weights = plan
            times = _key_times(clip, [channel])
            parts = hat_weights(channel.evaluate(times), key_weights)  # (T, K)
            out += [Channel(channel.target, name, 0, times, parts[:, k]) for k, name in enumerate(names)]
        return out

    return _copy(scene, meshes, {}, channels)


# -- mirrored joints -----------------------------------------------------------------


def _mirrored_joints(scene: Scene) -> set[Node]:
    """Joints whose rest world matrix is mirrored (negative determinant)."""
    return {node for node, world in scene.walk() if node.joint and np.linalg.det(world[:3, :3]) < 0}


def _unmirror(scene: Scene, flipped: set[Node]) -> Scene:
    return reframe(scene, {node: FLIP for node in flipped})


def reframe(scene: Scene, frames: dict[Node, NDArray]) -> Scene:
    """Turn the local frame of each node in `frames` by its matrix F without moving anything:
    world' = world @ F, so local' = F_parent^-1 @ local @ F_self, a mesh's geometric transform
    becomes F^-1 @ geometric and a joint's inverse bind F^-1 @ inverse bind. Animated nodes whose
    local matrix changes are baked per frame. Used to unmirror joints and to turn cameras
    between the -Z (USD) and +X (FBX) view axes.
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
        animated = [n for n in affected if any(c.target is n for c in clip.channels)]
        if not animated:
            return clip.channels
        out = [c for c in clip.channels if c.is_blend_shape or c.target not in animated]
        for node in animated:
            times = _key_times(clip, [c for c in clip.channels if c.target is node])
            local = local_frame(node, node.transform.matrix(**clip.node_values(times)[node]))
            translation, rotation, scale = decompose_matrix(local, node.transform.rotation_order)
            rotation = np.degrees(np.unwrap(np.radians(rotation), axis=0))
            for prop, values in (("translation", translation), ("rotation", rotation), ("scale", scale)):
                out += [Channel(node, prop, axis, times, values[:, axis]) for axis in range(3)]
        return out

    return _copy(scene, meshes, transforms, channels)


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
