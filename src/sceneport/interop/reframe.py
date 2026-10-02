"""Rewrite a scene without changing what it looks like: shallow copies and node reframing.

Used where a format expresses the same scene differently (see fbx/compat.py, and
USD dome lights whose image pole is Z). The input scene is never modified:
changed parts are copies, unchanged meshes and arrays are shared.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from numpy.typing import NDArray

from ..math import TICKS_PER_SECOND, decompose_matrix
from ..scene.animation import Animation, Channel
from ..scene.types import Mesh, Node, Scene, Transform

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
        target = mesh_map[id(channel.target)] if channel.is_blend_shape else node_map[channel.target]
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
        animated = [n for n in affected if any(c.target is n for c in clip.channels)]
        if not animated:
            return clip.channels
        out = [c for c in clip.channels if c.is_blend_shape or c.target not in animated]
        for node in animated:
            times = key_times(clip, [c for c in clip.channels if c.target is node])
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
