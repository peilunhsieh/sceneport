"""Scene rewrites applied before writing FBX, for readers that can't handle everything FBX can hold.

Both keep what the scene looks like at every frame; they change how it's expressed.
(Cameras are also turned from looking down -Z to FBX's +X, and lights from shining down -Z
to FBX's -Y, here, always; see interop/reframe.py.)

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

from ...scene.animation import Animation, Channel
from ...scene.types import BlendShape, Mesh, Node, Scene
from ..reframe import copy_scene, key_times, mirrored_joints, reframe, unmirror

# A camera looking down -Z (sceneport, USD) turned to look down +X (FBX): a 90 degree turn about Y,
# as Blender's FBX exporter does (MAT_CONVERT_CAMERA).
CAMERA_TO_FBX = np.array([[0.0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
# A light shining down -Z (sceneport, USD) turned to shine down -Y (FBX): a 90 degree turn about X,
# as Blender's FBX exporter does (MAT_CONVERT_LIGHT).
LIGHT_TO_FBX = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def prepare(scene: Scene, *, unmirror_joints: bool = True, split_inbetweens: bool = True) -> Scene:
    """The scene to write: in-betweens split and mirrored joints unmirrored, as requested (a copy when anything changes)."""
    if split_inbetweens and any(shape.inbetweens for mesh in scene.meshes for shape in mesh.blend_shapes):
        scene = _split_inbetweens(scene)
    if unmirror_joints:
        flipped = mirrored_joints(scene)
        if flipped:
            scene = unmirror(scene, flipped)
    # Always: FBX cameras look down +X and lights shine down -Y (a dome isn't an FBX light: left as is).
    frames = {node: CAMERA_TO_FBX for node, _ in scene.walk() if node.camera is not None}
    frames.update({node: LIGHT_TO_FBX for node, _ in scene.walk() if node.light is not None and node.light.type != "dome"})
    if frames:
        scene = reframe(scene, frames)
    return scene


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
                shapes.append(BlendShape(name, offsets, indices=shape.indices, normal_offsets=normals, weight=float(default),
                                          range=shape.range))
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
            times = key_times(clip, [channel])
            parts = hat_weights(channel.evaluate(times), key_weights)  # (T, K)
            out += [Channel(channel.target, name, 0, times, parts[:, k]) for k, name in enumerate(names)]
        return out

    return copy_scene(scene, meshes, {}, channels)
