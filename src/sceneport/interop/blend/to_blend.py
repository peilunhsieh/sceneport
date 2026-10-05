"""Build Blender data from a Scene, inside Blender (imports bpy).

So far: the node hierarchy (empties and mesh objects with their transforms),
meshes (points, faces, custom normals, UV maps) and blend shapes as shape keys.
The scene is first moved to Blender's Z-up meters (compat.prepare). A shape key's
slider range is its BlendShape.range, widened to whole numbers where the shape's
default or animated weights fall outside it (Blender clamps values to the slider). Skins, materials, cameras,
lights and animation aren't written yet: they're skipped with a warning (joints
become empties).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import bpy
import numpy as np

from ...math import _axes
from ...scene.types import Mesh, Node, Primvar, Scene
from .compat import prepare

SLIDER_LIMIT = 10.0  # Blender's widest shape key slider range is -10..10


def write_blend(scene: Scene, path: str) -> None:
    """Replace Blender's current data with `scene` and save it to `path`."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene_to_blender(scene)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).resolve()), check_existing=False)


def scene_to_blender(scene: Scene, *, collection=None) -> dict[Node, bpy.types.Object]:
    """Create objects for `scene` in `collection` (default: the current scene's) and return them by
    (prepared copy's) node."""
    _warn_unsupported(scene)
    scene = prepare(scene)
    collection = collection or bpy.context.scene.collection
    weights = _shape_weights(scene)
    meshes: dict[int, bpy.types.Mesh] = {}
    objects: dict[Node, bpy.types.Object] = {}

    def build(node: Node, parent) -> None:
        data = None
        if node.mesh is not None and not node.joint:
            first = id(node.mesh) not in meshes
            if first:
                meshes[id(node.mesh)] = _mesh(node.mesh)
            data = meshes[id(node.mesh)]
        obj = bpy.data.objects.new(node.name, data)
        collection.objects.link(obj)
        if data is not None and first and node.mesh.blend_shapes:
            _shape_keys(obj, node.mesh, weights)
        t = node.transform
        obj.rotation_mode = _axes(t.rotation_order)  # Blender's Euler modes use the same names: "XYZ" turns about X first
        obj.location = t.translation
        obj.rotation_euler = [math.radians(a) for a in t.rotation]
        obj.scale = t.scale
        obj.parent = parent
        objects[node] = obj
        for child in node.children:
            build(child, obj)

    for root in scene.nodes:
        build(root, None)
    return objects


def _mesh(mesh: Mesh) -> bpy.types.Mesh:
    data = bpy.data.meshes.new(mesh.name)
    counts = mesh.face_vertex_counts
    data.vertices.add(len(mesh.vertices))
    data.vertices.foreach_set("co", mesh.vertices.astype(np.float32).ravel())
    data.loops.add(len(mesh.face_vertex_indices))
    data.loops.foreach_set("vertex_index", mesh.face_vertex_indices.astype(np.int32))
    data.polygons.add(len(counts))
    data.polygons.foreach_set("loop_start", np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32))
    data.update(calc_edges=True)
    if data.validate(verbose=False, clean_customdata=False):
        warnings.warn(f"mesh {mesh.name!r}: Blender removed invalid geometry (e.g. repeated corners or faces)")
    if len(data.loops) != len(mesh.face_vertex_indices):
        warnings.warn(f"mesh {mesh.name!r}: corners changed while validating; UVs and normals are not written")
        return data
    for name, uv in mesh.uvs.items():
        layer = data.uv_layers.new(name=name, do_init=False)
        layer.data.foreach_set("uv", _per_corner(uv, mesh).astype(np.float32).ravel())
    if mesh.normals is not None:
        data.shade_smooth()
        normals = _per_corner(mesh.normals, mesh)
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        data.normals_split_custom_set(normals.astype(np.float32))
    return data


def _per_corner(primvar: Primvar, mesh: Mesh) -> np.ndarray:
    values = primvar.expanded()
    if primvar.interpolation == "vertex":
        return values[mesh.face_vertex_indices]
    if primvar.interpolation == "uniform":
        return np.repeat(values, mesh.face_vertex_counts, axis=0)
    if primvar.interpolation == "constant":
        return np.repeat(values[:1], len(mesh.face_vertex_indices), axis=0)
    return values


def _shape_weights(scene: Scene) -> dict[tuple[int, str], list[float]]:
    """Every weight each shape takes: its default plus its animation keys, by (id(mesh), shape name)."""
    out: dict[tuple[int, str], list[float]] = {}
    for mesh in scene.meshes:
        for shape in mesh.blend_shapes:
            out[(id(mesh), shape.name)] = [shape.weight]
    for clip in scene.animations:
        for channel in clip.channels:
            if channel.is_blend_shape and (id(channel.target), channel.property) in out:
                out[(id(channel.target), channel.property)] += [float(channel.values.min()), float(channel.values.max())]
    return out


def _shape_keys(obj, mesh: Mesh, weights: dict) -> None:
    obj.shape_key_add(name="Basis", from_mix=False)
    base = mesh.vertices.astype(np.float64)
    clamped = []
    for shape in mesh.blend_shapes:
        key = obj.shape_key_add(name=shape.name, from_mix=False)
        coords = base.copy()
        if shape.indices is None:
            coords += shape.offsets
        else:
            coords[shape.indices] += shape.offsets
        key.data.foreach_set("co", coords.astype(np.float32).ravel())
        used = weights[(id(mesh), shape.name)]
        low, high = shape.range
        low = low if min(used) >= low else math.floor(min(used))
        high = high if max(used) <= high else math.ceil(max(used))
        if low < -SLIDER_LIMIT or high > SLIDER_LIMIT:
            clamped.append(shape.name)
        key.slider_min = max(low, -SLIDER_LIMIT)
        key.slider_max = min(high, SLIDER_LIMIT)
        key.value = shape.weight
    if clamped:
        warnings.warn(f"mesh {mesh.name!r}: shape keys {clamped[:3]}{'...' if len(clamped) > 3 else ''} use weights "
                      f"beyond Blender's -10..10 slider limit; they are clamped there")  # fmt: skip


def _warn_unsupported(scene: Scene) -> None:
    nodes = [node for node, _ in scene.walk()]
    skipped = [
        what
        for what, present in (
            ("skins (joints become empties)", any(m.skin is not None for m in scene.meshes) or any(n.joint for n in nodes)),
            ("materials", bool(scene.materials) or any(n.materials for n in nodes)),
            ("cameras", any(n.camera is not None for n in nodes)),
            ("lights", any(n.light is not None for n in nodes)),
            ("animation", bool(scene.animations)),
            ("blend shape normal offsets (Blender computes shape normals)",
             any(s.normal_offsets is not None for m in scene.meshes for s in m.blend_shapes)),
        )
        if present
    ]  # fmt: skip
    if skipped:
        warnings.warn(f"the .blend writer doesn't write {', '.join(skipped)} yet; they are skipped")
