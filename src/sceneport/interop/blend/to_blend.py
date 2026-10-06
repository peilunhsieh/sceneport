"""Build Blender data from a Scene, inside Blender (imports bpy).

So far: the node hierarchy (empties and mesh objects with their transforms),
meshes (points, faces, custom normals, UV maps), blend shapes as shape keys, and
joints and skins as armatures, vertex groups and Armature modifiers. The scene is
first moved to Blender's Z-up meters (compat.prepare). A shape key's slider range
is its BlendShape.range, widened to whole numbers where the shape's default or
animated weights fall outside it (Blender clamps values to the slider).

Armatures follow compat.rigs: each joint tree's plain parent node becomes the
armature object (compat.prepare adds one where needed, and moves skinned meshes
under it); bones rest in the bind pose and are posed to the scene's current pose. A node under a joint is parented to its bone, with a parent
inverse that cancels Blender's offset to the bone's tail. Materials, cameras,
lights and animation aren't written yet: they're skipped with a warning.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

from ...math import _axes
from ...scene.types import Mesh, Node, Primvar, Scene
from .compat import Rig, prepare, rigs

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
    layout = rigs(scene)
    held = {rig.node: rig for rig in layout}
    rig_of = {bone.joint: rig for rig in layout for bone in rig.bones}
    meshes: dict[int, bpy.types.Mesh] = {}
    objects: dict[Node, bpy.types.Object] = {}
    rig_objects: dict[int, bpy.types.Object] = {}  # id(Rig) -> its armature object
    under_bones: list[tuple[Node, Node]] = []  # (node, parent joint): parented to a bone once bones exist

    def build(node: Node, parent_obj, parent_joint: Node | None = None) -> None:
        if node.joint:
            if node.mesh is not None or node.camera is not None or node.light is not None:
                warnings.warn(f"joint {node.name!r}: Blender bones can't hold a mesh, camera or light; it is skipped")
            for child in node.children:
                build(child, None, node)
            return
        data = None
        if node in held:
            data = bpy.data.armatures.new(node.name)
        elif node.mesh is not None:
            first = id(node.mesh) not in meshes
            if first:
                meshes[id(node.mesh)] = _mesh(node.mesh)
            data = meshes[id(node.mesh)]
        obj = bpy.data.objects.new(node.name, data)
        collection.objects.link(obj)
        if node.mesh is not None and node not in held and first and node.mesh.blend_shapes:
            _shape_keys(obj, node.mesh, weights)
        _set_transform(obj, node)
        obj.parent = parent_obj
        objects[node] = obj
        if node in held:
            rig_objects[id(held[node])] = obj
        if parent_joint is not None:
            under_bones.append((node, parent_joint))
        for child in node.children:
            build(child, obj)

    for root in scene.nodes:
        build(root, None)

    bone_names = {}
    for rig in layout:
        bone_names.update(_bones(rig, rig_objects[id(rig)]))
    world = scene.world_matrices()
    for node, joint in under_bones:
        obj, rig = objects[node], rig_of[joint]
        arm = rig_objects[id(rig)]
        tail = np.eye(4)
        tail[1, 3] = arm.data.bones[bone_names[joint]].length
        parent_matrix = rig.world @ rig.bone(joint).pose @ tail  # Blender parents to the posed bone's tail
        obj.parent, obj.parent_type, obj.parent_bone = arm, "BONE", bone_names[joint]
        obj.matrix_parent_inverse = Matrix((np.linalg.inv(parent_matrix) @ world[joint]).tolist())
    for node, obj in objects.items():
        if node.mesh is not None and node.mesh.skin is not None and node not in held:
            _skin(obj, node.mesh, rig_of, rig_objects, bone_names)
    return objects


def _set_transform(obj, node: Node) -> None:
    t = node.transform
    obj.rotation_mode = _axes(t.rotation_order)  # Blender's Euler modes use the same names: "XYZ" turns about X first
    obj.location = t.translation
    obj.rotation_euler = [math.radians(a) for a in t.rotation]
    obj.scale = t.scale


def _bones(rig: Rig, obj) -> dict[Node, str]:
    """Create the rig's bones in rest pose (edit mode), then pose them; returns each joint's bone name."""
    view_layer = bpy.context.view_layer
    previous = view_layer.objects.active
    view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    names: dict[Node, str] = {}
    edit_bones = obj.data.edit_bones
    for bone in rig.bones:
        edit = edit_bones.new(bone.joint.name)
        edit.head, edit.tail = (0.0, 0.0, 0.0), (0.0, bone.length, 0.0)
        edit.matrix = Matrix(bone.rest.tolist())  # head, direction (+Y) and roll; keeps the length
        if bone.parent is not None:
            edit.parent = edit_bones[names[bone.parent]]
        edit.use_connect = False
        names[bone.joint] = edit.name
    bpy.ops.object.mode_set(mode="OBJECT")
    view_layer.objects.active = previous
    # Pose: Blender evaluates pose = parent pose @ parent rest^-1 @ rest @ basis (full inheritance), so the
    # basis comes from the rest matrices Blender actually stored.
    rests = {bone.joint: np.array(obj.data.bones[names[bone.joint]].matrix_local) for bone in rig.bones}
    poses = {bone.joint: bone.pose for bone in rig.bones}
    for bone in rig.bones:
        parent = np.eye(4) if bone.parent is None else poses[bone.parent] @ np.linalg.inv(rests[bone.parent])
        basis = np.linalg.inv(parent @ rests[bone.joint]) @ bone.pose
        obj.pose.bones[names[bone.joint]].matrix_basis = Matrix(basis.tolist())
    return names


def _skin(obj, mesh: Mesh, rig_of: dict, rig_objects: dict, bone_names: dict) -> None:
    """Vertex groups named after the bones, with the skin's weights, and an Armature modifier."""
    skin = mesh.skin
    used = {id(rig_of[j]) for j in skin.joints if j in rig_of}
    if len(used) > 1:
        warnings.warn(f"mesh {mesh.name!r}: its skin uses joints of several armatures; only the first one deforms it")
    rows = np.repeat(np.arange(len(skin.joint_indices)), skin.joint_indices.shape[1])
    cols, values = skin.joint_indices.ravel(), skin.joint_weights.ravel()
    keep = values != 0
    pair = rows[keep] * len(skin.joints) + cols[keep]
    pairs, inverse = np.unique(pair, return_inverse=True)
    sums = np.bincount(inverse, weights=values[keep])  # a joint listed twice for a vertex: weights add
    vertices, joints = pairs // len(skin.joints), pairs % len(skin.joints)
    for j, joint in enumerate(skin.joints):
        group = obj.vertex_groups.new(name=bone_names.get(joint, joint.name))
        mine = joints == j
        for weight in np.unique(sums[mine]):
            group.add(vertices[mine][sums[mine] == weight].tolist(), float(weight), "REPLACE")
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig_objects[id(rig_of[skin.joints[0]])]


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
