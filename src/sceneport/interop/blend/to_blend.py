"""Build Blender data from a Scene, inside Blender (imports bpy).

So far: the node hierarchy (empties and mesh objects with their transforms),
meshes (points, faces, custom normals, UV maps), blend shapes as shape keys, and
joints and skins as armatures, vertex groups and Armature modifiers. The scene is
first moved to Blender's Z-up meters (compat.prepare). A shape key's slider range
is its BlendShape.range, widened to whole numbers where the shape's default or
animated weights fall outside it (Blender clamps values to the slider).

Armatures follow compat.rigs: each joint tree's plain parent node becomes the
armature object (compat.prepare adds one where needed, and moves skinned meshes
under it); bones rest in the bind pose and are posed to the scene's current pose.

Animation: one Action per clip, with a slot per animated object, armature or
shape key set (Blender 5's slotted actions). The first clip is assigned and sets
the scene's fps and frame range (frame = seconds x fps); the others are kept
with a fake user. Object transform and shape key weight keys are written as they
are (cubic keys get Bezier handles that give sceneport's curve exactly); bones
are sampled on every frame and key, relative to their rest (bind) pose.

Per-vertex animation (point samples) of the first clip goes to a sidecar USD
file, `<name>_cache.usdc` next to the .blend (see cache.py), read back by a
Mesh Sequence Cache modifier first in each moving object's stack. The modifier
reads vertices only, so the .blend's own faces, UVs and materials stay
editable, and interpolates between samples on subframes. Meshes whose samples
never move get no cache.

Collections become Blender collections, nested the same way (under the scene's
collection); an object in any collection is linked only there, others stay in
the scene's collection. A joint has no object: its armature is linked instead.
extras["blender"] values are set on the collection where it has that property
(e.g. hide_render, color_tag). A node under a joint is parented to its bone, with a parent
inverse that cancels Blender's offset to the bone's tail.

Cameras and lights become camera / light objects and the first dome light the
World (cameras_lights.py), the first camera the scene's; their animated settings
are F-curves on the camera / light data, and a dome's turning or brightness is
sampled onto the World's nodes.

Materials are Principled BSDF setups built from each material's PBR reading
(materials.py), assigned to slots on the mesh data (on the object where a shared
mesh's users differ), with the faces' slot indices.

Scripts: scene.extras["blender"]["texts"] ({name: source}) become Text datablocks,
saved in the .blend and run on demand from Blender's Text Editor (Run Script,
Alt+P); nothing runs by itself.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from numpy.typing import NDArray

from ...math import _axes, complete_rotation, quaternion_continuity, quaternion_from_matrix
from ...scene.animation import Animation, Channel, _auto_slopes
from ...scene.splats import GaussianSplats
from ...scene.types import Mesh, Node, Primvar, Scene
from . import cameras_lights
from .materials import MaterialWriter, assign
from .cache import moving, write_point_cache
from ..reframe import key_times
from .compat import Rig, pose_tracks, prepare, rigs

SLIDER_LIMIT = 10.0  # Blender's widest shape key slider range is -10..10


def write_blend(scene: Scene, path: str, *, pack_images: bool = False, scripts=()) -> None:
    """Replace Blender's current data with `scene` and save it to `path` (plus `<name>_cache.usdc` beside it
    when meshes have per-vertex animation). `pack_images`: pack the texture images found on disk.
    `scripts`: (path, source, params) of Python run, in order, as __main__ before saving, with `params` as
    the global `params`."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    target = Path(path).resolve()
    scene_to_blender(scene, cache_path=target.with_name(f"{target.stem}_cache.usdc"), blend_path=target,
                     pack_images=pack_images)  # fmt: skip
    for script_path, source, params in scripts:
        exec(compile(source, script_path, "exec"), {"__name__": "__main__", "__file__": script_path, "params": params})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).resolve()), check_existing=False)


def scene_to_blender(
    scene: Scene,
    *,
    collection=None,
    cache_path: str | Path | None = None,
    blend_path: str | Path | None = None,
    pack_images: bool = False,
) -> dict[Node, bpy.types.Object]:
    """Create objects for `scene` in `collection` (default: the current scene's) and return them by
    (prepared copy's) node. `cache_path`: the USD file to write per-vertex animation to (without it, that
    animation is skipped with a warning); it is referenced relative to `blend_path`, where the .blend
    will be saved, when given (else to the open .blend, or absolute). `pack_images`: pack the texture
    images found on disk into the .blend (embedded ones always are)."""
    _warn_unsupported(scene)
    scene = prepare(scene)
    collection = collection or bpy.context.scene.collection
    weights = _shape_weights(scene)
    layout = rigs(scene)
    held = {rig.node: rig for rig in layout}
    rig_of = {bone.joint: rig for rig in layout for bone in rig.bones}
    meshes: dict[int, bpy.types.Mesh] = {}
    objects: dict[Node, bpy.types.Object] = {}
    blend_dir = Path(blend_path).parent if blend_path else (Path(bpy.data.filepath).parent if bpy.data.filepath else None)
    domes: list[Node] = []
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
        if node.camera is not None or node.light is not None:
            if data is not None:
                warnings.warn(f"{node.name!r}: a Blender object can't be a mesh or armature and a camera / light; "
                              "its camera / light is skipped")  # fmt: skip
            elif node.camera is not None:
                data = cameras_lights.camera_data(node.camera, node.name)
            elif node.light.type == "dome":
                domes.append(node)  # the World; the node stays an empty
            else:
                data = cameras_lights.light_data(node.light, node.name, node.name)
        if node.splats is not None:
            if data is not None or node in domes:
                warnings.warn(f"{node.name!r}: a Blender object can't hold Gaussian splats and a mesh, armature, camera "
                              "or light; its splats are skipped")  # fmt: skip
            else:
                data = _splats(node.splats)
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
    writer, first_slots = MaterialWriter(blend_dir, pack_images), {}
    for node, obj in objects.items():
        if node.materials and obj.type == "MESH":
            assign(obj, node.materials, writer, first_slots)
    for material in scene.materials:  # every material, bound or not: unused ones survive saving
        data = writer.material(material)
        if data.users == 0:
            data.use_fake_user = True
    cameras = [node for node in objects if node.camera is not None and objects[node].type == "CAMERA"]
    if cameras:
        cameras_lights.use_camera(objects[cameras[0]], cameras[0].camera)
        others = [c.name for c in cameras[1:] if c.camera.exposure != cameras[0].camera.exposure]
        if others:
            warnings.warn(f"cameras {others[:3]}{'...' if len(others) > 3 else ''}: Blender has one exposure for the "
                          f"view, the scene camera's ({cameras[0].name!r}); theirs isn't written")  # fmt: skip
    world_matrices = scene.world_matrices()
    if domes:
        cameras_lights.world_from_dome(domes[0].light, world_matrices[domes[0]][:3, :3], blend_dir)
        if len(domes) > 1:
            warnings.warn(f"Blender has one World: dome lights {[d.name for d in domes[1:]]} are skipped")

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
        if node.mesh is not None and node.mesh.vertex_groups and node not in held:
            _vertex_groups(obj, node.mesh)
    if scene.collections:
        _collections(scene, collection, objects, {j: rig_objects[id(r)] for j, r in rig_of.items()})
    for index, clip in enumerate(scene.animations):
        _action(scene, clip, index == 0, objects, meshes, layout, rig_objects, bone_names, domes[:1])
    _point_caches(scene, objects, cache_path, blend_path)
    cameras_lights.fit_eevee_light_threshold()
    _texts(scene)
    return objects


def _texts(scene: Scene) -> None:
    """scene.extras["blender"]["texts"] ({name: source}) as Text datablocks (Blender adds .001 to taken names)."""
    for name, source in scene.extras.get("blender", {}).get("texts", {}).items():
        if not isinstance(source, str):
            warnings.warn(f"text {name!r}: its source isn't a string; not written")
            continue
        text = bpy.data.texts.new(name)
        text.write(source)
        text.cursor_set(0)  # opened at the top in the Text Editor


def _point_caches(scene: Scene, objects: dict, cache_path, blend_path) -> None:
    """Mesh Sequence Cache modifiers reading the first clip's moving point samples from `cache_path`."""
    later = [clip.name for clip in scene.animations[1:] if clip.point_samples]
    if later:
        warnings.warn(f"clips {later}: per-vertex animation is cached for the first clip only; not written")
    samples = [s for s in scene.animations[0].point_samples if moving(s)] if scene.animations else []
    if not samples:
        return
    if cache_path is None:
        warnings.warn("per-vertex animation needs a cache file path; it is not written")
        return
    cache_path = Path(cache_path)
    prim_paths = write_point_cache(cache_path, samples, scene.animations[0].fps)
    known = set(bpy.data.cache_files)
    bpy.ops.cachefile.open(filepath=str(cache_path))
    cache = next(c for c in bpy.data.cache_files if c not in known)
    blend_dir = Path(blend_path).parent if blend_path else (Path(bpy.data.filepath).parent if bpy.data.filepath else None)
    if blend_dir is not None:  # "//" paths are relative to the .blend
        cache.filepath = "//" + os.path.relpath(cache_path.resolve(), blend_dir.resolve()).replace(os.sep, "/")
    shaped = []
    for sample, prim_path in zip(samples, prim_paths):
        for node, obj in objects.items():
            if node.mesh is not sample.mesh:
                continue
            modifier = obj.modifiers.new("MeshSequenceCache", "MESH_SEQUENCE_CACHE")
            modifier.cache_file = cache
            modifier.object_path = prim_path
            modifier.read_data = {"VERT"}  # faces, UVs and materials come from the .blend's mesh
            modifier.use_vertex_interpolation = True  # linear between samples, as sceneport evaluates them
            obj.modifiers.move(len(obj.modifiers) - 1, 0)  # before the armature: skinning deforms the cached points
            if obj.data.shape_keys:
                shaped.append(obj.name)
    if shaped:
        warnings.warn(f"objects {shaped[:3]}{'...' if len(shaped) > 3 else ''} have shape keys and per-vertex animation; "
                      "the cache replaces the shaped points, so their shape keys have no effect")  # fmt: skip


INTERPOLATION = {"constant": 0, "linear": 1, "cubic": 2}  # Blender's CONSTANT, LINEAR, BEZIER
OBJECT_PATHS = {"translation": "location", "rotation": "rotation_euler", "scale": "scale", "orientation": "rotation_quaternion"}


def _action(scene: Scene, clip: Animation, assign: bool, objects: dict, meshes: dict, layout: list[Rig],
            rig_objects: dict, bone_names: dict, domes: list[Node]) -> None:  # fmt: skip
    """One Action for `clip`; `assign`: give every animated datablock this action (the first clip)."""
    fps = scene.animations[0].fps
    if assign:
        render = bpy.context.scene.render
        render.fps = max(1, round(fps))
        render.fps_base = render.fps / fps
        start, end = clip.time_range()
        bpy.context.scene.frame_start, bpy.context.scene.frame_end = round(start * fps), round(end * fps)
        bpy.context.scene.frame_current = bpy.context.scene.frame_start
    action = bpy.data.actions.new(clip.name)
    strip = action.layers.new("Layer").strips.new(type="KEYFRAME")
    bags: dict[int, tuple] = {}  # id(datablock) -> (datablock, slot, channelbag)

    def bag(datablock, id_type: str):
        if id(datablock) not in bags:
            slot = action.slots.new(id_type=id_type, name=datablock.name)
            bags[id(datablock)] = (datablock, slot, strip.channelbags.new(slot))
        return bags[id(datablock)][2]

    for channel in clip.channels:
        if channel.is_transform and not channel.target.joint and channel.target in objects:
            obj = objects[channel.target]
            if channel.property == "orientation":
                obj.rotation_mode = "QUATERNION"
            factor = math.pi / 180.0 if channel.property == "rotation" else 1.0
            _fcurve(bag(obj, "OBJECT"), OBJECT_PATHS[channel.property], channel.component, channel, factor, fps, obj.name)
        elif channel.is_blend_shape and id(channel.target) in meshes and meshes[id(channel.target)].shape_keys:
            key = meshes[id(channel.target)].shape_keys
            if channel.property in key.key_blocks:
                _fcurve(bag(key, "KEY"), f'key_blocks["{channel.property}"].value', 0, channel, 1.0, fps, None)
    unwritten = _attribute_fcurves(clip, objects, bag, fps)
    if unwritten:
        warnings.warn(f"clip {clip.name!r}: Blender can't animate {', '.join(sorted(unwritten))}; not written")
    world = bpy.context.scene.world
    if domes and world is not None and world.node_tree is not None:
        _dome_fcurves(scene, clip, domes[0], world, bag, fps)
    for rig in layout:
        tracks = pose_tracks(scene, rig, clip)
        if tracks is not None:
            _pose_fcurves(bag(rig_objects[id(rig)], "OBJECT"), rig, rig_objects[id(rig)], bone_names, *tracks, fps)
    start, end = clip.time_range()
    action.use_frame_range = True
    action.frame_start, action.frame_end = start * fps, max(end * fps, start * fps + 1)
    if not assign:
        action.use_fake_user = True  # kept in the file though nothing uses it
        return
    for datablock, slot, _ in bags.values():
        data = datablock.animation_data or datablock.animation_data_create()
        data.action = action
        data.action_slot = slot


def _attribute_fcurves(clip: Animation, objects: dict, bag, fps: float) -> set[str]:
    """Camera and light channels as F-curves on the object's camera / light data; returns what can't be."""
    unwritten = set()
    for channel in clip.channels:
        if channel.attribute is None or channel.target not in objects:
            continue
        kind, field = channel.attribute
        data = objects[channel.target].data
        if kind == "camera" and isinstance(data, bpy.types.Camera):
            camera = channel.target.camera
            if field == "exposure":  # the view's, for the scene camera only
                scene = bpy.context.scene
                if scene.camera is objects[channel.target]:
                    _fcurve(bag(scene, "SCENE"), "view_settings.exposure", 0, channel, 1.0, fps, None)
                else:
                    unwritten.add("exposure of cameras other than the scene camera")
            elif field in ("horizontal_aperture_offset", "vertical_aperture_offset"):  # shift = offset / sensor width
                if any(c.target is channel.target and c.attribute == ("camera", "horizontal_aperture") for c in clip.channels):
                    times = channel.times  # the width moves too: divide by it on the offset's keys
                    width = clip.attribute_values(channel.target, "camera", times)["horizontal_aperture"]
                    channel = Channel(channel.target, channel.property, 0, times, channel.values / width, channel.interpolation)
                    factor = 1.0
                else:
                    factor = 1.0 / camera.horizontal_aperture
                path = "shift_x" if field.startswith("horizontal") else "shift_y"
                _fcurve(bag(data, "CAMERA"), path, 0, channel, factor, fps, None)
            elif field in cameras_lights.CAMERA_PATHS:
                path, factor = cameras_lights.CAMERA_PATHS[field]
                _fcurve(bag(data, "CAMERA"), path, 0, channel, factor, fps, None)
            else:
                unwritten.add(f"camera {field}")
        elif kind == "light" and isinstance(data, bpy.types.Light):
            paths = cameras_lights.light_paths(channel.target.light)
            if field not in paths:
                unwritten.add(f"light {field}")
                continue
            path, factor = paths[field]
            _fcurve(bag(data, "LIGHT"), path, channel.component if field == "color" else 0, channel, factor, fps, None)
            if field == "temperature":
                data.use_temperature = True
    return unwritten


def _dome_fcurves(scene: Scene, clip: Animation, dome: Node, world, bag, fps: float) -> None:
    """The World's Mapping rotation and Background strength, sampled on every frame and key, when the dome's
    orientation (its or an ancestor's transform) or brightness is animated."""
    parents = scene.parents()
    chain, node = set(), dome
    while node is not None:
        chain.add(node)
        node = parents[node]
    nodes = world.node_tree.nodes
    mapping = next((n for n in nodes if n.bl_idname == "ShaderNodeMapping"), None)
    turning = [c for c in clip.channels if c.is_transform and c.target in chain]
    if turning and mapping is not None:
        times = key_times(clip, turning)
        rotation = clip.world_matrices(scene, times)[dome][:, :3, :3]
        euler = np.unwrap(cameras_lights.dome_mapping_rotation(rotation), axis=0)
        for axis in range(3):
            channel = Channel(dome, "rotation", axis, times, euler[:, axis])
            _fcurve(bag(world.node_tree, "NODETREE"), f'nodes["{mapping.name}"].inputs[2].default_value', axis, channel, 1.0, fps, None)
    bright = [c for c in clip.channels if c.target is dome and c.attribute in (("light", "intensity"), ("light", "exposure"))]
    if bright:
        background = next(n for n in nodes if n.bl_idname == "ShaderNodeBackground")
        times = key_times(clip, bright)
        values = clip.attribute_values(dome, "light", times)
        channel = Channel(dome, "light:intensity", 0, times, values["intensity"] * 2.0 ** values["exposure"])
        _fcurve(bag(world.node_tree, "NODETREE"), f'nodes["{background.name}"].inputs[1].default_value', 0, channel, 1.0, fps, None)


def _fcurve(bag, path: str, index: int, channel: Channel, factor: float, fps: float, group: str | None) -> None:
    """Keys of `channel` (values times `factor`); cubic keys get free Bezier handles at a third of each
    segment along sceneport's automatic slopes, which reproduces its Hermite curve exactly."""
    curve = bag.fcurves.new(path, index=index)
    if group is not None:
        curve.group = bag.groups.get(group) or bag.groups.new(group)
    frames = channel.times * fps
    values = channel.values * factor
    count = len(frames)
    points = curve.keyframe_points
    points.add(count)
    points.foreach_set("co", np.column_stack([frames, values]).astype(np.float32).ravel())
    points.foreach_set("interpolation", [INTERPOLATION[m] for m in channel.interpolation])
    if count > 1 and any(m == "cubic" for m in channel.interpolation):
        slopes = _auto_slopes(frames, values)  # per frame
        gaps = np.diff(frames)
        before = np.concatenate([[gaps[0]], gaps]) / 3.0
        after = np.concatenate([gaps, [gaps[-1]]]) / 3.0
        points.foreach_set("handle_left_type", [0] * count)  # FREE: kept as written
        points.foreach_set("handle_right_type", [0] * count)
        points.foreach_set("handle_left", np.column_stack([frames - before, values - slopes * before]).astype(np.float32).ravel())
        points.foreach_set("handle_right", np.column_stack([frames + after, values + slopes * after]).astype(np.float32).ravel())
    curve.update()


def _pose_fcurves(bag, rig: Rig, obj, bone_names: dict, times: np.ndarray, poses: dict, fps: float) -> None:
    """Pose bone location / rotation_quaternion / scale keys on every sampled time (linear), from the posed
    matrices: basis = (parent pose @ parent rest^-1 @ rest)^-1 @ pose, with the rests Blender stored."""
    rests = {b.joint: np.array(obj.data.bones[bone_names[b.joint]].matrix_local) for b in rig.bones}
    for bone in rig.bones:
        parent = np.eye(4) if bone.parent is None else poses[bone.parent] @ np.linalg.inv(rests[bone.parent])
        basis = np.linalg.inv(parent @ rests[bone.joint]) @ poses[bone.joint]
        scale = np.linalg.norm(basis[:, :3, :3], axis=1)
        rotation = complete_rotation(basis[:, :3, :3] / np.where(scale == 0, 1, scale)[:, None, :], scale != 0)
        quaternion = quaternion_continuity(quaternion_from_matrix(rotation))
        name = bone_names[bone.joint]
        for prop, values in (("location", basis[:, :3, 3]), ("rotation_quaternion", quaternion), ("scale", scale)):
            for axis in range(values.shape[1]):
                channel = Channel(bone.joint, "translation", 0, times, values[:, axis])
                _fcurve(bag, f'pose.bones["{name}"].{prop}', axis, channel, 1.0, fps, name)


def _collections(scene: Scene, parent, objects: dict, armature_of: dict) -> None:
    made: dict[int, bpy.types.Collection] = {}
    linked, joints = set(), []

    def build(collection, parent_collection) -> None:
        if id(collection) in made:  # nested under several collections: link it again
            parent_collection.children.link(made[id(collection)])
            return
        data = made[id(collection)] = bpy.data.collections.new(collection.name)
        parent_collection.children.link(data)
        for key, value in collection.extras.get("blender", {}).items():
            if hasattr(data, key):
                try:
                    setattr(data, key, value)
                except (AttributeError, TypeError, ValueError):
                    warnings.warn(f"collection {collection.name!r}: can't set {key}={value!r}")
        for node in collection.nodes:
            obj = objects.get(node)
            if obj is None and node in armature_of:
                joints.append(node.name)
                obj = armature_of[node]
            if obj is not None and obj.name not in data.objects:
                data.objects.link(obj)
                linked.add(obj)
        for child in collection.children:
            build(child, data)

    for top in scene.collections:
        build(top, parent)
    for obj in linked:  # in a collection: not also loose in the scene's own collection
        if obj.name in parent.objects:
            parent.objects.unlink(obj)
    if joints:
        warnings.warn(f"joints {joints[:3]}{'...' if len(joints) > 3 else ''} are in collections; Blender bones aren't "
                      "objects, so their armature is linked instead")  # fmt: skip


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


def _vertex_groups(obj, mesh: Mesh) -> None:
    """The mesh's vertex groups (vertices of weight 0 left out). A name a skin's bone group already
    has would deform the mesh with that bone, so such a group is skipped."""
    for name, weights in mesh.vertex_groups.items():
        if name in obj.vertex_groups:
            warnings.warn(f"mesh {mesh.name!r}: vertex group {name!r} has a bone's name; it is skipped")
            continue
        group = obj.vertex_groups.new(name=name)
        for weight in np.unique(weights[weights != 0]):
            group.add(np.flatnonzero(weights == weight).tolist(), float(weight), "REPLACE")


def _splats(splats: GaussianSplats) -> bpy.types.PointCloud:
    """A point cloud with Blender 5.3's Gaussian splat attributes (its PLY importer's layout): position,
    scale, rotation (quaternion), radiance:base (SH DC and opacity), radiance:sh_0.. (the higher SH
    coefficients, RGB each). Blender 5.3+ marks it as Gaussian splats; earlier versions keep a plain
    point cloud, drawn as spheres of `radius` (the largest scale) in `color` (the view-independent
    color, linear), which 5.3's Set Point Cloud Type node can turn into splats. Other per-splat
    attributes (e.g. a parent triangle index) are kept under their names."""
    data = bpy.data.pointclouds.new(splats.name)
    data.resize(len(splats))

    def put(name: str, kind: str, values: NDArray) -> None:
        attribute = data.attributes.get(name) or data.attributes.new(name, kind, "POINT")
        field = {"FLOAT_VECTOR": "vector", "FLOAT4": "vector", "FLOAT_COLOR": "color"}.get(kind, "value")
        attribute.data.foreach_set(field, np.ascontiguousarray(values, dtype=np.int32 if kind == "INT" else np.float32).ravel())

    put("position", "FLOAT_VECTOR", splats.positions)
    put("scale", "FLOAT_VECTOR", splats.scales)
    put("rotation", "QUATERNION", splats.orientations)
    put("radiance:base", "FLOAT4", np.column_stack([splats.sh[:, 0], splats.opacities]))
    for k in range(1, splats.sh.shape[1]):
        put(f"radiance:sh_{k - 1}", "FLOAT_VECTOR", splats.sh[:, k])
    put("radius", "FLOAT", splats.scales.max(axis=1))
    srgb = splats.base_colors()
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    put("color", "FLOAT_COLOR", np.column_stack([linear, splats.opacities]))
    for name, values in splats.attributes.items():
        if values.ndim != 1 or name in data.attributes:
            warnings.warn(f"splat attribute {name!r} isn't one value per splat or clashes with a Blender attribute; it is skipped")
            continue
        put(name, "INT" if np.issubdtype(values.dtype, np.integer) else "FLOAT", values)
    if hasattr(data, "type"):  # Blender 5.3+: Gaussian splats rather than points
        data.type = "GAUSSIAN_SPLAT"
    return data


def _mesh(mesh: Mesh) -> bpy.types.Mesh:
    data = bpy.data.meshes.new(mesh.name)
    counts = mesh.face_vertex_counts
    data.vertices.add(len(mesh.vertices))
    data.vertices.foreach_set("co", mesh.vertices.astype(np.float32).ravel())
    data.loops.add(len(mesh.face_vertex_indices))
    data.loops.foreach_set("vertex_index", mesh.face_vertex_indices.astype(np.int32))
    data.polygons.add(len(counts))
    data.polygons.foreach_set("loop_start", (np.cumsum(counts) - counts).astype(np.int32))  # also right with no faces
    if mesh.material_indices is not None:
        data.polygons.foreach_set("material_index", mesh.material_indices.astype(np.int32))
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
            ("blend shape normal offsets (Blender computes shape normals)",
             any(s.normal_offsets is not None for m in scene.meshes for s in m.blend_shapes)),
        )
        if present
    ]  # fmt: skip
    if skipped:
        warnings.warn(f"the .blend writer doesn't write {', '.join(skipped)} yet; they are skipped")
