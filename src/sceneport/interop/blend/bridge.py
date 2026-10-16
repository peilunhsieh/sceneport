"""The sceneport side inside Blender: runs one action per background Blender process.

`main(job_dir)` is called by runner.run_in_blender's boot code. It reads
`request.pkl` ({"action", "payload"}), runs ACTIONS[action](payload) and always
writes `response.pkl` ({"result", "warnings", "error"}): warnings are recorded as
(category name, message) and an exception as its traceback, so the host reports
them as if they were raised locally. This is the only sceneport module that
imports bpy.
"""

from __future__ import annotations

import pickle
import sys
import traceback
import warnings
from pathlib import Path


def ping(payload: dict | None) -> dict:
    """Blender's and its Python's versions; a diagnostic. `payload` may hold "warn" (a message to
    warn with) and "fail" (a message to raise), to exercise the error and warning paths."""
    import bpy
    import numpy

    payload = payload or {}
    if "warn" in payload:
        warnings.warn(payload["warn"])
    if "fail" in payload:
        raise RuntimeError(payload["fail"])
    return {
        "blender": tuple(bpy.app.version),
        "python": tuple(sys.version_info[:3]),
        "numpy": numpy.__version__,
        "sys_path": list(sys.path),
    }


def write_blend(payload: dict) -> None:
    """Save payload["scene"] as the .blend payload["path"] (packing images when payload["pack_images"])."""
    from .to_blend import write_blend

    write_blend(payload["scene"], payload["path"], pack_images=payload.get("pack_images", False))


def inspect_blend(payload: dict) -> dict:
    """What a .blend holds, as plain data (a diagnostic, used by the tests): per object its name, type,
    parent (and parent bone), rotation mode and world matrix; for meshes their points, faces, corner
    normals, UV maps, shape keys (points, value, slider range), vertex groups, modifiers and evaluated
    world-space points (shape keys and armatures applied); for armatures their bones (parent, length,
    rest matrix and posed matrix, in armature space). Returns {"objects": [...], "collections": the scene
    collection's tree (name, objects, hide_render, hide_viewport, color_tag, children), "scene": fps and
    frame range, "actions": names, fake users, slots and F-curve paths, "world": the World's nodes,
    "frames": per frame in payload["frames"], each object's world matrix, (meshes) evaluated world points,
    (cameras / lights) their settings, the World's mapping rotation and strength, and the scene camera's
    view (Blender's world_to_camera_view) of payload["points"] (world space)}."""
    import bpy
    import numpy as np

    bpy.ops.wm.open_mainfile(filepath=payload["path"], load_ui=False)

    def get(collection, attr, width, dtype=np.float64):
        out = np.empty(len(collection) * width, dtype=dtype)
        collection.foreach_get(attr, out)
        return out.reshape(-1, width) if width > 1 else out

    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_keys = ("type", "lens", "sensor_fit", "sensor_width", "sensor_height", "shift_x", "shift_y", "clip_start",
                   "clip_end", "ortho_scale")
    light_keys = ("type", "energy", "exposure", "shadow_soft_size", "spot_size", "spot_blend", "shape", "size", "size_y",
                  "angle", "normalize", "use_temperature", "temperature", "diffuse_factor", "specular_factor", "use_shadow")

    def settings(data, keys) -> dict:
        out = {k: getattr(data, k) for k in keys if hasattr(data, k)}
        if hasattr(data, "color"):
            out["color"] = tuple(data.color)
        if hasattr(data, "dof"):
            out.update(use_dof=data.dof.use_dof, focus_distance=data.dof.focus_distance, aperture_fstop=data.dof.aperture_fstop)
        return out

    def world_state() -> dict | None:
        world = scene.world
        if world is None or world.node_tree is None:
            return None
        nodes = {n.bl_idname: n for n in world.node_tree.nodes}
        mapping, background = nodes.get("ShaderNodeMapping"), nodes.get("ShaderNodeBackground")
        environment = nodes.get("ShaderNodeTexEnvironment")
        return {"nodes": sorted(nodes), "strength": background.inputs["Strength"].default_value if background else None,
                "mapping": (mapping.vector_type, tuple(mapping.inputs["Rotation"].default_value)) if mapping else None,
                "image": environment.image.filepath if environment and environment.image else None}  # fmt: skip

    scene = bpy.context.scene
    out = []
    for obj in bpy.data.objects:
        info = {
            "name": obj.name, "type": obj.type, "parent": obj.parent.name if obj.parent else None,
            "rotation_mode": obj.rotation_mode, "matrix_world": np.array(obj.matrix_world),
            "parent_bone": obj.parent_bone if obj.parent_type == "BONE" else None,
        }  # fmt: skip
        if obj.type == "CAMERA":
            info["camera"] = settings(obj.data, camera_keys)
        if obj.type == "LIGHT":
            info["light"] = settings(obj.data, light_keys)
        if obj.type == "ARMATURE":
            info["bones"] = [
                {"name": b.name, "parent": b.parent.name if b.parent else None, "length": b.length,
                 "rest": np.array(b.matrix_local), "pose": np.array(obj.pose.bones[b.name].matrix)}
                for b in obj.data.bones
            ]  # fmt: skip
        if obj.type == "MESH":
            mesh = obj.data
            info["mesh"] = mesh.name
            info["vertices"] = get(mesh.vertices, "co", 3)
            info["face_vertex_counts"] = get(mesh.polygons, "loop_total", 1, np.int32)
            info["face_vertex_indices"] = get(mesh.loops, "vertex_index", 1, np.int32)
            info["corner_normals"] = get(mesh.corner_normals, "vector", 3)
            info["custom_normals"] = mesh.has_custom_normals
            info["uvs"] = {layer.name: get(layer.data, "uv", 2) for layer in mesh.uv_layers}
            keys = mesh.shape_keys.key_blocks if mesh.shape_keys else []
            info["shape_keys"] = [
                {"name": k.name, "points": get(k.data, "co", 3), "value": k.value, "slider_min": k.slider_min,
                 "slider_max": k.slider_max, "relative_key": k.relative_key.name}
                for k in keys
            ]  # fmt: skip
            info["vertex_groups"] = [g.name for g in obj.vertex_groups]
            info["material_slots"] = [(s.material.name if s.material else None, s.link) for s in obj.material_slots]
            info["material_indices"] = get(mesh.polygons, "material_index", 1, np.int32)
            info["modifiers"] = [(m.type, getattr(m, "object", None) and m.object.name) for m in obj.modifiers]
            evaluated = obj.evaluated_get(depsgraph).to_mesh()
            points = get(evaluated.vertices, "co", 3)
            world = np.array(obj.matrix_world)
            info["world_vertices"] = points @ world[:3, :3].T + world[:3, 3]
            obj.evaluated_get(depsgraph).to_mesh_clear()
        out.append(info)

    def collection(c) -> dict:
        return {"name": c.name, "objects": [o.name for o in c.objects], "hide_render": c.hide_render,
                "hide_viewport": c.hide_viewport, "color_tag": c.color_tag,
                "children": [collection(child) for child in c.children]}  # fmt: skip

    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    frames = {}
    for frame in payload.get("frames", []):
        scene.frame_set(int(frame))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        state = {}
        for obj in bpy.data.objects:
            world = np.array(obj.matrix_world)
            entry = {"matrix_world": world}
            if obj.type == "MESH":
                evaluated = obj.evaluated_get(depsgraph).to_mesh()
                entry["world_vertices"] = get(evaluated.vertices, "co", 3) @ world[:3, :3].T + world[:3, 3]
                obj.evaluated_get(depsgraph).to_mesh_clear()
            if obj.type == "CAMERA":
                entry["camera"] = settings(obj.data, camera_keys)
            if obj.type == "LIGHT":
                entry["light"] = settings(obj.data, light_keys)
            state[obj.name] = entry
        state["_world"] = world_state()
        state["_exposure"] = scene.view_settings.exposure
        if scene.camera is not None and "points" in payload:
            state["_view"] = np.array([tuple(world_to_camera_view(scene, scene.camera, Vector(p))) for p in payload["points"]])
        frames[frame] = state
    actions = [
        {"name": a.name, "fake_user": a.use_fake_user, "frame_range": tuple(a.frame_range),
         "slots": [s.name_display for s in a.slots],
         "fcurves": sorted({f.data_path for layer in a.layers for strip in layer.strips for bag in strip.channelbags
                            for f in bag.fcurves})}
        for a in bpy.data.actions
    ]  # fmt: skip
    return {
        "objects": out,
        "collections": collection(scene.collection),
        "scene": {"fps": scene.render.fps / scene.render.fps_base, "frame_range": (scene.frame_start, scene.frame_end),
                  "eevee_light_threshold": scene.eevee.light_threshold, "exposure": scene.view_settings.exposure},
        "actions": actions,
        "cache_files": [c.filepath for c in bpy.data.cache_files],
        "texts": {t.name: t.as_string() for t in bpy.data.texts},
        "world": world_state(),
        "materials": {m.name: _material_summary(m) for m in bpy.data.materials if m.node_tree is not None},
        "scene_camera": scene.camera.name if scene.camera else None,
        "render": (scene.render.resolution_x, scene.render.resolution_y),
        "frames": frames,
    }


def import_usd(payload: dict) -> None:
    """Import payload["usd"] with Blender's own USD importer (into its default scene, which has a World for
    dome lights) and save it as payload["path"]: a reference for comparing with sceneport's .blend writer."""
    import bpy

    bpy.ops.wm.read_factory_settings()
    for obj in list(bpy.data.objects):  # the default cube, camera and light
        bpy.data.objects.remove(obj)
    bpy.ops.wm.usd_import(filepath=payload["usd"])
    bpy.ops.wm.save_as_mainfile(filepath=payload["path"], check_existing=False)


def _material_summary(material) -> dict:
    """A material's nodes (type, name, settings, unlinked inputs) and links, as plain data."""
    nodes = []
    for node in material.node_tree.nodes:
        entry = {"type": node.bl_idname, "name": node.name, "label": node.label}
        for attr in ("operation", "mode", "vector_type", "space", "uv_map", "extension", "interpolation"):
            if hasattr(node, attr):
                entry[attr] = getattr(node, attr)
        if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
            entry["image"] = (node.image.filepath, node.image.colorspace_settings.name, node.image.packed_file is not None)
        values = {}
        for i, socket in enumerate(node.inputs):
            if not socket.is_linked and hasattr(socket, "default_value"):
                value = socket.default_value
                values[f"{i}:{socket.name}"] = tuple(value) if hasattr(value, "__len__") else value
        entry["inputs"] = values
        nodes.append(entry)
    links = [(l.from_node.name, l.from_socket.name, l.to_node.name, l.to_socket.name) for l in material.node_tree.links]
    return {"render_method": material.surface_render_method, "fake_user": material.use_fake_user, "nodes": nodes, "links": links}


def render_eevee_center(payload: dict) -> float:
    """Render payload["path"] with EEVEE from the front, looking at the origin (a diagnostic): the mean
    linear value of the image's center, so tests can tell a lit subject from a black one."""
    import math

    import bpy
    import numpy as np

    bpy.ops.wm.open_mainfile(filepath=payload["path"], load_ui=False)
    scene = bpy.context.scene
    camera = bpy.data.objects.new("_probe", bpy.data.cameras.new("_probe"))
    scene.collection.objects.link(camera)
    camera.location, camera.rotation_euler = (0.0, -0.6, 0.0), (math.radians(90.0), 0.0, 0.0)
    camera.data.lens = 200.0
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "OPEN_EXR"
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        scene.render.filepath = tmp + "/probe.exr"
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(scene.render.filepath)
        pixels = np.array(image.pixels[:]).reshape(32, 32, 4)
    return float(pixels[12:20, 12:20, :3].mean())


ACTIONS = {"ping": ping, "write_blend": write_blend, "inspect_blend": inspect_blend, "import_usd": import_usd,
           "render_eevee_center": render_eevee_center}


def main(job_dir: str) -> None:
    job = Path(job_dir)
    response = {"result": None, "warnings": [], "error": None}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with open(job / "request.pkl", "rb") as f:
                request = pickle.load(f)
            action = ACTIONS.get(request["action"])
            if action is None:
                raise ValueError(f"unknown bridge action {request['action']!r}; known: {', '.join(ACTIONS)}")
            response["result"] = action(request["payload"])
        except Exception:
            response["error"] = traceback.format_exc()
    response["warnings"] = [(w.category.__name__, str(w.message)) for w in caught]
    with open(job / "response.pkl", "wb") as f:
        pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)
