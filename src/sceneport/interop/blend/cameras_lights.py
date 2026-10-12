"""Cameras, lights and the World for the .blend writer, inside Blender (imports bpy).

Values follow Blender's own USD importer, so a scene reaches Blender the same way
through sceneport's .blend or through Blender's USD import of sceneport's USD:

- Cameras: sensor width / height from the apertures with a horizontal sensor fit,
  shift = offset / sensor width, clipping, depth of field only when an f-stop is
  set; orthographic view width (the horizontal aperture, in mm here meaning
  thousandths of a meter: USD's tenths of a unit) as ortho_scale. Blender's
  cameras look down -Z with +Y up, as sceneport's.
- Lights: energy = intensity x pi (a sun's strength = intensity x 4), with
  exposure, color temperature, normalize and the diffuse / specular factors as
  Blender's own properties; sphere radius = shadow_soft_size, a spot's size =
  2 x cone angle and blend = softness, rect / disk lights = area lights (a disk's
  size = 2 x radius), cylinders become point lights (warned). One difference: a
  sun's angle is used as is (USD's angular diameter, as Blender's), where
  Blender's USD importer doubles it.
- EEVEE skips a point, spot or area light beyond the distance where its power
  falls to the scene's light threshold, sqrt(power / threshold), with power =
  max(color) x energy x 2^exposure / 100 (eevee_light.cc), and fades it towards
  there. Physically scaled lights (e.g. a light stage's 0.37 W lamps 1 m from
  the subject) fall below Blender's default 0.01 and render black in EEVEE
  while lit in Cycles, so the threshold is lowered until every such light
  reaches 4x the scene's size (fading under 0.5%); it is never raised.
- A dome light becomes the World: Environment Texture -> Background (strength =
  intensity x 2^exposure), oriented by a Mapping node (point type, XYZ Euler)
  set to R_default @ R^-1 for the dome's world rotation R, R_default being
  Blender's own World orientation as a dome (Euler 90, 0, 90 in Z-up), as
  Blender's USD importer does. Only the first dome becomes the World.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import bpy
import numpy as np

from ...math import euler_from_matrix, euler_matrix
from ...scene.camera import Camera
from ...scene.light import Light
from ..fbx.light import _watts_per_unit

DEFAULT_WORLD = euler_matrix([90.0, 0.0, 90.0])[:3, :3]  # Blender's World orientation, as a dome's rotation
RENDER_WIDTH = 1920

# Animated camera / light fields -> (Blender data path, factor from sceneport's value).
CAMERA_PATHS = {
    "focal_length": ("lens", 1.0), "near": ("clip_start", 1.0), "far": ("clip_end", 1.0),
    "focus_distance": ("dof.focus_distance", 1.0), "f_stop": ("dof.aperture_fstop", 1.0),
    "horizontal_aperture": ("sensor_width", 1.0), "vertical_aperture": ("sensor_height", 1.0),
}  # fmt: skip


def light_paths(light: Light) -> dict[str, tuple[str, float]]:
    """Animated light fields -> (Blender data path, factor) for this light's Blender type."""
    paths = {
        "intensity": ("energy", _watts_per_unit(light.type)), "exposure": ("exposure", 1.0), "color": ("color", 1.0),
        "temperature": ("temperature", 1.0), "diffuse": ("diffuse_factor", 1.0), "specular": ("specular_factor", 1.0),
    }  # fmt: skip
    if light.type in ("sphere", "cylinder"):
        paths["radius"] = ("shadow_soft_size", 1.0)
    if light.type == "disk":
        paths["radius"] = ("size", 2.0)
    if light.type == "rect":
        paths.update(width=("size", 1.0), height=("size_y", 1.0))
    if light.type == "distant":
        paths["angle"] = ("angle", math.pi / 180.0)
    if light.is_spot:
        paths.update(cone_angle=("spot_size", math.pi / 90.0), cone_softness=("spot_blend", 1.0))
    return paths


def camera_data(camera: Camera, name: str):
    data = bpy.data.cameras.new(name)
    data.type = "ORTHO" if camera.projection == "orthographic" else "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width, data.sensor_height = camera.horizontal_aperture, camera.vertical_aperture
    data.lens = camera.focal_length
    data.shift_x = camera.horizontal_aperture_offset / camera.horizontal_aperture
    data.shift_y = camera.vertical_aperture_offset / camera.horizontal_aperture
    if camera.projection == "orthographic":
        data.ortho_scale = camera.horizontal_aperture / 1000.0
    data.clip_start, data.clip_end = camera.near, camera.far
    if camera.focus_distance is not None:
        data.dof.focus_distance = camera.focus_distance
    data.dof.use_dof = camera.f_stop is not None
    if camera.f_stop is not None:
        data.dof.aperture_fstop = camera.f_stop
    return data


def use_camera(obj, camera: Camera) -> None:
    """Make `obj` the scene's camera, with the render height giving the film's aspect."""
    scene = bpy.context.scene
    scene.camera = obj
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = max(1, round(RENDER_WIDTH * camera.vertical_aperture / camera.horizontal_aperture))
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0


def light_data(light: Light, name: str, node_name: str):
    lost = []
    if light.type == "cylinder":
        lost.append("cylinder shape (written as a point light)")
    if np.any(light.shadow_color != 0):
        lost.append("shadow color")
    if light.texture is not None:
        lost.append("texture")
    if lost:
        warnings.warn(f"light {node_name!r}: Blender can't hold its {', '.join(lost)}")
    kind = {"distant": "SUN", "rect": "AREA", "disk": "AREA"}.get(light.type, "SPOT" if light.is_spot else "POINT")
    data = bpy.data.lights.new(name, kind)
    data.color = light.color
    data.energy = light.intensity * _watts_per_unit(light.type)
    data.exposure = light.exposure
    data.use_temperature = light.temperature is not None
    if light.temperature is not None:
        data.temperature = light.temperature
    data.normalize = light.normalize
    data.diffuse_factor, data.specular_factor = light.diffuse, light.specular
    data.use_shadow = light.cast_shadows
    if kind in ("POINT", "SPOT"):
        data.shadow_soft_size = light.radius
    if kind == "SPOT":
        data.spot_size = math.radians(2.0 * light.cone_angle)
        data.spot_blend = light.cone_softness
    if kind == "SUN":
        data.angle = math.radians(light.angle)
    if light.type == "rect":
        data.shape, data.size, data.size_y = "RECTANGLE", light.width, light.height
    if light.type == "disk":
        data.shape, data.size = "DISK", 2.0 * light.radius
    return data


def world_from_dome(light: Light, rotation: np.ndarray, blend_dir: Path | None, name: str = "World"):
    """The scene's World showing the dome `light` with world rotation `rotation` (3x3, Blender's axes)."""
    world = bpy.data.worlds.new(name)
    bpy.context.scene.world = world
    if world.node_tree is None:  # older Blender: worlds start without nodes (use_nodes goes away in Blender 6)
        world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = next(n for n in nodes if n.bl_idname == "ShaderNodeBackground")
    background.inputs["Strength"].default_value = light.intensity * 2.0**light.exposure
    color = (*light.color, 1.0)
    if light.temperature is not None:
        warnings.warn("the environment's color temperature isn't written to the Blender World")
    if light.texture is None:
        background.inputs["Color"].default_value = color
        return world
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.vector_type = "POINT"
    mapping.inputs["Rotation"].default_value = dome_mapping_rotation(rotation)
    texture = nodes.new("ShaderNodeTexEnvironment")
    texture.projection = "MIRROR_BALL" if light.texture_format == "mirroredBall" else "EQUIRECTANGULAR"
    if light.texture_format in ("angular", "cubeMapVerticalCross"):
        warnings.warn(f"Blender's World has no {light.texture_format} environment images; read as lat-long")
    texture.image = _image(light.texture, blend_dir)
    links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    if np.allclose(light.color, 1.0):
        links.new(texture.outputs["Color"], background.inputs["Color"])
    else:
        tint = nodes.new("ShaderNodeMix")
        tint.data_type, tint.blend_type = "RGBA", "MULTIPLY"
        tint.inputs["Factor"].default_value = 1.0
        links.new(texture.outputs["Color"], tint.inputs["A"])
        tint.inputs["B"].default_value = color
        links.new(tint.outputs["Result"], background.inputs["Color"])
    return world


def dome_mapping_rotation(rotation: np.ndarray) -> list[float]:
    """The Mapping node's XYZ Euler radians for a dome with world rotation `rotation` (or (F, 3, 3))."""
    local = np.eye(4)[None].repeat(1 if rotation.ndim == 2 else len(rotation), 0)
    local[:, :3, :3] = DEFAULT_WORLD @ np.swapaxes(rotation.reshape(-1, 3, 3), -1, -2)
    radians = np.radians(euler_from_matrix(local, "XYZ"))
    return radians[0].tolist() if rotation.ndim == 2 else radians


def _image(asset, blend_dir: Path | None):
    path = asset.resolved or asset.path
    try:
        image = bpy.data.images.load(path, check_existing=True)
    except RuntimeError:
        warnings.warn(f"environment image {asset.path!r} not found; the World references it anyway")
        image = bpy.data.images.new(Path(asset.path).name, 1, 1)
        image.source = "FILE"
        image.filepath = path
    if blend_dir is not None and os.path.isabs(image.filepath):
        try:
            image.filepath = "//" + os.path.relpath(image.filepath, blend_dir).replace(os.sep, "/")
        except ValueError:  # another drive (Windows): keep it absolute
            pass
    return image


REACH = 4.0  # EEVEE lights reach this many times the scene's size: their edge fade then costs under 0.5%


def fit_eevee_light_threshold(scene=None) -> float:
    """Lower the scene's EEVEE light threshold so no point / spot / area light is cut off within REACH x the
    size of everything in the scene; returns the threshold (unchanged when the lights are bright enough)."""
    scene = scene or bpy.context.scene
    bpy.context.view_layer.update()
    corners = []
    for obj in scene.objects:
        world = np.array(obj.matrix_world)
        local = np.array([tuple(c) for c in obj.bound_box]) if obj.type == "MESH" else np.zeros((1, 3))
        corners.append(local @ world[:3, :3].T + world[:3, 3])
    if not corners:
        return scene.eevee.light_threshold
    points = np.concatenate(corners)
    reach = REACH * max(float(np.linalg.norm(points.max(0) - points.min(0))), 1e-3)
    threshold = scene.eevee.light_threshold
    for obj in scene.objects:
        if obj.type != "LIGHT" or obj.data.type == "SUN" or obj.data.use_custom_distance:
            continue
        power = _min_power(obj.data)
        if power > 0.0:
            threshold = min(threshold, power / reach**2)
    if threshold < scene.eevee.light_threshold:
        scene.eevee.light_threshold = max(threshold, 1e-16)  # EEVEE's own floor
    return scene.eevee.light_threshold


def _min_power(light) -> float:
    """EEVEE's influence power of `light` (max color x energy x 2^exposure / 100 x the strongest surface factor),
    at its dimmest animated energy / exposure; frames where it is off (energy 0) don't count: it needs no reach
    then (e.g. a light stage's one-light-at-a-time sequence keeps every light off most of the time)."""
    color = np.array(light.color) * (np.array(light.temperature_color) if light.use_temperature else 1.0)
    energy = _min_keyed(light, "energy", light.energy, nonzero=True)
    exposure = _min_keyed(light, "exposure", light.exposure)
    factor = max(light.diffuse_factor, light.specular_factor, getattr(light, "transmission_factor", 1.0))
    return float(color.max()) * abs(energy) * 2.0**exposure / 100.0 * factor


def _min_keyed(datablock, path: str, value: float, nonzero: bool = False) -> float:
    """The smallest of `value` and the keys animating `path` (`nonzero`: ignoring zeros; 0 if all are), in every
    action that has a slot for `datablock`: the assigned one, and others (e.g. later clips, kept with a fake user)
    through their slot named after it, as the writer names slots. Switching actions then keeps the lights lit."""
    data = datablock.animation_data
    assigned = (data.action, data.action_slot) if data is not None and data.action is not None else (None, None)
    keys = []
    for action in bpy.data.actions:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    slot = bag.slot
                    if (action, slot) != assigned and not (
                        slot.target_id_type == datablock.id_type and slot.name_display == datablock.name
                    ):
                        continue
                    keys += [k.co[1] for curve in bag.fcurves if curve.data_path == path for k in curve.keyframe_points]
    values = [v for v in (value, *keys) if not nonzero or v != 0.0]
    return min(values, default=0.0)
