"""FBX cameras: a Model (subtype "Camera") with a NodeAttribute (subtype "Camera") holding the lens.

Lens properties as Blender's FBX exporter writes and its importer reads them:
FocalLength (mm), FilmWidth / FilmHeight / FilmOffsetX / FilmOffsetY (inches),
NearPlane / FarPlane / FocusDistance (scene units), CameraProjectionType (0
perspective, 1 orthographic). FBX cameras look down their node's +X axis; the
-Z <-> +X turn is done on the nodes (interop/reframe.py), not here.

Render settings: FBX has no scene resolution; like Blender's exporter, every
camera gets it (AspectRatioMode 2, fixed resolution; AspectWidth / AspectHeight
in pixels; PixelAspectRatio), and GlobalSettings' DefaultCamera names the camera
rendered.
"""

from __future__ import annotations

import math
import warnings

from ...scene.camera import Camera
from ...scene.types import Node, RenderSettings
from .binary.node import FBXNode, Prop
from .document import FBXDocument, FBXObject, Properties70, Property

MM_PER_INCH = 25.4
FIXED_RESOLUTION = 2  # AspectRatioMode: AspectWidth / AspectHeight are the resolution in pixels
# Animatable Camera field -> (FBX property, factor from sceneport's value to FBX's). f_stop has no FBX property.
ANIMATED_PROPERTIES = {
    "focal_length": ("FocalLength", 1.0), "horizontal_aperture": ("FilmWidth", 1 / MM_PER_INCH),
    "vertical_aperture": ("FilmHeight", 1 / MM_PER_INCH), "horizontal_aperture_offset": ("FilmOffsetX", 1 / MM_PER_INCH),
    "vertical_aperture_offset": ("FilmOffsetY", 1 / MM_PER_INCH), "near": ("NearPlane", 1.0), "far": ("FarPlane", 1.0),
    "focus_distance": ("FocusDistance", 1.0),
}  # fmt: skip


def fbx_values(fields: set[str], values: dict) -> dict[str, "np.ndarray"]:
    """FBX property curves ({property: (F,) values}) for the animated camera `fields`, from attribute_values()."""
    return {ANIMATED_PROPERTIES[f][0]: values[f] * ANIMATED_PROPERTIES[f][1] for f in fields if f in ANIMATED_PROPERTIES}


def from_fbx_curve(prop: str) -> tuple[str, float] | None:
    """(camera field, factor from FBX's value to sceneport's) for an animated FBX camera property."""
    return next(((f, 1 / scale) for f, (p, scale) in ANIMATED_PROPERTIES.items() if p == prop), None)


def read_camera(doc: FBXDocument, model: FBXObject) -> Camera | None:
    """The lens of a camera Model, from its NodeAttribute (FBX SDK defaults where unset)."""
    attribute = next((a for a in doc.children_of(model, "NodeAttribute") if a.subtype == "Camera"), None)
    if attribute is None:
        return None

    def get(name: str, default: float) -> float:
        value = doc.get_property(attribute, name, default)
        return float(value[0] if isinstance(value, tuple) else value)

    film_width, film_height = get("FilmWidth", 0.816), get("FilmHeight", 0.612)
    return Camera(
        projection="orthographic" if int(get("CameraProjectionType", 0)) == 1 else "perspective",
        focal_length=get("FocalLength", 34.89327621672628),
        horizontal_aperture=film_width * MM_PER_INCH,
        vertical_aperture=film_height * MM_PER_INCH,
        horizontal_aperture_offset=get("FilmOffsetX", 0.0) * MM_PER_INCH,
        vertical_aperture_offset=get("FilmOffsetY", 0.0) * MM_PER_INCH,
        near=get("NearPlane", 10.0),
        far=get("FarPlane", 4000.0),
        focus_distance=get("FocusDistance", 0.0) or None,
        f_stop=None,  # FBX has no f-stop; UseDepthOfField alone can't give one
        extras={"fbx": {"use_depth_of_field": bool(get("UseDepthOfField", 0))}},
    )


def read_render(doc: FBXDocument, cameras: dict[int, Node]) -> RenderSettings | None:
    """The render settings, from `cameras` (camera Model id -> its node): the camera GlobalSettings'
    DefaultCamera names, if it's one of them, else the first with a fixed resolution, and that camera's
    resolution and pixel aspect ratio. None when no camera has a fixed resolution."""
    default = doc.global_settings.get("DefaultCamera", "")
    ordered = sorted(cameras.items(), key=lambda item: cameras[item[0]].name != default)  # DefaultCamera's first
    for model_id, node in ordered:
        attribute = next((a for a in doc.children_of(doc.objects[model_id], "NodeAttribute") if a.subtype == "Camera"), None)
        if attribute is None or int(doc.get_property(attribute, "AspectRatioMode", 0)) != FIXED_RESOLUTION:
            continue
        width, height = (round(float(doc.get_property(attribute, name, 0.0))) for name in ("AspectWidth", "AspectHeight"))
        if width < 1 or height < 1:
            continue
        return RenderSettings((width, height), float(doc.get_property(attribute, "PixelAspectRatio", 1.0)),
                              node if node.name == default else None)  # fmt: skip
    return None


def write_camera(doc: FBXDocument, node: Node, render: RenderSettings | None = None) -> FBXObject:
    """The NodeAttribute of a camera node, laid out as Blender writes it, with the image `render` gives."""
    camera = node.camera
    if camera.exposure != 0.0:
        warnings.warn(f"camera {node.name!r}: FBX has no camera exposure; its {camera.exposure:+g} stops aren't written")
    width, height = camera.horizontal_aperture / MM_PER_INCH, camera.vertical_aperture / MM_PER_INCH
    props = Properties70()
    props.set("CameraProjectionType", "enum", 1 if camera.projection == "orthographic" else 0)
    if render is not None:
        props.set("AspectRatioMode", "enum", FIXED_RESOLUTION)
        props.set("AspectWidth", "double", float(render.resolution[0]))
        props.set("AspectHeight", "double", float(render.resolution[1]))
        props.set("PixelAspectRatio", "double", float(render.pixel_aspect_ratio))
    props.set("FocalLength", "double", camera.focal_length)
    props.set("FilmWidth", "double", width)
    props.set("FilmHeight", "double", height)
    props.set("FilmAspectRatio", "double", width / height if height else 1.0)
    props.set("FilmOffsetX", "double", camera.horizontal_aperture_offset / MM_PER_INCH)
    props.set("FilmOffsetY", "double", camera.vertical_aperture_offset / MM_PER_INCH)
    props.set("ApertureMode", "enum", 3)  # the field of view follows the focal length
    props.set("GateFit", "enum", 2)  # fit horizontally
    for name, degrees in (("FieldOfView", camera.horizontal_fov), ("FieldOfViewX", camera.horizontal_fov),
                          ("FieldOfViewY", camera.vertical_fov)):  # fmt: skip
        props.add(Property(name, name, "", "A", (Prop("D", float(degrees)),)))
    props.set("NearPlane", "double", camera.near)
    props.set("FarPlane", "double", camera.far)
    if camera.focus_distance is not None:
        props.set("FocusDistance", "double", camera.focus_distance)
    props.set("UseDepthOfField", "bool", int(camera.f_stop is not None))
    attribute = doc.add_object("NodeAttribute", node.name, "Camera", cls="NodeAttribute", props=props, children=[
        FBXNode("TypeFlags", [Prop("S", "Camera")]),
        FBXNode("GeometryVersion", [Prop("I", 124)]),
        FBXNode("Position", [Prop("D", 0.0), Prop("D", 0.0), Prop("D", 0.0)]),
        FBXNode("Up", [Prop("D", 0.0), Prop("D", 1.0), Prop("D", 0.0)]),
        FBXNode("LookAt", [Prop("D", 1.0), Prop("D", 0.0), Prop("D", 0.0)]),  # FBX cameras look down +X
        FBXNode("ShowInfoOnMoving", [Prop("I", 1)]),
        FBXNode("ShowAudio", [Prop("I", 0)]),
        FBXNode("AudioColor", [Prop("D", 0.0), Prop("D", 1.0), Prop("D", 0.0)]),
        FBXNode("CameraOrthoZoom", [Prop("D", 1.0)]),
    ])  # fmt: skip
    attribute.props_index = 0
    return attribute
