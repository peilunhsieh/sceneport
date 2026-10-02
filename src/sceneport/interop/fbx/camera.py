"""FBX cameras: a Model (subtype "Camera") with a NodeAttribute (subtype "Camera") holding the lens.

Lens properties as Blender's FBX exporter writes and its importer reads them:
FocalLength (mm), FilmWidth / FilmHeight / FilmOffsetX / FilmOffsetY (inches),
NearPlane / FarPlane / FocusDistance (scene units), CameraProjectionType (0
perspective, 1 orthographic). FBX cameras look down their node's +X axis; the
-Z <-> +X turn is done on the nodes (interop/reframe.py), not here.
"""

from __future__ import annotations

import math

from ...scene.camera import Camera
from ...scene.types import Node
from .binary.node import FBXNode, Prop
from .document import FBXDocument, FBXObject, Properties70, Property

MM_PER_INCH = 25.4


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


def write_camera(doc: FBXDocument, node: Node) -> FBXObject:
    """The NodeAttribute of a camera node, laid out as Blender writes it."""
    camera = node.camera
    width, height = camera.horizontal_aperture / MM_PER_INCH, camera.vertical_aperture / MM_PER_INCH
    props = Properties70()
    props.set("CameraProjectionType", "enum", 1 if camera.projection == "orthographic" else 0)
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
