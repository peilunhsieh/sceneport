"""FBX lights: a Model (subtype "Light") with a NodeAttribute (subtype "Light") holding the light's properties.

Written and read the way Blender's FBX exporter and importer do: LightType 0 point,
1 directional, 2 spot, 3 area; Color; Intensity; CastShadows / ShadowColor; spot
OuterAngle (the full cone, degrees) and InnerAngle = OuterAngle * (1 - softness);
quadratic decay. FBX lights shine down their node's -Y axis; the -Z <-> -Y turn is
done on the nodes (interop/reframe.py), not here.

FBX has no physical light units. Intensity follows Blender's two conventions, so a
light keeps its brightness between Blender, FBX and USD: FBX Intensity = Blender
watts x 100 (Blender's FBX exporter), and USD intensity = watts / pi, or a sun's
strength / 4 (Blender's USD exporter).

FBX can't hold dome lights, area sizes, cylinders, colour temperature or the
diffuse / specular / normalize controls: they are dropped with a warning.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from ...scene.light import Light
from ...scene.types import Node
from .binary.node import FBXNode, Prop
from .document import FBXDocument, FBXObject, Properties70

POINT, DIRECTIONAL, SPOT, AREA, VOLUME = range(5)
# Animatable Light fields FBX can hold -> the FBX properties computed from them.
ANIMATED_PROPERTIES = {
    "intensity": ("Intensity",), "exposure": ("Intensity",), "color": ("Color",), "shadow_color": ("ShadowColor",),
    "cone_angle": ("OuterAngle", "InnerAngle"), "cone_softness": ("InnerAngle",),
}  # fmt: skip


def fbx_values(light: Light, fields: set[str], values: dict) -> dict[str, np.ndarray]:
    """FBX property curves ({property: (F,) or (F, 3) values}) for the animated light `fields`, from
    attribute_values(), with the same conversions as write_light (watts x 100, the full spot cone)."""
    props = {p for f in fields for p in ANIMATED_PROPERTIES.get(f, ())}
    out = {}
    if "Intensity" in props:
        out["Intensity"] = values["intensity"] * 2.0 ** values["exposure"] * _watts_per_unit(light.type) * 100.0
    if "Color" in props:
        out["Color"] = values["color"]
    if "ShadowColor" in props:
        out["ShadowColor"] = values["shadow_color"]
    if {"OuterAngle", "InnerAngle"} & props and light.is_spot:
        outer = 2.0 * values["cone_angle"]
        out["OuterAngle"] = outer
        out["InnerAngle"] = outer * (1.0 - values["cone_softness"])
    return out


def _watts_per_unit(light_type: str) -> float:
    """Blender watts (or sun strength) per unit of USD-style intensity (Blender's USD exporter, inverted)."""
    return 4.0 if light_type == "distant" else math.pi


def read_light(doc: FBXDocument, model: FBXObject) -> Light | None:
    """The light of a light Model, from its NodeAttribute."""
    attribute = next((a for a in doc.children_of(model, "NodeAttribute") if a.subtype == "Light"), None)
    if attribute is None:
        return None

    def get(name: str, default):
        return doc.get_property(attribute, name, default)

    fbx_type = int(get("LightType", POINT))
    kind = {POINT: "sphere", DIRECTIONAL: "distant", SPOT: "sphere", AREA: "rect"}.get(fbx_type)
    if kind is None:
        warnings.warn(f"{model.name!r}: FBX light type {fbx_type} (volume) read as a point light")
        kind = "sphere"
    watts = float(get("Intensity", 100.0)) / 100.0
    light = Light(
        kind,
        color=np.array(get("Color", (1.0, 1.0, 1.0)), dtype=np.float64),
        intensity=watts / _watts_per_unit(kind),
        cast_shadows=bool(get("CastShadows", 1)),
        shadow_color=np.array(get("ShadowColor", (0.0, 0.0, 0.0)), dtype=np.float64),
    )
    if kind == "sphere":
        light.radius = 0.0  # FBX point lights have no size
    if fbx_type == SPOT:
        outer = float(get("OuterAngle", 45.0))
        inner = float(get("InnerAngle", 0.0))
        light.cone_angle = outer / 2.0
        light.cone_softness = 1.0 - inner / outer if outer else 0.0
    return light


def write_light(doc: FBXDocument, node: Node) -> FBXObject:
    """The NodeAttribute of a light node (not a dome: FBX has none), laid out as Blender writes it."""
    light = node.light
    lost = []
    if light.type == "cylinder":
        lost.append("cylinder shape (written as a point light)")
    if light.type in ("rect", "disk", "cylinder"):
        lost.append("size")
    if light.type == "disk":
        lost.append("disk shape (written as a rectangular area light)")
    if light.temperature is not None:
        lost.append("colour temperature")
    if (light.diffuse, light.specular, light.normalize) != (1.0, 1.0, False):
        lost.append("diffuse / specular / normalize")
    if lost:
        warnings.warn(f"light {node.name!r}: FBX can't hold its {', '.join(lost)}")

    fbx_type = {"distant": DIRECTIONAL, "rect": AREA, "disk": AREA}.get(light.type, SPOT if light.is_spot else POINT)
    watts = light.intensity * 2.0**light.exposure * _watts_per_unit(light.type)
    props = Properties70()
    props.set("LightType", "enum", fbx_type)
    props.set("CastLight", "bool", 1)
    props.set("Color", "Color", [float(c) for c in light.color], flags="A")
    props.set("Intensity", "Number", watts * 100.0, flags="A")
    props.set("DecayType", "enum", 2)  # quadratic, as Blender writes
    props.set("CastShadows", "bool", int(light.cast_shadows))
    props.set("ShadowColor", "Color", [float(c) for c in light.shadow_color], flags="A")
    if fbx_type == SPOT:
        outer = 2.0 * light.cone_angle
        props.set("OuterAngle", "Number", outer, flags="A")
        props.set("InnerAngle", "Number", outer * (1.0 - light.cone_softness), flags="A")
    if fbx_type == AREA:
        props.set("AreaLightShape", "enum", 0)  # rectangle
    attribute = doc.add_object("NodeAttribute", node.name, "Light", cls="NodeAttribute", props=props, children=[
        FBXNode("GeometryVersion", [Prop("I", 124)]),
        FBXNode("TypeFlags", [Prop("S", "Light")]),
    ])  # fmt: skip
    attribute.props_index = 1
    return attribute
