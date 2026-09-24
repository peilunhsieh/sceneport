"""Default PropertyTemplates written into the Definitions section.

Values match what Blender writes. Readers use a template to fill in properties
an object leaves out, so these double as the defaults when reading.
"""

from __future__ import annotations

from .properties import Properties70, Property


def _props(*entries: tuple) -> Properties70:
    out = Properties70()
    for name, type_, value, *flags in entries:
        out.add(Property.make(name, type_, value, flags=flags[0] if flags else ""))
    return out


_ZERO3 = (0.0, 0.0, 0.0)
_ONE3 = (1.0, 1.0, 1.0)


def _fbx_node() -> Properties70:
    entries = [
        ("QuaternionInterpolate", "enum", 0),
        ("RotationOffset", "Vector3D", _ZERO3),
        ("RotationPivot", "Vector3D", _ZERO3),
        ("ScalingOffset", "Vector3D", _ZERO3),
        ("ScalingPivot", "Vector3D", _ZERO3),
        ("TranslationActive", "bool", 0),
        ("TranslationMin", "Vector3D", _ZERO3),
        ("TranslationMax", "Vector3D", _ZERO3),
        *((f"Translation{m}{a}", "bool", 0) for m in ("Min", "Max") for a in "XYZ"),
        ("RotationOrder", "enum", 0),
        ("RotationSpaceForLimitOnly", "bool", 0),
        *((f"RotationStiffness{a}", "double", 0.0) for a in "XYZ"),
        ("AxisLen", "double", 10.0),
        ("PreRotation", "Vector3D", _ZERO3),
        ("PostRotation", "Vector3D", _ZERO3),
        ("RotationActive", "bool", 0),
        ("RotationMin", "Vector3D", _ZERO3),
        ("RotationMax", "Vector3D", _ZERO3),
        *((f"Rotation{m}{a}", "bool", 0) for m in ("Min", "Max") for a in "XYZ"),
        ("InheritType", "enum", 0),
        ("ScalingActive", "bool", 0),
        ("ScalingMin", "Vector3D", _ZERO3),
        ("ScalingMax", "Vector3D", _ONE3),
        *((f"Scaling{m}{a}", "bool", 0) for m in ("Min", "Max") for a in "XYZ"),
        ("GeometricTranslation", "Vector3D", _ZERO3),
        ("GeometricRotation", "Vector3D", _ZERO3),
        ("GeometricScaling", "Vector3D", _ONE3),
        *((f"{k}{a}", "double", 0.0) for k in ("MinDampRange", "MaxDampRange", "MinDampStrength", "MaxDampStrength") for a in "XYZ"),
        *((f"PreferedAngle{a}", "double", 0.0) for a in "XYZ"),
        ("LookAtProperty", "object", None),
        ("UpVectorProperty", "object", None),
        ("Show", "bool", 1),
        ("NegativePercentShapeSupport", "bool", 1),
        ("DefaultAttributeIndex", "int", -1),
        ("Freeze", "bool", 0),
        ("LODBox", "bool", 0),
        ("Lcl Translation", "Lcl Translation", _ZERO3, "A"),
        ("Lcl Rotation", "Lcl Rotation", _ZERO3, "A"),
        ("Lcl Scaling", "Lcl Scaling", _ONE3, "A"),
        ("Visibility", "Visibility", 1.0, "A"),
        ("Visibility Inheritance", "Visibility Inheritance", 1),
    ]
    return _props(*entries)


def _fbx_mesh() -> Properties70:
    return _props(
        ("Color", "ColorRGB", (0.8, 0.8, 0.8)),
        ("BBoxMin", "Vector3D", _ZERO3),
        ("BBoxMax", "Vector3D", _ZERO3),
        ("Primary Visibility", "bool", 1),
        ("Casts Shadows", "bool", 1),
        ("Receive Shadows", "bool", 1),
    )


# Object node name -> (template name, factory). A fresh copy is built on each call.
_TEMPLATES = {
    "Model": ("FbxNode", _fbx_node),
    "Geometry": ("FbxMesh", _fbx_mesh),
}


def default_template(kind: str) -> tuple[str, Properties70] | None:
    """(template name, properties) for object node name `kind`, or None if we have none."""
    entry = _TEMPLATES.get(kind)
    return None if entry is None else (entry[0], entry[1]())


def default_global_settings() -> Properties70:
    """Y up, +Z front, +X right, 1 unit = 1 cm, 24 fps (as written by Blender)."""
    return _props(
        ("UpAxis", "int", 1),
        ("UpAxisSign", "int", 1),
        ("FrontAxis", "int", 2),
        ("FrontAxisSign", "int", 1),
        ("CoordAxis", "int", 0),
        ("CoordAxisSign", "int", 1),
        ("OriginalUpAxis", "int", -1),
        ("OriginalUpAxisSign", "int", 1),
        ("UnitScaleFactor", "double", 1.0),
        ("OriginalUnitScaleFactor", "double", 1.0),
        ("AmbientColor", "ColorRGB", _ZERO3),
        ("DefaultCamera", "KString", "Producer Perspective"),
        ("TimeMode", "enum", 11),
        ("TimeSpanStart", "KTime", 0),
        ("TimeSpanStop", "KTime", 46186158000),
        ("CustomFrameRate", "double", 24.0),
    )
