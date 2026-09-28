"""Map between Scene units/axes and FBX GlobalSettings.

The file's GlobalSettings are kept in `scene.extras["fbx"]["global_settings"]`
so that saving a loaded scene writes them back unchanged (frame rate, time span,
...), with only the units and axes updated from the Scene fields.
"""

from __future__ import annotations

import math

from ...math import seconds_to_ticks
from ...scene.types import Scene
from .animation import time_mode_for
from .document.properties import Properties70
from .document.templates import default_global_settings

EXTRAS_KEY = "global_settings"

# Scene field -> (FBX axis property, FBX sign property)
AXIS_PROPERTIES = {
    "up_axis": ("UpAxis", "UpAxisSign"),
    "front_axis": ("FrontAxis", "FrontAxisSign"),
    "right_axis": ("CoordAxis", "CoordAxisSign"),
}


def axis_name(axis: int, sign: int) -> str:
    """(1, 1) -> "+Y"."""
    return ("+" if sign >= 0 else "-") + "XYZ"[axis]


def axis_value(name: str) -> tuple[int, int]:
    """"+Y" -> (1, 1)."""
    return "XYZ".index(name[1]), 1 if name[0] == "+" else -1


def read_global_settings(settings: Properties70, scene: Scene) -> None:
    """Set `scene`'s units and axes from `settings` and keep a copy in scene.extras."""
    defaults = default_global_settings()
    scene.meters_per_unit = settings.get("UnitScaleFactor", 1.0) / 100
    for field, (axis, sign) in AXIS_PROPERTIES.items():
        value = settings.get(axis, defaults.get(axis)), settings.get(sign, defaults.get(sign))
        setattr(scene, field, axis_name(*value))
    scene.extras.setdefault("fbx", {})[EXTRAS_KEY] = settings.copy()


def write_global_settings(scene: Scene) -> Properties70:
    """GlobalSettings for `scene`: the kept copy if any, else defaults, updated with units and axes."""
    kept = scene.extras.get("fbx", {}).get(EXTRAS_KEY)
    settings = kept.copy() if kept is not None else default_global_settings()

    unit_scale = scene.meters_per_unit * 100  # FBX counts centimeters per unit
    if not math.isclose(settings.get("UnitScaleFactor", 1.0), unit_scale, rel_tol=1e-12):
        settings.set("UnitScaleFactor", "double", unit_scale)
        if kept is None:
            settings.set("OriginalUnitScaleFactor", "double", unit_scale)
    for field, (axis, sign) in AXIS_PROPERTIES.items():
        value, direction = axis_value(getattr(scene, field))
        settings.set(axis, "int", value)
        settings.set(sign, "int", direction)
    if scene.animations:  # frame rate and time span of the first clip
        clip = scene.animations[0]
        mode, custom = time_mode_for(clip.fps)
        settings.set("TimeMode", "enum", mode)
        settings.set("CustomFrameRate", "double", custom)
        start, end = clip.time_range()
        settings.set("TimeSpanStart", "KTime", seconds_to_ticks(start), subtype="Time")
        settings.set("TimeSpanStop", "KTime", seconds_to_ticks(end), subtype="Time")
    return settings
