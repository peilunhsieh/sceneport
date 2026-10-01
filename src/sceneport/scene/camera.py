"""Cameras: lens settings attached to a Node (the node's transform places and aims it).

A camera looks down its node's local -Z axis with +Y up (the USD, glTF and Blender
convention). Lens values use physical units: focal length, film (aperture) size
and film offset in millimeters; clipping planes and focus distance in scene units.
Orthographic cameras keep the same fields (USD's convention: the apertures give
the view size).
Formats convert on read and write: USD stores lens values in tenths of a scene
unit, FBX stores film sizes in inches and aims cameras down +X.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Projection = Literal["perspective", "orthographic"]


@dataclass
class Camera:
    projection: Projection = "perspective"
    focal_length: float = 50.0  # mm
    horizontal_aperture: float = 36.0  # film width, mm
    vertical_aperture: float = 24.0  # film height, mm
    horizontal_aperture_offset: float = 0.0  # film shift, mm
    vertical_aperture_offset: float = 0.0
    near: float = 0.1  # clipping, scene units
    far: float = 1000.0
    focus_distance: float | None = None  # scene units; None = not set
    f_stop: float | None = None  # None = no depth of field
    # Format-specific data kept for round trips, keyed by format.
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def horizontal_fov(self) -> float:
        """Horizontal field of view in degrees (perspective)."""
        return math.degrees(2 * math.atan(self.horizontal_aperture / (2 * self.focal_length)))

    @property
    def vertical_fov(self) -> float:
        """Vertical field of view in degrees (perspective)."""
        return math.degrees(2 * math.atan(self.vertical_aperture / (2 * self.focal_length)))
