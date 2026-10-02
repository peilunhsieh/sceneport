"""Lights attached to a Node (the node's transform places and orients them).

Types follow USD (UsdLux): "dome" (an environment lit by an image all around the
scene, an HDRI), "distant" (sun), "sphere" (point light; a spotlight with a cone),
"rect", "disk" and "cylinder" (area lights). Lights shine down the node's local -Z
axis (distant, rect, disk, and spot cones). A dome's image pole (the "up" of a
lat-long image) lies along the node's +Y axis, USD's classic DomeLight convention;
rotating the node turns the environment. `Scene.environment` finds the scene's
dome light.

Values are USD's: brightness is intensity x 2^exposure, times color (and the color
temperature when set); sizes are in scene units, angles in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .material import AssetPath

LightType = Literal["dome", "distant", "sphere", "rect", "disk", "cylinder"]
LIGHT_TYPES = ("dome", "distant", "sphere", "rect", "disk", "cylinder")
TextureFormat = Literal["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"]


@dataclass
class Light:
    type: LightType = "dome"
    color: NDArray[np.float64] = field(default_factory=lambda: np.ones(3))
    intensity: float = 1.0
    exposure: float = 0.0  # stops: brightness is intensity * 2**exposure
    temperature: float | None = None  # color temperature in Kelvin, multiplied into color; None = not used
    diffuse: float = 1.0  # multipliers on the light's diffuse and specular response
    specular: float = 1.0
    normalize: bool = False  # True: total power doesn't change with the light's size
    radius: float = 0.5  # sphere, disk, cylinder; a sphere with radius 0 is a true point light
    width: float = 1.0  # rect
    height: float = 1.0
    length: float = 1.0  # cylinder
    angle: float = 0.53  # distant: apparent size in degrees (the sun is about 0.53)
    cone_angle: float | None = None  # spot (sphere only): half-angle of the cone in degrees; None = no cone
    cone_softness: float = 0.0  # 0 = hard edge, 1 = soft
    cast_shadows: bool = True
    shadow_color: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    texture: AssetPath | None = None  # dome: the environment image; rect: an image on the light
    texture_format: TextureFormat = "automatic"  # dome; USD's default: deduced from the file (lat-long for an HDRI)
    # Format-specific data kept for round trips, keyed by format.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in LIGHT_TYPES:
            raise ValueError(f"light type must be one of {LIGHT_TYPES}, got {self.type!r}")
        self.color = np.asarray(self.color, dtype=np.float64).reshape(3)
        self.shadow_color = np.asarray(self.shadow_color, dtype=np.float64).reshape(3)

    @property
    def is_spot(self) -> bool:
        return self.type == "sphere" and self.cone_angle is not None
