"""Lights attached to a Node (the node's transform places and orients them).

For now one type: "dome", an environment light lit by an image all around the
scene (an HDRI). Its image's pole (the "up" of a lat-long image) lies along the
node's local +Y axis, USD's DomeLight convention; rotating the node turns the
environment. `Scene.environment` finds the scene's dome light.

Brightness follows USD: intensity x 2^exposure, times color.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .material import AssetPath

LightType = Literal["dome"]
TextureFormat = Literal["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"]


@dataclass
class Light:
    type: LightType = "dome"
    color: NDArray[np.float64] = field(default_factory=lambda: np.ones(3))
    intensity: float = 1.0
    exposure: float = 0.0  # stops: brightness is intensity * 2**exposure
    texture: AssetPath | None = None  # the environment image (dome)
    texture_format: TextureFormat = "automatic"  # USD's default: deduced from the file (lat-long for an HDRI)
    # Format-specific data kept for round trips, keyed by format.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.color = np.asarray(self.color, dtype=np.float64).reshape(3)
