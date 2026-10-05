"""Blender (.blend) support through an installed Blender, run in the background.

`runner` finds Blender and runs actions in it; `bridge` and `to_blend` are the code
that runs inside Blender (the only modules importing bpy); `compat` prepares a
scene for Blender (pure numpy). Nothing is installed into Blender.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...scene.types import Scene
from .runner import Blender, BlenderError, find_blender, run_in_blender


def save(
    scene: Scene,
    path: str | os.PathLike,
    *,
    blender: Blender | str | os.PathLike | None = None,
    slider_range: tuple[float, float] | None = None,
    timeout: float | None = None,
) -> None:
    """Write `scene` as a .blend file with an installed Blender (see find_blender for `blender`).

    The data is converted to Blender's Z-up meters. Shape key sliders are widened to hold every weight
    the scene uses, and at least `slider_range` when given (e.g. (-3, 3) to edit weights by hand).
    So far meshes and blend shapes are written; skins, materials, cameras, lights and animation are
    skipped with a warning.
    """
    options = {} if slider_range is None else {"slider_range": tuple(slider_range)}
    payload = {"scene": scene, "path": str(Path(path).resolve()), "options": options}
    run_in_blender("write_blend", payload, blender=blender, timeout=timeout)


def load(path: str | os.PathLike) -> Scene:
    raise NotImplementedError("reading .blend files isn't supported yet; sceneport can write them")


__all__ = ["Blender", "BlenderError", "find_blender", "run_in_blender", "save", "load"]
