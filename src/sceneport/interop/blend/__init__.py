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
    timeout: float | None = None,
) -> None:
    """Write `scene` as a .blend file with an installed Blender (see find_blender for `blender`).

    The data is converted to Blender's Z-up meters. Each shape key's slider range is its blend shape's
    `range`, widened where the shape's default or animated weights fall outside it.
    So far meshes, blend shapes, joints (armatures, posed as the scene is), skins, collections and
    animation (an action per clip, the first assigned) are written; per-vertex animation goes to
    `<name>_cache.usdc` beside the .blend, read by Mesh Sequence Cache modifiers. Cameras, lights and
    the first dome light (as the World) and materials (Principled BSDF setups from their PBR reading)
    are written too. Scripts in scene.extras["blender"]["texts"] ({name: source}) are saved as Text
    datablocks, run on demand from Blender's Text Editor.
    """
    payload = {"scene": scene, "path": str(Path(path).resolve())}
    run_in_blender("write_blend", payload, blender=blender, timeout=timeout)


def load(path: str | os.PathLike) -> Scene:
    raise NotImplementedError("reading .blend files isn't supported yet; sceneport can write them")


__all__ = ["Blender", "BlenderError", "find_blender", "run_in_blender", "save", "load"]
