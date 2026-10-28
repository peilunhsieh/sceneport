"""Blender (.blend) support through an installed Blender, run in the background.

`runner` finds Blender and runs actions in it; `bridge` and `to_blend` are the code
that runs inside Blender (the only modules importing bpy); `compat` prepares a
scene for Blender (pure numpy). Nothing is installed into Blender.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from ...scene.types import Scene
from .runner import Blender, BlenderError, find_blender, run_in_blender


def save(
    scene: Scene,
    path: str | os.PathLike,
    *,
    blender: Blender | str | os.PathLike | None = None,
    timeout: float | None = None,
    pack_images: bool = False,
    scripts: Sequence[str | os.PathLike | tuple[str | os.PathLike, Mapping]] = (),
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

    Texture images (materials' and the World's environment image) are referenced by paths relative to the
    .blend; with `pack_images` the ones found on disk are packed into it too (they keep those paths, where Blender's Unpack Resources puts them).
    Images embedded in the source (FBX) are always packed.

    `scripts`: Python files run inside the same Blender after the scene is built and before the file
    is saved, in order, as `__main__` (with `__file__` set), for setup sceneport doesn't do itself:
    node trees, modifiers, object settings. They run with Blender's own Python and bpy, see the
    scene's objects by their node names, and shouldn't save the file themselves (`bpy.data.filepath`
    is still empty then). Their errors and warnings come back as from the rest of the save. A script given
    as (path, params) sees `params` (a dict, pickled across, so plain values, paths and arrays) as the
    global `params`; one given as a path sees an empty dict.
    """
    entries = [script if isinstance(script, tuple) else (script, {}) for script in scripts]
    sources = [(str(Path(script).resolve()), Path(script).read_text(), dict(params)) for script, params in entries]
    payload = {"scene": scene, "path": str(Path(path).resolve()), "pack_images": pack_images, "scripts": sources}
    run_in_blender("write_blend", payload, blender=blender, timeout=timeout)


def load(path: str | os.PathLike) -> Scene:
    raise NotImplementedError("reading .blend files isn't supported yet; sceneport can write them")


__all__ = ["Blender", "BlenderError", "find_blender", "run_in_blender", "save", "load"]
