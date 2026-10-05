"""Blender (.blend) support through an installed Blender, run in the background.

`runner` finds Blender and runs actions in it; `bridge` is the code that runs
inside Blender (the only module importing bpy). Nothing is installed into Blender.
"""

from .runner import Blender, BlenderError, find_blender, run_in_blender

__all__ = ["Blender", "BlenderError", "find_blender", "run_in_blender"]
