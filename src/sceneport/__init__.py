"""Lightweight 3D scene I/O in Python + numpy."""

__version__ = "0.1.0"

import os  # noqa: E402
from pathlib import Path  # noqa: E402

from .scene import (  # noqa: E402
    AXES, Animation, AssetPath, BlendShape, Camera, Channel, Light, Connection, Inbetween, Material, Mesh, Node, NodeGraph, Port,
    PointSamples, Primvar, Scene, ShaderNode, Skin, Transform,
)  # fmt: skip


def _formats():
    from .interop import blend, fbx, obj, ply, usd

    return {".fbx": fbx, **{ext: usd for ext in usd.EXTENSIONS}, ".obj": obj, ".ply": ply, ".blend": blend}


def load(path: str | os.PathLike) -> Scene:
    """Load a scene; the format is picked from the file extension."""
    return _format_for(path).load(path)


def save(scene: Scene, path: str | os.PathLike, **kwargs) -> None:
    """Save `scene`; the format is picked from the file extension. `kwargs` go to the format's save."""
    _format_for(path).save(scene, path, **kwargs)


def _format_for(path: str | os.PathLike):
    suffix = Path(path).suffix.lower()
    formats = _formats()
    if suffix not in formats:
        raise ValueError(f"unsupported file extension {suffix!r}; supported: {', '.join(formats)}")
    return formats[suffix]


__all__ = [
    "Scene", "Node", "Mesh", "Primvar", "Transform", "BlendShape", "Inbetween", "Skin", "Animation", "Channel",
    "PointSamples", "Camera", "Light", "Material", "ShaderNode", "NodeGraph", "Port", "Connection", "AssetPath", "AXES", "load", "save",
]
