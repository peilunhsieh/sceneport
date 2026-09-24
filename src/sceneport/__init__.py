"""Lightweight 3D scene I/O in Python + numpy."""

__version__ = "0.1.0"

import os  # noqa: E402
from pathlib import Path  # noqa: E402

from .scene import AXES, Mesh, Node, Primvar, Scene, Transform  # noqa: E402


def _formats():
    from .interop import fbx, usd

    return {".fbx": fbx, **{ext: usd for ext in usd.EXTENSIONS}}


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


__all__ = ["Scene", "Node", "Mesh", "Primvar", "Transform", "AXES", "load", "save"]
