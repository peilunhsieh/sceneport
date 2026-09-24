"""Layer 1: binary FBX codec (bytes <-> generic FBXNode tree)."""

from .node import FBXFile, FBXNode, Prop
from .reader import FBXParseError, loads, read
from .writer import dumps, write

__all__ = ["FBXFile", "FBXNode", "Prop", "FBXParseError", "read", "loads", "write", "dumps"]
