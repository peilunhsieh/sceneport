"""Layer 3: plain numpy scene model and its conversion to/from FBXDocument."""

from .from_fbx import load, scene_from_document
from .to_fbx import document_from_scene, save
from .types import Mesh, Node, Primvar, Scene, Transform

__all__ = [
    "Scene", "Node", "Mesh", "Primvar", "Transform",
    "load", "save", "scene_from_document", "document_from_scene",
]  # fmt: skip
