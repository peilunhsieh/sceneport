"""Native FBX support: binary codec, document structure, and Scene conversion.

Layers, lowest first: `binary` (bytes <-> node tree), `document` (Objects,
Connections, Properties70, ...), then `from_fbx`/`to_fbx` (document <-> Scene).
"""

from .from_fbx import load, scene_from_document
from .to_fbx import document_from_scene, save

__all__ = ["load", "save", "scene_from_document", "document_from_scene"]
