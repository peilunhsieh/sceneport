"""Layer 2: FBX document structure on top of the node tree."""

from .document import NAME_SEP, ROOT_ID, Connection, FBXDocument, FBXObject
from .properties import Properties70, Property

__all__ = ["FBXDocument", "FBXObject", "Connection", "Properties70", "Property", "ROOT_ID", "NAME_SEP"]
