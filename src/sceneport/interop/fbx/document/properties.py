"""Read/write helpers for Properties70 "P" entries.

Each entry is `P: name, type, subtype, flags, *values`, e.g.
`P: "Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ..binary.node import FBXNode, Prop

# FBX property type -> (default subtype, value type code, value count).
# Mirrors FBX_PROPERTIES_DEFINITIONS in Blender's fbx_utils.py.
PROPERTY_TYPES: dict[str, tuple[str, str | None, int]] = {
    "bool": ("", "I", 1),
    "int": ("Integer", "I", 1),
    "ULongLong": ("", "L", 1),
    "double": ("Number", "D", 1),
    "Number": ("", "D", 1),
    "enum": ("", "I", 1),
    "Vector3D": ("Vector", "D", 3),
    "Vector": ("", "D", 3),
    "ColorRGB": ("Color", "D", 3),
    "Color": ("", "D", 3),
    "KString": ("", "S", 1),
    "KTime": ("Time", "L", 1),
    "DateTime": ("", "S", 1),
    "object": ("", None, 0),
    "Compound": ("", None, 0),
    "Lcl Translation": ("", "D", 3),
    "Lcl Rotation": ("", "D", 3),
    "Lcl Scaling": ("", "D", 3),
    "Visibility": ("", "D", 1),
    "Visibility Inheritance": ("", "I", 1),
}


@dataclass
class Property:
    """One Properties70 entry. `raw` keeps the typed values as read or written."""

    name: str
    type: str
    subtype: str = ""
    flags: str = ""
    raw: tuple[Prop, ...] = ()

    @property
    def value(self) -> Any:
        """None for no values, a scalar for one value, a tuple for several."""
        if not self.raw:
            return None
        if len(self.raw) == 1:
            return self.raw[0].value
        return tuple(p.value for p in self.raw)

    @classmethod
    def make(
        cls, name: str, type: str, value: Any = None, *, subtype: str | None = None, flags: str = ""
    ) -> Property:
        """Build an entry for a known FBX property `type`, e.g. make("Lcl Scaling", "Lcl Scaling", (1, 1, 1))."""
        try:
            default_subtype, code, count = PROPERTY_TYPES[type]
        except KeyError:
            raise ValueError(f"unknown property type {type!r}; build Property(...) with raw values") from None
        values = () if count == 0 else (value,) if count == 1 else tuple(value)
        if len(values) != count:
            raise ValueError(f"{name!r} ({type}) expects {count} values, got {len(values)}")
        if code in ("I", "L"):
            values = tuple(int(v) for v in values)
        elif code == "D":
            values = tuple(float(v) for v in values)
        return cls(name, type, default_subtype if subtype is None else subtype, flags, tuple(Prop(code, v) for v in values))

    @classmethod
    def from_node(cls, node: FBXNode) -> Property:
        name, type_, subtype, flags = (p.value for p in node.props[:4])
        return cls(name, type_, subtype, flags, tuple(node.props[4:]))

    def to_node(self) -> FBXNode:
        head = [Prop("S", self.name), Prop("S", self.type), Prop("S", self.subtype), Prop("S", self.flags)]
        return FBXNode("P", head + list(self.raw))


class Properties70:
    """Ordered collection of Property entries, keyed by name."""

    def __init__(self, entries: list[Property] | None = None):
        self._entries: dict[str, Property] = {}
        for entry in entries or ():
            self.add(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Property]:
        return iter(self._entries.values())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Property:
        return self._entries[name]

    def get(self, name: str, default: Any = None) -> Any:
        """Value of entry `name`, or `default` if missing."""
        entry = self._entries.get(name)
        return default if entry is None else entry.value

    def add(self, entry: Property) -> None:
        self._entries[entry.name] = entry

    def set(self, name: str, type: str, value: Any = None, *, subtype: str | None = None, flags: str = "") -> None:
        self.add(Property.make(name, type, value, subtype=subtype, flags=flags))

    def remove(self, name: str) -> None:
        self._entries.pop(name, None)

    def copy(self) -> Properties70:
        return Properties70(list(self))

    @classmethod
    def from_node(cls, node: FBXNode | None) -> Properties70:
        if node is None:
            return cls()
        return cls([Property.from_node(c) for c in node.children if c.name == "P"])

    def to_node(self) -> FBXNode:
        return FBXNode("Properties70", children=[e.to_node() for e in self])
