"""FBXNode: a named record holding typed property values and child nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, NamedTuple

# Footer id written by Blender (and accepted by the FBX SDK); real SDK files derive it from CreationTime.
DEFAULT_FOOTER_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"


class Prop(NamedTuple):
    """A typed property value. `type` is an FBX type code, see `fbxio.binary.types`."""

    type: str
    value: Any


@dataclass
class FBXNode:
    name: str
    props: list[Prop] = field(default_factory=list)
    children: list[FBXNode] = field(default_factory=list)
    # Whether a null record terminates this node. None means use the default rule
    # (children present, or no props); the reader sets it to preserve the source bytes.
    sentinel: bool | None = field(default=None, repr=False, compare=False)

    @property
    def values(self) -> list[Any]:
        return [p.value for p in self.props]

    def find(self, name: str) -> FBXNode | None:
        return next(self.iter(name), None)

    def find_all(self, name: str) -> list[FBXNode]:
        return list(self.iter(name))

    def iter(self, name: str) -> Iterator[FBXNode]:
        return (c for c in self.children if c.name == name)

    def add(self, name: str, *props: Prop) -> FBXNode:
        child = FBXNode(name, list(props))
        self.children.append(child)
        return child


@dataclass
class FBXFile:
    """A whole binary FBX file: version plus top-level nodes."""

    version: int = 7400
    nodes: list[FBXNode] = field(default_factory=list)
    footer_id: bytes = DEFAULT_FOOTER_ID

    def find(self, name: str) -> FBXNode | None:
        return next((n for n in self.nodes if n.name == name), None)
