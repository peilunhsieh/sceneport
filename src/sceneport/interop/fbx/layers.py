"""Encode/decode LayerElements (normals, UVs, materials) across mapping and reference modes.

FBX MappingInformationType  <->  USD interpolation
    ByVertice / ByVertex    <->  vertex
    ByPolygonVertex         <->  faceVarying
    ByPolygon               <->  uniform
    AllSame                 <->  constant
ReferenceInformationType IndexToDirect (or Index) means the values are indexed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .binary.node import FBXNode, Prop
from .document.document import FBXObject
from ...scene.types import Primvar

MAPPING_TO_INTERPOLATION = {
    "ByVertice": "vertex",
    "ByVertex": "vertex",
    "ByPolygonVertex": "faceVarying",
    "ByPolygon": "uniform",
    "AllSame": "constant",
}
INTERPOLATION_TO_MAPPING = {
    "vertex": "ByVertice",
    "faceVarying": "ByPolygonVertex",
    "uniform": "ByPolygon",
    "constant": "AllSame",
}


@dataclass(frozen=True)
class LayerKind:
    element: str  # e.g. "LayerElementNormal"
    values: str  # child holding the values, e.g. "Normals"
    index: str  # child holding the indices, e.g. "NormalsIndex"
    components: int
    version: int


NORMAL = LayerKind("LayerElementNormal", "Normals", "NormalsIndex", 3, 101)
UV = LayerKind("LayerElementUV", "UV", "UVIndex", 2, 101)


def read_layers(geometry: FBXObject, kind: LayerKind) -> list[tuple[str, Primvar]]:
    """All `kind` layer elements of `geometry` as (name, primvar), in file order."""
    out = []
    for node in geometry.children:
        if node.name != kind.element:
            continue
        name = _value(node, "Name", "")
        mapping = _value(node, "MappingInformationType", "")
        interpolation = MAPPING_TO_INTERPOLATION.get(mapping)
        if interpolation is None:
            warnings.warn(f"{geometry.name}: skipping {kind.element} {name!r} with mapping {mapping!r}")
            continue
        values = _value(node, kind.values)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64).reshape(-1, kind.components)
        indices = None
        if _value(node, "ReferenceInformationType", "Direct") in ("IndexToDirect", "Index"):
            indices = np.asarray(_value(node, kind.index), dtype=np.int32)
        out.append((name, Primvar(values, interpolation, indices)))
    return out


def write_layer(kind: LayerKind, index: int, name: str, primvar: Primvar) -> FBXNode:
    values = np.asarray(primvar.values, dtype=np.float64).reshape(-1)
    children = [
        FBXNode("Version", [Prop("I", kind.version)]),
        FBXNode("Name", [Prop("S", name)]),
        FBXNode("MappingInformationType", [Prop("S", INTERPOLATION_TO_MAPPING[primvar.interpolation])]),
        FBXNode("ReferenceInformationType", [Prop("S", "Direct" if primvar.indices is None else "IndexToDirect")]),
        FBXNode(kind.values, [Prop("d", values)]),
    ]
    if primvar.indices is not None:
        children.append(FBXNode(kind.index, [Prop("i", np.asarray(primvar.indices, dtype=np.int32))]))
    return FBXNode(kind.element, [Prop("I", index)], children)


def write_layer_table(layers: list[list[tuple[str, int]]]) -> list[FBXNode]:
    """`Layer` nodes. layers[i] lists the (element name, typed index) pairs in layer i."""
    out = []
    for i, entries in enumerate(layers):
        children = [FBXNode("Version", [Prop("I", 100)])]
        for element, typed_index in entries:
            children.append(
                FBXNode(
                    "LayerElement",
                    children=[FBXNode("Type", [Prop("S", element)]), FBXNode("TypedIndex", [Prop("I", typed_index)])],
                )
            )
        out.append(FBXNode("Layer", [Prop("I", i)], children))
    return out


def _value(node: FBXNode, child: str, default=None):
    c = node.find(child)
    return default if c is None or not c.props else c.props[0].value
