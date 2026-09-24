"""Blend shapes: FBX BlendShape deformers <-> Mesh.blend_shapes.

FBX layout (as Blender writes it):

    Geometry "Mesh"  <-  Deformer "BlendShape"
                             <-  SubDeformer "BlendShapeChannel"  (one per BlendShape;
                                    DeformPercent = weight * 100, FullWeights = [percent per shape])
                                     <-  Geometry "Shape"  (one per in-between plus the full shape;
                                            Indexes, Vertices and Normals hold offsets)
"""

from __future__ import annotations

import warnings

import numpy as np

from ...scene.types import BlendShape, Inbetween, Mesh
from .binary.node import FBXNode, Prop
from .document import FBXDocument, FBXObject


def read_blend_shapes(doc: FBXDocument, geometry: FBXObject, n_vertices: int) -> list[BlendShape]:
    shapes = []
    for deformer in doc.children_of(geometry, "Deformer"):
        if deformer.subtype != "BlendShape":
            continue
        for channel in doc.children_of(deformer, "Deformer"):
            if channel.subtype == "BlendShapeChannel" and (shape := _read_channel(doc, channel, n_vertices)):
                shapes.append(shape)
    return shapes


def _read_channel(doc: FBXDocument, channel: FBXObject, n_vertices: int) -> BlendShape | None:
    targets = [g for g in doc.children_of(channel, "Geometry") if g.subtype == "Shape"]
    if not targets:
        return None
    full = channel.child_value("FullWeights")
    if full is None or len(full) != len(targets):
        full = [100.0 * (i + 1) / len(targets) for i in range(len(targets))]  # FBX default: evenly spaced
    levels = sorted(zip((float(w) for w in full), targets), key=lambda level: level[0])
    top = levels[-1][0]
    if top != 100.0:
        warnings.warn(f"{channel.name}: full shape at {top}%; treating it as weight 1")

    index_arrays = [np.asarray(t.child_value("Indexes", np.zeros(0)), dtype=np.int32) for _, t in levels]
    if all(np.array_equal(a, index_arrays[-1]) for a in index_arrays):
        indices = index_arrays[-1]
    else:  # in-betweens touch different vertices: use the union, zero where a shape has no offset
        indices = np.unique(np.concatenate(index_arrays)).astype(np.int32)

    def scatter(target: FBXObject, own: np.ndarray, key: str) -> np.ndarray | None:
        values = target.child_value(key)
        if values is None or len(values) == 0:
            return None
        values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        if own is indices or np.array_equal(own, indices):
            return values
        out = np.zeros((len(indices), 3))
        out[np.searchsorted(indices, own)] = values
        return out

    offsets = [scatter(t, idx, "Vertices") for (_, t), idx in zip(levels, index_arrays)]
    normals = [scatter(t, idx, "Normals") for (_, t), idx in zip(levels, index_arrays)]
    normals = [None if n is None or not n.any() else n for n in normals]
    zeros = np.zeros((len(indices), 3))

    percent = doc.get_property(channel, "DeformPercent")
    if percent is None:
        percent = channel.child_value("DeformPercent", 0.0)
    dense = len(indices) == n_vertices and np.array_equal(indices, np.arange(n_vertices))
    return BlendShape(
        channel.name,
        offsets[-1] if offsets[-1] is not None else zeros,
        indices=None if dense else indices,
        normal_offsets=normals[-1],
        weight=float(percent) / 100.0,
        inbetweens=[
            Inbetween(w / top, o if o is not None else zeros, n)
            for (w, _), o, n in zip(levels[:-1], offsets[:-1], normals[:-1])
        ],
    )


def write_blend_shapes(doc: FBXDocument, geometry: FBXObject, mesh: Mesh) -> None:
    if not mesh.blend_shapes:
        return
    deformer = doc.add_object("Deformer", mesh.name, "BlendShape", cls="Deformer", children=[_int("Version", 100)])
    deformer.props_index = 1
    doc.connect(deformer, geometry)

    for shape in mesh.blend_shapes:
        levels = sorted(shape.inbetweens, key=lambda i: i.weight)
        full_weights = np.array([i.weight * 100.0 for i in levels] + [100.0])
        channel = doc.add_object(
            "Deformer",
            shape.name,
            "BlendShapeChannel",
            cls="SubDeformer",
            children=[
                _int("Version", 100),
                FBXNode("DeformPercent", [Prop("D", shape.weight * 100.0)]),
                FBXNode("FullWeights", [Prop("d", full_weights)]),
            ],
        )
        channel.props_index = 1
        channel.props.set("DeformPercent", "Number", shape.weight * 100.0, flags="A")
        doc.connect(channel, deformer)

        indices = np.arange(mesh.n_vertices, dtype=np.int32) if shape.indices is None else shape.indices
        entries = [(f"{shape.name}_{i.weight * 100:g}", i.offsets, i.normal_offsets) for i in levels]
        entries.append((shape.name, shape.offsets, shape.normal_offsets))
        for name, offsets, normals in entries:
            normals = np.zeros_like(offsets) if normals is None else normals
            target = doc.add_object(
                "Geometry",
                name,
                "Shape",
                children=[
                    _int("Version", 100),
                    FBXNode("Indexes", [Prop("i", np.asarray(indices, dtype=np.int32))]),
                    FBXNode("Vertices", [Prop("d", np.asarray(offsets, dtype=np.float64).reshape(-1))]),
                    FBXNode("Normals", [Prop("d", np.asarray(normals, dtype=np.float64).reshape(-1))]),
                ],
            )
            target.props_index = 1
            doc.connect(target, channel)


def _int(name: str, value: int) -> FBXNode:
    return FBXNode(name, [Prop("I", value)])
