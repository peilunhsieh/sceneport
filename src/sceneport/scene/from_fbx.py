"""Convert an FBXDocument into a Scene."""

from __future__ import annotations

import os
import warnings

import numpy as np

from ..binary import read
from ..document import FBXDocument, FBXObject
from . import layers
from .types import ROTATION_ACTIVE_FIELDS, TRANSFORM_PROPERTIES, Mesh, Node, Scene, Transform


def load(path: str | os.PathLike) -> Scene:
    return scene_from_document(FBXDocument.from_file(read(path)))


def scene_from_document(doc: FBXDocument) -> Scene:
    scene = Scene(settings=doc.global_settings.copy())
    meshes: dict[int, Mesh] = {}  # geometry id -> Mesh, so shared geometry stays shared
    nodes: dict[int, Node] = {}

    models = list(doc.iter_objects("Model"))
    for model in models:
        node = nodes[model.id] = Node(model.name, _read_transform(doc, model))
        for geom in doc.children_of(model, "Geometry"):
            if geom.subtype != "Mesh":
                continue
            if geom.id not in meshes:
                meshes[geom.id] = mesh_from_geometry(geom)
            node.mesh = meshes[geom.id]
            break

    for model in models:
        parents = doc.parents_of(model, "Model")
        if parents:
            nodes[parents[0].id].children.append(nodes[model.id])
        else:
            scene.nodes.append(nodes[model.id])
    return scene


def _read_transform(doc: FBXDocument, model: FBXObject) -> Transform:
    values = {field: doc.get_property(model, prop) for field, prop in TRANSFORM_PROPERTIES.items()}
    values = {k: v for k, v in values.items() if v is not None}
    values["rotation_order"] = doc.get_property(model, "RotationOrder", 0)
    if not doc.get_property(model, "RotationActive", 0):
        for field in ROTATION_ACTIVE_FIELDS:
            values.pop(field, None)
    return Transform(**values)


def mesh_from_geometry(geom: FBXObject) -> Mesh:
    vertices = np.asarray(geom.child_value("Vertices", np.zeros(0)), dtype=np.float64)
    pvi = np.array(geom.child_value("PolygonVertexIndex", np.zeros(0)), dtype=np.int32)
    if len(pvi) and pvi[-1] >= 0:
        warnings.warn(f"{geom.name}: PolygonVertexIndex doesn't end a polygon; closing the last one")
        pvi[-1] = ~pvi[-1]
    ends = np.flatnonzero(pvi < 0)
    counts = np.diff(ends, prepend=-1)
    indices = np.where(pvi < 0, ~pvi, pvi)

    normals = layers.read_layers(geom, layers.NORMAL)
    uvs = {}
    for i, (name, primvar) in enumerate(layers.read_layers(geom, layers.UV)):
        key = name or f"UVMap{i}"
        while key in uvs:
            key += "_"
        uvs[key] = primvar
    return Mesh(
        geom.name,
        vertices,
        counts,
        indices,
        normals=normals[0][1] if normals else None,
        uvs=uvs,
    )
