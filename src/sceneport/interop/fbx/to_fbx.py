"""Convert a Scene into an FBXDocument."""

from __future__ import annotations

import os

import numpy as np

from ...scene.types import TRANSFORM_PROPERTIES, Mesh, Node, Scene, Transform
from . import layers
from .binary import write
from .binary.node import FBXNode, Prop
from .document import ROOT_ID, FBXDocument, FBXObject, Properties70
from .settings import write_global_settings


def save(scene: Scene, path: str | os.PathLike, *, version: int = 7400) -> None:
    write(document_from_scene(scene, version=version).to_file(), path)


def document_from_scene(scene: Scene, *, version: int = 7400) -> FBXDocument:
    doc = FBXDocument(version=version, global_settings=write_global_settings(scene))
    geometries: dict[int, FBXObject] = {}  # id(Mesh) -> Geometry, so shared meshes are written once

    def add(node: Node, parent_id: int) -> None:
        model = doc.add_object("Model", node.name, "Mesh" if node.mesh is not None else "Null")
        model.props = _transform_props(node.transform)
        model.props.set("InheritType", "enum", 1)
        if node.mesh is not None:
            model.props.set("DefaultAttributeIndex", "int", 0)
        model.children = [
            FBXNode("Version", [Prop("I", 232)]),
            FBXNode("MultiLayer", [Prop("I", 0)]),
            FBXNode("MultiTake", [Prop("I", 0)]),
            FBXNode("Shading", [Prop("C", True)]),
            FBXNode("Culling", [Prop("S", "CullingOff")]),
        ]
        model.props_index = 1  # after Version, as Blender writes it
        doc.connect(model, parent_id)

        if node.mesh is not None:
            key = id(node.mesh)
            if key not in geometries:
                geometries[key] = _add_geometry(doc, node.mesh)
            doc.connect(geometries[key], model)
        for child in node.children:
            add(child, model.id)

    for root in scene.nodes:
        add(root, ROOT_ID)
    return doc


def _transform_props(t: Transform) -> Properties70:
    props = Properties70()
    defaults = Transform()
    for field, name in TRANSFORM_PROPERTIES.items():
        value = getattr(t, field)
        if np.array_equal(value, getattr(defaults, field)):
            continue
        if name.startswith("Lcl "):
            props.set(name, name, value, flags="A")
        else:
            props.set(name, "Vector3D", value)
    if t.rotation_order != 0:
        props.set("RotationOrder", "enum", t.rotation_order)
    if any(name in props for name in ("PreRotation", "PostRotation", "RotationOrder")):
        props.set("RotationActive", "bool", 1)
    return props


def _add_geometry(doc: FBXDocument, mesh: Mesh) -> FBXObject:
    mesh.validate()
    pvi = mesh.face_vertex_indices.copy()
    ends = np.cumsum(mesh.face_vertex_counts) - 1
    pvi[ends] = ~pvi[ends]

    children = [
        FBXNode("GeometryVersion", [Prop("I", 124)]),
        FBXNode("Vertices", [Prop("d", mesh.vertices.reshape(-1))]),
        FBXNode("PolygonVertexIndex", [Prop("i", pvi)]),
    ]
    # Layer 0 holds the normals and the first UV set; layer i holds UV set i.
    table: list[list[tuple[str, int]]] = [[] for _ in range(max(1, len(mesh.uvs)))]
    if mesh.normals is not None:
        children.append(layers.write_layer(layers.NORMAL, 0, "", mesh.normals))
        table[0].append((layers.NORMAL.element, 0))
    for i, (name, primvar) in enumerate(mesh.uvs.items()):
        children.append(layers.write_layer(layers.UV, i, name, primvar))
        table[i].append((layers.UV.element, i))
    if any(table):
        children += layers.write_layer_table(table)

    return doc.add_object("Geometry", mesh.name, "Mesh", children=children)
