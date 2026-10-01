"""Convert a Scene into an FBXDocument."""

from __future__ import annotations

import os
import warnings

import numpy as np

from ...scene.types import TRANSFORM_PROPERTIES, Mesh, Node, Scene, Transform
from . import layers
from .animation import write_animations
from .blendshapes import write_blend_shapes
from .binary import write
from .binary.node import FBXNode, Prop
from .compat import prepare
from .document import ROOT_ID, FBXDocument, FBXObject, Properties70
from .materials import MaterialWriter, material_layer
from .settings import write_global_settings
from .skin import write_bind_pose, write_skin


def save(
    scene: Scene,
    path: str | os.PathLike,
    *,
    version: int = 7400,
    unmirror_joints: bool = True,
    split_inbetweens: bool = True,
) -> None:
    """Write `scene` as binary FBX. See document_from_scene for the options."""
    directory = os.path.dirname(os.path.abspath(path))
    doc = document_from_scene(
        scene, version=version, directory=directory, unmirror_joints=unmirror_joints, split_inbetweens=split_inbetweens
    )
    write(doc.to_file(), path)


def document_from_scene(
    scene: Scene,
    *,
    version: int = 7400,
    directory: str | os.PathLike | None = None,
    unmirror_joints: bool = True,
    split_inbetweens: bool = True,
) -> FBXDocument:
    """`directory`: where the file will be written; texture paths are made relative to it.

    `unmirror_joints` rewrites mirrored joints (negative scale) without mirroring and `split_inbetweens`
    turns in-between shapes into plain ones, both without changing how the scene looks (see compat.py):
    Blender's importer can't handle either. Set them to False to write the scene as it is (e.g. for Maya).
    """
    if any(clip.point_samples for clip in scene.animations):
        warnings.warn("FBX can't hold per-vertex animation (point samples); it is not written (USD keeps it)")
    scene = prepare(scene, unmirror_joints=unmirror_joints, split_inbetweens=split_inbetweens)
    doc = FBXDocument(version=version, global_settings=write_global_settings(scene))
    geometries: dict[int, FBXObject] = {}  # id(Mesh) -> Geometry, so shared meshes are written once
    first_users: dict[int, tuple[Mesh, FBXObject]] = {}  # id(Mesh) -> (mesh, first Model using it)
    models: dict[Node, FBXObject] = {}
    shape_channels: dict[tuple[int, str], FBXObject] = {}  # (id(Mesh), shape name) -> BlendShapeChannel
    materials = MaterialWriter(doc, directory)
    for material in scene.materials:  # unbound ones too, in scene order
        materials.material(material)
    # Meshes with material slots on any of their nodes get a LayerElementMaterial.
    slotted = {id(node.mesh) for node, _ in scene.walk() if node.mesh is not None and node.materials}

    def add(node: Node, parent_id: int) -> None:
        model = doc.add_object("Model", node.name, _model_subtype(node))
        models[node] = model
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
        if node.joint:
            attribute = doc.add_object("NodeAttribute", node.name, model.subtype, cls="NodeAttribute", children=[
                FBXNode("TypeFlags", [Prop("S", "Skeleton")]),
            ])  # fmt: skip
            attribute.props_index = -1
            doc.connect(attribute, model)

        if node.mesh is not None:
            key = id(node.mesh)
            if key not in geometries:
                geometries[key], channels = _add_geometry(doc, node.mesh, key in slotted)
                shape_channels.update(((key, name), channel) for name, channel in channels.items())
                first_users[key] = (node.mesh, model)
            doc.connect(geometries[key], model)
            materials.connect_slots(node, model)
        for child in node.children:
            add(child, model.id)

    for root in scene.nodes:
        add(root, ROOT_ID)

    # Skins last, once every joint has a Model.
    for key, (mesh, mesh_model) in first_users.items():
        if mesh.skin is None:
            continue
        write_skin(doc, geometries[key], mesh, models)
        bind_world = mesh.skin.bind_world()
        entries = [(mesh_model, mesh.skin.geom_bind)]
        entries += [(models[joint], bind_world[j]) for j, joint in enumerate(mesh.skin.joints)]
        write_bind_pose(doc, mesh.name, entries)
    write_animations(doc, scene, models, shape_channels)
    return doc


def _model_subtype(node: Node) -> str:
    if node.joint:
        return node.extras.get("fbx", {}).get("subtype", "LimbNode")
    return "Mesh" if node.mesh is not None else "Null"


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


def _add_geometry(doc: FBXDocument, mesh: Mesh, with_materials: bool) -> tuple[FBXObject, dict[str, FBXObject]]:
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
    if with_materials:
        children.append(material_layer(mesh.material_indices))
        table[0].append(("LayerElementMaterial", 0))
    if any(table):
        children += layers.write_layer_table(table)

    geometry = doc.add_object("Geometry", mesh.name, "Mesh", children=children)
    return geometry, write_blend_shapes(doc, geometry, mesh)
