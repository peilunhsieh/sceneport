"""Convert an FBXDocument into a Scene."""

from __future__ import annotations

import os
import warnings

import numpy as np

from ...scene.types import ROTATION_ACTIVE_FIELDS, TRANSFORM_PROPERTIES, Mesh, Node, Scene, Transform
from . import layers
from .animation import fps_from_settings, read_animations
from .blendshapes import read_blend_shapes
from .binary import read
from .document import FBXDocument, FBXObject
from .camera import read_camera
from ..reframe import reframe
from .compat import CAMERA_TO_FBX, LIGHT_TO_FBX
from .light import read_light
from .materials import read_material_indices, read_material_slots, read_materials
from .settings import read_global_settings
from .skin import JOINT_SUBTYPES, read_skin


def load(path: str | os.PathLike) -> Scene:
    return scene_from_document(FBXDocument.from_file(read(path)), os.path.dirname(os.path.abspath(path)))


def scene_from_document(doc: FBXDocument, directory: str | None = None) -> Scene:
    """`directory` (the FBX file's folder) resolves relative texture paths."""
    scene = Scene()
    read_global_settings(doc.global_settings, scene)
    materials = read_materials(doc, directory)
    scene.materials = list(materials.values())
    meshes: dict[int, Mesh] = {}  # geometry id -> Mesh, so shared geometry stays shared
    nodes: dict[int, Node] = {}
    shapes: dict[int, tuple[Mesh, str]] = {}  # BlendShapeChannel id -> (mesh, shape name)
    attributes: dict[int, tuple[Node, str]] = {}  # camera / light NodeAttribute id -> (node, "camera" | "light")

    models = list(doc.iter_objects("Model"))
    for model in models:
        node = nodes[model.id] = Node(model.name, _read_transform(doc, model), joint=model.subtype in JOINT_SUBTYPES)
        if model.subtype == "Camera":
            node.camera = read_camera(doc, model)
        elif model.subtype == "Light":
            node.light = read_light(doc, model)
        if node.camera is not None or node.light is not None:
            kind = "camera" if node.camera is not None else "light"
            for attribute in doc.children_of(model, "NodeAttribute"):
                attributes[attribute.id] = (node, kind)
        if node.joint and model.subtype != "LimbNode":
            node.extras["fbx"] = {"subtype": model.subtype}  # e.g. "Root"; LimbNode is the default on write
        for geom in doc.children_of(model, "Geometry"):
            if geom.subtype != "Mesh":
                continue
            if geom.id not in meshes:
                mesh = meshes[geom.id] = mesh_from_geometry(geom)
                mesh.material_indices = read_material_indices(geom, mesh.n_faces)
                channels: dict[int, str] = {}
                mesh.blend_shapes = read_blend_shapes(doc, geom, mesh.n_vertices, channels)
                shapes.update((cid, (mesh, name)) for cid, name in channels.items())
            node.mesh = meshes[geom.id]
            node.materials = read_material_slots(doc, model, materials)
            break

    for model in models:
        parents = doc.parents_of(model, "Model")
        if parents:
            nodes[parents[0].id].children.append(nodes[model.id])
        else:
            scene.nodes.append(nodes[model.id])

    # Skins last: their joints can be anywhere in the hierarchy.
    for geom_id, mesh in meshes.items():
        mesh.skin = read_skin(doc, doc.objects[geom_id], mesh.n_vertices, nodes)
    scene.animations = read_animations(doc, nodes, shapes, fps_from_settings(doc.global_settings), attributes)
    # FBX cameras look down +X and lights shine down -Y; sceneport's both down -Z.
    frames = {node: np.linalg.inv(CAMERA_TO_FBX) for node in nodes.values() if node.camera is not None}
    frames.update({node: np.linalg.inv(LIGHT_TO_FBX) for node in nodes.values() if node.light is not None})
    if frames:
        scene = reframe(scene, frames)
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
