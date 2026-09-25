"""Skinning: FBX Skin deformers and bind poses <-> Mesh.skin.

FBX layout (as Blender and the FBX SDK write it):

    Geometry "Mesh"  <-  Deformer "Skin"
                             <-  SubDeformer "Cluster"  (one per joint)  <-  Model "LimbNode" (the joint)
                                    Indexes, Weights: the vertices this joint moves
                                    TransformLink: the joint's world matrix at bind time
                                    Transform: inverse(TransformLink) @ the mesh's world matrix at bind time
    Pose "BindPose": PoseNode { Node, Matrix } for the mesh and every joint (world, at bind time)

FBX matrices are 16 values with the translation last (row-vector layout);
they are transposed to and from our column-vector matrices.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from ...scene.types import Mesh, Node, Skin
from .binary.node import FBXNode, Prop
from .document import FBXDocument, FBXObject

JOINT_SUBTYPES = ("LimbNode", "Root", "Limb")  # FBX Model subtypes that are skeleton joints


def matrix_from_fbx(values) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64).reshape(4, 4).T


def matrix_to_fbx(matrix: NDArray) -> NDArray[np.float64]:
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float64).T).reshape(-1)


# -- reading --------------------------------------------------------------------


def read_skin(doc: FBXDocument, geometry: FBXObject, n_vertices: int, nodes: dict[int, Node]) -> Skin | None:
    """The Skin deformer on `geometry`, with joints mapped to Nodes by Model id."""
    skins = [d for d in doc.children_of(geometry, "Deformer") if d.subtype == "Skin"]
    if not skins:
        return None
    if len(skins) > 1:
        warnings.warn(f"{geometry.name}: {len(skins)} skin deformers; only the first is read")
    deformer = skins[0]
    if (kind := deformer.child_value("SkinningType", "Linear")) != "Linear":
        warnings.warn(f"{geometry.name}: {kind} skinning is read as linear blend skinning")

    joints, inverse_bind, geom_bind = [], [], None
    vertex_ids, joint_ids, weights = [], [], []
    for cluster in doc.children_of(deformer, "Deformer"):
        links = [m for m in doc.children_of(cluster, "Model") if m.id in nodes]
        if cluster.subtype != "Cluster" or not links:
            continue
        link_values = cluster.child_value("TransformLink")
        if link_values is None:
            warnings.warn(f"{geometry.name}: cluster {cluster.name!r} has no TransformLink; skipped")
            continue
        link = matrix_from_fbx(link_values)
        transform_values = cluster.child_value("Transform")
        transform = np.eye(4) if transform_values is None else matrix_from_fbx(transform_values)
        if geom_bind is None:
            geom_bind = link @ transform

        j = len(joints)
        joints.append(nodes[links[0].id])
        inverse_bind.append(np.linalg.inv(link))
        indexes = cluster.child_value("Indexes")
        if indexes is not None and len(indexes):
            vertex_ids.append(np.asarray(indexes, dtype=np.int64))
            joint_ids.append(np.full(len(indexes), j, dtype=np.int32))
            weights.append(np.asarray(cluster.child_value("Weights"), dtype=np.float64))
    if not joints:
        return None

    joint_indices, joint_weights = _per_vertex(
        n_vertices,
        np.concatenate(vertex_ids) if vertex_ids else np.zeros(0, np.int64),
        np.concatenate(joint_ids) if joint_ids else np.zeros(0, np.int32),
        np.concatenate(weights) if weights else np.zeros(0),
    )
    return Skin(joints, joint_indices, joint_weights, np.stack(inverse_bind), geom_bind)


def _per_vertex(n_vertices: int, vertex_ids, joint_ids, weights) -> tuple[NDArray, NDArray]:
    """Per-joint (vertex, weight) lists -> (N, K) arrays, heaviest influence first."""
    order = np.lexsort((-weights, vertex_ids))
    vertex_ids, joint_ids, weights = vertex_ids[order], joint_ids[order], weights[order]
    counts = np.bincount(vertex_ids, minlength=n_vertices)
    k = max(int(counts.max()) if len(counts) else 0, 1)
    slot = np.arange(len(vertex_ids)) - np.repeat(np.cumsum(counts) - counts, counts)
    joint_indices = np.zeros((n_vertices, k), dtype=np.int32)
    joint_weights = np.zeros((n_vertices, k))
    joint_indices[vertex_ids, slot] = joint_ids
    joint_weights[vertex_ids, slot] = weights
    return joint_indices, joint_weights


# -- writing --------------------------------------------------------------------


def write_skin(doc: FBXDocument, geometry: FBXObject, mesh: Mesh, models: dict[Node, FBXObject]) -> None:
    skin = mesh.skin
    deformer = doc.add_object("Deformer", mesh.name, "Skin", cls="Deformer", children=[
        FBXNode("Version", [Prop("I", 101)]),
        FBXNode("Link_DeformAcuracy", [Prop("D", 50.0)]),
        FBXNode("SkinningType", [Prop("S", "Linear")]),
    ])  # fmt: skip
    deformer.props_index = 1
    doc.connect(deformer, geometry)

    bind_world = skin.bind_world()
    for j, joint in enumerate(skin.joints):
        if joint not in models:
            raise ValueError(f"mesh {mesh.name!r}: skin joint {joint.name!r} is not in the scene")
        vertices, slots = np.nonzero((skin.joint_indices == j) & (skin.joint_weights != 0))
        children = [
            FBXNode("Version", [Prop("I", 100)]),
            FBXNode("UserData", [Prop("S", ""), Prop("S", "")]),
        ]
        if len(vertices):
            children += [
                FBXNode("Indexes", [Prop("i", vertices.astype(np.int32))]),
                FBXNode("Weights", [Prop("d", skin.joint_weights[vertices, slots])]),
            ]
        children += [
            FBXNode("Transform", [Prop("d", matrix_to_fbx(skin.inverse_bind[j] @ skin.geom_bind))]),
            FBXNode("TransformLink", [Prop("d", matrix_to_fbx(bind_world[j]))]),
        ]
        cluster = doc.add_object("Deformer", joint.name, "Cluster", cls="SubDeformer", children=children)
        cluster.props_index = 1
        doc.connect(cluster, deformer)
        doc.connect(models[joint], cluster)


def write_bind_pose(doc: FBXDocument, name: str, entries: list[tuple[FBXObject, NDArray]]) -> None:
    """A BindPose listing each (model, world matrix at bind time)."""
    pose_nodes = [
        FBXNode("PoseNode", children=[
            FBXNode("Node", [Prop("L", model.id)]),
            FBXNode("Matrix", [Prop("d", matrix_to_fbx(matrix))]),
        ])  # fmt: skip
        for model, matrix in entries
    ]
    pose = doc.add_object("Pose", name, "BindPose", children=[
        FBXNode("Type", [Prop("S", "BindPose")]),
        FBXNode("Version", [Prop("I", 100)]),
        FBXNode("NbPoseNodes", [Prop("I", len(pose_nodes))]),
        *pose_nodes,
    ])  # fmt: skip
    pose.props_index = -1  # FBX SDK poses have no Properties70
