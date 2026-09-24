"""Convert between fbxio meshes/scenes and trimesh.

trimesh stores triangles with one normal and one UV per vertex, so converting
to it triangulates faces (fan) and, by default, splits vertices along normal
and UV seams. `metadata["fbx_vertex_index"]` maps each trimesh vertex back to
the FBX vertex it came from.
"""

from __future__ import annotations

import warnings

import numpy as np
import trimesh
from numpy.typing import NDArray

from ..math import decompose_matrix
from ..scene.types import Mesh, Node, Primvar, Scene, Transform

VERTEX_MAP_KEY = "fbx_vertex_index"


def to_trimesh(mesh: Mesh, *, split_seams: bool = True, uv_set: str | None = None) -> trimesh.Trimesh:
    """Triangulated trimesh of `mesh` with its normals and one UV set (default: the first).

    split_seams=True: a vertex is duplicated wherever its normal or UV differs
    between faces, so hard edges and UV seams come out right. Vertices no face
    uses are dropped. A mesh without seams keeps its vertex order.

    split_seams=False: vertices stay one-to-one with `mesh.vertices` (useful when
    vertex order matters, e.g. blend shapes). Normals are averaged per vertex;
    UVs are kept only if every vertex has a single UV.
    """
    corner_tris = mesh.triangulate_corners()
    uv = _pick_uv(mesh, uv_set)
    corner_normals = None if mesh.normals is None else mesh.corner_values(mesh.normals)
    corner_uvs = None if uv is None else mesh.corner_values(uv)

    if split_seams:
        columns = [mesh.face_vertex_indices[:, None].astype(np.float64)]
        columns += [c for c in (corner_normals, corner_uvs) if c is not None]
        _, first_corner, inverse = np.unique(np.hstack(columns), axis=0, return_index=True, return_inverse=True)
        vertex_map = mesh.face_vertex_indices[first_corner]
        faces = inverse.reshape(-1)[corner_tris]
        normals = None if corner_normals is None else corner_normals[first_corner]
        uvs = None if corner_uvs is None else corner_uvs[first_corner]
    else:
        vertex_map = np.arange(mesh.n_vertices)
        faces = mesh.face_vertex_indices[corner_tris]
        normals = None if mesh.normals is None else _vertex_normals(mesh, corner_normals)
        uvs = None if uv is None else _vertex_uvs(mesh, uv, corner_uvs)

    visual = trimesh.visual.TextureVisuals(uv=uvs) if uvs is not None else None
    tm = trimesh.Trimesh(
        vertices=mesh.vertices[vertex_map],
        faces=faces,
        vertex_normals=normals,
        visual=visual,
        process=False,
        metadata={"name": mesh.name, VERTEX_MAP_KEY: vertex_map},
    )
    return tm


def from_trimesh(tm: trimesh.Trimesh, name: str | None = None, *, normals: bool = True) -> Mesh:
    """Mesh with tm's triangles, vertex normals (computed by trimesh if missing) and UVs if present."""
    faces = np.asarray(tm.faces, dtype=np.int32)
    uvs = {}
    uv = getattr(tm.visual, "uv", None)
    if uv is not None and len(uv) == len(tm.vertices):
        uvs["UVMap"] = Primvar(np.asarray(uv, dtype=np.float64), "vertex")
    return Mesh(
        name or tm.metadata.get("name") or "mesh",
        np.asarray(tm.vertices, dtype=np.float64),
        np.full(len(faces), 3, dtype=np.int32),
        faces.reshape(-1),
        normals=Primvar(np.asarray(tm.vertex_normals, dtype=np.float64), "vertex") if normals else None,
        uvs=uvs,
    )


def to_trimesh_scene(scene: Scene, **kwargs) -> trimesh.Scene:
    """Flat trimesh.Scene: one graph node per mesh node, placed at its world transform.

    `kwargs` go to `to_trimesh`. Nodes sharing a Mesh share one trimesh geometry.
    """
    out = trimesh.Scene()
    geom_names: dict[int, str] = {}
    for node, world in scene.walk():
        if node.mesh is None:
            continue
        matrix = world @ node.transform.geometric_matrix()
        node_name = _unique(node.name, out.graph.nodes)
        key = id(node.mesh)
        if key in geom_names:
            out.graph.update(frame_to=node_name, frame_from=out.graph.base_frame, matrix=matrix, geometry=geom_names[key])
        else:
            geom_name = _unique(node.mesh.name, out.geometry)
            out.add_geometry(to_trimesh(node.mesh, **kwargs), node_name=node_name, geom_name=geom_name, transform=matrix)
            geom_names[key] = geom_name
    return out


def from_trimesh_scene(tscene: trimesh.Scene, **kwargs) -> Scene:
    """Scene with one root node per geometry instance, transforms taken from world matrices.

    `kwargs` go to `from_trimesh`. Instances of one trimesh geometry share one Mesh.
    Shear in a world matrix cannot be represented and is dropped.
    """
    scene = Scene()
    meshes: dict[str, Mesh] = {}
    for node_name in tscene.graph.nodes_geometry:
        matrix, geom_name = tscene.graph[node_name]
        geometry = tscene.geometry[geom_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if geom_name not in meshes:
            meshes[geom_name] = from_trimesh(geometry, geom_name, **kwargs)
        t, r, s = decompose_matrix(matrix)
        scene.add(Node(str(node_name), Transform(translation=t, rotation=r, scale=s), mesh=meshes[geom_name]))
    return scene


def _pick_uv(mesh: Mesh, uv_set: str | None) -> Primvar | None:
    if uv_set is not None:
        return mesh.uvs[uv_set]
    return next(iter(mesh.uvs.values()), None)


def _vertex_normals(mesh: Mesh, corner_normals: NDArray) -> NDArray:
    if mesh.normals.interpolation == "vertex":
        return mesh.normals.expanded()
    summed = np.zeros((mesh.n_vertices, 3))
    np.add.at(summed, mesh.face_vertex_indices, corner_normals)
    length = np.linalg.norm(summed, axis=1, keepdims=True)
    return summed / np.where(length == 0, 1, length)


def _vertex_uvs(mesh: Mesh, uv: Primvar, corner_uvs: NDArray) -> NDArray | None:
    if uv.interpolation == "vertex":
        return uv.expanded()
    per_vertex = np.zeros((mesh.n_vertices, 2))
    per_vertex[mesh.face_vertex_indices] = corner_uvs
    if not np.array_equal(per_vertex[mesh.face_vertex_indices], corner_uvs):
        warnings.warn(f"{mesh.name}: UVs have seams; dropped because split_seams=False")
        return None
    return per_vertex


def _unique(name: str, taken) -> str:
    candidate, i = name, 1
    while candidate in taken:
        candidate, i = f"{name}.{i:03d}", i + 1
    return candidate
