"""Read and write Wavefront OBJ (geometry; materials on write).

Writing: one `o` group per mesh node, with vertices in world space (like
Blender's exporter). Skinned meshes are written as posed by the scene's current
joint transforms and blend shapes at their default weights; skins, blend shapes
and animation themselves can't be stored (a warning says so). Polygons keep
their size; normals (`vn`) and the first UV set (`vt`) keep their indexed,
per-corner form (`f v/vt/vn`), so UV seams and vertex order survive.

Materials: the node's material slots are written to an MTL file next to the OBJ
(same name, see mtl.py), referenced with `mtllib`; faces switch material with
`usemtl` where their slot changes (a bare `usemtl` for an empty slot), as
Blender does.

Reading: one Node + Mesh per `o` group (or per `g` group if the file has no
`o`). Normals indexed like the vertices come back per vertex, others per corner.
OBJ has no units or axes; the Scene defaults (meters, Y up) are used.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..scene.types import Mesh, Node, Primvar, Scene
from ..scene.material import Material
from .export import mesh_in_world, warn_dropped
from .mtl import material_block


def save(
    scene: Scene, path: str | os.PathLike, *, precision: int = 9, materials: bool = True, pbr_extensions: bool = False
) -> None:
    """Write `scene` as OBJ, plus an MTL file when nodes have material slots.

    `precision`: significant digits (9 is exact for float32 data, 17 for float64).
    `materials=False` writes geometry only. `pbr_extensions` writes the PBR MTL
    keys (Pr, Pm, map_Pr, ...) instead of Ns / Ka / map_Ns / map_refl.
    """
    path = Path(path)
    warn_dropped(scene, "OBJ", materials=not materials)
    number = f"%.{precision}g"
    lines = ["# written by sceneport"]
    names: dict[int, str] = {}  # id(Material) -> its unique MTL name
    used: list[Material] = []
    offsets = np.zeros(3, dtype=np.int64)  # v, vt, vn written so far (OBJ indices are global, 1-based)
    for node, mesh, vertices, normals in mesh_in_world(scene):
        if len(mesh.uvs) > 1:
            warnings.warn(f"OBJ holds one UV set; {mesh.name!r} keeps {next(iter(mesh.uvs))!r} only")
        lines.append(f"o {_obj_name(node.name)}")
        lines += [f"v {number} {number} {number}" % tuple(v) for v in vertices]

        uv_values, uv_index = _indexed_corners(mesh, next(iter(mesh.uvs.values()), None))
        lines += [f"vt {number} {number}" % tuple(uv) for uv in uv_values]
        normal_values, normal_index = _indexed_corners(mesh, normals)
        lines += [f"vn {number} {number} {number}" % tuple(n) for n in normal_values]

        corners = (mesh.face_vertex_indices + 1 + offsets[0]).astype(str).astype(object)
        if uv_index is not None:
            corners = corners + "/" + (uv_index + 1 + offsets[1]).astype(str)
        if normal_index is not None:
            corners = corners + ("/" if uv_index is not None else "//") + (normal_index + 1 + offsets[2]).astype(str)
        slots = _slot_names(node.materials, names, used) if materials else []
        slot_of_face = mesh.material_indices if mesh.material_indices is not None else np.zeros(mesh.n_faces, dtype=np.int32)
        current = None
        for f, face in enumerate(np.split(corners, np.cumsum(mesh.face_vertex_counts)[:-1])):
            if slots:
                slot = int(slot_of_face[f])
                name = slots[slot] if slot < len(slots) else ""
                if name != current:
                    lines.append(f"usemtl {name}".rstrip())
                    current = name
            lines.append("f " + " ".join(face))
        offsets += [len(vertices), len(uv_values), len(normal_values)]
    if used:  # only materials that some node uses; the MTL file sits next to the OBJ
        lines.insert(1, f"mtllib {path.with_suffix('.mtl').name}")
        blocks = [f"# written by sceneport, {len(used)} materials"]
        for material in sorted(used, key=lambda m: names[id(m)]):  # sorted by name, as Blender writes them
            blocks += ["", *material_block(names[id(material)], material, path.parent, pbr_extensions=pbr_extensions)]
        path.with_suffix(".mtl").write_text("\n".join(blocks) + "\n")
    path.write_text("\n".join(lines) + "\n")


def _slot_names(slots: list[Material | None], names: dict[int, str], used: list[Material]) -> list[str]:
    """MTL names of a node's slots ("" for an empty slot); each Material gets one unique name."""
    out = []
    for material in slots:
        if material is None:
            out.append("")
            continue
        if id(material) not in names:
            base = _obj_name(material.name)
            name, i = base, 1
            while name in names.values():
                name, i = f"{base}.{i:03d}", i + 1
            names[id(material)] = name
            used.append(material)
        out.append(names[id(material)])
    return out


def _obj_name(name: str) -> str:
    return "_".join(name.split()) or "mesh"  # OBJ names are single tokens


def _indexed_corners(mesh: Mesh, primvar: Primvar | None) -> tuple[NDArray, NDArray | None]:
    """(values, per-corner indices) for a primvar, keeping its own indexing where OBJ allows it."""
    if primvar is None:
        return np.zeros((0, 3)), None
    if primvar.interpolation == "faceVarying" and primvar.indices is not None:
        return primvar.values, primvar.indices
    if primvar.interpolation == "faceVarying":
        return primvar.values, np.arange(mesh.n_corners)
    if primvar.interpolation == "vertex":
        return primvar.expanded(), mesh.face_vertex_indices
    values, index = np.unique(mesh.corner_values(primvar), axis=0, return_inverse=True)
    return values, index.reshape(-1)


def load(path: str | os.PathLike) -> Scene:
    positions, uvs, normals = [], [], []
    groups: list[tuple[str, list[list[tuple[int, int, int]]]]] = []  # (name, faces of (v, vt, vn))
    has_objects = False
    faces = None
    for raw in Path(path).read_text(errors="replace").splitlines():
        parts = raw.split("#", 1)[0].split()
        if not parts:
            continue
        key = parts[0]
        if key == "v":
            positions.append([float(x) for x in parts[1:4]])
        elif key == "vt":
            uvs.append([float(x) for x in (parts[1:3] + ["0"])[:2]])
        elif key == "vn":
            normals.append([float(x) for x in parts[1:4]])
        elif key == "o" or (key == "g" and not has_objects):
            has_objects = has_objects or key == "o"
            name = " ".join(parts[1:]) or "mesh"
            if faces is not None and not faces:  # an empty group directly before this one: rename it
                groups[-1] = (name, faces)
            else:
                faces = []
                groups.append((name, faces))
        elif key == "f":
            if faces is None:
                faces = []
                groups.append((Path(path).stem, faces))
            counts = (len(positions), len(uvs), len(normals))
            faces.append([_corner(token, counts) for token in parts[1:]])

    positions, uvs, normals = (np.array(a, dtype=np.float64).reshape(-1, n) for a, n in ((positions, 3), (uvs, 2), (normals, 3)))
    scene = Scene()
    for name, group_faces in groups:
        if group_faces:
            scene.add(Node(name, mesh=_group_mesh(name, group_faces, positions, uvs, normals)))
    return scene


def _corner(token: str, counts: tuple[int, int, int]) -> tuple[int, int, int]:
    """'v', 'v/vt', 'v//vn' or 'v/vt/vn' -> 0-based (v, vt, vn), -1 when absent; negative = relative."""
    fields = (token.split("/") + ["", ""])[:3]
    out = []
    for field, count in zip(fields, counts):
        if not field:
            out.append(-1)
        else:
            index = int(field)
            out.append(index - 1 if index > 0 else count + index)
    return tuple(out)


def _group_mesh(name: str, faces, positions, uvs, normals) -> Mesh:
    counts = np.array([len(f) for f in faces], dtype=np.int32)
    corners = np.array([c for f in faces for c in f], dtype=np.int64).reshape(-1, 3)
    used, local = np.unique(corners[:, 0], return_inverse=True)  # this group's vertices, in file order
    mesh = Mesh(name, positions[used], counts, local.reshape(-1))

    if (corners[:, 1] >= 0).all() and len(uvs):
        values, index = np.unique(corners[:, 1], return_inverse=True)
        mesh.uvs["UVMap"] = Primvar(uvs[values], "faceVarying", index.reshape(-1))
    if (corners[:, 2] >= 0).all() and len(normals):
        by_vertex = np.full(len(used), -1)
        by_vertex[local.reshape(-1)] = corners[:, 2]
        if np.array_equal(by_vertex[local.reshape(-1)], corners[:, 2]):  # one normal per vertex
            mesh.normals = Primvar(normals[by_vertex], "vertex")
        else:
            values, index = np.unique(corners[:, 2], return_inverse=True)
            mesh.normals = Primvar(normals[values], "faceVarying", index.reshape(-1))
    return mesh
