"""Read and write Wavefront OBJ (geometry, with materials in MTL files).

Writing: one `o` group per mesh node, with vertices in world space (like
Blender's exporter), in meters unless `meters_per_unit` says otherwise. Skinned meshes are written as posed by the scene's current
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
Materials come from the `mtllib` files (plus an MTL file with the OBJ's name, as
Blender does); `usemtl` sets the material of the faces that follow, carrying
over into the next group. A group's materials, in order of first use, are its
node's slots, with per-face material_indices when there are several. Names
missing from the libraries get an empty Material (with a warning).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..scene.types import Mesh, Node, Primvar, Scene
from ..scene.material import Material
from .export import mesh_in_world, unit_scale, warn_dropped
from .mtl import load_library, material_block


def save(
    scene: Scene,
    path: str | os.PathLike,
    *,
    precision: int = 9,
    materials: bool = True,
    pbr_extensions: bool = False,
    meters_per_unit: float | None = 1.0,
) -> None:
    """Write `scene` as OBJ, plus an MTL file when nodes have material slots.

    OBJ has no units; coordinates are converted to `meters_per_unit` (default
    meters, what Blender and most tools assume), e.g. a centimeter FBX scene is
    scaled by 0.01. `meters_per_unit=None` writes the scene's own units.
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
    for node, mesh, vertices, normals in mesh_in_world(scene, unit_scale(scene, meters_per_unit)):
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
    path = Path(path)
    positions, uvs, normals = [], [], []
    # (name, faces of (v, vt, vn), material name of each face or None)
    groups: list[tuple[str, list[list[tuple[int, int, int]]], list[str | None]]] = []
    libraries: list[str] = []
    has_objects = False
    faces, face_materials, material = None, None, None
    for raw in path.read_text(errors="replace").splitlines():
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
                groups[-1] = (name, faces, face_materials)
            else:
                faces, face_materials = [], []
                groups.append((name, faces, face_materials))
        elif key == "f":
            if faces is None:
                faces, face_materials = [], []
                groups.append((path.stem, faces, face_materials))
            counts = (len(positions), len(uvs), len(normals))
            faces.append([_corner(token, counts) for token in parts[1:]])
            face_materials.append(material)
        elif key == "usemtl":
            material = raw.strip()[len("usemtl") :].strip() or None  # a bare usemtl: no material
        elif key == "mtllib":
            library = raw.strip()[len("mtllib") :].strip().strip('"')
            if library and library not in libraries:
                libraries.append(library)

    positions, uvs, normals = (np.array(a, dtype=np.float64).reshape(-1, n) for a, n in ((positions, 3), (uvs, 2), (normals, 3)))
    scene = Scene()
    materials = _load_libraries(path, libraries)
    for name, group_faces, group_materials in groups:
        if group_faces:
            node = scene.add(Node(name, mesh=_group_mesh(name, group_faces, positions, uvs, normals)))
            _assign_slots(node, group_materials, materials)
    scene.materials = list(materials.values())
    return scene


def _load_libraries(path: Path, libraries: list[str]) -> dict[str, Material]:
    """Materials of the mtllib files, then of an MTL file named like the OBJ (Blender reads that one too)."""
    materials: dict[str, Material] = {}
    default = path.with_suffix(".mtl").name
    for library in libraries + ([default] if default not in libraries else []):
        file = path.parent / library.replace("\\", "/")
        if file.is_file():
            load_library(file, materials)
        elif library != default:
            warnings.warn(f"OBJ {path.name}: material library {library!r} not found")
    return materials


def _assign_slots(node: Node, face_materials: list[str | None], materials: dict[str, Material]) -> None:
    """The group's materials in order of first use become the node's slots."""
    order = list(dict.fromkeys(face_materials))
    if order == [None]:
        return
    for name in order:
        if name is not None and name not in materials:
            warnings.warn(f"OBJ: material {name!r} is not in any material library; using an empty one")
            materials[name] = Material(name)
    node.materials = [materials[name] if name is not None else None for name in order]
    if len(order) > 1:
        index = {name: i for i, name in enumerate(order)}
        node.mesh.material_indices = np.array([index[name] for name in face_materials], dtype=np.int32)


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
