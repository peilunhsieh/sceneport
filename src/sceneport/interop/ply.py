"""Read and write PLY (Stanford polygon format; geometry only).

Writing: one mesh per file, so several meshes are joined (world space, like
Blender's exporter, in meters unless `meters_per_unit` says otherwise), with a warning. Binary little-endian by default
(`ascii=True` for text), float32 coordinates (`double=True` for float64).
Vertices carry x/y/z and per-vertex normals; faces carry `vertex_indices`.
UVs (`uvs=`):
  "corner" (default): a per-face `texcoord` list (u0 v0 u1 v1 ..., the MeshLab
      convention). Lossless and keeps the vertices, but Blender ignores it.
      Per-corner normals are averaged per vertex, with a warning.
  "vertex": per-vertex `s`/`t`, which Blender and most tools read. Vertices are
      split where a vertex has different UVs (or normals) on different faces,
      so the vertex count and order change (with a warning).
  None: no UVs.
Skins, blend shapes and animation can't be stored.

Reading: ASCII and binary (either byte order). Per-corner `texcoord` lists and
per-vertex `s`/`t` (or `u`/`v`, `texture_u`/`texture_v`) UVs are both read;
elements other than vertex and face are skipped. A file without faces whose
vertices have `opacity`, `scale_*` and `rot_*` is read as 3D Gaussian splats
(the 3DGS reference layout, decoded to the physical values of Node.splats;
other per-splat properties are kept in its `attributes`).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..scene.splats import SH_C0, GaussianSplats
from ..scene.types import Mesh, Node, Primvar, Scene
from .export import mesh_in_world, unit_scale, warn_dropped

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip
_UV_NAMES = (("s", "t"), ("u", "v"), ("texture_u", "texture_v"), ("texture_s", "texture_t"))


# -- writing --------------------------------------------------------------------


def save(
    scene: Scene,
    path: str | os.PathLike,
    *,
    ascii: bool = False,
    double: bool = False,
    uvs: str | None = "corner",
    meters_per_unit: float | None = 1.0,
) -> None:
    """Write `scene` as one PLY mesh. PLY has no units: coordinates are converted to
    `meters_per_unit` (default meters); None writes the scene's own units."""
    if uvs not in ("corner", "vertex", None):
        raise ValueError(f'uvs must be "corner", "vertex" or None, got {uvs!r}')
    warn_dropped(scene, "PLY")
    parts = list(mesh_in_world(scene, unit_scale(scene, meters_per_unit)))
    if not parts:
        raise ValueError("PLY needs at least one mesh")
    if len(parts) > 1:
        warnings.warn(f"PLY holds one mesh; {len(parts)} meshes are joined into one")
    with_uvs = [uvs is not None and bool(mesh.uvs) for _, mesh, _, _ in parts]
    if any(with_uvs) and not all(with_uvs):
        warnings.warn("some joined meshes have no UVs; their corners get (0, 0)")
    with_normals = [n is not None for _, _, _, n in parts]
    write_normals = all(with_normals)

    per_vertex_uvs = uvs == "vertex" and any(with_uvs)
    vertices, counts, indices, normals, corner_uvs, vertex_uvs = [], [], [], [], [], []
    offset = added = 0
    for (_, mesh, world_vertices, world_normals), has_uvs in zip(parts, with_uvs):
        uv = next(iter(mesh.uvs.values())) if has_uvs else None
        uv_corners = mesh.corner_values(uv) if uv is not None else np.zeros((mesh.n_corners, 2))
        if per_vertex_uvs:  # one output vertex per unique (vertex, UV, normal) corner
            columns = [mesh.face_vertex_indices[:, None].astype(np.float64), uv_corners]
            if write_normals:
                columns.append(mesh.corner_values(world_normals))
            _, first, inverse = np.unique(np.hstack(columns), axis=0, return_index=True, return_inverse=True)
            source = mesh.face_vertex_indices[first]
            vertices.append(world_vertices[source])
            vertex_uvs.append(uv_corners[first])
            if write_normals:
                normals.append(columns[2][first])
            indices.append(inverse.reshape(-1) + offset)
            offset += len(first)
            added += len(first) - mesh.n_vertices
        else:
            vertices.append(world_vertices)
            indices.append(mesh.face_vertex_indices + offset)
            offset += mesh.n_vertices
            if write_normals:
                normals.append(_vertex_normals(mesh, world_normals))
            if any(with_uvs):
                corner_uvs.append(uv_corners)
        counts.append(mesh.face_vertex_counts)
    if added > 0:
        warnings.warn(f'uvs="vertex" split vertices at UV seams: {added} vertices added, vertex order changed')
    vertices = np.concatenate(vertices)
    counts, indices = np.concatenate(counts), np.concatenate(indices)
    normals = np.concatenate(normals) if write_normals else None
    uvs = np.concatenate(corner_uvs) if corner_uvs else None
    vertex_uvs = np.concatenate(vertex_uvs) if vertex_uvs else None

    real = "double" if double else "float"
    count_type = "uchar" if counts.max(initial=0) < 256 else "int"
    header = ["ply", f"format {'ascii' if ascii else 'binary_little_endian'} 1.0", "comment written by sceneport",
              f"element vertex {len(vertices)}", *(f"property {real} {a}" for a in "xyz")]  # fmt: skip
    if normals is not None:
        header += [f"property {real} {a}" for a in ("nx", "ny", "nz")]
    if vertex_uvs is not None:
        header += [f"property {real} s", f"property {real} t"]
    header += [f"element face {len(counts)}", f"property list {count_type} int vertex_indices"]
    if uvs is not None:
        header.append(f"property list {count_type} {real} texcoord")
    header.append("end_header")

    vertex_data = np.hstack([a for a in (vertices, normals, vertex_uvs) if a is not None])
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if ascii:
            _write_ascii(f, vertex_data, counts, indices, uvs)
        else:
            _write_binary(f, vertex_data, counts, indices, uvs, "<f8" if double else "<f4", _TYPES[count_type])


def _vertex_normals(mesh: Mesh, normals: Primvar | None) -> NDArray:
    if normals.interpolation == "vertex":
        return normals.expanded()
    warnings.warn(f"PLY stores normals per vertex; {mesh.name!r}'s per-corner normals are averaged")
    summed = np.zeros((mesh.n_vertices, 3))
    np.add.at(summed, mesh.face_vertex_indices, mesh.corner_values(normals))
    return summed / np.maximum(np.linalg.norm(summed, axis=1, keepdims=True), 1e-300)


def _write_ascii(f, vertex_data, counts, indices, uvs) -> None:
    f.write("".join(" ".join(f"{x:.9g}" for x in row) + "\n" for row in vertex_data).encode("ascii"))
    starts = np.cumsum(counts) - counts
    lines = []
    for start, count in zip(starts, counts):
        line = f"{count} " + " ".join(str(i) for i in indices[start : start + count])
        if uvs is not None:
            line += f" {2 * count} " + " ".join(f"{x:.9g}" for x in uvs[start : start + count].ravel())
        lines.append(line)
    f.write(("\n".join(lines) + "\n").encode("ascii"))


def _write_binary(f, vertex_data, counts, indices, uvs, real: str, count_type: str) -> None:
    f.write(np.ascontiguousarray(vertex_data, dtype=real).tobytes())
    if len(counts) and (counts == counts[0]).all():  # one face size: a single structured array
        k = int(counts[0])
        fields = [("n", count_type), ("indices", "<i4", (k,))]
        if uvs is not None:
            fields += [("uv_n", count_type), ("uvs", real, (2 * k,))]
        faces = np.zeros(len(counts), dtype=fields)
        faces["n"] = k
        faces["indices"] = indices.reshape(-1, k)
        if uvs is not None:
            faces["uv_n"] = 2 * k
            faces["uvs"] = uvs.reshape(-1, 2 * k)
        f.write(faces.tobytes())
        return
    starts = np.cumsum(counts) - counts
    chunks = []
    for start, count in zip(starts, counts):
        chunks.append(np.array([count], dtype=count_type).tobytes() + indices[start : start + count].astype("<i4").tobytes())
        if uvs is not None:
            chunks.append(np.array([2 * count], dtype=count_type).tobytes() + uvs[start : start + count].astype(real).tobytes())
    f.write(b"".join(chunks))


# -- reading --------------------------------------------------------------------


def load(path: str | os.PathLike) -> Scene:
    data = Path(path).read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError(f"{path}: not a PLY file")
    body_start = data.index(b"\n", end) + 1
    fmt, elements = _parse_header(data[:end].decode("ascii", errors="replace"))

    if fmt == "ascii":
        tokens = data[body_start:].split()
        values, _ = _read_ascii(tokens, elements)
    else:
        values = _read_binary(data, body_start, elements, "<" if fmt == "binary_little_endian" else ">")

    vertex, face = values.get("vertex", {}), values.get("face", {})
    index_lists = face.get("vertex_indices", face.get("vertex_index", []))
    if not len(index_lists) and all(p in vertex for p in ("opacity", "scale_0", "rot_0")):
        splats = _splats(vertex, Path(path).stem)
        return Scene(nodes=[Node(splats.name, splats=splats)])

    positions = np.stack([np.asarray(vertex[a], dtype=np.float64) for a in "xyz"], axis=1)
    counts = np.array([len(i) for i in index_lists], dtype=np.int32)
    indices = np.concatenate(index_lists).astype(np.int32) if len(index_lists) else np.zeros(0, np.int32)
    mesh = Mesh(Path(path).stem, positions, counts, indices)

    if all(a in vertex for a in ("nx", "ny", "nz")):
        mesh.normals = Primvar(np.stack([np.asarray(vertex[a], np.float64) for a in ("nx", "ny", "nz")], axis=1), "vertex")
    if "texcoord" in face and len(face["texcoord"]):
        corner_uvs = np.concatenate(face["texcoord"]).astype(np.float64).reshape(-1, 2)
        values_, index = np.unique(corner_uvs, axis=0, return_inverse=True)
        mesh.uvs["UVMap"] = Primvar(values_, "faceVarying", index.reshape(-1))
    else:
        for u, v in _UV_NAMES:
            if u in vertex and v in vertex:
                mesh.uvs["UVMap"] = Primvar(np.stack([vertex[u], vertex[v]], axis=1).astype(np.float64), "vertex")
                break
    return Scene(nodes=[Node(mesh.name, mesh=mesh)])


def _splats(vertex: dict, name: str) -> GaussianSplats:
    """Decode the 3DGS reference layout: log scales, logit opacity, unnormalized w-x-y-z quaternion and
    SH as f_dc_0..2 plus f_rest_* (all of red's higher-order coefficients, then green's, then blue's).
    Two scales (2DGS surfels) get a zero third; red/green/blue colors stand in for missing f_dc.
    Every other property except the unused normals is kept in `attributes`."""

    def column(prop):
        return np.asarray(vertex[prop], dtype=np.float64)

    def numbered(prefix):  # prefix_0, prefix_1, ... in numeric order
        found = [p for p in vertex if p.startswith(prefix) and p[len(prefix) :].isdigit()]
        return sorted(found, key=lambda p: int(p[len(prefix) :]))

    used = {"x", "y", "z", "nx", "ny", "nz", "opacity"}
    positions = np.stack([column(a) for a in "xyz"], axis=1)
    n = len(positions)

    scale_props, rot_props = numbered("scale_"), numbered("rot_")
    if len(scale_props) not in (2, 3) or len(rot_props) != 4:
        raise ValueError(f"{name}: expected 2 or 3 scale_* and 4 rot_* properties, got {len(scale_props)} and {len(rot_props)}")
    scales = np.zeros((n, 3))
    scales[:, : len(scale_props)] = np.exp(np.stack([column(p) for p in scale_props], axis=1))
    orientations = np.stack([column(p) for p in rot_props], axis=1)
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    orientations = np.where(norms > 0, orientations / np.where(norms > 0, norms, 1.0), [1.0, 0.0, 0.0, 0.0])
    opacities = 0.5 * (1.0 + np.tanh(0.5 * column("opacity")))  # sigmoid, without overflow for large logits
    used.update(scale_props + rot_props)

    dc_props, rest_props = numbered("f_dc_"), numbered("f_rest_")
    if len(dc_props) == 3:
        dc = np.stack([column(p) for p in dc_props], axis=1)
        used.update(dc_props)
    elif all(c in vertex for c in ("red", "green", "blue")):  # plain 8-bit colors: the degree-0 term giving them
        dc = (np.stack([column(c) for c in ("red", "green", "blue")], axis=1) / 255.0 - 0.5) / SH_C0
        used.update(("red", "green", "blue"))
    else:
        dc = np.zeros((n, 3))
    rest = np.zeros((n, 0, 3))
    if rest_props:
        if len(rest_props) % 3 or round(np.sqrt(len(rest_props) // 3 + 1)) ** 2 != len(rest_props) // 3 + 1:
            raise ValueError(f"{name}: {len(rest_props)} f_rest_* properties isn't 3 x (a square - 1)")
        rest = np.stack([column(p) for p in rest_props], axis=1).reshape(n, 3, -1).transpose(0, 2, 1)
        used.update(rest_props)
    sh = np.concatenate([dc[:, None], rest], axis=1)

    attributes = {p: np.asarray(values) for p, values in vertex.items() if p not in used}
    return GaussianSplats(name, positions, scales, orientations, opacities, sh, attributes)


def _parse_header(text: str):
    fmt, elements = None, []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1][2].append((parts[4], _TYPES[parts[2]], _TYPES[parts[3]]))
            else:
                elements[-1][2].append((parts[2], None, _TYPES[parts[1]]))
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unsupported PLY format {fmt!r}")
    return fmt, elements


def _read_ascii(tokens, elements):
    out, pos = {}, 0
    for name, count, props in elements:
        columns = {p[0]: [] for p in props}
        for _ in range(count):
            for prop, count_type, item_type in props:
                if count_type is None:
                    columns[prop].append(float(tokens[pos]))
                    pos += 1
                else:
                    n = int(tokens[pos])
                    columns[prop].append(np.array([float(t) for t in tokens[pos + 1 : pos + 1 + n]]).astype(item_type))
                    pos += 1 + n
        out[name] = columns
    return out, pos


def _read_binary(data: bytes, pos: int, elements, endian: str):
    out = {}
    for name, count, props in elements:
        if all(count_type is None for _, count_type, _ in props):  # fixed-size rows: one numpy read
            dtype = np.dtype([(p, endian + t) for p, _, t in props])
            rows = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
            out[name] = {p: rows[p] for p, _, _ in props}
            pos += dtype.itemsize * count
            continue
        rows, pos = _read_binary_lists(data, pos, count, props, endian)
        out[name] = rows
    return out


def _read_binary_lists(data: bytes, pos: int, count: int, props, endian: str):
    """Rows with list properties. Fast path when every row's lists have the first row's lengths."""
    columns = {p[0]: [] for p in props}
    if count == 0:
        return columns, pos
    fields, offset = [], pos
    for prop, count_type, item_type in props:  # the first row's layout
        if count_type is None:
            fields.append((prop, endian + item_type))
            offset += np.dtype(item_type).itemsize
        else:
            n = int(np.frombuffer(data, dtype=endian + count_type, count=1, offset=offset)[0])
            fields += [(prop + "#n", endian + count_type), (prop, endian + item_type, (n,))]
            offset += np.dtype(count_type).itemsize + n * np.dtype(item_type).itemsize
    dtype = np.dtype(fields)
    if pos + dtype.itemsize * count <= len(data):
        rows = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
        lengths_ok = all(
            (rows[prop + "#n"] == dtype[prop].shape[0]).all() for prop, count_type, _ in props if count_type is not None
        )
        if lengths_ok:
            for prop, count_type, _ in props:
                columns[prop] = list(rows[prop]) if count_type is not None else rows[prop]
            return columns, pos + dtype.itemsize * count
    for _ in range(count):  # mixed list lengths: row by row
        for prop, count_type, item_type in props:
            if count_type is None:
                columns[prop].append(np.frombuffer(data, dtype=endian + item_type, count=1, offset=pos)[0])
                pos += np.dtype(item_type).itemsize
            else:
                n = int(np.frombuffer(data, dtype=endian + count_type, count=1, offset=pos)[0])
                pos += np.dtype(count_type).itemsize
                columns[prop].append(np.frombuffer(data, dtype=endian + item_type, count=n, offset=pos))
                pos += n * np.dtype(item_type).itemsize
    return columns, pos
