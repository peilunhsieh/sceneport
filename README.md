# fbxio

Minimal FBX reader/writer in pure Python + numpy, with conversion to and from
[trimesh](https://trimesh.org).

Structure follows Blender's
[io_scene_fbx](https://github.com/blender/blender-addons/tree/main/io_scene_fbx),
split into three layers:

| Layer | Package | Responsibility |
|---|---|---|
| Codec | `fbxio.binary` | Bytes ⇄ generic `FBXNode` tree (binary FBX 7.4 / 7.5) |
| Document | `fbxio.document` | FBX structure: header, GlobalSettings, Definitions, Objects, Connections, `Properties70` |
| Scene | `fbxio.scene` | Plain numpy data classes (Scene, Node, Mesh, BlendShape, ...) ⇄ document |

## Usage

```python
import fbxio

scene = fbxio.load("model.fbx")
for node, world in scene.walk():  # world: 4x4 matrix
    if node.mesh is not None:
        mesh = node.mesh
        mesh.vertices             # (N, 3)
        mesh.face_vertex_counts   # (F,) corners per face
        mesh.face_vertex_indices  # (C,) vertex index per corner
        mesh.triangles()          # (T, 3) fan triangulation
        mesh.corner_values(mesh.uvs["UVMap"])  # (C, 2) one UV per corner

fbxio.save(scene, "out.fbx")
```

### trimesh

```python
from fbxio.interop.trimesh import to_trimesh, from_trimesh, to_trimesh_scene, from_trimesh_scene

tm = to_trimesh(mesh)                     # triangulated; vertices split at normal/UV seams
tm = to_trimesh(mesh, split_seams=False)  # same vertices as the FBX mesh (e.g. for blend shapes)
tm.metadata["fbx_vertex_index"]           # trimesh vertex -> FBX vertex

tscene = to_trimesh_scene(scene)          # flat: each mesh node at its world transform
scene = from_trimesh_scene(tscene)
mesh = from_trimesh(tm, "Name")
```

With `split_seams=False`, normals are averaged per vertex and UVs are dropped
(with a warning) if a vertex has more than one UV.

## Inspecting files

`fbxio dump` prints the node tree of a binary FBX file:

```bash
fbxio dump model.fbx               # indented tree, first 8 items of each array
fbxio dump model.fbx -d 2          # only the top two levels
fbxio dump model.fbx --full        # every array item
fbxio dump model.fbx --json        # JSON, complete arrays, raw bytes as hex
```

`python -m fbxio dump ...` works too.

```
Objects
  Geometry: L 477939210, S 'Cube.001\x00\x01Geometry', S 'Mesh'
    Vertices: d[24] [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, ...]
    PolygonVertexIndex: i[24] [0, 4, 6, -3, 3, 2, 6, -8, ...]
```

Each property shows its FBX type code, then its value (`d[24]` is a float64
array with 24 items).

## Testing

From this folder:

```bash
uv run --group dev pytest
```

From the root of a uv workspace that includes fbxio as a member:

```bash
uv run --package fbxio --group dev pytest path/to/fbxio/tests
```

Test files live in `tests/fixtures/`. `cube.fbx` was exported by Blender and
must round-trip byte-exactly.

## Roadmap

- [x] Phase 0 — binary codec, byte-exact round trip, `fbxio dump`
- [x] Phase 1 — static mesh (vertices, polygons, normals, UVs), trimesh conversion
- [ ] Phase 2 — blend shapes
- [ ] Phase 3 — materials / textures
- [ ] Phase 4 — skeleton / skinning
- [ ] Phase 5 — animation

## Conventions

- Binary FBX only. Reads 7400 and 7500, writes 7400.
- No implicit axis/unit conversion; helpers in `fbxio.math` are opt-in.
- Geometry is stored as numpy arrays.
- Polygons use USD-style `face_vertex_counts` + `face_vertex_indices`.
- Per-vertex data such as normals and UVs keeps FBX's indexed form, using USD
  interpolation names (`vertex`, `faceVarying`, `uniform`, `constant`); helpers
  expand it to one value per corner.
