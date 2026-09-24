# sceneport

Lightweight 3D scene I/O in pure Python + numpy. Reads and writes FBX natively
and converts to and from [trimesh](https://trimesh.org).

At the center is a format-neutral scene model; each file format or library is
a converter around it:

| Package | Responsibility |
|---|---|
| `sceneport.scene` | Plain numpy data classes: Scene, Node, Mesh, Primvar, Transform |
| `sceneport.interop.fbx` | Native FBX read/write |
| `sceneport.interop.trimesh` | Conversion to and from trimesh |

The FBX support follows Blender's
[io_scene_fbx](https://github.com/blender/blender-addons/tree/main/io_scene_fbx),
split into layers:

| Layer | Package | Responsibility |
|---|---|---|
| Codec | `sceneport.interop.fbx.binary` | Bytes ⇄ generic `FBXNode` tree (binary FBX 7.4 / 7.5) |
| Document | `sceneport.interop.fbx.document` | FBX structure: header, GlobalSettings, Definitions, Objects, Connections, `Properties70` |
| Conversion | `sceneport.interop.fbx` | Document ⇄ Scene |

## Usage

```python
import sceneport

scene = sceneport.load("model.fbx")
for node, world in scene.walk():  # world: 4x4 matrix
    if node.mesh is not None:
        mesh = node.mesh
        mesh.vertices             # (N, 3)
        mesh.face_vertex_counts   # (F,) corners per face
        mesh.face_vertex_indices  # (C,) vertex index per corner
        mesh.triangles()          # (T, 3) fan triangulation
        mesh.corner_values(mesh.uvs["UVMap"])  # (C, 2) one UV per corner

scene.meters_per_unit, scene.up_axis    # e.g. 0.01, "+Y"

sceneport.save(scene, "out.fbx")
```

### trimesh

```python
from sceneport.interop.trimesh import to_trimesh, from_trimesh, to_trimesh_scene, from_trimesh_scene

tm = to_trimesh(mesh)                     # triangulated; vertices split at normal/UV seams
tm = to_trimesh(mesh, split_seams=False)  # same vertices as the mesh (e.g. for blend shapes)
tm.metadata["source_vertex_index"]        # trimesh vertex -> mesh vertex

tscene = to_trimesh_scene(scene)          # flat: each mesh node at its world transform
scene = from_trimesh_scene(tscene)
mesh = from_trimesh(tm, "Name")
```

With `split_seams=False`, normals are averaged per vertex and UVs are dropped
(with a warning) if a vertex has more than one UV.

## Inspecting files

`sceneport dump` prints the node tree of a binary FBX file:

```bash
sceneport dump model.fbx               # indented tree, first 8 items of each array
sceneport dump model.fbx -d 2          # only the top two levels
sceneport dump model.fbx --full        # every array item
sceneport dump model.fbx --json        # JSON, complete arrays, raw bytes as hex
```

`python -m sceneport dump ...` works too.

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

From the root of a uv workspace that includes sceneport as a member:

```bash
uv run --package sceneport --group dev pytest path/to/sceneport/tests
```

Test files live in `tests/fixtures/`. `cube.fbx` was exported by Blender and
must round-trip byte-exactly.

## Roadmap

- [x] Phase 0 — binary codec, byte-exact round trip, `sceneport dump`
- [x] Phase 1 — static mesh (vertices, polygons, normals, UVs), trimesh conversion
- [ ] Phase 2 — blend shapes
- [ ] Phase 3 — materials / textures
- [ ] Phase 4 — skeleton / skinning
- [ ] Phase 5 — animation

## Conventions

- `load`/`save` pick the format from the file extension (`.fbx` for now).
- `Scene.meters_per_unit` and `up_axis`/`front_axis`/`right_axis` (signed names
  like `"+Y"`) describe the data; nothing is converted on load or save. New
  scenes default to meters, Y up.
- Format-specific data needed for round trips lives in `Scene.extras`, e.g. the
  FBX file's GlobalSettings in `extras["fbx"]`.
- FBX: binary only. Reads 7400 and 7500, writes 7400.
- Geometry is stored as numpy arrays.
- Polygons use USD-style `face_vertex_counts` + `face_vertex_indices`.
- Per-vertex data such as normals and UVs keeps FBX's indexed form, using USD
  interpolation names (`vertex`, `faceVarying`, `uniform`, `constant`); helpers
  expand it to one value per corner.
