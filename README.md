# fbxio

Minimal FBX reader/writer in pure Python + numpy. No `bpy` required.

Structure follows Blender's
[io_scene_fbx](https://github.com/blender/blender-addons/tree/main/io_scene_fbx),
split into three layers:

| Layer | Package | Responsibility |
|---|---|---|
| Codec | `fbxio.binary` | Bytes ⇄ generic `FBXNode` tree (binary FBX 7.4 / 7.5) |
| Document | `fbxio.document` | FBX structure: header, GlobalSettings, Definitions, Objects, Connections, `Properties70` |
| Scene | `fbxio.scene` | Plain numpy data classes (Scene, Node, Mesh, BlendShape, ...) ⇄ document |

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
- [ ] Phase 1 — static mesh (vertices, polygons, normals, UVs)
- [ ] Phase 2 — blend shapes
- [ ] Phase 3 — materials / textures
- [ ] Phase 4 — skeleton / skinning
- [ ] Phase 5 — animation

## Conventions

- Binary FBX only. Reads 7400 and 7500, writes 7400.
- No implicit axis/unit conversion; helpers in `fbxio.math` are opt-in.
- Geometry is stored as numpy arrays.
