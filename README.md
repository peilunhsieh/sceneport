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

## Roadmap

- [ ] Phase 0 — binary codec, byte-exact round trip, `fbxio dump`
- [ ] Phase 1 — static mesh (vertices, polygons, normals, UVs)
- [ ] Phase 2 — blend shapes
- [ ] Phase 3 — materials / textures
- [ ] Phase 4 — skeleton / skinning
- [ ] Phase 5 — animation

## Conventions

- Binary FBX only. Reads 7400 and 7500, writes 7400.
- No implicit axis/unit conversion; helpers in `fbxio.math` are opt-in.
- Geometry is stored as numpy arrays.
