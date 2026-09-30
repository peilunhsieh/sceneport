# sceneport

Lightweight 3D scene I/O in Python + numpy. Reads and writes FBX natively,
USD through [OpenUSD](https://openusd.org) (`usd-core`), and converts to and
from [trimesh](https://trimesh.org).

At the center is a format-neutral scene model; each file format or library is
a converter around it:

| Package | Responsibility |
|---|---|
| `sceneport.scene` | Plain numpy data classes: Scene, Node, Mesh, Primvar, Transform, Material |
| `sceneport.interop.fbx` | Native FBX read/write (materials: read only) |
| `sceneport.interop.usd` | USD read/write (`.usd`, `.usda`, `.usdc`, `.usdz`) via usd-core |
| `sceneport.interop.obj`, `sceneport.interop.ply` | OBJ and PLY read/write (geometry only) |
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

# Blend shapes: offsets per vertex (sparse with `indices`, or dense)
mesh.blend_shapes.append(sceneport.BlendShape("Smile", offsets, indices=vertex_ids))
mesh.deformed_vertices({"Smile": 0.5})   # vertices with shapes applied, (N, 3)
mesh.deformed_vertices(w)                # w: (S,) -> (N, 3), or (F, S) frames -> (F, N, 3)
mesh.blend_shape_matrix()                # (S, N, 3) dense basis, for linear algebra

# Skinning: joints are Nodes with joint=True; a Skin binds the mesh to them
skin = mesh.skin                         # joints, joint_indices/joint_weights (N, K), inverse_bind (J, 4, 4)
scene.find("l_lowleg").transform.rotation += [0, 0, 60]
mesh.skinned_vertices(skin.joint_world(scene))  # world-space (N, 3); (F, J, 4, 4) poses -> (F, N, 3)
skin.dense_weights()                     # (N, J) weights, SMPL-style

# Animation: clips of keyframed channels, evaluated for any times
clip = scene.animations[0]               # e.g. one per FBX animation stack
times = clip.frame_times()               # one per frame at clip.fps
clip.skinned_vertices(scene, mesh, times)  # (F, N, 3)
clip.world_matrices(scene, times)        # {node: (F, 4, 4)}

# Materials (read from USD and FBX): generic shading node graphs
node.materials                           # the node's material slots
mesh.material_indices                    # (F,) slot of each face, or None = slot 0
material = node.materials[0]
surface = material.surface()             # e.g. the UsdPreviewSurface node; surface("mtlx") for MaterialX
source, output, port = material.trace(surface.inputs["diffuseColor"])  # e.g. a UsdUVTexture, "outputs:rgb"
source.inputs["file"].value              # AssetPath("tex/wood.png", resolved=...)
pbr = material.pbr()                     # metallic-roughness view: UsdPreviewSurface, MaterialX, FBX Phong
pbr.base_color, pbr.roughness, pbr.base_color_texture.file, pbr.notes  # notes: what was approximated or skipped

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

### USD

```python
sceneport.save(scene, "model.usda")   # also .usd, .usdc, .usdz
scene = sceneport.load("model.usd")  # composed stage: references, payloads, instances
```

- One Xform prim per node, with a Mesh prim child (`subdivisionScheme = "none"`).
- Transforms become an xformOp stack that matches `Transform` exactly (pivots
  and pre/post rotation included); other stacks are read via the local matrix.
- The first UV set is written as `st`; original UV set names are kept in
  customData and restored on load.
- Invalid prim names are made valid (`Cube.001` -> `Cube_001`); the original
  name is kept as the prim's `displayName`.
- USD only records the up axis (Y or Z). Front/right axes are implied on load
  (+Z/+X for Y up, -Y/+X for Z up). Points are stored as float32.
- Blend shapes use UsdSkel: the node becomes a SkelRoot, each shape a
  BlendShape prim under the Mesh, and a jointless Skeleton's SkelAnimation
  holds the default weights.
- Skeletons use UsdSkel: each joint tree becomes a Skeleton prim (joint paths,
  original names in `jointNames`, rest and bind matrices). Non-joint children of
  joints are written under the Skeleton, placed at their joint's rest pose, and
  re-attached on load. When a skinned mesh and its skeleton share no ancestor,
  a SkelRoot is generated around them and dropped again on load.
- Materials are read (not yet written) as generic UsdShade graphs, whatever
  the node library (UsdPreviewSurface, MaterialX, renderer-specific).
  `materialBind` GeomSubsets become per-face material slots.

## Converting files

```bash
sceneport convert model.fbx model.usd   # the formats follow the extensions
sceneport convert model.usdz model.fbx  # .fbx, .usd, .usda, .usdc, .usdz, .obj, .ply
sceneport convert scan.ply scan.usd
```

Meshes, blend shapes, skins and animation carry over between FBX and USD.
For PLY, `sceneport.save(scene, "x.ply", uvs="vertex")` writes per-vertex UVs
that Blender reads (splitting vertices at UV seams); the default per-corner UVs
are lossless but only MeshLab-style readers see them; `uvs=None` leaves UVs out.
OBJ and PLY hold geometry only: meshes are written in world space as currently
posed (blend shapes at their default weights), with a warning about what is
dropped. See the conventions below for what each format can hold.

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
- [x] Phase 1b — USD read/write for everything in Phase 1
- [x] Phase 2 — blend shapes (with in-betweens), FBX and USD
- [ ] Phase 3 — materials / textures (reading USD and FBX, PBR interpretation done)
- [x] Phase 4 — skeleton / skinning (linear blend skinning), FBX and USD
- [x] Phase 5 — animation (keyframes, joints/nodes/blend shape weights), FBX and USD

## Conventions

The reasoning behind these is in
[docs/design.md](docs/design.md).


- `load`/`save` pick the format from the file extension (`.fbx`, `.usd`, `.usda`,
  `.usdc`, `.usdz`, `.obj`, `.ply`).
- `Scene.meters_per_unit` and `up_axis`/`front_axis`/`right_axis` (signed names
  like `"+Y"`) describe the data; nothing is converted on load or save. New
  scenes default to meters, Y up.
- Format-specific data needed for round trips lives in `Scene.extras`, e.g. the
  FBX file's GlobalSettings in `extras["fbx"]`.
- FBX: binary only. Reads 7400 and 7500, writes 7400.
- Geometry is stored as numpy arrays.
- Polygons use USD-style `face_vertex_counts` + `face_vertex_indices`.
- Rigs follow glTF: joints are ordinary Nodes (`joint=True`), so other nodes can
  hang from them; `Mesh.skin` lists its joints with inverse bind matrices and
  per-vertex (N, K) joint indices and weights.
- Blend shapes store offsets (not absolute positions), a default `weight`
  (1 = fully applied) and optional in-betweens; weights interpolate linearly
  through the in-betweens, as in FBX and UsdSkel.
- Per-vertex data such as normals and UVs keeps FBX's indexed form, using USD
  interpolation names (`vertex`, `faceVarying`, `uniform`, `constant`); helpers
  expand it to one value per corner.
