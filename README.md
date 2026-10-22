# sceneport

Lightweight 3D scene I/O in Python + numpy. Reads and writes FBX natively,
USD through [OpenUSD](https://openusd.org) (`usd-core`), OBJ and PLY, writes
Blender `.blend` files through an installed [Blender](https://www.blender.org),
and converts to and from [trimesh](https://trimesh.org).

At the center is a format-neutral scene model; each file format or library is
a converter around it:

| Package | Responsibility |
|---|---|
| `sceneport.scene` | Plain numpy data classes: Scene, Node, Mesh, Primvar, Transform, Material, Camera, Light, GaussianSplats |
| `sceneport.interop.fbx` | Native FBX read/write |
| `sceneport.interop.usd` | USD read/write (`.usd`, `.usda`, `.usdc`, `.usdz`) via usd-core |
| `sceneport.interop.obj`, `sceneport.interop.ply` | OBJ (+ MTL) and PLY read/write |
| `sceneport.interop.blend` | `.blend` writing, run inside an installed Blender 5.2+ |
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
clip.vertices(mesh, times)               # (F, N, 3) in mesh space, incl. per-vertex animation (point_samples)
clip.camera_at(cam_node, times)          # [Camera] per time; channels like Channel(cam_node, "camera:focal_length", 0, ...)
clip.light_at(light_node, times)         # [Light] per time; e.g. "light:intensity", "light:color" (components 0-2)

# Materials (USD and FBX): generic shading node graphs
node.materials                           # the node's material slots
mesh.material_indices                    # (F,) slot of each face, or None = slot 0
material = node.materials[0]
surface = material.surface()             # e.g. the UsdPreviewSurface node; surface("mtlx") for MaterialX
source, output, port = material.trace(surface.inputs["diffuseColor"])  # e.g. a UsdUVTexture, "outputs:rgb"
source.inputs["file"].value              # AssetPath("tex/wood.png", resolved=...)
pbr = material.pbr()                     # metallic-roughness view: UsdPreviewSurface, MaterialX, FBX Phong
pbr.base_color, pbr.roughness, pbr.base_color_texture.file, pbr.notes  # notes: what was approximated or skipped

# Collections (USD and .blend): named, nestable groups of nodes, without transforms
props = Collection("Props", [table, shelf], children=[Collection("Chairs", [chair])])
scene.collections.append(props)          # a node may be in several collections, or none
scene.find_collection("Chairs").nodes    # direct members (a member's children aren't implied)
props.all_nodes()                        # members, nested collections included

# Gaussian splats (3DGS PLY, .blend): physical values on Node.splats
splats = sceneport.load("point_cloud.ply").nodes[0].splats
splats.positions, splats.scales          # (N, 3); scales are standard deviations
splats.orientations, splats.opacities    # (N, 4) w, x, y, z; (N,) 0..1
splats.sh, splats.base_colors()          # (N, K, 3) SH coefficients (3DGS order); view-independent RGB
splats.attributes                        # other per-splat values, e.g. {"binding_0": ...}
splats.transformed(matrix)               # moved by a 4x4 matrix, axes and SH colors turned too

# Cameras: lens settings on a node, looking down its -Z; look_at_matrix aims one
from sceneport.math import decompose_matrix, look_at_matrix
camera = Node("camera", Transform(*decompose_matrix(look_at_matrix(eye, target))), camera=Camera(focal_length=50.0))

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
- Materials are read and written as generic UsdShade graphs, whatever the
  node library (UsdPreviewSurface, MaterialX, renderer-specific).
  `materialBind` GeomSubsets are per-face material slots. Materials from FBX
  also get a UsdPreviewSurface network; embedded FBX textures are written to
  a `textures` folder (packaged in `.usdz`).

### Blender

sceneport writes `.blend` files by running an installed **Blender 5.2 or newer**
in the background, so the file is built by Blender itself. Nothing is installed
into Blender, and there's no Python version to match: sceneport puts its own
source on the background Blender's path for the run. Reading `.blend` files
isn't supported yet.

```python
sceneport.save(scene, "scene.blend")                  # Z-up meters, as Blender's own data
sceneport.save(scene, "scene.blend", blender="/Applications/Blender.app")
sceneport.save(scene, "scene.blend", pack_images=True)  # texture images stored inside the .blend
sceneport.save(scene, "scene.blend", scripts=["setup.py"])  # your bpy setup, run before saving
```

Blender is found from `blender=` (an executable, or the `.app` on macOS), then
`$SCENEPORT_BLENDER`, then `blender` on PATH, then the newest Blender 5.2+ in
the usual install locations. Each save starts Blender, about a second plus the
time to build the scene; errors inside Blender come back as `BlenderError` with
Blender's traceback, and its warnings as Python warnings.

```python
from sceneport.interop.blend import find_blender, run_in_blender
find_blender()              # Blender(path=..., version=(5, 2, 2))
run_in_blender("ping")      # {"blender": (5, 2, 2), "python": ..., "numpy": ...}
```

**What's written.** The scene is converted to Blender's own axes and units (Z
up, meters), so objects carry no extra rotation.

| Scene | Blender |
|---|---|
| Nodes | Empties and objects, with their parents and transforms |
| Meshes | Meshes with custom normals, UV maps and material slots |
| Blend shapes | Shape keys; each slider spans the shape's `range` (default 0..1, e.g. `(-3, 3)` for a morphable model's coefficients), widened to fit its weights, since Blender clamps weights to the slider |
| Joints and skins | One armature per joint tree, bones resting in the bind pose and posed as the scene is; vertex groups and an Armature modifier, with the mesh a child of its armature |
| Animation | One action per clip: the first is assigned and sets the scene's fps and frame range, the others are kept with a fake user (switch in Blender) |
| Per-vertex animation | A `<name>_cache.usdc` beside the .blend, read by Mesh Sequence Cache modifiers: keep the two files together |
| Cameras and lights | Cameras and lights with their animated settings; the first camera is the scene's camera (its `exposure`, in stops, is the view's exposure: Blender has none per camera), the first dome light the World |
| Materials | Principled BSDF setups built from each material's PBR reading (textures, UV transforms, channels, normal maps); images are linked by paths relative to the .blend, or packed into it with `pack_images=True`; images embedded in an FBX are always packed |
| Gaussian splats | Point clouds with Blender 5.3's splat attributes (position, scale, rotation, radiance:base, radiance:sh_*), marked as splats on Blender 5.3+ and plain points on 5.2; view-dependent colors turned to Z-up |
| Collections | Blender collections, nested the same way |
| `scene.extras["blender"]["texts"]` | Text datablocks (`{name: source}`): scripts saved in the file, run from Blender's Text Editor (Alt+P); nothing runs by itself |

**Compared with Blender's own USD importer.** Values follow what Blender's USD
importer sets for the same scene, so a scene looks the same either way, except
where its result would be wrong:

- Shape key sliders hold every weight (the importer keeps 0..1 and clamps).
- Opacity is coverage (Alpha), as Blender's FBX and glTF importers read
  transparency; the USD importer reads it as glass (Transmission).
- A sun's angle is used as is (USD's angular diameter); the importer doubles it.
- The render size follows the scene camera's film aspect.
- EEVEE's light threshold is lowered when dim, physically scaled lights would
  otherwise be cut off and render black in EEVEE while Cycles lights them.

## Converting files

```bash
sceneport convert model.fbx model.usd   # the formats follow the extensions
sceneport convert model.usdz model.fbx  # .fbx, .usd, .usda, .usdc, .usdz, .obj, .ply
sceneport convert scan.ply scan.usd
sceneport convert walk.usd walk_for_blender.usd --unmirror-joints
sceneport convert walk.usd walk.blend     # needs Blender 5.2+ installed (see Blender below)
```

Rigs with mirrored joints (negative scale, e.g. a right side built by mirroring
the left) are valid USD, but Blender drops their animation on import; sceneport
warns about them, and `--unmirror-joints` (or `unmirror_joints=True`) writes them
without mirroring and with the same deformation. FBX export always does this.

Meshes, blend shapes, skins, animation, materials and cameras carry over between FBX
and USD, as do lights (sun, point, spot, area; FBX keeps fewer of their settings);
HDRI environment (dome) lights are USD only (`scene.environment`,
`scene.add_environment("sky.exr")`) (materials as Phong in FBX; see the conventions below). Per-vertex
animation (time-sampled points) is kept by USD only. For Blender, FBX export
splits in-between shapes into plain ones and rewrites mirrored joints (negative
scale) without mirroring, since its importer can't handle either; both keep the
same deformation (`split_inbetweens=False` / `unmirror_joints=False` to write
them as they are, e.g. for Maya).
For PLY, `sceneport.save(scene, "x.ply", uvs="vertex")` writes per-vertex UVs
that Blender reads (splitting vertices at UV seams); the default per-corner UVs
are lossless but only MeshLab-style readers see them; `uvs=None` leaves UVs out.
3D Gaussian splat PLYs (the 3DGS reference layout, as written by most splatting
tools) are read as `Node.splats`, decoded to physical scales, opacities and unit
quaternions, with extra per-splat properties kept in `splats.attributes`; a
scene with splats and no meshes is written back in the same layout.
OBJ and PLY hold geometry: meshes are written in world space as currently
posed (blend shapes at their default weights), in meters (`meters_per_unit=None`
keeps the scene's units), with a warning about what is dropped. OBJ also reads and writes materials in MTL files, like Blender
(`materials=False` to skip, `pbr_extensions=True` for the Pr/Pm keys). See the conventions below for what each format can hold.

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

Larger third-party sample files aren't stored in the repository. To download
all of them (to `tests/samples/`, only those missing) and run every test:

```bash
uv run --group dev pytest --fetch-samples
```

Without `--fetch-samples`, nothing is downloaded and tests that need a missing
sample are skipped (`-rs` shows why). The samples can also be fetched on their
own:

```bash
uv run python tests/fetch_samples.py --list   # samples, sources and licenses
uv run python tests/fetch_samples.py          # download all
uv run python tests/fetch_samples.py NAME     # or only some
```

Tests that run Blender use `--blender PATH`, `$SCENEPORT_BLENDER` or the newest
Blender 5.2+ installed, and are skipped when there is none.

Each sample's URL and license are listed in `tests/fetch_samples.py`. A sample
that's already there isn't downloaded again (delete it to refresh); zip samples
are unpacked and the zip deleted. Some samples have restrictive licenses (e.g.
Pixar's USD Asset EULA): keep them, and anything converted from them, out of
the repository.

## Roadmap

- [x] Phase 0 — binary codec, byte-exact round trip, `sceneport dump`
- [x] Phase 1 — static mesh (vertices, polygons, normals, UVs), trimesh conversion
- [x] Phase 1b — USD read/write for everything in Phase 1
- [x] Phase 2 — blend shapes (with in-betweens), FBX and USD
- [x] Phase 3 — materials / textures (USD graphs, FBX Phong, PBR interpretation)
- [x] Phase 4 — skeleton / skinning (linear blend skinning), FBX and USD
- [x] Phase 5 — animation (keyframes, joints/nodes/blend shape weights), FBX and USD
- [ ] Phase 6 — `.blend` through an installed Blender: running Blender, writing meshes, shape keys, armatures, skins, animation, cameras, lights and PBR materials (done); reading
- [x] Gaussian splats — the 3DGS PLY (read and write) and `.blend` (write); not yet USD

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
- Gaussian splats (`Node.splats`) hold physical values (scales, opacities,
  unit quaternions, raw SH coefficients in the 3DGS order), not a file's
  encoding; other per-splat fields stay in `splats.attributes`.
- Per-vertex data such as normals and UVs keeps FBX's indexed form, using USD
  interpolation names (`vertex`, `faceVarying`, `uniform`, `constant`); helpers
  expand it to one value per corner.
