# sceneport design notes

Decisions behind sceneport's structure and data model, with the reasoning, so
future changes can keep them consistent (or revisit them deliberately).

## Architecture

- **Format-neutral scene at the center.** `sceneport.scene` holds plain numpy
  data classes. Every file format or library is a converter around it, under
  `sceneport.interop` (`fbx`, `usd`, `trimesh`). No format is the "main" one.
  Cost accepted: each feature needs a neutral design plus one converter per
  format, and round trips through the Scene are *equivalent*, not byte-exact.
- **FBX is implemented natively** (no FBX SDK), in layers:
  binary codec (bytes ⇄ node tree, byte-exact round trip) → document
  (Objects, Connections, Properties70, templates) → conversion to/from Scene.
  Byte-exact fidelity is only guaranteed at the codec level.
- **USD goes through OpenUSD (`usd-core`)** rather than a custom parser, so
  composition (references, payloads, instances), `.usdc`, `.usdz` and UsdSkel
  evaluation come for free.
- **Dependencies:** numpy, trimesh and usd-core are all required, for simplicity.
- `sceneport.load` / `sceneport.save` pick the converter from the file extension.

## Coordinate systems and units

- **Nothing is converted on load or save.** `Scene.meters_per_unit` and the
  signed axes `up_axis` / `front_axis` / `right_axis` (e.g. `"+Y"`, with FBX's
  meaning of up/front/right) describe the data as it is.
- New scenes default to meters, Y up.
- Format-specific settings needed for round trips live in `Scene.extras`
  (e.g. the FBX file's GlobalSettings in `extras["fbx"]`), and writers start
  from them. USD only records the up axis (Y or Z); front/right are implied on load.

## Precision

- **In memory, everything is float64** (vertices, offsets, weights, matrices,
  transforms), whatever the source stored.
- **On disk, each format keeps its own types:**

  | Data | FBX | USD |
  |---|---|---|
  | Vertices, normals, UVs, blend shape offsets | float64 | float32 |
  | Skin weights | float64 | float32 |
  | Transforms (xformOps), bind/rest matrices | float64 | float64 |
  | Animation key values | float32 (`KeyValueFloat`) | float32 (SkelAnimation arrays) |
  | Animation key times | int64 ticks (46186158000 per second) | double time codes |

- Consequences: FBX round trips of geometry are exact; a trip through USD
  rounds geometry and weights to float32 (fine for data that started as
  float32, as most scanned/3DMM data does). Transforms stay exact in both.
  `Mesh.blend_shape_matrix(dtype=np.float32)` exists to halve memory for large
  bases.

## Meshes

- **Polygons use USD's layout:** `face_vertex_counts` + `face_vertex_indices`.
  Triangles, quads and n-gons are all kept; triangulation is a helper, never
  implicit. FBX's negative end-of-polygon markers convert directly.
- **Per-vertex data (`Primvar`) keeps FBX's indexed form** (values + optional
  indices) with USD interpolation names (`vertex`, `faceVarying`, `uniform`,
  `constant`); `Mesh.corner_values` expands to one value per corner.
- **USD UV naming:** the first UV set is written as `st` (what USD tools look
  for); every UV primvar keeps its original name in customData, restored on load.
- USD meshes are written with `subdivisionScheme = "none"` (USD's default,
  Catmull-Clark, would render them smoothed).
- **trimesh** is triangles-only with one UV/normal per vertex: `to_trimesh`
  splits vertices at seams by default, or keeps them one-to-one
  (`split_seams=False`) when vertex order matters.

## OBJ and PLY

- Native readers and writers (not trimesh, which triangulates and splits
  vertices at UV seams). Both keep polygons of any size, vertex order and
  per-corner UVs: OBJ through `vt` plus `f v/vt/vn` indices (written in our
  indexed form directly), PLY through a per-face `texcoord` list (the MeshLab
  convention). PLY also reads per-vertex `s`/`t` UVs.
- Geometry (plus MTL materials for OBJ, below). Writers bake each mesh into world space as currently posed
  (skinned by the scene's joints, blend shapes at default weights) — what
  Blender's exporters do — and warn about dropped skins, blend shapes, vertex groups and
  animation. Neither format has units, and tools read them as meters, so
  writers convert to meters by default (`meters_per_unit=` another unit, or
  None for the scene's own); a centimeter FBX becomes the same size in Blender
  whether imported as FBX or OBJ. Readers use meters (the Scene default). No
  axis conversion: axes are written as they are.
- OBJ: one `o` group per mesh node (read back as one node each; `g` groups when
  a file has no `o`); one UV set; normals come back per vertex when they are
  indexed like the vertices. Text with 9 significant digits by default (exact
  for float32; `precision=17` for float64).
- PLY: one mesh per file, so meshes are joined. PLY has no standard for UVs:
  `uvs="corner"` (default) writes MeshLab's per-face `texcoord` lists, lossless
  but ignored by Blender (per-corner normals are averaged per vertex);
  `uvs="vertex"` writes per-vertex `s`/`t` like Blender's exporter, splitting
  vertices at UV/normal seams (vertex count and order change, with a warning);
  `uvs=None` writes no UVs. Binary little-endian by default (`ascii=True`), float32
  coordinates by default (`double=True`).

## Names

- USD prim names must be identifiers: names are made valid and unique
  (`Cube.001` → `Cube_001`) and the original is kept as the prim's
  `displayName` (joints: in the Skeleton's `jointNames`).
- FBX blend shapes are named like Blender does: after the Shape geometry when
  a channel has one target, after the channel when it has in-betweens.

## Collections

- `Scene.collections` holds `Collection`s: named groups of nodes for
  organizing a scene, nestable (`children`), with format data in `extras`
  (e.g. `extras["blender"]["hide_render"]`). They don't touch transforms or the
  node tree: a group with a transform is a parent node. Membership is explicit
  (a member's children aren't implied) and a node may be in several collections
  or none, as in Blender, Maya sets and USD collections. Scene rewrites
  (`reframe.copy_scene`) point collections at the copied nodes.
- USD: one `UsdCollectionAPI` per collection on the default prim, with
  `expansionRule = explicitOnly` and nested collections included by their
  collection path (`</root.collection:Chairs>`); collection names are made
  valid identifiers and the originals, the order and the plain parts of
  `extras` are kept in the prim's `customData` (`sceneport:collections`). A
  joint has no prim: its Skeleton is included (read back as its root joint).
  Other tools' collections on the default prim are read through USD's own
  membership rules (expansion, excludes, expressions), mapped to nodes, flat;
  linking collections (`lightLink`, `shadowLink`, `filterLink`) are skipped.
  Blender's USD exporter writes no collections (prims follow parenting) and its
  importer ignores these.
- .blend: Blender collections nested the same way; objects in any collection
  are linked only there, and `extras["blender"]` values are set where the
  collection has that property. A joint's armature stands in for it.
- FBX (Maya / Max display layers and selection sets) isn't read or written yet:
  saving warns.

## Transforms

- `Transform` follows FBX's full formula (translation, Euler rotation with
  rotation order, scale, pre/post rotation, rotation/scaling pivots and
  offsets, and a geometric transform for the node's own mesh). It doubles as
  the neutral form because USD can express it exactly.
- Pre/post rotation and rotation order only apply when FBX's `RotationActive`
  is set (as in Blender and the FBX SDK).
- USD: written as an xformOp stack that reproduces the formula exactly
  (pivots via inverse ops); that stack is read back exactly, other stacks are
  decomposed from the local matrix (shear is lost).
- Decomposing a matrix: a mirror (negative determinant) becomes a negative X
  scale; an axis with zero scale (a node squashed flat, e.g. to hide it) has
  no direction, so it gets one that completes the others to a rotation. Any
  completion recomposes to the same matrix; guessing from the zero columns
  would turn the remaining axes too.

## Blend shapes

- `BlendShape` stores **offsets** (not absolute positions), optionally sparse
  (`indices`), a default `weight` (1 = fully applied), optional normal offsets
  and **in-betweens**. Weights interpolate piecewise-linearly through the
  in-betweens and extrapolate beyond [0, 1], matching FBX and UsdSkel.
  In-betweens exist because FBX (Maya) and USD support them; Blender and glTF do not.
- `range` (default (0, 1)) is the span of weights a shape is meant for, a hint
  for editors: only .blend stores it (the shape key slider range). Weights
  outside it still apply.
- **FBX export splits in-betweens by default** (`split_inbetweens=False` keeps
  them): Blender's FBX importer stops with an error on them. Weights
  interpolate linearly through the in-betweens, so a shape equals a sum of
  plain shapes ("<name>_<weight>" plus the full one) with "hat" function
  weights; weight curves are rewritten per frame. Exact at every frame.
- `Mesh.deformed_vertices(weights)` takes a dict, an `(S,)` array or an
  `(F, S)` array of frames (strict 1-D/2-D, shape axis last). Evaluation loops
  over shapes on their own vertices, so sparse shapes stay cheap and
  in-betweens work. `Mesh.blend_shape_matrix()` gives the dense `(S, N, 3)`
  basis for linear algebra and rejects shapes with in-betweens.
- Neither format stores a weight *range*; tools apply their own (Blender's
  sliders default to 0–1), so ranges are set on the tool side.
- USD: BlendShape prims under the Mesh; weights in a SkelAnimation — on the
  mesh's skeleton if skinned, otherwise on a jointless Skeleton under a
  SkelRoot made from the node. Apple's Quick Look shows nothing for these
  (it ignores a jointless Skeleton); accepted, since the files are valid USD
  and other tools read them. Exporters differ here: Maya's (maya-usd) also
  writes a jointless Skeleton, Blender's writes one dummy joint (`joint1`,
  every point bound to it) because "some DCCs seem to require joint names".

## Rigs

- **Joints are ordinary Nodes (`joint=True`), glTF-style.** This keeps FBX's
  structure (other nodes can hang from joints, e.g. collision helpers) while
  `Skin` provides arrays for computation. Alternatives considered: joints only
  as arrays in a Skeleton object (USD/SMPL style; breaks attachments) and
  joints only as nodes (FBX style; no array view).
- `Node` compares by identity (`@dataclass(eq=False)`), so nodes can be dict
  keys and two distinct joints are never "equal".
- `Skin` (on `Mesh.skin`): ordered joint nodes, per-vertex `(N, K)` joint
  indices and weights, `(J, 4, 4)` inverse bind matrices and the mesh's bind
  matrix. `Mesh.skinned_vertices` applies blend shapes then linear blend
  skinning, per pose or per frame. `Skin.dense_weights()` gives `(N, J)`.
- FBX: `TransformLink` is the joint's bind matrix; `Transform` is
  `inverse(TransformLink) @ mesh bind matrix`. Matrices are stored with the
  translation last and transposed on read/write. Repeated connections (FBX SDK
  files contain them) are followed once.
- USD: each joint tree becomes a Skeleton prim (joints parents-first, rest and
  bind matrices). Non-joint children of joints are written under the Skeleton
  at their joint's rest pose and re-attached on load. Mesh and Skeleton get a
  common SkelRoot (their nearest shared ancestor, or a generated wrapper that
  is dropped again on load). Only linear blend skinning is supported.
- USD reading resolves a Skeleton's animation like UsdSkel does: its own
  `skel:animationSource`, or one inherited from an ancestor (Pixar's UsdSkel
  examples bind it on the SkelRoot).
- USD SkelAnimations store translation, quaternion and scale per joint, so a
  mirrored joint (negative determinant, e.g. a rig's right side) must keep its
  reflection in the scale (written as (-1, -1, -1) x size); positive scales
  would turn the reflection into a wrong rotation.
- **Mirrored joints and Blender.** Mirrored joints are valid in USD and FBX, but
  Blender can't make bones of them: its USD importer drops the whole skeleton's
  animation, its FBX importer misplaces the mirrored chains.
  `interop/reframe.unmirror` re-expresses them without mirroring (each one's
  frame turned by -I; skinning and joint positions unchanged, the joints' axes
  point the other way). FBX export does this by default (`unmirror_joints=False`
  keeps them); USD keeps them by default, faithful to the source, and warns when
  a scene has mirrored joints (`unmirror_joints=True`, or
  `sceneport convert --unmirror-joints`, for Blender). Detection uses the rest
  pose; a joint mirrored in only its rest or only its bind pose can't be fully
  unmirrored (the skin itself mirrors), and is reported with a warning.

## Materials

- **Generic node graphs first.** A `Material` holds `ShaderNode`s and
  `NodeGraph`s keyed by their path relative to the material, an interface
  (`inputs`) and `outputs` per render context (`surface`, `mtlx:surface`,
  `ri:surface`, ...), following UsdShade. Nodes keep their identity as authored
  (`shader_id` such as `UsdPreviewSurface` or `ND_standard_surface_surfaceshader`,
  or source assets / inline code per source type), so any node library is read
  without knowing what its nodes mean, and nothing is lost.
- `Port`: USD value type name, value (Python scalars/str, `AssetPath`, float64
  arrays), connections (`Connection(node, "outputs:x" | "inputs:x")`, node ""
  = the material's interface) and color space. `Material.trace` follows
  connections through node graph and material interfaces to the producing
  shader output or value; `Material.surface(context)` finds the surface shader.
- **Slots, like FBX and Blender:** `Node.materials` lists the node's material
  slots and `Mesh.material_indices` gives each face's slot (None = slot 0).
  Keeping slots on the node lets instances of one mesh use different materials.
  `Scene.materials` lists every material, bound or not.
- USD: bindings are resolved with UsdShade (inheritance, collections). Each
  `materialBind` GeomSubset becomes a slot, in order; faces in no subset get one
  more slot with the mesh's own binding.
- FBX: a Material has no graph, just Properties70, so it becomes a Material
  with one `Surface` node (`FbxSurfacePhong` / `FbxSurfaceLambert` /
  `FbxSurfaceMaterial`) whose inputs are the effective properties (object over
  template, any exporter-specific ones included). Textures connected to a
  property become `FbxFileTexture` / `FbxLayeredTexture` nodes connected to that
  input, so `trace` works as for USD. File paths come from RelativeFilename /
  FileName (resolved next to the FBX file); embedded Video content is kept as
  bytes in the texture node's `extras["fbx"]["content"]`. Slots are the Model's
  connected materials in order; `LayerElementMaterial` gives per-polygon slots.
- **Interpretation is a separate layer.** `Material.pbr(context)` (scene/pbr.py)
  reads a graph into a `PBRMaterial`: the UsdPreviewSurface / glTF
  metallic-roughness parameters, each a value plus an optional `TextureRef`
  (file, channels, UV set, scale/bias, UV transform, wrap, color space). It
  understands UsdPreviewSurface, MaterialX standard_surface and OpenPBR, and
  FBX Phong/Lambert, following textures through pass-through nodes (normal
  map, extract/separate, convert). FBX Phong is converted as Blender's FBX
  importer does (roughness = 1 - sqrt(shininess) / 10, metallic =
  ReflectionFactor, its opacity rules). Anything else (math nodes, unknown
  libraries) is skipped and listed in `notes`, so the graph stays the source of
  truth and the PBR view is a convenience.
- **Writing keeps graphs, converts only when a format needs it.** USD writes
  any graph as UsdShade prims (invalid names made valid, originals in
  displayName / customData). FBX can only hold Phong: graphs that came from FBX
  are written back unchanged (property types kept in the node's
  `extras["fbx"]["types"]`, and in USD customData when not the inferred type);
  others go through `pbr()` and Blender's exporter formulas (shininess =
  ((1 - roughness) * 10)^2, ReflectionFactor = metallic, its opacity and
  texture-slot rules). An FBX graph written to USD keeps its surface on
  `outputs:fbx:surface` and gets a generated UsdPreviewSurface network on
  `outputs:surface`, so USD renderers see it and FBX -> USD -> FBX is lossless.
- **Slots on write:** FBX connects the node's materials to its Model in slot
  order and writes a `LayerElementMaterial` (empty slots get a "default"
  material). USD binds slot 0 to the mesh, or writes one `materialBind`
  GeomSubset per slot; a shared mesh's later prims block the bindings they
  bring along from the first one, so each node gets exactly its own slots.
- **Texture files:** USD, OBJ and FBX paths (FBX: RelativeFilename, with the
  absolute FileName next to it) are made relative to the written file when the
  texture exists on disk, as Blender does; FBX-embedded images are written to a `textures`
  folder next to the USD file (inside the package for `.usdz`, which leaves out
  textures it can't find, with a warning). PLY drops materials.
- **OBJ writes an MTL file** next to it (interop/mtl.py), following Blender's
  OBJ exporter: values from `pbr()` (Ns = (1 - roughness)^2 * 1000, Ka =
  metallic, Kd, Ks, Ke, Ni, d, Blender's illum choice), optional PBR keys
  (`pbr_extensions=True`: Pr, Pm, Pc, Pcr, map_Pr, map_Pm), texture maps with
  -o / -s / -bm, `usemtl` wherever a face's slot changes, materials sorted by
  name. Texture paths follow the same rules as USD (relative, embedded images
  to `textures/`). UV rotation has no MTL option (warning).
- **OBJ reads MTL back** the FBX way: a material keeps its raw MTL keys on an
  `MtlSurface` node and each map line becomes an `MtlTexture` node (options as
  inputs, path resolved next to the MTL), so OBJ -> OBJ writes the MTL back
  unchanged. `pbr()` interprets it like Blender's OBJ importer (illum flags;
  roughness = 1 - sqrt(Ns / 1000); metallic = mean(Ka) when illum reflects; Pr /
  Pm override; Pc / 4). `usemtl` state carries over between groups as in
  Blender; each group's materials in order of first use are its slots. In USD
  the MTL surface is kept on `outputs:mtl:surface` next to a generated
  UsdPreviewSurface, as for FBX.

## FBX export for Blender

- `sceneport.save(scene, "x.fbx")` rewrites two things Blender's FBX importer
  can't handle, without changing what the scene looks like (interop/fbx/compat.py,
  on a shallow copy; the scene itself is never modified):
  - in-between shapes are split (see Blend shapes);
  - mirrored joints (negative scale, e.g. a rig's right side) are unmirrored:
    each one's frame is turned by -I, so local' = F_parent @ local @ F_self and
    inverse bind' = F @ inverse bind. Skinning and joint positions are
    unchanged; animated joints in mirrored chains are baked per frame.
    `unmirror_joints=False` keeps the negative scales (Maya reads them).
- Key times are merged on whole FBX ticks: two float times for the same frame
  must not become two keys on one tick, which Blender's importer misreads.

## USD stage layout

- One top-level prim, which is the stage's default prim: viewers and
  references only see the default prim's subtree (Apple's Quick Look shows
  nothing else). A scene with several top-level nodes is written under a
  generated `root` Xform (as Blender's exporter does), tagged in customData
  and dropped again on load; a single top-level node is the default prim itself.
  The wrapper also carries Blender's own tag (`Blender:generated`), so Blender's
  importer skips it like the `/root` of its own exports instead of adding an empty;
  likewise sceneport drops a `/root` with Blender's tag when reading a Blender export.
- Materials go in a `_materials` Scope under the default prim.
- A node is an Xform prim (its transform) with its mesh in a Mesh child prim
  (the FBX geometric transform goes there; child nodes stay under the Xform),
  like Maya's transform and shape. The Mesh prim is named after the node, and
  the mesh's own name is its `displayName` (read back as `Mesh.name`): Blender's
  importer merges an Xform with its only Mesh child into one object named after
  the Mesh prim, and ignores `displayName`. E.g. MHR's `body_mesh` node with
  mesh `mesh` imports as object `body_mesh`, not `mesh`.
- `metersPerUnit` is read even when an old writer stored it as a float
  (Blender 2.80), which USD's typed getter rejects.

## Cameras

- `Node.camera` (scene/camera.py): lens settings on a node; the node's
  transform places and aims it, looking down local -Z with +Y up (USD, glTF,
  Blender). Physical units: focal length, apertures and offsets in mm,
  clipping and focus distance in scene units. `math.look_at_matrix(eye,
  target, up)` gives the world matrix that aims a camera node (-Z at the
  target, +Y toward `up`).
- USD: the node's prim is a `Camera`; lens values are in tenths of a scene unit
  (mm per unit = 100 x metersPerUnit, as Blender converts).
- FBX: a camera Model with a "Camera" NodeAttribute (film sizes in inches).
  FBX cameras look down +X, so camera nodes are turned 90 degrees about Y on
  write and back on read (`interop/reframe.py`, which also keeps their children in
  place), as Blender's exporter does.
- Viewers that frame a scene with its camera (Apple's Quick Look) need it: a
  scene's bounds can be useless (e.g. a node squashed to zero scale makes
  USD's bounds ~1e38).

- `Camera.exposure` (stops, log2; animatable): the image is brightened by
  2^exposure. USD: the camera's `exposure` attribute (written when not 0).
  FBX has none (warned). Blender has no exposure per camera: the scene camera's
  is the view's (Color Management > Exposure), keyed on the scene when animated;
  other cameras' exposure is warned about.

- `Scene.render` (RenderSettings, None when unset): the image size in pixels,
  the pixel aspect ratio (a pixel's width / height) and the camera node
  rendered (None: the scene's first camera). One per scene, as USD and Blender
  hold them. The image aspect and the camera's film aspect are independent;
  saving warns when they differ, since formats fit the film differently
  (Blender's horizontal sensor fit and FBX keep the film's width, USD's default
  conform policy expands the film to cover the image).
  - USD: a `RenderSettings` prim at /Render/RenderSettings (in a Scope, outside
    the default prim, as USD's render conventions have it), named by the
    stage's `renderSettingsPrimPath`: `resolution`, `pixelAspectRatio` and a
    `camera` relationship, written for the first camera when none is named.
    Reading takes that prim, else the first RenderSettings prim.
  - FBX has no scene resolution: every camera gets it, as Blender's exporter
    writes it (AspectRatioMode 2, fixed resolution; AspectWidth / AspectHeight;
    PixelAspectRatio), and GlobalSettings' DefaultCamera names the camera
    rendered. Reading takes DefaultCamera's camera, else the first with a fixed
    resolution (the render camera then unset: Blender's files name its
    viewport, "Producer Perspective"). Without cameras nothing is written
    (warned).
  - Blender: resolution_x / resolution_y at 100%, pixel_aspect_x /
    pixel_aspect_y (each at least 1 in Blender), and the scene camera. Without
    render settings the width is 1920 and the height follows the scene camera's
    film aspect.

## Lights

- `Node.light` (scene/light.py): a light on a node, placed and oriented by the
  node, like cameras. Types and values follow USD (UsdLux): "dome" (an
  environment image all around, an HDRI), "distant" (sun), "sphere" (point; a
  spotlight with `cone_angle`), "rect", "disk", "cylinder". Brightness is
  intensity x 2^exposure, times color (and colour temperature when set); sizes in
  scene units, angles in degrees. Lights shine down the node's -Z.
- **The environment is a node with a scene-level shortcut**, not a scene
  setting: as a node it reuses transforms (rotating the node turns the sky),
  animation, naming and reframing, and USD round trips stay exact (several
  domes, their place in the hierarchy). `Scene.environment` returns the first
  dome node and `Scene.add_environment(texture, rotation=...)` creates one, for
  the common one-sky case (Blender's World). A dome image's pole lies along the
  node's +Y (USD's classic DomeLight convention).
- USD: each type is its UsdLux prim on the node's own prim (a classic
  `DomeLight`, `DistantLight`, `SphereLight` + `ShapingAPI` for spots,
  `RectLight`, `DiskLight`, `CylinderLight`), `ShadowAPI` only when shadows are
  off or coloured, and an extent on sized lights (viewers cull by it).
  `DomeLight_1` is read too; one whose `poleAxis` puts the pole on Z gets its
  node turned to +Y (interop/reframe.py). Geometry, portal and plugin lights are
  read as plain nodes with a warning.
- FBX (interop/fbx/light.py), following Blender's exporter and importer: types
  point / directional / spot / area, Color, CastShadows, ShadowColor, spot
  `OuterAngle` = the full cone and `InnerAngle` = outer x (1 - softness). FBX
  lights shine down -Y, so light nodes are turned 90 degrees about X on write and
  back on read (Blender's `MAT_CONVERT_LIGHT`). FBX has no light units:
  Intensity = Blender watts x 100 (Blender's FBX exporter) and USD intensity =
  watts / pi, or sun strength / 4 (Blender's USD exporter), so brightness
  survives Blender <-> FBX <-> USD. Dome lights (FBX has none: an empty),
  area sizes, cylinders, colour temperature and diffuse / specular / normalize
  are dropped with a warning. (Blender's FBX importer reads point, sun and spot
  only: area lights arrive as point lights.)
- Light and camera settings are animatable: see Animation.

## Gaussian splats

**Model.** `Node.splats` (scene/splats.py) holds 3D Gaussians placed by the
node, like a mesh, in **physical values, not a file's encoding**:

- positions; per-axis scales (standard deviations in scene units; a 2DGS surfel
  has a zero third scale); unit quaternions (w, x, y, z); opacities 0..1;
- real SH color coefficients (N, (degree + 1)^2, 3) in the 3DGS order and signs,
  kept as stored. The 3DGS convention adds 0.5 to the evaluated color, so a
  splat's color toward direction d is 0.5 + `sh_basis(d)` @ sh and its
  view-independent color 0.5 + SH_C0 x sh[:, 0] (`base_colors()`);
- `attributes`: any other per-splat values a file carries (a method's own
  fields, e.g. GaussianAvatars' parent triangle `binding_0`), by name, so they
  can be written back.

These are the quantities of OpenUSD's `ParticleField3DGaussianSplat`, so a
reader decodes once (the 3DGS reference PLY stores log scales, logit opacities
and unnormalized quaternions) and nothing downstream needs to know a file's
encoding.

**Moving splats.** `GaussianSplats.transformed(matrix)` maps every Gaussian by a
4x4 matrix: positions move and covariances become A Σ Aᵀ. A rotation times a
uniform scale rotates the orientations and scales the scales; any other matrix
re-derives each splat's axes from its covariance, largest first, so flat 2DGS
splats stay flat. SH colors turn with the splats: each band maps onto itself
linearly, and `rotate_sh` fits its matrix from the basis at sample directions
(exact). Skews and uneven scales don't map view directions rigidly, so there the
closest rotation is used, with a warning. Writers use it for node transforms and
units (PLY), and `change_basis` for axis changes (Blender's Z-up).

**PLY** (interop/ply.py), the 3DGS reference layout:

- Reading: a file without faces whose vertices have `opacity`, `scale_*` and
  `rot_*` is splats: exp of the log scales, sigmoid of the logit opacity, the
  quaternion normalized, SH from `f_dc_0..2` plus `f_rest_*` (all of red's
  higher-order coefficients, then green's, then blue's; the degree follows from
  their count). Two scales (2DGS) get a zero third; `red`/`green`/`blue` stand in
  for a missing `f_dc`. The unused normals are dropped; every other property
  goes to `attributes`. Compressed formats (.splat, .spz, compressed PLY) aren't
  read.
- Writing: a scene with splats and no meshes is written in the same layout (a
  PLY file holds a mesh or splats, so with meshes too the splats are dropped
  with a warning), in world space and meters like meshes. Flat splats get two
  scales; opacities are clipped to 1e-12..1 - 1e-12 so the logit stays finite;
  extra attributes keep their names and numeric types. Several splat nodes are
  joined: SH padded to the highest degree, attributes not on every node dropped
  with a warning. Read and written back, the GaussianAvatars sample matches
  exactly except quaternion normalization (relative error 1e-7).

**.blend** (interop/blend/to_blend.py): a splat node becomes a point cloud with
Blender 5.3's splat attributes, the layout its PLY importer makes: `position`,
`scale`, `rotation` (quaternion), `radiance:base` (SH DC and opacity),
`radiance:sh_0..` (the higher SH coefficients, RGB each), plus `radius` (the
largest scale) and `color` (the view-independent color, linear) for Blender's
point display, and the extra attributes by name. Blender 5.3+ marks it as
Gaussian splats; 5.2 has no splat type and keeps plain points (5.3 turns them
into splats by setting the point cloud's type). Blender renders the higher SH
bands and the rotations, and blends splats in linear light (unlike the 3DGS
reference, which blends the colors as they are), so faint, bright splats look
brighter there.

**Other formats.** USD, FBX and OBJ don't store splats yet: each warns that they
aren't written, so they're never dropped silently.

## Animation

- **Keyframes, not baked samples.** `Scene.animations` holds clips
  (`Animation`: name, fps, optional start/end) of `Channel`s. A channel animates
  one scalar: a component of a node's `translation` / `rotation` / `scale` /
  `orientation`, or one blend shape weight (target = the Mesh, property = the
  shape name). Keys carry a per-key interpolation mode for the segment they
  start: constant, linear or cubic (Hermite with automatic, Catmull-Rom style
  tangents; FBX's stored tangent data is kept in `extras` for exact FBX round
  trips but not evaluated). Values hold before the first and after the last key.
- **Evaluation is frame-batched:** `Animation.world_matrices` / `joint_world` /
  `blend_weights` / `skinned_vertices` take an array of times and feed
  `Mesh.skinned_vertices` / `deformed_vertices`. `Transform.matrix()` accepts
  per-frame overrides for this.
- **Rotation channels come in two kinds**, like Blender's rotation modes:
  `rotation` (Euler degrees in the node's rotation order; FBX and USD xformOps)
  and `orientation` (quaternion w, x, y, z replacing the Euler term; UsdSkel and
  Blender bones). Quaternion components are interpolated one by one and
  normalized (as in Blender), so keys must be sign-continuous
  (`math.quaternion_continuity`). Data stays in the kind it was loaded in;
  writers convert only where a format requires it (FBX has no quaternion
  curves: orientation becomes Euler keys at every frame, unwrapped).
- **FBX:** one clip per AnimationStack (empty stacks skipped; only the first
  layer is read, with a warning if there are more). Rotation curves are Euler in
  the node's order; blend shape curves (`DeformPercent`) are percent. Key times
  are int64 ticks, values float32. The first clip sets the file's TimeMode and
  time span.
- **USD:** one timeline per stage. The first clip is on it: its fps becomes
  `timeCodesPerSecond`, its range `startTimeCode`/`endTimeCode`, and its name is
  recorded in the root layer's customLayerData. Every clip becomes a
  SkelAnimation (joint translations, quaternion rotations, scales, blend shape
  weights) under each skeleton it animates; the first is the bound
  `skel:animationSource`. Non-joint node animation is written as xformOp time
  samples, for the first clip only. USD interpolates samples linearly, so clips
  are baked to one sample per frame. Rest values and default weights are kept
  as attribute defaults next to the samples, and loading keeps them.
- **Time codes** are doubles: seconds × fps, snapped to whole frames when only
  floating-point noise is off, so FBX keys on frames land on integer codes.
- On load, USD joint keys that never leave the rest value are not turned into channels.
- **Camera and light settings** animate through channels whose property is
  `"camera:<field>"` or `"light:<field>"` (target = the node; e.g.
  `"camera:focal_length"`, `"light:color"` with components 0-2). Fields are
  the dataclass fields, in the same units (`ATTRIBUTE_PROPERTIES`); validation
  needs the node to carry a camera / light. `Animation.attribute_values`,
  `camera_at` and `light_at` evaluate them (unanimated fields keep the stored
  value). Reframing a camera / light node (FBX turns, the DomeLight_1 pole)
  bakes its transform but keeps these channels.
- USD writes them as attribute time samples, one per frame, for the first clip
  only (later clips warn), in USD's units (lens in tenths of a unit,
  `clippingRange` from near / far, a temperature turns on
  `enableColorTemperature`); attributes with more than one sample are read back.
- FBX writes them as AnimationCurveNodes on the camera / light NodeAttribute,
  baked per frame because one FBX property can combine several fields:
  FocalLength, FilmWidth/Height and offsets (inches), Near/FarPlane,
  FocusDistance; light Intensity (intensity and exposure, Blender's watt
  conventions), Color, ShadowColor, and spot OuterAngle / InnerAngle (cone
  angle and softness). Other fields (f-stop, sizes, temperature, ...) warn.
- Blender (5.2) imports only FBX FocalLength and FocusDistance curves and no
  light animation from FBX or USD. Its FBX exporter writes animated
  FocusDistance in mm (x1000) while the static property is in scene units, and
  its importer divides curves by 1000; sceneport writes scene units in both, so
  Blender reads an animated focus distance from sceneport's FBX 1000x too small.
- **Per-vertex animation** (point caches): a clip's `point_samples` hold
  time-sampled vertex positions per mesh (`PointSamples`, mesh space; linear
  between samples, held outside), which replace the rest positions before blend
  shapes and skinning (`Animation.vertices` / `skinned_vertices`). USD reads and
  writes them as time-sampled `points` on the Mesh prim (timeline clip only,
  rest points kept as the default), with the mesh's `extent` time-sampled
  alongside: viewers cull by extent (Quick Look stops drawing a mesh whose
  points left a stale one). FBX, OBJ and PLY can't hold them and warn.

## Blender (.blend)

- **An installed Blender does the work**, run in the background
  (`interop/blend/runner.py`): a .blend is Blender's memory dumped to disk, so
  only Blender reads and writes it reliably. Not the `bpy` package from PyPI (one
  Python version per release, 250-400 MB, GPL code in the process) and not a
  pure-Python parser (Blender's internals change every release; modifiers and
  drivers need Blender to evaluate). Blender 5.2 or newer only: 5.x has a single
  animation API (slotted actions).
- **Nothing is installed into Blender.** Each run puts a temporary folder on
  Blender's `sys.path` holding only a link to the sceneport package (a copy
  where symlinks aren't allowed), so sceneport runs on Blender's own Python 3.13,
  numpy and pxr, and no host packages leak in (a venv's site-packages would bring
  numpy built for another Python). So sceneport keeps its import-time
  dependencies to numpy and stays compatible with Blender's Python and numpy 2.
- **One action per run:** `Blender -b --factory-startup --python-exit-code 1
  --python-expr <boot> -- <path dir> <job dir>` (no user preferences or add-ons).
  `bridge.main` reads `request.pkl`, runs `bridge.ACTIONS[action]` and always
  writes `response.pkl` with the result, the warnings raised (re-raised on the
  host) and any traceback (raised as `BlenderError`). A run that leaves no
  response (a crash) raises with the end of Blender's output. Pickles are safe
  here: both sides run the same sceneport source and the files never leave the
  private job folder. A run costs about a second.
- `bridge.py` and `to_blend.py` are the only modules that import bpy.
- **Writing** (`to_blend.py`, after `compat.prepare`): the scene is moved to
  Blender's axes and units, Z up, -Y front, meters (`reframe.change_basis`), so
  the data itself is Z-up instead of root objects carrying a 90 degree turn (what
  Blender's FBX / USD importers do). The change M = s * C (C a signed axis
  permutation) conjugates every local matrix, so plain transforms stay plain (the
  Euler order permuted, e.g. Y-up ZXY is Z-up YXZ) and their channels map
  one-to-one with unchanged keys; pivots / pre / post rotations are baked.
  Cameras and lights are turned back to look down their -Z; a mirroring change
  reverses face winding. In-betweens are split (as for FBX) and geometric
  transforms baked into (copies of) the meshes.
- Meshes are built with `foreach_set` (points, corners, faces), then validated;
  normals become custom normals on smooth faces (without normals, faces stay
  flat, Blender's default), UV maps per corner.
- Blend shapes become shape keys relative to a "Basis" key. The slider range is
  `BlendShape.range` (default 0..1; a hint for editors that no other format
  stores), widened to whole numbers where the shape's default or animated
  weights fall outside it, within Blender's -10..10: Blender clamps values to
  the slider, e.g. GNM identity weights of -4 would become 0. Blender computes shape
  key normals itself, so normal offsets are dropped.
- **Armatures** (`compat.rigs`, pure numpy, then `to_blend._bones`): each tree
  of joints is one armature, held by the joints' parent node when it's a plain
  node (a USD Skeleton, an FBX "Armature" null), else by a new node inserted
  there (a skeleton at the top level, e.g. MHR's FBX), named after the root
  joint ("body_world" for MHR; the root bone keeps the name too, since Blender
  keeps object and bone names apart) or "Armature" for several root joints. Skinned meshes
  not under their armature are moved under it, keeping their world placement:
  both as Blender's FBX importer does, whose convention is deformed meshes as
  children of their armature. Blender's armature modifier deforms by P @ R^-1 (pose times
  inverse rest, in armature space) and has no separate bind pose, so bones rest
  in the bind pose: with A the armature's world matrix and B a joint's bind
  world matrix (from its skin, else its current one), rest R = A^-1 @ B with
  scale removed (edit bones can't scale), and pose P = A^-1 @ J @ B^-1 @ A @ R
  for current world J, which is A^-1 @ J when the bind has no scale. The pose
  bone basis is computed from the rest matrices Blender stored (pose = parent
  pose @ parent rest^-1 @ rest @ basis). Mirrored joints are unmirrored first.
  A bone's frame is its joint's: it points along the joint's +Y (rigs whose
  joints point down X show sideways bones; deformation is unaffected). Bones
  reach the child joint ahead along +Y, else the nearest child, else half
  their parent's length.
- **Skins** become vertex groups named after the bones (a joint listed twice
  for a vertex has its weights added) and an Armature modifier. The modifier
  deforms the mesh as its object places it, so a mesh whose node moved since
  binding is re-bound first (points moved to the node's space). Nodes under a
  joint are parented to its bone, with a parent inverse cancelling Blender's
  offset to the bone's tail. A skin spanning several armatures deforms with the
  first one only (warned).
- **Animation**: one Action per clip, with a slot per animated datablock
  (object, armature object, shape key set: Blender 5's slotted actions). The
  first clip is assigned to them and sets the scene's fps (`fps / fps_base` for
  fractional rates), frame range and current frame; the others keep a fake
  user. Frame = seconds x fps, so time 0 is frame 0 (as Blender's USD importer;
  its FBX importer starts at 1). Object transform and shape key weight keys are
  written as they are, `change_basis` having mapped them one-to-one (Euler in
  radians; `orientation` as `rotation_quaternion` with the object's rotation
  mode set to quaternion); constant / linear keys map directly and cubic ones
  get FREE Bezier handles a third of the way along each segment on sceneport's
  automatic slopes, which reproduces its Hermite curve exactly. Bones are
  sampled on every frame and key: their Blender values are relative to the
  rest (bind) pose, P(t) = A(t)^-1 @ J(t) @ B^-1 @ A @ R, so keys don't carry
  over one-to-one. A skinned mesh moved under its armature has its transform
  channels baked in its new parent's space.
- **Per-vertex animation** (point samples, the first clip's) can't live in a
  .blend: a CacheFile only references a file and can't be packed, and shape
  keys per frame or packed Geometry Nodes bakes would be no more editable than
  a cache. So the meshes whose samples move are written to a sidecar
  `<name>_cache.usdc` (`blend/cache.py`: topology plus time-sampled points in
  the prepared scene's Z-up meters, one time code per frame) and read by a Mesh
  Sequence Cache modifier, first in the object's stack (an armature after it
  skins the cached points), through a CacheFile referenced relative to the
  .blend (`//name_cache.usdc`). The modifier reads vertices only, so the .blend's
  faces, UVs and materials stay editable, and interpolates between samples.
  Meshes whose samples never leave their rest points (exporters sample static
  meshes too: 119 of Spring's 148) get no cache. A mesh with shape keys loses
  their effect under the cache (warned). Making such animation editable needs a
  rig: fitting one (skinning decomposition) is a possible later addition.
- **Cameras and lights** (`blend/cameras_lights.py`) use the values Blender's
  own USD importer sets for the same scene, so both routes into Blender agree:
  horizontal sensor fit with shift = offset / sensor width; energy = intensity
  x pi (sun x 4) with exposure, color temperature, normalize and the diffuse /
  specular factors as Blender's own light properties; spot size = 2 x cone;
  rect / disk = area lights (disk size = 2 x radius); cylinders = point lights
  (warned); no shadow color or light textures (warned). Two deliberate
  differences: a sun's `angle` is used as is (USD's angular diameter, the same
  as Blender's; Blender's importer doubles it), and the render height is set so
  the scene camera's film aspect holds (the importer keeps 1920 x 1080). Their
  animated settings are F-curves on the camera / light data (`lens`, `energy`,
  `spot_size`, ...; shift keys divide by the sensor width, on the offset's keys
  if the width is animated too).
- **EEVEE's light threshold is fitted.** EEVEE cuts a point / spot / area light
  off where its power falls to the scene's light threshold, at sqrt(power /
  threshold) with power = max(color) x energy x 2^exposure / 100 (Blender's
  `eevee_light.cc`), and fades it towards there. Physically scaled lights, e.g.
  a light stage's 0.37 W lamps (20 lux at 1 m) 1 m from the subject, reach only
  0.6 m at Blender's default 0.01, so the scene renders black in EEVEE while
  Cycles (which has no threshold) lights it. After writing, the threshold is
  lowered until every such light (at its dimmest non-zero animated energy in
  any clip, assigned or kept with a fake user: a light that's off needs no reach) reaches 4x
  the size of everything in the scene, where the fade costs under 0.5%; it is
  never raised, so ordinary lights leave Blender's default. Lights with a
  custom distance and suns are left alone.
- **Scripts:** `scene.extras["blender"]["texts"]` ({name: source}) become
  Text datablocks, saved inside the .blend and run on demand from Blender's
  Text Editor (Run Script, Alt+P), e.g. a script switching many lights between
  actions (Blender assigns actions per datablock, with no control for many at
  once). They aren't registered: Blender only auto-runs a file's scripts when
  the user allows it, so nothing runs by itself.
- **The first dome light becomes the World** (Blender has one): Environment
  Texture -> Background with strength = intensity x 2^exposure, through a
  Mapping node (point, XYZ Euler) set to R_default @ R^-1, R the dome's world
  rotation and R_default = Euler(90, 0, 90), Blender's own World orientation as
  a dome. This is the rule Blender's USD importer follows (checked on Z-up and
  Y-up domes). A turning or brightening dome is sampled onto the Mapping and
  Background nodes. The dome node stays an empty. Note that Blender's USD
  importer only fills a World the scene already has.
- **Materials** (`blend/materials.py`): each Material becomes a Principled
  BSDF built from `Material.pbr()`, laid out as Blender's USD importer builds
  UsdPreviewSurface materials (compared node for node in the tests): constants
  on the Principled inputs (Emission Strength 1, Specular Tint for the specular
  workflow); textures as Image Texture nodes (sRGB for colors, Non-Color for
  data, wrap as the image extension) through a shared UV Map node and a Mapping
  node (point type, UsdTransform2d's translation / rotation / scale); one
  channel through Separate Color (alpha from the image's Alpha); scale / bias as
  multiply-add nodes; normal maps through a Normal Map node (strength =
  normal_scale), re-encoded to 0..1 when their scale / bias decode to -1..1;
  displacement through a Displacement node; occlusion kept as an unconnected
  image node. Opacity is coverage (Principled Alpha, blended; a threshold makes
  a dithered cutout), as Blender's FBX and glTF importers read transparency;
  Blender's USD importer instead reads an unthresholded opacity as glass
  (Transmission Weight = 1 - opacity) and doesn't threshold a constant one.
  Slots live on the mesh data (on the object where a shared mesh's users
  differ), faces take the material indices, unused materials keep a fake user.
  Image paths are relative to the .blend; images embedded in an FBX are packed
  (FBX paths may be Windows-style: separators are normalized), and
  `pack_images=True` packs the ones found on disk as well. They keep their
  relative paths, so Blender's Unpack Resources writes them back there; a
  missing file can't be packed and stays a reference. Only the PBR
  reading is kept: source-specific graphs (FBX Phong values, MaterialX
  networks) aren't stored in the .blend.
- **Setup scripts** (`scripts=`): Python files run in the same Blender after
  the scene is built and before saving, as `__main__`, for scene-specific setup
  that isn't file conversion (node trees, modifiers, object settings). One
  Blender run instead of a second one that reopens and resaves the file; the
  scripts' errors come back as BlenderError like the rest of the save. A
  script given as `(path, params)` reads `params` as a global dict (pickled
  across with the scene), so one script can serve several files; a plain
  path gets an empty one.
