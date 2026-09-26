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

## Names

- USD prim names must be identifiers: names are made valid and unique
  (`Cube.001` → `Cube_001`) and the original is kept as the prim's
  `displayName` (joints: in the Skeleton's `jointNames`).
- FBX blend shapes are named like Blender does: after the Shape geometry when
  a channel has one target, after the channel when it has in-betweens.

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

## Blend shapes

- `BlendShape` stores **offsets** (not absolute positions), optionally sparse
  (`indices`), a default `weight` (1 = fully applied), optional normal offsets
  and **in-betweens**. Weights interpolate piecewise-linearly through the
  in-betweens and extrapolate beyond [0, 1], matching FBX and UsdSkel.
  In-betweens exist because FBX (Maya) and USD support them; Blender and glTF do not.
- `Mesh.deformed_vertices(weights)` takes a dict, an `(S,)` array or an
  `(F, S)` array of frames (strict 1-D/2-D, shape axis last). Evaluation loops
  over shapes on their own vertices, so sparse shapes stay cheap and
  in-betweens work. `Mesh.blend_shape_matrix()` gives the dense `(S, N, 3)`
  basis for linear algebra and rejects shapes with in-betweens.
- Neither format stores a weight *range*; tools apply their own (Blender's
  sliders default to 0–1), so ranges are set on the tool side.
- USD: BlendShape prims under the Mesh; weights in a SkelAnimation — on the
  mesh's skeleton if skinned, otherwise on a jointless Skeleton under a
  SkelRoot made from the node.

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

## Animation (planned)

- **Keyframes, not baked samples.** Clips (`Scene.animations`) hold channels of
  keys (`times`, `values`) with an interpolation mode (constant, linear,
  cubic); evaluation samples them at any times and feeds the frame-batched
  `skinned_vertices` / `deformed_vertices`. FBX tangent data is kept in
  `extras` for exact FBX round trips.
- **Rotation channels come in two kinds**, like Blender's rotation modes:
  Euler (in the node's rotation order; what FBX and USD xformOps use) and
  quaternion (what UsdSkel and Blender pose bones use). Data stays in the kind
  it was loaded in; writers convert only when a format requires it.
- **Blend shape weight channels** are part of the model (FBX `DeformPercent`
  curves, USD `blendShapeWeights`), so body and face animation can play together.
- **FBX:** one clip per AnimationStack; empty stacks are skipped; only the
  first layer of a stack is read at first (warning if there are more).
- **USD:** one timeline per stage. Every clip is written as its own
  SkelAnimation prim; the first (or chosen) clip is the active
  `skel:animationSource`, and non-joint node animation is written only for it.
  USD time samples interpolate linearly (no tangents; splines are scalar-only),
  so curves USD can't represent are baked at the clip's frame rate.
- Frame rate: `Animation.fps` ⇄ FBX `TimeMode` / USD `timeCodesPerSecond`.
