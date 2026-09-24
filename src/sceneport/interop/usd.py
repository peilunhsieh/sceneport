"""Read and write USD (.usd, .usda, .usdc, .usdz) through OpenUSD (usd-core).

Writing: one Xform prim per Node and a Mesh prim (child of the Xform) per mesh,
with subdivisionScheme "none". Transforms become an xformOp stack that expresses
Transform exactly (pivots and pre/post rotation included); the geometric
transform goes on the Mesh prim. Shared meshes are written once and internally
referenced. Invalid prim names are made valid, with the original kept as the
prim's displayName. The first UV set is written as "st" (the USD convention);
every UV primvar keeps its original name in customData.

Blend shapes: a node whose mesh has blend shapes is written as a SkelRoot;
each shape is a BlendShape prim under the Mesh (in-betweens included), bound
with SkelBindingAPI, and a jointless Skeleton's SkelAnimation holds the
default weights. Skeleton prims are skipped when reading.

Reading: composed stages (references, payloads, instances) are read through
usd-core. Xformable prims become Nodes; other prims (Scope, Material, ...) are
flattened away. Transform stacks that match the one written here are read
exactly; others are decomposed from the local matrix.

USD only records the up axis (Y or Z); front and right axes are implied
(+Z/+X for Y up, -Y/+X for Z up). Points are stored as float32.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdSkel, UsdUtils, Vt

from ..math import ROTATION_ORDERS, decompose_matrix
from ..scene.types import BlendShape, Inbetween, Mesh, Node, Primvar, Scene, Transform

EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")
NAME_KEY = "sceneport:name"  # customData key holding a primvar's original name
IMPLIED_AXES = {"+Y": ("+Z", "+X"), "+Z": ("-Y", "+X")}  # up -> (front, right)

_TIME = Usd.TimeCode.EarliestTime()
_PREDICATE = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
_DOUBLE = UsdGeom.XformOp.PrecisionDouble

# The xformOp stack for Transform, in order; "rotate" stands for xformOp:rotate<ORDER>.
# Each entry: (op name, Transform field or None for the inverse half of a pivot pair).
_OP_STACK = [
    ("xformOp:translate", "translation"),
    ("xformOp:translate:rotateOffset", "rotation_offset"),
    ("xformOp:translate:rotatePivot", "rotation_pivot"),
    ("xformOp:rotateXYZ:preRotation", "pre_rotation"),
    ("rotate", "rotation"),
    ("!invert!xformOp:rotateXYZ:postRotation", "post_rotation"),
    ("!invert!xformOp:translate:rotatePivot", None),
    ("xformOp:translate:scaleOffset", "scaling_offset"),
    ("xformOp:translate:scalePivot", "scaling_pivot"),
    ("xformOp:scale", "scale"),
    ("!invert!xformOp:translate:scalePivot", None),
]
_OP_INDEX = {name: i for i, (name, _) in enumerate(_OP_STACK)}


# -- public API -----------------------------------------------------------------


def load(path: str | os.PathLike) -> Scene:
    path = os.path.abspath(path)
    if (layer := Sdf.Layer.Find(path)) is not None:
        layer.Reload(force=True)  # the layer registry may hold an older version of this file
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise OSError(f"could not open USD stage {path!r}")
    return scene_from_stage(stage)


def save(scene: Scene, path: str | os.PathLike) -> None:
    path = Path(path)
    stage = stage_from_scene(scene)
    if path.suffix.lower() == ".usdz":
        with tempfile.TemporaryDirectory() as tmp:
            layer_path = Path(tmp) / f"{path.stem}.usdc"
            _export(stage, layer_path)
            if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer_path)), str(path)):
                raise OSError(f"could not write {path}")
    else:
        _export(stage, path)


def stage_from_scene(scene: Scene) -> Usd.Stage:
    """An in-memory stage holding `scene`."""
    if scene.up_axis not in IMPLIED_AXES:
        raise ValueError(f"USD only supports +Y or +Z up, got {scene.up_axis!r}")
    if (scene.front_axis, scene.right_axis) != IMPLIED_AXES[scene.up_axis]:
        front, right = IMPLIED_AXES[scene.up_axis]
        warnings.warn(
            f"USD can't record front/right axes; readers will assume front {front}, right {right} "
            f"(scene has {scene.front_axis}, {scene.right_axis})"
        )
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, scene.meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if scene.up_axis == "+Y" else UsdGeom.Tokens.z)

    written: dict[int, Sdf.Path] = {}  # id(Mesh) -> path of its Mesh prim
    taken: set[str] = set()
    root_paths = [_write_node(stage, Sdf.Path.absoluteRootPath, node, taken, written) for node in scene.nodes]
    if root_paths:
        stage.SetDefaultPrim(stage.GetPrimAtPath(root_paths[0]))
    return stage


def scene_from_stage(stage: Usd.Stage) -> Scene:
    up = "+" + UsdGeom.GetStageUpAxis(stage)
    front, right = IMPLIED_AXES[up]
    scene = Scene(meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage), up_axis=up, front_axis=front, right_axis=right)
    meshes: dict[Sdf.Path, Mesh] = {}
    for prim in stage.GetPseudoRoot().GetFilteredChildren(_PREDICATE):
        _read_prim(prim, scene.nodes, meshes)
    return scene


# -- writing --------------------------------------------------------------------


def _export(stage: Usd.Stage, path: Path) -> None:
    if not stage.GetRootLayer().Export(str(path)):
        raise OSError(f"could not write {path}")


def _prim_name(name: str, taken: set[str]) -> str:
    base = Tf.MakeValidIdentifier(name or "node")
    candidate, i = base, 1
    while candidate in taken:
        candidate, i = f"{base}_{i}", i + 1
    taken.add(candidate)
    return candidate


def _write_node(stage: Usd.Stage, parent: Sdf.Path, node: Node, taken: set[str], written: dict[int, Sdf.Path]) -> Sdf.Path:
    path = parent.AppendChild(_prim_name(node.name, taken))
    has_shapes = node.mesh is not None and bool(node.mesh.blend_shapes)
    # Blend shapes need a SkelRoot above the mesh.
    xform = (UsdSkel.Root if has_shapes else UsdGeom.Xform).Define(stage, path)
    _set_display_name(xform.GetPrim(), node.name)
    _write_transform(xform, node.transform)

    child_names: set[str] = set()
    if node.mesh is not None:
        mesh_path = path.AppendChild(_prim_name(node.mesh.name, child_names))
        if id(node.mesh) in written:
            prim = stage.DefinePrim(mesh_path)
            prim.GetReferences().AddInternalReference(written[id(node.mesh)])
            usd_mesh = UsdGeom.Mesh(prim)
            usd_mesh.ClearXformOpOrder()
        else:
            usd_mesh = _write_mesh(stage, mesh_path, node.mesh)
            written[id(node.mesh)] = mesh_path
        _set_display_name(usd_mesh.GetPrim(), node.mesh.name)
        _write_geometric(usd_mesh, node.transform)
        if has_shapes:
            _bind_blend_shapes(stage, path.AppendChild(_prim_name("Skel", child_names)), usd_mesh, node.mesh)

    for child in node.children:
        _write_node(stage, path, child, child_names, written)
    return path


def _set_display_name(prim: Usd.Prim, name: str) -> None:
    if prim.GetName() != name:
        prim.SetDisplayName(name)


def _add_op(xf: UsdGeom.Xformable, kind: str, value=None, suffix: str = "", inverse: bool = False) -> None:
    op = getattr(xf, f"Add{kind}Op")(precision=_DOUBLE, opSuffix=suffix, isInverseOp=inverse)
    if value is not None:
        op.GetAttr().Set(Gf.Vec3d(*(float(v) for v in value)))


def _write_transform(xf: UsdGeom.Xformable, t: Transform) -> None:
    def nonzero(v):
        return bool(np.any(v))

    if nonzero(t.translation):
        _add_op(xf, "Translate", t.translation)
    if nonzero(t.rotation_offset):
        _add_op(xf, "Translate", t.rotation_offset, "rotateOffset")
    if nonzero(t.rotation_pivot):
        _add_op(xf, "Translate", t.rotation_pivot, "rotatePivot")
    if nonzero(t.pre_rotation):
        _add_op(xf, "RotateXYZ", t.pre_rotation, "preRotation")
    if nonzero(t.rotation) or t.rotation_order != 0:
        _add_op(xf, f"Rotate{ROTATION_ORDERS[t.rotation_order]}", t.rotation)
    if nonzero(t.post_rotation):
        _add_op(xf, "RotateXYZ", t.post_rotation, "postRotation", inverse=True)
    if nonzero(t.rotation_pivot):
        _add_op(xf, "Translate", suffix="rotatePivot", inverse=True)
    if nonzero(t.scaling_offset):
        _add_op(xf, "Translate", t.scaling_offset, "scaleOffset")
    if nonzero(t.scaling_pivot):
        _add_op(xf, "Translate", t.scaling_pivot, "scalePivot")
    if not np.array_equal(t.scale, [1, 1, 1]):
        _add_op(xf, "Scale", t.scale)
    if nonzero(t.scaling_pivot):
        _add_op(xf, "Translate", suffix="scalePivot", inverse=True)


def _write_geometric(xf: UsdGeom.Xformable, t: Transform) -> None:
    geometric = Transform(translation=t.geometric_translation, rotation=t.geometric_rotation, scale=t.geometric_scale)
    _write_transform(xf, geometric)


def _write_mesh(stage: Usd.Stage, path: Sdf.Path, mesh: Mesh) -> UsdGeom.Mesh:
    mesh.validate()
    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    points = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(mesh.face_vertex_counts)))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(mesh.face_vertex_indices)))
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if len(points):
        usd_mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0), points.max(0)])))

    primvars = UsdGeom.PrimvarsAPI(usd_mesh)
    if mesh.normals is not None:
        _write_primvar(primvars, "normals", Sdf.ValueTypeNames.Normal3fArray, Vt.Vec3fArray, mesh.normals)
    names = {"normals"}
    for i, (name, uv) in enumerate(mesh.uvs.items()):
        usd_name = _prim_name("st" if i == 0 else name, names)
        _write_primvar(primvars, usd_name, Sdf.ValueTypeNames.TexCoord2fArray, Vt.Vec2fArray, uv, original=name)
    return usd_mesh


def _shape_tokens(mesh: Mesh) -> list[str]:
    """Valid, unique names for the mesh's blend shapes: BlendShape prim names and skel:blendShapes tokens."""
    taken: set[str] = set()
    return [_prim_name(shape.name, taken) for shape in mesh.blend_shapes]


def _bind_blend_shapes(stage: Usd.Stage, skel_path: Sdf.Path, usd_mesh: UsdGeom.Mesh, mesh: Mesh) -> None:
    """BlendShape prims under the mesh (unless referenced in), plus a jointless Skeleton whose
    SkelAnimation carries the default weights."""
    tokens = _shape_tokens(mesh)
    binding = UsdSkel.BindingAPI.Apply(usd_mesh.GetPrim())
    if not binding.GetBlendShapeTargetsRel().HasAuthoredTargets():  # not already there via a reference
        targets = []
        for token, shape in zip(tokens, mesh.blend_shapes):
            usd_shape = UsdSkel.BlendShape.Define(stage, usd_mesh.GetPath().AppendChild(token))
            _set_display_name(usd_shape.GetPrim(), shape.name)
            usd_shape.CreateOffsetsAttr(_vec3f(shape.offsets))
            if shape.indices is not None:
                usd_shape.CreatePointIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(shape.indices, np.int32)))
            if shape.normal_offsets is not None:
                usd_shape.CreateNormalOffsetsAttr(_vec3f(shape.normal_offsets))
            for inbetween in shape.inbetweens:
                usd_inbetween = usd_shape.CreateInbetween(Tf.MakeValidIdentifier(f"w{inbetween.weight:g}"))
                usd_inbetween.SetWeight(float(inbetween.weight))
                usd_inbetween.SetOffsets(_vec3f(inbetween.offsets))
                if inbetween.normal_offsets is not None:
                    usd_inbetween.SetNormalOffsets(_vec3f(inbetween.normal_offsets))
            targets.append(usd_shape.GetPath())
        binding.CreateBlendShapesAttr(Vt.TokenArray(tokens))
        binding.CreateBlendShapeTargetsRel().SetTargets(targets)

    skeleton = UsdSkel.Skeleton.Define(stage, skel_path)
    animation = UsdSkel.Animation.Define(stage, skel_path.AppendChild("Anim"))
    animation.CreateBlendShapesAttr(Vt.TokenArray(tokens))
    animation.CreateBlendShapeWeightsAttr(Vt.FloatArray([float(shape.weight) for shape in mesh.blend_shapes]))
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([animation.GetPath()])
    binding.CreateSkeletonRel().SetTargets([skel_path])


def _vec3f(values) -> Vt.Vec3fArray:
    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 3))


def _write_primvar(api: UsdGeom.PrimvarsAPI, name: str, type_name, array_type, pv: Primvar, original: str | None = None):
    primvar = api.CreatePrimvar(name, type_name, pv.interpolation)
    primvar.Set(array_type.FromNumpy(np.ascontiguousarray(pv.values, dtype=np.float32)))
    if pv.indices is not None:
        primvar.SetIndices(Vt.IntArray.FromNumpy(np.ascontiguousarray(pv.indices, dtype=np.int32)))
    if original is not None:
        primvar.GetAttr().SetCustomDataByKey(NAME_KEY, original)


# -- reading --------------------------------------------------------------------


def _display_name(prim: Usd.Prim) -> str:
    return prim.GetDisplayName() or prim.GetName()


def _xformable_children(prim: Usd.Prim) -> list[Usd.Prim]:
    return [c for c in prim.GetFilteredChildren(_PREDICATE) if c.IsA(UsdGeom.Xformable)]


def _read_prim(prim: Usd.Prim, siblings: list[Node], meshes: dict[Sdf.Path, Mesh]) -> None:
    if prim.IsA(UsdSkel.Skeleton):  # joints and skinning are not supported yet
        return
    children = list(prim.GetFilteredChildren(_PREDICATE))
    if not prim.IsA(UsdGeom.Xformable):  # Scope, Material, ...: no transform, keep looking below
        for child in children:
            _read_prim(child, siblings, meshes)
        return

    node = Node(_display_name(prim), _read_transform(UsdGeom.Xformable(prim)))
    siblings.append(node)
    if prim.IsA(UsdGeom.Mesh):
        node.mesh = _get_mesh(prim, meshes)
    else:
        # A leaf Mesh child is the node's own geometry, with its transform as the geometric transform.
        for child in children:
            if child.IsA(UsdGeom.Mesh) and not _xformable_children(child):
                node.mesh = _get_mesh(child, meshes)
                _set_geometric(node.transform, _read_transform(UsdGeom.Xformable(child)))
                children.remove(child)
                break
    for child in children:
        _read_prim(child, node.children, meshes)


def _read_transform(xf: UsdGeom.Xformable) -> Transform:
    ops = xf.GetOrderedXformOps()
    values: dict = {}
    positions = []
    for op in ops:
        name = op.GetOpName()
        order = name[len("xformOp:rotate") :] if name.startswith("xformOp:rotate") and ":" not in name[8:] else None
        if order in ROTATION_ORDERS:
            name = "rotate"
            values["rotation_order"] = ROTATION_ORDERS.index(order)
        if name not in _OP_INDEX:
            positions = None
            break
        positions.append(_OP_INDEX[name])
        field = _OP_STACK[_OP_INDEX[name]][1]
        if field is not None:
            values[field] = np.asarray(op.GetAttr().Get(_TIME), dtype=np.float64)
    names = {op.GetOpName() for op in ops}
    pivots_paired = all(
        (f"xformOp:translate:{p}" in names) == (f"!invert!xformOp:translate:{p}" in names)
        for p in ("rotatePivot", "scalePivot")
    )
    if positions is not None and positions == sorted(set(positions)) and pivots_paired:
        return Transform(**values)
    matrix = np.array(xf.GetLocalTransformation(_TIME)).T  # USD matrices are row-vector
    translation, rotation, scale = decompose_matrix(matrix)
    return Transform(translation=translation, rotation=rotation, scale=scale)


def _set_geometric(target: Transform, local: Transform) -> None:
    simple = Transform(translation=local.translation, rotation=local.rotation, scale=local.scale)
    if local.rotation_order != 0 or not np.allclose(simple.matrix(), local.matrix()):
        simple = Transform(*decompose_matrix(local.matrix()))
    target.geometric_translation = simple.translation
    target.geometric_rotation = simple.rotation
    target.geometric_scale = simple.scale


def _get_mesh(prim: Usd.Prim, meshes: dict[Sdf.Path, Mesh]) -> Mesh:
    """Mesh for `prim`, shared with the mesh it internally references if that was read already."""
    refs = prim.GetMetadata("references")
    if refs is not None:
        for ref in refs.GetAddedOrExplicitItems():
            if not ref.assetPath and ref.primPath in meshes:
                return meshes[ref.primPath]
    mesh = meshes[prim.GetPath()] = _read_mesh(prim)
    return mesh


def _read_mesh(prim: Usd.Prim) -> Mesh:
    usd_mesh = UsdGeom.Mesh(prim)
    vertices = _array(usd_mesh.GetPointsAttr().Get(_TIME), np.float64).reshape(-1, 3)
    counts = _array(usd_mesh.GetFaceVertexCountsAttr().Get(_TIME), np.int32)
    indices = _array(usd_mesh.GetFaceVertexIndicesAttr().Get(_TIME), np.int32)

    primvars = UsdGeom.PrimvarsAPI(usd_mesh)
    normals = None
    if (pv := primvars.GetPrimvar("normals")) and pv.HasValue():
        normals = _read_primvar(pv, 3)
    elif usd_mesh.GetNormalsAttr().HasValue():
        values = _array(usd_mesh.GetNormalsAttr().Get(_TIME), np.float64).reshape(-1, 3)
        normals = Primvar(values, _interpolation(usd_mesh.GetNormalsInterpolation()))

    uvs = {}
    uv_type = Sdf.ValueTypeNames.Float2Array.type
    candidates = [pv for pv in primvars.GetPrimvarsWithValues() if pv.GetTypeName().type == uv_type]
    candidates.sort(key=lambda pv: (pv.GetPrimvarName() != "st", pv.GetPrimvarName()))
    for pv in candidates:
        name = pv.GetAttr().GetCustomDataByKey(NAME_KEY) or str(pv.GetPrimvarName())
        uvs[name] = _read_primvar(pv, 2)

    mesh = Mesh(_display_name(prim), vertices, counts, indices, normals=normals, uvs=uvs)
    mesh.blend_shapes = _read_blend_shapes(prim, len(vertices))
    if usd_mesh.GetOrientationAttr().Get(_TIME) == UsdGeom.Tokens.leftHanded:
        _flip_winding(mesh)
    return mesh


def _read_blend_shapes(prim: Usd.Prim, n_vertices: int) -> list[BlendShape]:
    binding = UsdSkel.BindingAPI(prim)
    targets = binding.GetBlendShapeTargetsRel().GetTargets() if binding.GetBlendShapeTargetsRel() else []
    if not targets:
        return []
    tokens = list(binding.GetBlendShapesAttr().Get() or [])
    weights = {}
    skeleton = binding.GetInheritedSkeleton()
    if skeleton:
        animation = UsdSkel.Animation(UsdSkel.BindingAPI(skeleton.GetPrim()).GetInheritedAnimationSource())
        if animation:
            names = animation.GetBlendShapesAttr().Get() or []
            values = animation.GetBlendShapeWeightsAttr().Get(_TIME) or []
            weights = dict(zip(names, values))

    shapes = []
    for token, target in zip(tokens, targets):
        usd_shape = UsdSkel.BlendShape(prim.GetStage().GetPrimAtPath(target))
        if not usd_shape:
            continue
        indices_attr = usd_shape.GetPointIndicesAttr()
        indices = _array(indices_attr.Get(_TIME), np.int32) if indices_attr.HasAuthoredValue() else None
        inbetweens = [
            Inbetween(i.GetWeight(), _array(i.GetOffsets(), np.float64), _optional_vec3(i.GetNormalOffsets()))
            for i in usd_shape.GetInbetweens()
        ]
        shapes.append(BlendShape(
            usd_shape.GetPrim().GetDisplayName() or str(token),
            _array(usd_shape.GetOffsetsAttr().Get(_TIME), np.float64),
            indices=indices,
            normal_offsets=_optional_vec3(usd_shape.GetNormalOffsetsAttr().Get(_TIME)),
            weight=float(weights.get(token, 0.0)),
            inbetweens=inbetweens,
        ))  # fmt: skip
    return shapes


def _optional_vec3(values) -> NDArray | None:
    return None if values is None or len(values) == 0 else np.array(values, dtype=np.float64)


def _read_primvar(pv: UsdGeom.Primvar, components: int) -> Primvar:
    values = _array(pv.Get(_TIME), np.float64).reshape(-1, components)
    indices = _array(pv.GetIndices(_TIME), np.int32) if pv.IsIndexed() else None
    return Primvar(values, _interpolation(pv.GetInterpolation()), indices)


def _interpolation(token: str) -> str:
    return "vertex" if token == UsdGeom.Tokens.varying else str(token)


def _array(value, dtype) -> NDArray:
    return np.zeros(0, dtype) if value is None else np.array(value, dtype=dtype)


def _flip_winding(mesh: Mesh) -> None:
    """Reverse each face's corner order (leftHanded -> rightHanded), with faceVarying data."""
    counts = mesh.face_vertex_counts.astype(np.int64)
    starts = np.repeat(mesh.face_offsets(), counts)
    local = np.arange(mesh.n_corners) - starts
    perm = starts + np.repeat(counts, counts) - 1 - local
    mesh.face_vertex_indices = mesh.face_vertex_indices[perm]
    for pv in [mesh.normals, *mesh.uvs.values()]:
        if pv is None or pv.interpolation != "faceVarying":
            continue
        if pv.indices is not None:
            pv.indices = pv.indices[perm]
        else:
            pv.values = pv.values[perm]
