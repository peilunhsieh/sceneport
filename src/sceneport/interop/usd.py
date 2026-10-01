"""Read and write USD (.usd, .usda, .usdc, .usdz) through OpenUSD (usd-core).

Writing: one Xform prim per Node and a Mesh prim (child of the Xform) per mesh,
with subdivisionScheme "none". Transforms become an xformOp stack that expresses
Transform exactly (pivots and pre/post rotation included); the geometric
transform goes on the Mesh prim. Shared meshes are written once and internally
referenced. Invalid prim names are made valid, with the original kept as the
prim's displayName. The first UV set is written as "st" (the USD convention);
every UV primvar keeps its original name in customData.

Blend shapes: each shape is a BlendShape prim under the Mesh (in-betweens
included), bound with SkelBindingAPI. The default weights live in a
SkelAnimation: on the mesh's skeleton if it is skinned, else on a jointless
Skeleton under a SkelRoot made from the mesh's node.

Skeletons: USD joints are not prims. Each tree of joint Nodes becomes a Skeleton
prim (named after its root joint, where that joint would be), with joints listed
parents first, original names in jointNames, local rest and world bind matrices.
Non-joint children of joints (attachments) are written under the Skeleton prim:
their first xformOp places them at their joint's rest pose and customData names
the joint, so they are re-attached on load. A skinned mesh binds with skel:joints
(its own joint order), jointIndices/jointWeights and geomBindTransform. Mesh and
Skeleton need a common SkelRoot: their nearest shared ancestor, or a generated
wrapper (tagged in customData and dropped again on load).

Reading: composed stages (references, payloads, instances) are read through
usd-core. Xformable prims become Nodes; other prims (Scope, Material, ...) are
flattened away. Transform stacks that match the one written here are read
exactly; others (and joint rest matrices) are decomposed from matrices.

Materials: every UsdShade Material <-> a generic Material graph (shaders, node
graphs, port values and connections, whatever the node library). Materials are
written under a "_materials" Scope in the first root prim; names that aren't
valid identifiers (prims and ports) are made valid, with the original in
displayName / customData. A graph whose surface is an FBX surface keeps it
under outputs:fbx:surface and gets a generated UsdPreviewSurface network as
outputs:surface (likewise an MTL surface, under outputs:mtl:surface). Texture files resolved on disk are referenced relative to the
written file; textures embedded in an FBX are written to a "textures" folder.
Reading resolves a mesh's bound material like USD does (inherited, collection
and purpose-less bindings); "materialBind" GeomSubsets become the node's
material slots and the mesh's per-face material_indices. Writing binds slot 0
to the mesh, or, with material_indices, one GeomSubset per slot.

USD only records the up axis (Y or Z); front and right axes are implied
(+Z/+X for Y up, -Y/+X for Z up). Points and weights are stored as float32.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, UsdSkel, UsdUtils, Vt

from ..math import (
    ROTATION_ORDERS,
    decompose_matrix,
    euler_from_matrix,
    quaternion_continuity,
    quaternion_from_matrix,
    quaternion_matrix,
)
from ..scene.animation import Animation, Channel, PointSamples, rest_value
from ..scene.material import AssetPath, Connection, Material, NodeGraph, Port, ShaderNode
from ..scene.pbr import NATIVE_SURFACES, native_surface, preview_surface_nodes
from .export import texture_path
from .fbx.materials import inferred_fbx_type
from ..scene.types import BlendShape, Inbetween, Mesh, Node, Primvar, Scene, Skin, Transform

EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")
NAME_KEY = "sceneport:name"  # customData: a primvar's original name
JOINT_KEY = "sceneport:joint"  # customData: joint token an attachment hangs from
GENERATED_KEY = "sceneport:generated"  # customData: SkelRoot added only to satisfy UsdSkel
JOINT_OP = "sceneportJoint"  # xformOp suffix placing an attachment at its joint
ACTIVE_CLIP_KEY = "sceneport:activeClip"  # root layer customLayerData: name of the clip on the timeline
FBX_TYPE_KEY = "sceneport:fbxType"  # customData of a shader port: its FBX [type, subtype, flags], if not the inferred one
IMPLIED_AXES = {"+Y": ("+Z", "+X"), "+Z": ("-Y", "+X")}  # up -> (front, right)

_TIME = Usd.TimeCode.EarliestTime()
_DEFAULT = Usd.TimeCode.Default()
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
    if path.suffix.lower() == ".usdz":
        with tempfile.TemporaryDirectory() as tmp:
            # Textures written next to the layer get packaged; missing ones would make packaging fail.
            stage = stage_from_scene(scene, directory=Path(tmp), missing_textures="drop")
            layer_path = Path(tmp) / f"{path.stem}.usdc"
            _export(stage, layer_path)
            if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer_path)), str(path)):
                raise OSError(f"could not write {path}")
    else:
        _export(stage_from_scene(scene, directory=path.parent), path)


def stage_from_scene(scene: Scene, *, directory: Path | None = None, missing_textures: str = "keep") -> Usd.Stage:
    """An in-memory stage holding `scene`.

    `directory` is where the stage will be saved: texture paths are made relative
    to it and embedded (FBX) textures are written to its "textures" folder.
    Without it, texture paths are written as authored and embedded ones are dropped.
    `missing_textures="drop"` leaves out (with a warning) texture paths that don't resolve to a file.
    """
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
    _Writer(stage, scene, directory, missing_textures == "drop").write()
    return stage


def scene_from_stage(stage: Usd.Stage) -> Scene:
    up = "+" + UsdGeom.GetStageUpAxis(stage)
    front, right = IMPLIED_AXES[up]
    scene = Scene(meters_per_unit=_meters_per_unit(stage), up_axis=up, front_axis=front, right_axis=right)
    _Reader().read(stage, scene)
    return scene


# -- writing --------------------------------------------------------------------


def _meters_per_unit(stage: Usd.Stage) -> float:
    """The stage's metersPerUnit, also when an old writer stored it as a float (e.g. Blender 2.80)."""
    try:
        return UsdGeom.GetStageMetersPerUnit(stage)
    except Tf.ErrorException:  # "Requested type double ... does not match retrieved type float"
        return float(stage.GetRootLayer().pseudoRoot.GetInfo("metersPerUnit"))


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


@dataclass(eq=False)
class _SkeletonPlan:
    """One Skeleton prim: a set of joint trees sharing a parent node."""

    roots: list[Node]
    parent: Node | None
    joints: list[Node] = field(default_factory=list)  # parents first
    tokens: dict[Node, str] = field(default_factory=dict)  # joint -> USD joint path
    skel_space: dict[Node, NDArray] = field(default_factory=dict)  # joint -> rest matrix relative to the Skeleton
    path: Sdf.Path | None = None
    shape_weights: dict[str, float] = field(default_factory=dict)  # blend shape token -> default weight
    shape_targets: dict[tuple[int, str], str] = field(default_factory=dict)  # (id(Mesh), shape name) -> token
    child_names: set[str] = field(default_factory=set)  # prim names used under the Skeleton

    def build(self) -> None:
        self.joints, self.tokens, self.skel_space = [], {}, {}
        root_names: set[str] = set()
        stack = [(root, None, np.eye(4), root_names) for root in reversed(self.roots)]
        while stack:
            joint, parent, parent_matrix, taken = stack.pop()
            name = _prim_name(joint.name, taken)
            self.tokens[joint] = name if parent is None else f"{self.tokens[parent]}/{name}"
            self.skel_space[joint] = parent_matrix @ joint.transform.matrix()
            self.joints.append(joint)
            child_names: set[str] = set()
            for child in reversed([c for c in joint.children if c.joint]):
                stack.append((child, joint, self.skel_space[joint], child_names))


class _Writer:
    def __init__(self, stage: Usd.Stage, scene: Scene, directory: Path | None = None, drop_missing: bool = False):
        self.stage = stage
        self.scene = scene
        self.directory = directory
        self.drop_missing = drop_missing
        self.material_users: list[tuple[Sdf.Path, Node, Mesh]] = []  # every mesh prim, with its node
        self.subset_names: dict[int, list[str]] = {}  # id(Mesh) -> its GeomSubset names, shared by every user
        self.parents = scene.parents()
        self.world = scene.world_matrices()
        self.written: dict[int, Sdf.Path] = {}  # id(Mesh) -> path of its Mesh prim
        self.plans: dict[Node, _SkeletonPlan] = {}  # root joint -> plan
        self.skel_roots: set[Node] = set()  # nodes written as SkelRoot
        self.wrapped: list[Node] = []  # root nodes placed under a generated SkelRoot
        self.bindings: list[tuple[UsdGeom.Mesh, Mesh, _SkeletonPlan]] = []
        self.own_skeletons: list[_SkeletonPlan] = []  # jointless skeletons of unskinned blend shape meshes
        for clip in scene.animations:
            clip.validate()
        # Only the first clip is on the timeline (USD has one per stage); it also drives node transforms.
        self.active = scene.animations[0] if scene.animations else None
        self.fps = self.active.fps if self.active else 24.0
        self.active_times = self.active.frame_times() if self.active else np.zeros(0)
        self.active_values = self.active.node_values(self.active_times) if self.active else {}
        # Points are time-sampled on the Mesh prim, so like node animation only the timeline's clip can have them.
        self.active_points = {id(p.mesh): p for p in self.active.point_samples} if self.active else {}
        for clip in scene.animations[1:]:
            if clip.point_samples:
                warnings.warn(f"clip {clip.name!r}: USD holds per-vertex animation for the first clip only; not written")
        self._plan()

    # planning

    def _root_joint(self, joint: Node) -> Node:
        while (parent := self.parents.get(joint)) is not None and parent.joint:
            joint = parent
        return joint

    def _ancestors(self, node: Node | None) -> list[Node]:
        chain = []
        while node is not None:
            chain.append(node)
            node = self.parents[node]
        return chain[::-1]

    def _plan(self) -> None:
        for node in self.parents:
            if node.joint and not (self.parents[node] is not None and self.parents[node].joint):
                self.plans[node] = _SkeletonPlan([node], self.parents[node])
        # A skin spanning several joint trees puts them in one Skeleton.
        for node in self.parents:
            if node.mesh is None or node.mesh.skin is None:
                continue
            roots = list(dict.fromkeys(self._root_joint(j) for j in node.mesh.skin.joints))
            if any(not r.joint or r not in self.parents for r in roots):
                raise ValueError(f"mesh {node.mesh.name!r}: skin joints must be joint nodes in the scene")
            plans = list(dict.fromkeys(self.plans[r] for r in roots))
            if len({p.parent for p in plans}) > 1:
                raise ValueError(f"mesh {node.mesh.name!r}: skin joints must share one parent node")
            if len(plans) > 1:
                merged = _SkeletonPlan([r for p in plans for r in p.roots], plans[0].parent)
                for r in merged.roots:
                    self.plans[r] = merged
        for plan in set(self.plans.values()):
            plan.build()
        # Skinned meshes and their skeletons need a common SkelRoot.
        for node in self.parents:
            if node.mesh is None or node.mesh.skin is None:
                continue
            plan = self.plans[self._root_joint(node.mesh.skin.joints[0])]
            mesh_chain, skel_chain = self._ancestors(node), self._ancestors(plan.parent)
            common = [a for a, b in zip(mesh_chain, skel_chain) if a is b]
            if common:
                self.skel_roots.add(common[-1])
            else:
                for root in (mesh_chain[0], skel_chain[0] if skel_chain else plan.roots[0]):
                    if root not in self.wrapped:
                        self.wrapped.append(root)

    # writing

    def write(self) -> None:
        if self.active is not None:
            start, end = self.active.time_range()
            self.stage.SetTimeCodesPerSecond(self.fps)
            self.stage.SetFramesPerSecond(self.fps)
            self.stage.SetStartTimeCode(_time_code(start, self.fps))
            self.stage.SetEndTimeCode(_time_code(end, self.fps))
            self.stage.GetRootLayer().customLayerData = {ACTIVE_CLIP_KEY: self.active.name}
        taken: set[str] = set()
        root_paths = []
        wrapper: tuple[Sdf.Path, set[str]] | None = None
        for node in self.scene.nodes:
            if node in self.wrapped:
                if wrapper is None:
                    path = Sdf.Path.absoluteRootPath.AppendChild(_prim_name("SkelRoot", taken))
                    UsdSkel.Root.Define(self.stage, path).GetPrim().SetCustomDataByKey(GENERATED_KEY, True)
                    wrapper = (path, set())
                    root_paths.append(path)
                self._write_node(wrapper[0], node, wrapper[1])
            elif (path := self._write_node(Sdf.Path.absoluteRootPath, node, taken)) is not None:
                root_paths.append(path)
        if root_paths:
            self.stage.SetDefaultPrim(self.stage.GetPrimAtPath(root_paths[0]))
        self._write_materials(root_paths[0] if root_paths else Sdf.Path.absoluteRootPath)
        self._finish()

    def _write_node(
        self, parent: Sdf.Path, node: Node, taken: set[str], joint: tuple[str, NDArray] | None = None
    ) -> Sdf.Path | None:
        if node.joint:
            plan = self.plans.get(node)
            if plan is not None and plan.path is None:
                self._write_skeleton(parent, plan, taken)
            return plan.path if plan is not None else None

        path = parent.AppendChild(_prim_name(node.name, taken))
        mesh = node.mesh
        own_skeleton = mesh is not None and bool(mesh.blend_shapes) and mesh.skin is None
        prim_type = UsdSkel.Root if own_skeleton or node in self.skel_roots else UsdGeom.Xform
        xform = prim_type.Define(self.stage, path)
        _set_display_name(xform.GetPrim(), node.name)
        if joint is not None:  # attachment of a joint: start at the joint's rest pose
            token, matrix = joint
            xform.AddTransformOp(precision=_DOUBLE, opSuffix=JOINT_OP).Set(Gf.Matrix4d(matrix.T.tolist()))
            xform.GetPrim().SetCustomDataByKey(JOINT_KEY, token)
        animated = self.active_values.get(node, {})
        ops = _write_transform(xform, node.transform, force={"rotation" if p == "orientation" else p for p in animated})
        if animated:
            self._write_samples(ops, node, animated)

        child_names: set[str] = set()
        if mesh is not None:
            usd_mesh = self._write_mesh_prim(path.AppendChild(_prim_name(mesh.name, child_names)), mesh)
            _write_geometric(usd_mesh, node.transform)
            self.material_users.append((usd_mesh.GetPath(), node, mesh))
            tokens = _write_blend_shape_targets(self.stage, usd_mesh, mesh) if mesh.blend_shapes else []
            if mesh.skin is not None:
                plan = self.plans[self._root_joint(mesh.skin.joints[0])]
                plan.shape_weights.update(zip(tokens, (float(s.weight) for s in mesh.blend_shapes)))
                plan.shape_targets.update(((id(mesh), s.name), t) for s, t in zip(mesh.blend_shapes, tokens))
                self.bindings.append((usd_mesh, mesh, plan))
            elif own_skeleton:
                skel_path = path.AppendChild(_prim_name("Skel", child_names))
                UsdSkel.Skeleton.Define(self.stage, skel_path)
                own = _SkeletonPlan([], node, path=skel_path)
                own.shape_weights = dict(zip(tokens, (float(s.weight) for s in mesh.blend_shapes)))
                own.shape_targets = {(id(mesh), s.name): t for s, t in zip(mesh.blend_shapes, tokens)}
                self.own_skeletons.append(own)
                UsdSkel.BindingAPI.Apply(usd_mesh.GetPrim()).CreateSkeletonRel().SetTargets([skel_path])

        for child in node.children:
            self._write_node(path, child, child_names)
        return path

    def _write_materials(self, parent: Sdf.Path) -> None:
        materials = {id(m): m for m in self.scene.materials}
        for _, node, _ in self.material_users:
            materials.update((id(m), m) for m in node.materials if m is not None)
        if not materials:
            return
        taken = {child.GetName() for child in self.stage.GetPrimAtPath(parent).GetChildren()} if parent != Sdf.Path.absoluteRootPath else set()
        scope = parent.AppendChild(_prim_name("_materials", taken))
        UsdGeom.Scope.Define(self.stage, scope)
        # The first UV set is written as "st": textures reading it by its original name must say "st".
        first_uvs: dict[int, set[str]] = {}
        for _, node, mesh in self.material_users:
            if mesh.uvs:
                for m in node.materials:
                    first_uvs.setdefault(id(m), set()).add(next(iter(mesh.uvs)))
        names: set[str] = set()
        written = {
            key: _write_material(self.stage, scope.AppendChild(_prim_name(m.name, names)), m, first_uvs.get(key, set()),
                                 self.directory, self.drop_missing)
            for key, m in materials.items()
        }  # fmt: skip
        for mesh_path, node, mesh in self.material_users:
            prim = self.stage.GetPrimAtPath(mesh_path)
            # A shared mesh's later prims reference the first one, bindings included: block what doesn't apply.
            referenced = mesh_path != self.written[id(mesh)]
            if not node.materials:
                if referenced:
                    self._unbind(prim, mesh)
                continue
            api = UsdShade.MaterialBindingAPI.Apply(prim)
            indices = mesh.material_indices
            if indices is None and len(node.materials) == 1:
                if node.materials[0] is not None:
                    api.Bind(written[id(node.materials[0])])
                elif referenced:
                    self._unbind(prim, mesh)
                continue
            if indices is None:
                indices = np.zeros(mesh.n_faces, dtype=np.int32)
            if id(mesh) not in self.subset_names:
                subset_names: set[str] = set()
                self.subset_names[id(mesh)] = [_prim_name(m.name if m is not None else f"slot{i}", subset_names)
                                               for i, m in enumerate(node.materials)]  # fmt: skip
            subset_names = self.subset_names[id(mesh)]
            for i, material in enumerate(node.materials):
                name = subset_names[i] if i < len(subset_names) else f"slot{i}"
                faces = Vt.IntArray.FromNumpy(np.flatnonzero(indices == i).astype(np.int32))
                subset = api.CreateMaterialBindSubset(name, faces, UsdGeom.Tokens.face)
                subset_api = UsdShade.MaterialBindingAPI.Apply(subset.GetPrim())
                if material is not None:
                    subset_api.Bind(written[id(material)])
                elif referenced:
                    subset_api.UnbindDirectBinding()
            api.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)

    def _unbind(self, prim: Usd.Prim, mesh: Mesh) -> None:
        """Block the bindings a referenced mesh prim brings along (on the prim and its subsets)."""
        UsdShade.MaterialBindingAPI.Apply(prim).UnbindDirectBinding()
        for name in self.subset_names.get(id(mesh), []):
            subset = prim.GetChild(name)
            if subset:
                UsdShade.MaterialBindingAPI.Apply(subset).UnbindDirectBinding()

    def _write_mesh_prim(self, path: Sdf.Path, mesh: Mesh) -> UsdGeom.Mesh:
        if id(mesh) in self.written:
            prim = self.stage.DefinePrim(path)
            prim.GetReferences().AddInternalReference(self.written[id(mesh)])
            usd_mesh = UsdGeom.Mesh(prim)
            usd_mesh.ClearXformOpOrder()
        else:
            usd_mesh = _write_mesh(self.stage, path, mesh)
            self.written[id(mesh)] = path
            if id(mesh) in self.active_points:  # the rest points stay the default value
                samples, attr = self.active_points[id(mesh)], usd_mesh.GetPointsAttr()
                for t, points in zip(samples.times, samples.points):
                    attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32)), _time_code(t, self.fps))
        _set_display_name(usd_mesh.GetPrim(), mesh.name)
        return usd_mesh

    def _write_skeleton(self, parent: Sdf.Path, plan: _SkeletonPlan, taken: set[str]) -> None:
        plan.path = parent.AppendChild(_prim_name(plan.roots[0].name, taken))
        skeleton = UsdSkel.Skeleton.Define(self.stage, plan.path)
        bind = {joint: self.world[joint] for joint in plan.joints}
        for node in self.parents:  # prefer the skins' bind matrices over the current pose
            skin = node.mesh.skin if node.mesh is not None else None
            if skin is not None:
                bind.update((j, m) for j, m in zip(skin.joints, skin.bind_world()) if j in bind)
        skeleton.CreateJointsAttr(Vt.TokenArray([plan.tokens[j] for j in plan.joints]))
        skeleton.CreateJointNamesAttr(Vt.TokenArray([j.name for j in plan.joints]))
        skeleton.CreateRestTransformsAttr(_matrix_array([j.transform.matrix() for j in plan.joints]))
        skeleton.CreateBindTransformsAttr(_matrix_array([bind[j] for j in plan.joints]))

        child_names = plan.child_names
        for joint in plan.joints:
            if joint.mesh is not None:
                warnings.warn(f"joint {joint.name!r} has a mesh; USD joints can't carry geometry, so it is dropped")
            for child in joint.children:
                if not child.joint:
                    self._write_node(plan.path, child, child_names, (plan.tokens[joint], plan.skel_space[joint]))

    def _finish(self) -> None:
        for usd_mesh, mesh, plan in self.bindings:
            skin = mesh.skin
            binding = UsdSkel.BindingAPI.Apply(usd_mesh.GetPrim())
            binding.CreateSkeletonRel().SetTargets([plan.path])
            if binding.GetJointIndicesAttr().HasAuthoredValue():  # a referenced copy: already bound
                continue
            k = skin.joint_indices.shape[1]
            binding.CreateJointsAttr(Vt.TokenArray([plan.tokens[j] for j in skin.joints]))
            binding.CreateJointIndicesPrimvar(False, k).Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(skin.joint_indices.ravel())))
            weights = np.ascontiguousarray(skin.joint_weights.ravel(), dtype=np.float32)
            binding.CreateJointWeightsPrimvar(False, k).Set(Vt.FloatArray.FromNumpy(weights))
            binding.CreateGeomBindTransformAttr(Gf.Matrix4d(skin.geom_bind.T.tolist()))
        for plan in [*set(self.plans.values()), *self.own_skeletons]:
            self._write_skeleton_animations(plan)

    def _write_samples(self, ops: dict, node: Node, values: dict[str, NDArray]) -> None:
        """Time samples on the node's xformOps for the active clip (rest values stay as defaults)."""
        codes = [_time_code(t, self.fps) for t in self.active_times]
        for prop, frames in values.items():
            if prop == "orientation":  # xformOps rotate with Euler angles
                euler = euler_from_matrix(quaternion_matrix(frames), node.transform.rotation_order)
                prop, frames = "rotation", np.degrees(np.unwrap(np.radians(euler), axis=0))
            attr = ops[prop].GetAttr()
            for code, value in zip(codes, frames):
                attr.Set(Gf.Vec3d(*(float(v) for v in value)), float(code))

    def _write_skeleton_animations(self, plan: _SkeletonPlan) -> None:
        """One SkelAnimation per clip touching the plan's joints or blend shapes; the active clip is bound."""
        joints = set(plan.joints)
        clips = [
            clip for clip in self.scene.animations
            if any(c.target in joints if not c.is_blend_shape else (id(c.target), c.property) in plan.shape_targets
                   for c in clip.channels)
        ]  # fmt: skip
        source = None
        for clip in clips:
            path = plan.path.AppendChild(_prim_name(clip.name, plan.child_names))
            self._write_clip(path, plan, clip)
            if clip is self.active:
                source = path
        if source is None and plan.shape_weights:  # no animation on the timeline: hold the default weights
            source = plan.path.AppendChild(_prim_name("Anim", plan.child_names))
            _write_animation(self.stage, source, plan.shape_weights)
        if source is not None:
            skeleton = self.stage.GetPrimAtPath(plan.path)
            UsdSkel.BindingAPI.Apply(skeleton).CreateAnimationSourceRel().SetTargets([source])

    def _write_clip(self, path: Sdf.Path, plan: _SkeletonPlan, clip: Animation) -> None:
        animation = UsdSkel.Animation.Define(self.stage, path)
        _set_display_name(animation.GetPrim(), clip.name)
        times = clip.frame_times()
        codes = [_time_code(t, self.fps) for t in times]
        if plan.joints:
            values = clip.node_values(times)
            local = np.stack(
                [np.broadcast_to(j.transform.matrix(**values.get(j, {})), (len(times), 4, 4)) for j in plan.joints], axis=1
            )  # (F, J, 4, 4)
            translations = local[..., :3, 3]
            scales = np.linalg.norm(local[..., :3, :3], axis=-2)
            # A mirrored joint (negative determinant) needs a negative scale, or the rotation isn't one:
            # negating all three keeps it proper and matches the usual (-1, -1, -1) of mirrored rigs.
            scales *= np.where(np.linalg.det(local[..., :3, :3]) < 0, -1.0, 1.0)[..., None]
            rotations = quaternion_from_matrix(local[..., :3, :3] / np.where(scales == 0, 1, scales)[..., None, :])
            for f in range(1, len(times)):  # keep each joint's quaternion signs continuous over time
                rotations[f] *= np.where(np.sum(rotations[f] * rotations[f - 1], axis=-1, keepdims=True) < 0, -1, 1)
            animation.CreateJointsAttr(Vt.TokenArray([plan.tokens[j] for j in plan.joints]))
            t_attr, r_attr, s_attr = (animation.CreateTranslationsAttr(), animation.CreateRotationsAttr(), animation.CreateScalesAttr())
            for f, code in enumerate(codes):
                t_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(translations[f], dtype=np.float32)), float(code))
                r_attr.Set(Vt.QuatfArray.FromNumpy(np.ascontiguousarray(rotations[f][:, [1, 2, 3, 0]], dtype=np.float32)), float(code))
                s_attr.Set(Vt.Vec3hArray.FromNumpy(np.ascontiguousarray(scales[f], dtype=np.float16)), float(code))
        if plan.shape_targets:
            tokens = list(dict.fromkeys(plan.shape_targets.values()))
            index = {token: i for i, token in enumerate(tokens)}
            weights = np.tile([plan.shape_weights[t] for t in tokens], (len(times), 1))
            for channel in clip.channels:
                token = plan.shape_targets.get((id(channel.target), channel.property)) if channel.is_blend_shape else None
                if token is not None:
                    weights[:, index[token]] = channel.evaluate(times)
            animation.CreateBlendShapesAttr(Vt.TokenArray(tokens))
            w_attr = animation.CreateBlendShapeWeightsAttr(Vt.FloatArray([plan.shape_weights[t] for t in tokens]))
            for f, code in enumerate(codes):
                w_attr.Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(weights[f], dtype=np.float32)), float(code))


def _write_material(
    stage: Usd.Stage, path: Sdf.Path, material: Material, first_uvs: set[str], directory: Path | None,
    drop_missing: bool = False,
) -> UsdShade.Material:
    """Write `material`'s graph under `path`; an FBX surface also gets a UsdPreviewSurface network (see module doc)."""
    usd_material = UsdShade.Material.Define(stage, path)
    _set_display_name(usd_material.GetPrim(), material.name)
    nodes, outputs = dict(material.nodes), dict(material.outputs)
    native = native_surface(material, tuple(NATIVE_SURFACES))
    pbr = material.pbr() if native is not None else None
    if pbr is not None and pbr.source in NATIVE_SURFACES:  # no USD-readable surface yet
        context = NATIVE_SURFACES[pbr.source]
        outputs = {(f"{context}:surface" if key == "surface" else key): port for key, port in outputs.items()}
        prefix = "Preview"
        while any(p == prefix or p.startswith(prefix) for p in nodes):
            prefix += "_"
        preview, outputs["surface"] = preview_surface_nodes(pbr, prefix)
        for node in preview.values():  # UV set names as written on the meshes
            if node.shader_id == "UsdPrimvarReader_float2":
                name = node.inputs["varname"].value
                node.inputs["varname"].value = "st" if name in first_uvs else Tf.MakeValidIdentifier(name)
        nodes.update(preview)

    prim_paths: dict[str, Sdf.Path] = {"": path}
    taken: dict[Sdf.Path, set[str]] = {}

    def prim_path(rel: str) -> Sdf.Path:
        if rel not in prim_paths:
            parent_rel, _, leaf = rel.rpartition("/")
            parent = prim_path(parent_rel)
            prim_paths[rel] = parent.AppendChild(_prim_name(leaf, taken.setdefault(parent, set())))
            if rel not in nodes:  # a grouping prim that isn't a node
                _set_display_name(stage.DefinePrim(prim_paths[rel]), leaf)
        return prim_paths[rel]

    apis: dict[str, UsdShade.ConnectableAPI] = {"": UsdShade.ConnectableAPI(usd_material)}
    for rel in sorted(nodes, key=lambda p: p.count("/")):
        node, usd_path = nodes[rel], prim_path(rel)
        schema = UsdShade.Shader if isinstance(node, ShaderNode) else UsdShade.NodeGraph
        prim = schema.Define(stage, usd_path).GetPrim()
        _set_display_name(prim, rel.rpartition("/")[2])
        apis[rel] = UsdShade.ConnectableAPI(prim)
        if isinstance(node, ShaderNode):
            _write_shader_info(UsdShade.Shader(prim), node)

    port_names: dict[tuple[str, str, str], str] = {}  # (node, "inputs" | "outputs", name) -> USD port name
    created: list[tuple[Usd.Attribute, Port]] = []
    owners = [("", material.inputs, outputs, None)] + [(rel, n.inputs, n.outputs, n) for rel, n in nodes.items()]
    for rel, inputs, node_outputs, node in owners:
        types = node.extras.get("fbx", {}).get("types", {}) if node is not None else {}
        for kind, ports in (("inputs", inputs), ("outputs", node_outputs)):
            used: set[str] = set()
            for name, port in ports.items():
                usd_name = ":".join(Tf.MakeValidIdentifier(part) for part in name.split(":"))
                while usd_name in used:
                    usd_name += "_"
                used.add(usd_name)
                port_names[(rel, kind, name)] = usd_name
                type_name = _port_type(port)
                api = apis[rel]
                attr = (api.CreateInput if kind == "inputs" else api.CreateOutput)(usd_name, type_name).GetAttr()
                if usd_name != name:
                    attr.SetCustomDataByKey(NAME_KEY, name)
                if kind == "inputs" and name in types and tuple(types[name]) != inferred_fbx_type(port):
                    attr.SetCustomDataByKey(FBX_TYPE_KEY, Vt.StringArray([str(t) for t in types[name]]))
                if port.color_space:
                    attr.SetColorSpace(port.color_space)
                if port.value is not None:
                    try:
                        value = _usd_value(port.value, type_name, node if isinstance(node, ShaderNode) else None, directory,
                                           drop_missing)  # fmt: skip
                        ok = value is not None and attr.Set(value)
                    except (Tf.ErrorException, TypeError, ValueError):
                        ok = False
                    if not ok and isinstance(port.value, AssetPath):
                        warnings.warn(f"material {material.name!r}: texture {port.value.path!r} not found; left out of the package")
                    elif not ok:
                        warnings.warn(f"material {material.name!r}: value of {rel or '<material>'}.{name} ({type_name}) not written")
                created.append((attr, port))

    for attr, port in created:
        targets = []
        for connection in port.connections:
            owner = Sdf.Path(connection.node) if connection.node.startswith("/") else prim_paths.get(connection.node)
            if owner is None:
                warnings.warn(f"material {material.name!r}: connection to missing node {connection.node!r} dropped")
                continue
            kind, _, name = connection.port.partition(":")
            usd_name = port_names.get((connection.node, kind, name))
            usd_name = usd_name or ":".join(Tf.MakeValidIdentifier(part) for part in name.split(":"))
            targets.append(owner.AppendProperty(f"{kind}:{usd_name}"))
        if targets:
            attr.SetConnections(targets)
    return usd_material


def _write_shader_info(shader: UsdShade.Shader, node: ShaderNode) -> None:
    for source_type, asset in node.source_assets.items():
        shader.SetSourceAsset(Sdf.AssetPath(asset.path), source_type)
    for source_type, code in node.source_code.items():
        shader.SetSourceCode(code, source_type)
    if node.shader_id:
        shader.SetShaderId(node.shader_id)
    source = node.extras.get("usd", {}).get("implementationSource")
    if source:
        shader.GetImplementationSourceAttr().Set(source)


def _port_type(port: Port) -> Sdf.ValueTypeName:
    """The port's USD type, or one guessed from its value when the type name isn't a USD one."""
    found = Sdf.ValueTypeNames.Find(port.type_name) if port.type_name else None
    if found:
        return found
    value = port.value
    if isinstance(value, AssetPath):
        return Sdf.ValueTypeNames.Asset
    if isinstance(value, bool):
        return Sdf.ValueTypeNames.Bool
    if isinstance(value, int):
        return Sdf.ValueTypeNames.Int
    if isinstance(value, float):
        return Sdf.ValueTypeNames.Double
    if isinstance(value, str):
        return Sdf.ValueTypeNames.String
    if isinstance(value, np.ndarray) and value.ndim == 1 and 2 <= len(value) <= 4:
        return getattr(Sdf.ValueTypeNames, f"Double{len(value)}")
    return Sdf.ValueTypeNames.Token


def _usd_value(value, type_name: Sdf.ValueTypeName, node: ShaderNode | None, directory: Path | None, drop_missing: bool):
    if isinstance(value, AssetPath):
        path = _asset_path(value, node, directory, drop_missing)
        return None if path is None else Sdf.AssetPath(path)
    if isinstance(value, list) and value and isinstance(value[0], AssetPath):
        paths = [_asset_path(v, node, directory, drop_missing) for v in value]
        return Sdf.AssetPathArray([p for p in paths if p is not None])
    if isinstance(value, np.ndarray):
        cls = type_name.type.pythonClass  # e.g. Gf.Vec3d, Gf.Matrix4d, Vt.FloatArray
        return cls(value.tolist()) if cls is not None else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _asset_path(asset: AssetPath, node: ShaderNode | None, directory: Path | None, drop_missing: bool = False) -> str | None:
    content = node.extras.get("fbx", {}).get("content") if node is not None else None
    fallback = f"{Tf.MakeValidIdentifier(node.path)}.png" if node is not None else "texture.png"
    return texture_path(asset, directory, content=content, fallback_name=fallback, drop_missing=drop_missing)


def _time_code(seconds: float, fps: float) -> float:
    """Seconds -> time code, snapped to whole frames when floating-point noise is all that's off."""
    code = float(seconds) * fps
    return float(round(code)) if abs(code - round(code)) < 1e-9 else code


def _matrix_array(matrices) -> Vt.Matrix4dArray:
    """Column-vector 4x4 matrices -> USD's row-vector Matrix4dArray."""
    stacked = np.asarray(matrices, dtype=np.float64).reshape(-1, 4, 4)
    return Vt.Matrix4dArray.FromNumpy(np.ascontiguousarray(stacked.transpose(0, 2, 1)))


def _set_display_name(prim: Usd.Prim, name: str) -> None:
    if prim.GetName() != name:
        prim.SetDisplayName(name)


def _add_op(xf: UsdGeom.Xformable, kind: str, value=None, suffix: str = "", inverse: bool = False) -> UsdGeom.XformOp:
    op = getattr(xf, f"Add{kind}Op")(precision=_DOUBLE, opSuffix=suffix, isInverseOp=inverse)
    if value is not None:
        op.GetAttr().Set(Gf.Vec3d(*(float(v) for v in value)))
    return op


def _write_transform(xf: UsdGeom.Xformable, t: Transform, force: set[str] = frozenset()) -> dict[str, UsdGeom.XformOp]:
    """Write `t` as xformOps; returns the translate/rotate/scale ops. Properties in `force`
    ("translation", "rotation", "scale") get an op even at their identity value, for animation."""
    ops = {}

    def nonzero(v):
        return bool(np.any(v))

    if nonzero(t.translation) or "translation" in force:
        ops["translation"] = _add_op(xf, "Translate", t.translation)
    if nonzero(t.rotation_offset):
        _add_op(xf, "Translate", t.rotation_offset, "rotateOffset")
    if nonzero(t.rotation_pivot):
        _add_op(xf, "Translate", t.rotation_pivot, "rotatePivot")
    if nonzero(t.pre_rotation):
        _add_op(xf, "RotateXYZ", t.pre_rotation, "preRotation")
    if nonzero(t.rotation) or t.rotation_order != 0 or "rotation" in force:
        ops["rotation"] = _add_op(xf, f"Rotate{ROTATION_ORDERS[t.rotation_order]}", t.rotation)
    if nonzero(t.post_rotation):
        _add_op(xf, "RotateXYZ", t.post_rotation, "postRotation", inverse=True)
    if nonzero(t.rotation_pivot):
        _add_op(xf, "Translate", suffix="rotatePivot", inverse=True)
    if nonzero(t.scaling_offset):
        _add_op(xf, "Translate", t.scaling_offset, "scaleOffset")
    if nonzero(t.scaling_pivot):
        _add_op(xf, "Translate", t.scaling_pivot, "scalePivot")
    if not np.array_equal(t.scale, [1, 1, 1]) or "scale" in force:
        ops["scale"] = _add_op(xf, "Scale", t.scale)
    if nonzero(t.scaling_pivot):
        _add_op(xf, "Translate", suffix="scalePivot", inverse=True)
    return ops


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


def _write_blend_shape_targets(stage: Usd.Stage, usd_mesh: UsdGeom.Mesh, mesh: Mesh) -> list[str]:
    """BlendShape prims under the mesh, bound to it (unless already there via a reference). Returns the tokens."""
    tokens = _shape_tokens(mesh)
    binding = UsdSkel.BindingAPI.Apply(usd_mesh.GetPrim())
    if binding.GetBlendShapeTargetsRel().HasAuthoredTargets():
        return tokens
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
    return tokens


def _write_animation(stage: Usd.Stage, path: Sdf.Path, weights: dict[str, float]) -> None:
    """A SkelAnimation holding constant (default) blend shape weights."""
    animation = UsdSkel.Animation.Define(stage, path)
    animation.CreateBlendShapesAttr(Vt.TokenArray(list(weights)))
    animation.CreateBlendShapeWeightsAttr(Vt.FloatArray(list(weights.values())))


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


class _Reader:
    def __init__(self):
        self.meshes: dict[Sdf.Path, Mesh] = {}
        # Skeleton path -> (joint token -> Node, joint token -> world bind matrix)
        self.skeletons: dict[Sdf.Path, tuple[dict[str, Node], dict[str, NDArray]]] = {}
        self.pending_skins: list[tuple[Mesh, Usd.Prim]] = []
        # Skeleton path -> (SkelAnimation prims, the bound one); and blend shape token maps of meshes bound to it
        self.skeleton_animations: dict[Sdf.Path, tuple[list[Usd.Prim], Sdf.Path | None]] = {}
        self.bound_shapes: dict[Sdf.Path, list[tuple[Mesh, dict[str, str]]]] = {}
        self.node_channels: list[Channel] = []  # time-sampled xformOps: the timeline's clip
        self.point_samples: list[PointSamples] = []  # time-sampled mesh points: the timeline's clip too
        self.fps = 24.0
        self.materials: dict[Sdf.Path, Material] = {}

    def read(self, stage: Usd.Stage, scene: Scene) -> None:
        self.fps = stage.GetTimeCodesPerSecond()
        for prim in stage.Traverse(_PREDICATE):  # every material, in stage order; bound ones are shared from here
            if prim.IsA(UsdShade.Material):
                self.get_material(prim)
        for prim in stage.GetPseudoRoot().GetFilteredChildren(_PREDICATE):
            self.read_prim(prim, scene.nodes)
        scene.materials = list(self.materials.values())
        for mesh, prim in self.pending_skins:  # after all skeletons are known
            mesh.skin = self.read_skin(prim, mesh)
        scene.animations = self.read_animations(stage)

    def read_prim(self, prim: Usd.Prim, siblings: list[Node]) -> None:
        if prim.IsA(UsdSkel.Skeleton):
            self.read_skeleton(prim, siblings)
            return
        children = list(prim.GetFilteredChildren(_PREDICATE))
        generated = prim.IsA(UsdSkel.Root) and prim.GetCustomDataByKey(GENERATED_KEY)
        if generated or not prim.IsA(UsdGeom.Xformable):  # Scope, Material, ...: no transform, keep looking below
            for child in children:
                self.read_prim(child, siblings)
            return

        node = Node(_display_name(prim), _read_transform(UsdGeom.Xformable(prim)))
        siblings.append(node)
        self.read_node_samples(UsdGeom.Xformable(prim), node)
        if prim.IsA(UsdGeom.Mesh):
            node.mesh = self.get_mesh(prim)
            node.materials = self.read_material_slots(prim, node.mesh)
        else:
            # A leaf Mesh child is the node's own geometry, with its transform as the geometric transform.
            for child in children:
                if child.IsA(UsdGeom.Mesh) and not _xformable_children(child):
                    node.mesh = self.get_mesh(child)
                    node.materials = self.read_material_slots(child, node.mesh)
                    _set_geometric(node.transform, _read_transform(UsdGeom.Xformable(child)))
                    children.remove(child)
                    break
        for child in children:
            self.read_prim(child, node.children)

    def read_skeleton(self, prim: Usd.Prim, siblings: list[Node]) -> None:
        skeleton = UsdSkel.Skeleton(prim)
        tokens = [str(t) for t in skeleton.GetJointsAttr().Get() or []]
        names_attr = skeleton.GetJointNamesAttr()
        names = [str(n) for n in names_attr.Get()] if names_attr.HasAuthoredValue() else []
        if len(names) != len(tokens):
            names = [Sdf.Path(t).name for t in tokens]
        rest = _matrices(skeleton.GetRestTransformsAttr().Get())
        bind = _matrices(skeleton.GetBindTransformsAttr().Get())

        own = _read_transform(UsdGeom.Xformable(prim))
        target = siblings
        if tokens and not np.allclose(own.matrix(), np.eye(4)):  # keep a transformed Skeleton as a node
            holder = Node(_display_name(prim), own)
            siblings.append(holder)
            target = holder.children

        by_token: dict[str, Node] = {}
        bind_by_token: dict[str, NDArray] = {}
        for i, (token, name) in enumerate(zip(tokens, names)):
            parent = by_token.get(str(Sdf.Path(token).GetParentPath()))
            if rest is not None and len(rest) == len(tokens):
                local = rest[i]
            elif bind is not None:
                local = bind[i] if parent is None else np.linalg.inv(bind_by_token[_parent_token(token)]) @ bind[i]
            else:
                local = np.eye(4)
            joint = Node(name, Transform(*decompose_matrix(local)), joint=True)
            (parent.children if parent is not None else target).append(joint)
            by_token[token] = joint
            if bind is not None and len(bind) == len(tokens):
                bind_by_token[token] = bind[i]
        self.skeletons[prim.GetPath()] = (by_token, bind_by_token)
        animations = [c for c in prim.GetChildren() if c.IsA(UsdSkel.Animation)]
        # The bound source may be authored on the Skeleton or inherited from an ancestor (e.g. the SkelRoot).
        bound = UsdSkel.BindingAPI(prim).GetInheritedAnimationSource()
        source_path = bound.GetPath() if bound else None
        if bound and bound.IsA(UsdSkel.Animation) and all(a.GetPath() != source_path for a in animations):
            animations.append(bound)
        self.skeleton_animations[prim.GetPath()] = (animations, source_path)

        for child in prim.GetFilteredChildren(_PREDICATE):
            token = child.GetCustomDataByKey(JOINT_KEY)
            if token in by_token:
                self.read_prim(child, by_token[token].children)
            elif child.IsA(UsdGeom.Xformable):
                self.read_prim(child, target)

    def get_mesh(self, prim: Usd.Prim) -> Mesh:
        """Mesh for `prim`, shared with the mesh it internally references if that was read already."""
        refs = prim.GetMetadata("references")
        if refs is not None:
            for ref in refs.GetAddedOrExplicitItems():
                if not ref.assetPath and ref.primPath in self.meshes:
                    return self.meshes[ref.primPath]
        mesh = self.meshes[prim.GetPath()] = _read_mesh(prim)
        if (samples := _read_point_samples(prim, mesh, self.fps)) is not None:
            self.point_samples.append(samples)
        binding = UsdSkel.BindingAPI(prim)
        if binding.GetJointIndicesAttr().HasAuthoredValue():
            self.pending_skins.append((mesh, prim))
        skeleton = binding.GetInheritedSkeleton()
        if skeleton and mesh.blend_shapes:
            self.bound_shapes.setdefault(skeleton.GetPath(), []).append((mesh, _shape_token_names(prim)))
        return mesh

    def get_material(self, prim: Usd.Prim) -> Material:
        if prim.GetPath() not in self.materials:
            self.materials[prim.GetPath()] = _read_material(prim)
        return self.materials[prim.GetPath()]

    def bound_material(self, prim: Usd.Prim) -> Material | None:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        return self.get_material(material.GetPrim()) if material else None

    def read_material_slots(self, prim: Usd.Prim, mesh: Mesh) -> list[Material | None]:
        """The node's material slots: one per materialBind GeomSubset, plus the mesh's own binding for other faces.

        Sets `mesh.material_indices` when there are subsets (a shared mesh keeps the first ones read).
        """
        whole = self.bound_material(prim)
        subsets = [s for s in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
                   if s.GetElementTypeAttr().Get() == UsdGeom.Tokens.face]  # fmt: skip
        if not subsets:
            return [whole] if whole is not None else []
        slots: list[Material | None] = []
        indices = np.full(mesh.n_faces, -1, dtype=np.int32)
        for subset in subsets:
            faces = _array(subset.GetIndicesAttr().Get(_TIME), np.int64)
            indices[faces[(faces >= 0) & (faces < mesh.n_faces)]] = len(slots)
            slots.append(self.bound_material(subset.GetPrim()))
        if (indices < 0).any():  # faces in no subset use the mesh's own binding
            indices[indices < 0] = len(slots)
            slots.append(whole)
        if all(slot is None for slot in slots):  # e.g. a mesh reference's subsets, with bindings blocked
            return []
        if mesh.material_indices is None:
            mesh.material_indices = indices
        return slots

    def read_node_samples(self, xf: UsdGeom.Xformable, node: Node) -> None:
        """Channels for time-sampled xformOps (translate / rotate / scale; else the decomposed matrix)."""
        ops = [op for op in xf.GetOrderedXformOps() if op.GetOpName() != f"xformOp:transform:{JOINT_OP}"]
        if not any(op.GetAttr().GetNumTimeSamples() > 1 for op in ops):
            return
        fields = {"xformOp:translate": "translation", "xformOp:scale": "scale"}
        fields.update({f"xformOp:rotate{order}": "rotation" for order in ROTATION_ORDERS})
        animated = [op for op in ops if op.GetAttr().GetNumTimeSamples() > 1]
        if all(op.GetOpName() in fields for op in animated):
            for op in animated:
                codes = np.array(op.GetAttr().GetTimeSamples())
                values = np.array([op.GetAttr().Get(c) for c in codes], dtype=np.float64)
                self.node_channels += [
                    Channel(node, fields[op.GetOpName()], axis, codes / self.fps, values[:, axis]) for axis in range(3)
                ]
            return
        codes = np.array(xf.GetTimeSamples())  # other op stacks: decompose the local matrix at every sample
        matrices = np.stack([np.array(xf.GetLocalTransformation(c)).T for c in codes])
        joint_op = [op for op in xf.GetOrderedXformOps() if op.GetOpName() == f"xformOp:transform:{JOINT_OP}"]
        if joint_op:
            matrices = np.linalg.inv(np.array(joint_op[0].GetOpTransform(_TIME)).T) @ matrices
        translation, rotation, scale = decompose_matrix(matrices, node.transform.rotation_order)
        rotation = np.degrees(np.unwrap(np.radians(rotation), axis=0))
        for prop, values in (("translation", translation), ("rotation", rotation), ("scale", scale)):
            self.node_channels += [Channel(node, prop, axis, codes / self.fps, values[:, axis]) for axis in range(3)]

    def read_animations(self, stage: Usd.Stage) -> list[Animation]:
        """Clips from SkelAnimation prims (by name) plus node time samples (the timeline's clip)."""
        clips: dict[str, Animation] = {}
        active_name = stage.GetRootLayer().customLayerData.get(ACTIVE_CLIP_KEY)
        for skel_path, (animations, source_path) in self.skeleton_animations.items():
            by_token = self.skeletons[skel_path][0]
            for prim in animations:
                channels = self.read_skel_animation(UsdSkel.Animation(prim), by_token, self.bound_shapes.get(skel_path, []))
                if not channels:
                    continue
                name = _display_name(prim)
                clips.setdefault(name, Animation(name, fps=self.fps)).channels += channels
                if prim.GetPath() == source_path and active_name is None:
                    active_name = name
        if self.node_channels or self.point_samples:
            name = active_name or "animation"
            clip = clips.setdefault(name, Animation(name, fps=self.fps))
            clip.channels += self.node_channels
            clip.point_samples += self.point_samples
        if active_name in clips and stage.HasAuthoredTimeCodeRange():
            clips[active_name].start = stage.GetStartTimeCode() / self.fps
            clips[active_name].end = stage.GetEndTimeCode() / self.fps
        return sorted(clips.values(), key=lambda clip: clip.name != active_name)

    def read_skel_animation(self, animation: UsdSkel.Animation, by_token, bound_shapes) -> list[Channel]:
        """Channels from one SkelAnimation; keys that never leave the rest value are dropped."""
        channels = []
        joints = [by_token.get(str(t)) for t in animation.GetJointsAttr().Get() or []]
        for attr, prop in ((animation.GetTranslationsAttr(), "translation"), (animation.GetRotationsAttr(), "orientation"),
                           (animation.GetScalesAttr(), "scale")):  # fmt: skip
            codes = attr.GetTimeSamples()
            if not codes:
                continue
            data = np.stack([np.array(attr.Get(c), dtype=np.float64) for c in codes])  # (F, J, 3 or 4)
            if prop == "orientation":
                data = data[..., [3, 0, 1, 2]]  # USD stores (x, y, z, w)
            times = np.array(codes) / self.fps
            for j, node in enumerate(joints):
                if node is None or j >= data.shape[1]:
                    continue
                values, rest = data[:, j], rest_value(node, prop)
                if prop == "orientation":
                    values = quaternion_continuity(values)
                    if np.allclose(values, rest, atol=1e-6) or np.allclose(values, -rest, atol=1e-6):
                        continue
                    channels += [Channel(node, prop, c, times, values[:, c]) for c in range(4)]
                else:
                    channels += [
                        Channel(node, prop, c, times, values[:, c])
                        for c in range(3) if not np.allclose(values[:, c], rest[c], atol=1e-6)
                    ]  # fmt: skip
        attr = animation.GetBlendShapeWeightsAttr()
        codes = attr.GetTimeSamples()
        if codes:
            tokens = [str(t) for t in animation.GetBlendShapesAttr().Get() or []]
            data = np.stack([np.array(attr.Get(c), dtype=np.float64) for c in codes])
            times = np.array(codes) / self.fps
            for mesh, names in bound_shapes:
                defaults = {shape.name: shape.weight for shape in mesh.blend_shapes}
                for i, token in enumerate(tokens):
                    name = names.get(token)
                    if name in defaults and not np.allclose(data[:, i], defaults[name], atol=1e-6):
                        channels.append(Channel(mesh, name, 0, times, data[:, i]))
        return channels

    def read_skin(self, prim: Usd.Prim, mesh: Mesh) -> Skin | None:
        binding = UsdSkel.BindingAPI(prim)
        skeleton = binding.GetInheritedSkeleton()
        if not skeleton or skeleton.GetPath() not in self.skeletons:
            warnings.warn(f"{prim.GetPath()}: skinned to an unknown skeleton; skin skipped")
            return None
        by_token, bind_by_token = self.skeletons[skeleton.GetPath()]
        joints_attr = binding.GetJointsAttr()
        order = [str(t) for t in (joints_attr.Get() if joints_attr.HasAuthoredValue() else skeleton.GetJointsAttr().Get())]
        missing = [t for t in order if t not in by_token or t not in bind_by_token]
        if missing:
            warnings.warn(f"{prim.GetPath()}: joints {missing[:3]} missing from the skeleton or its bind pose; skin skipped")
            return None

        index_primvar, weight_primvar = binding.GetJointIndicesPrimvar(), binding.GetJointWeightsPrimvar()
        k = max(index_primvar.GetElementSize(), 1)
        joint_indices = _array(index_primvar.Get(_TIME), np.int32).reshape(-1, k)
        joint_weights = _array(weight_primvar.Get(_TIME), np.float64).reshape(-1, k)
        if index_primvar.GetInterpolation() == UsdGeom.Tokens.constant:  # rigid: one set for all points
            joint_indices = np.repeat(joint_indices[:1], mesh.n_vertices, axis=0)
            joint_weights = np.repeat(joint_weights[:1], mesh.n_vertices, axis=0)
        geom_attr = binding.GetGeomBindTransformAttr()
        geom_bind = np.array(geom_attr.Get()).T if geom_attr.HasAuthoredValue() else np.eye(4)
        inverse_bind = np.linalg.inv(np.stack([bind_by_token[t] for t in order]))
        return Skin([by_token[t] for t in order], joint_indices, joint_weights, inverse_bind, geom_bind)


def _read_material(prim: Usd.Prim) -> Material:
    """A Material prim as a generic graph: every Shader and NodeGraph below it, with ports and connections.

    Node paths are made of the prims' display names (the original names when written by sceneport).
    """
    root = prim.GetPath()
    paths = {root: ""}  # prim path -> node path relative to the material
    prims = []
    for child in Usd.PrimRange(prim, _PREDICATE):
        if child != prim:
            parent = paths.get(child.GetPath().GetParentPath(), str(child.GetPath().GetParentPath().MakeRelativePath(root)))
            paths[child.GetPath()] = f"{parent}/{_display_name(child)}" if parent else _display_name(child)
            prims.append(child)
    inputs, outputs = _read_ports(UsdShade.ConnectableAPI(prim), root, paths)
    material = Material(_display_name(prim), inputs=inputs, outputs=outputs)
    for child in prims:
        path = paths[child.GetPath()]
        if child.IsA(UsdShade.Shader):
            inputs, outputs = _read_ports(UsdShade.ConnectableAPI(child), root, paths)
            node = ShaderNode(path, inputs=inputs, outputs=outputs)
            _read_shader_info(UsdShade.Shader(child), node)
            types = {name: list(t) for name, t in _fbx_types(UsdShade.ConnectableAPI(child)).items()}
            if types:
                node.extras.setdefault("fbx", {})["types"] = types
            material.nodes[path] = node
        elif child.IsA(UsdShade.NodeGraph):  # also a Material nested in a Material
            inputs, outputs = _read_ports(UsdShade.ConnectableAPI(child), root, paths)
            material.nodes[path] = NodeGraph(path, inputs=inputs, outputs=outputs)
    return material


def _port_name(attr: Usd.Attribute) -> str:
    """A port's name without its inputs:/outputs: prefix: the original name if sceneport sanitized it."""
    return attr.GetCustomDataByKey(NAME_KEY) or attr.GetName().split(":", 1)[1]


def _fbx_types(api: UsdShade.ConnectableAPI) -> dict[str, list[str]]:
    out = {}
    for port in api.GetInputs():
        value = port.GetAttr().GetCustomDataByKey(FBX_TYPE_KEY)
        if value is not None and len(value) == 3:
            out[_port_name(port.GetAttr())] = [str(v) for v in value]
    return out


def _read_shader_info(shader: UsdShade.Shader, node: ShaderNode) -> None:
    if (value := shader.GetIdAttr().Get()) is not None:
        node.shader_id = str(value)
    for attr in shader.GetPrim().GetAuthoredAttributes():
        parts = attr.GetName().split(":")  # info:<source type>:sourceAsset / sourceCode
        if len(parts) != 3 or parts[0] != "info" or (value := attr.Get()) is None:
            continue
        if parts[2] == "sourceAsset":
            node.source_assets[parts[1]] = _shade_value(value)
        elif parts[2] == "sourceCode":
            node.source_code[parts[1]] = str(value)
    source = shader.GetImplementationSourceAttr()
    if source.HasAuthoredValue() and source.Get() != UsdShade.Tokens.id:
        node.extras["usd"] = {"implementationSource": str(source.Get())}


def _read_ports(
    api: UsdShade.ConnectableAPI, root: Sdf.Path, paths: dict[Sdf.Path, str]
) -> tuple[dict[str, Port], dict[str, Port]]:
    """(inputs, outputs) of a shader, node graph or material; connections name nodes by `paths` (see _read_material)."""
    stage = api.GetPrim().GetStage()
    ports = []
    for group in (api.GetInputs(), api.GetOutputs()):
        out = {}
        for port in group:
            attr = port.GetAttr()
            connections = []
            for source in attr.GetConnections():
                owner = source.GetPrimPath()
                node = paths.get(owner, str(owner))  # outside the material: absolute path
                target = stage.GetAttributeAtPath(source)
                name = source.name
                if target and target.GetCustomDataByKey(NAME_KEY):
                    name = f"{name.split(':', 1)[0]}:{target.GetCustomDataByKey(NAME_KEY)}"
                connections.append(Connection(node, name))
            out[_port_name(attr)] = Port(
                str(port.GetTypeName()),
                _shade_value(_static_value(attr)) if attr.HasAuthoredValue() else None,
                connections,
                str(attr.GetColorSpace()) if attr.HasColorSpace() else None,
            )
        ports.append(out)
    return ports[0], ports[1]


def _shade_value(value):
    """A port value as plain Python: scalars and str as they are, AssetPath for files, float64 arrays for the rest."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Sdf.AssetPath):
        return AssetPath(value.path, value.resolvedPath or None)
    if isinstance(value, Sdf.AssetPathArray):
        return [AssetPath(v.path, v.resolvedPath or None) for v in value]
    if isinstance(value, (Gf.Quatd, Gf.Quatf, Gf.Quath)):
        return np.array([value.GetReal(), *value.GetImaginary()], dtype=np.float64)  # w, x, y, z
    array = np.array(value)
    if array.dtype.kind == "f":
        return array.astype(np.float64)
    return array if array.dtype.kind in "iub" else array.tolist()


def _parent_token(token: str) -> str:
    return str(Sdf.Path(token).GetParentPath())


def _matrices(value) -> NDArray | None:
    """USD row-vector Matrix4dArray -> (J, 4, 4) column-vector matrices."""
    return None if value is None or len(value) == 0 else np.array(value, dtype=np.float64).transpose(0, 2, 1)


def _display_name(prim: Usd.Prim) -> str:
    return prim.GetDisplayName() or prim.GetName()


def _xformable_children(prim: Usd.Prim) -> list[Usd.Prim]:
    return [c for c in prim.GetFilteredChildren(_PREDICATE) if c.IsA(UsdGeom.Xformable)]


def _read_transform(xf: UsdGeom.Xformable) -> Transform:
    ops = list(xf.GetOrderedXformOps())
    joint_matrix = None
    if ops and ops[0].GetOpName() == f"xformOp:transform:{JOINT_OP}":  # attachment: drop the joint placement
        joint_matrix = np.array(ops.pop(0).GetOpTransform(_TIME)).T
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
        field_name = _OP_STACK[_OP_INDEX[name]][1]
        if field_name is not None:
            values[field_name] = np.asarray(_static_value(op.GetAttr()), dtype=np.float64)
    names = {op.GetOpName() for op in ops}
    pivots_paired = all(
        (f"xformOp:translate:{p}" in names) == (f"!invert!xformOp:translate:{p}" in names)
        for p in ("rotatePivot", "scalePivot")
    )
    if positions is not None and positions == sorted(set(positions)) and pivots_paired:
        return Transform(**values)
    matrix = np.array(xf.GetLocalTransformation(_TIME)).T  # USD matrices are row-vector
    if joint_matrix is not None:
        matrix = np.linalg.inv(joint_matrix) @ matrix
    translation, rotation, scale = decompose_matrix(matrix)
    return Transform(translation=translation, rotation=rotation, scale=scale)


def _set_geometric(target: Transform, local: Transform) -> None:
    simple = Transform(translation=local.translation, rotation=local.rotation, scale=local.scale)
    if local.rotation_order != 0 or not np.allclose(simple.matrix(), local.matrix()):
        simple = Transform(*decompose_matrix(local.matrix()))
    target.geometric_translation = simple.translation
    target.geometric_rotation = simple.rotation
    target.geometric_scale = simple.scale


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


def _read_point_samples(prim: Usd.Prim, mesh: Mesh, fps: float) -> PointSamples | None:
    """Time-sampled points of a Mesh prim (vertex animation), if it has more than one sample of the same size."""
    attr = UsdGeom.Mesh(prim).GetPointsAttr()
    codes = attr.GetTimeSamples()
    if len(codes) < 2:
        return None
    points = [np.array(attr.Get(code), dtype=np.float64).reshape(-1, 3) for code in codes]
    if any(len(p) != mesh.n_vertices for p in points):
        warnings.warn(f"{prim.GetPath()}: point samples change the vertex count; vertex animation skipped")
        return None
    return PointSamples(mesh, np.array(codes) / fps, np.stack(points))


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
            values = _static_value(animation.GetBlendShapeWeightsAttr()) or []
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


def _static_value(attr: Usd.Attribute):
    """An attribute's default (unanimated) value if authored, else its first time sample."""
    value = attr.Get(_DEFAULT)
    return value if value is not None else attr.Get(_TIME)


def _shape_token_names(prim: Usd.Prim) -> dict[str, str]:
    """skel:blendShapes token -> blend shape name, as _read_blend_shapes names them."""
    binding = UsdSkel.BindingAPI(prim)
    tokens = [str(t) for t in binding.GetBlendShapesAttr().Get() or []]
    targets = binding.GetBlendShapeTargetsRel().GetTargets() if binding.GetBlendShapeTargetsRel() else []
    out = {}
    for token, target in zip(tokens, targets):
        shape_prim = prim.GetStage().GetPrimAtPath(target)
        if shape_prim and shape_prim.IsA(UsdSkel.BlendShape):
            out[token] = shape_prim.GetDisplayName() or token
    return out


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
