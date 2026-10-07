"""Scene rewrites applied before building Blender data (pure numpy: no bpy here).

- In-betweens are split into plain shapes (Blender shape keys have none), as for FBX
  (see fbx/compat.py).
- Mirrored joints are unmirrored (reframe.unmirror): a Blender bone can't be mirrored.
- The scene moves to Blender's axes and units: Z up, -Y front, meters
  (reframe.change_basis), so the data is Z-up itself rather than turned by a
  rotation on the root objects.
- Geometric transforms (FBX's offset of a node's own geometry) are baked into the
  mesh: Blender has none. A mesh shared by nodes with different geometric
  transforms is copied per transform.
- A skinned mesh is re-bound where its node isn't where it was bound: Blender's
  armature modifier deforms the mesh as placed by its object, with no separate
  bind placement, so the points are moved to the node's space (geom_bind = the
  node's world matrix).
- Every tree of joints gets a plain parent node to become its armature object:
  the joints' parent when it's a plain node (e.g. a USD Skeleton or an FBX
  "Armature" null), else a new node inserted there (as Blender's FBX importer
  does), named after the root joint (e.g. MHR's "body_world"; Blender keeps
  object and bone names apart), or "Armature" for several root joints.
- A skinned mesh that isn't under its armature is moved there, keeping its world
  placement, as Blender's FBX importer does: Blender's convention is that
  deformed meshes are children of their armature.

The input scene is never modified.

`rigs(scene)` then lays out the armatures (pure numpy): each tree of joints is
one armature, held by its parent node.
Bones rest in the bind pose (Blender's armature modifier binds to the rest pose)
and are posed to the scene's current pose. A bone's frame is its joint's frame,
scale removed: the bone points along the joint's +Y.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ...math import complete_rotation, decompose_matrix
from ...scene.types import Inbetween, Mesh, Node, Primvar, Scene, Transform
from ..fbx.compat import _split_inbetweens
from ..reframe import change_basis, copy_scene, key_times, mirrored_joints, unmirror
from ...scene.animation import Animation, Channel

UP, FRONT, RIGHT = "+Z", "-Y", "+X"  # Blender's axes, in FBX's meaning of front


def prepare(scene: Scene) -> Scene:
    """The scene as Blender holds it (a copy when anything changes)."""
    if any(shape.inbetweens for mesh in scene.meshes for shape in mesh.blend_shapes):
        scene = _split_inbetweens(scene)
    if mirrored_joints(scene):
        scene = unmirror(scene)
    scene = change_basis(scene, up=UP, front=FRONT, right=RIGHT, meters_per_unit=1.0)
    scene = _bake_geometric(scene)
    scene = _rebind_skins(scene)
    if any(node.joint for node, _ in scene.walk()):
        scene = _armature_nodes(copy_scene(scene, {}, {}, lambda clip: clip.channels))  # a copy to restructure
    return scene


def _holds_armature(node: Node | None) -> bool:
    return node is not None and not node.joint and node.mesh is None and node.camera is None and node.light is None


def _armature_nodes(scene: Scene) -> Scene:
    """Give every joint tree a plain parent node and move skinned meshes under it (`scene` is restructured
    in place: pass a copy). A moved mesh node whose world placement is animated (it or an ancestor, old or
    new, has transform channels) gets its transform channels baked relative to its new parent."""
    parents = scene.parents()
    roots: dict[Node | None, list[Node]] = {}
    for node, _ in scene.walk():
        if node.joint and not (parents[node] is not None and parents[node].joint):
            roots.setdefault(parents[node], []).append(node)
    holder_of: dict[Node, Node] = {}  # root joint -> its armature node
    for parent, joints in roots.items():
        holder = parent
        if not _holds_armature(parent):
            holder = Node(joints[0].name if len(joints) == 1 else "Armature", children=joints)
            siblings = scene.nodes if parent is None else parent.children
            siblings[siblings.index(joints[0])] = holder
            for joint in joints:
                if joint in siblings:
                    siblings.remove(joint)
        for joint in joints:
            holder_of[joint] = holder

    parents = scene.parents()
    world = scene.world_matrices()

    def root_joint(joint: Node) -> Node:
        while parents[joint] is not None and parents[joint].joint:
            joint = parents[joint]
        return joint

    def ancestors(node: Node):
        while parents[node] is not None:
            node = parents[node]
            yield node

    for node in [n for n in world if n.mesh is not None and n.mesh.skin is not None and not n.joint]:
        holder = holder_of[root_joint(node.mesh.skin.joints[0])]
        if holder in ancestors(node):
            continue
        baked = _baked_locals(scene, node, holder, {node, *ancestors(node), holder, *ancestors(holder)})
        siblings = scene.nodes if parents[node] is None else parents[node].children
        siblings.remove(node)
        holder.children.append(node)
        translation, rotation, scale = decompose_matrix(np.linalg.inv(world[holder]) @ world[node], node.transform.rotation_order)
        node.transform = Transform(translation=translation, rotation=rotation, scale=scale,
                                   rotation_order=node.transform.rotation_order)  # fmt: skip
        for clip, (times, locals_) in baked.items():
            translation, rotation, scale = decompose_matrix(locals_, node.transform.rotation_order)
            rotation = np.degrees(np.unwrap(np.radians(rotation), axis=0))
            clip.channels = [c for c in clip.channels if not (c.target is node and c.is_transform)] + [
                Channel(node, prop, axis, times, values[:, axis])
                for prop, values in (("translation", translation), ("rotation", rotation), ("scale", scale))
                for axis in range(3)
            ]
    return scene


def _baked_locals(scene: Scene, node: Node, holder: Node, chain: set[Node]) -> dict[Animation, tuple]:
    """For each clip animating `node` or a node in `chain`: (times, (F, 4, 4) node world relative to holder)."""
    out = {}
    for clip in scene.animations:
        channels = [c for c in clip.channels if c.is_transform and c.target in chain]
        if channels:
            times = key_times(clip, channels)
            world = clip.world_matrices(scene, times)
            out[clip] = (times, np.linalg.inv(world[holder]) @ world[node])
    return out


def _bake_geometric(scene: Scene) -> Scene:
    baked = {node: node.transform.geometric_matrix() for node, _ in scene.walk()
             if node.mesh is not None and not np.array_equal(node.transform.geometric_matrix(), np.eye(4))}  # fmt: skip
    transforms = {
        node: dataclasses.replace(node.transform, geometric_translation=np.zeros(3), geometric_rotation=np.zeros(3),
                                  geometric_scale=np.ones(3))
        for node in baked
    }  # fmt: skip
    return _move_meshes(scene, baked, transforms)


def _rebind_skins(scene: Scene) -> Scene:
    """Skinned meshes whose node isn't at the bind placement, moved there: points p' = W^-1 @ geom_bind @ p."""
    moves = {}
    for node, world in scene.walk():
        skin = None if node.mesh is None else node.mesh.skin
        if skin is not None and not np.allclose(world, skin.geom_bind, atol=1e-9):
            moves[node] = np.linalg.inv(world) @ skin.geom_bind
    return _move_meshes(scene, moves, {}, rebind=True)


def _move_meshes(scene: Scene, moves: dict, transforms: dict, rebind: bool = False) -> Scene:
    """A copy with each node in `moves` given a copy of its mesh transformed by its matrix (shared by nodes
    with the same mesh and matrix) and nodes in `transforms` given those transforms. `rebind`: the copies'
    skins get geom_bind = their node's world matrix."""
    if not moves:
        return scene
    copy = copy_scene(scene, {}, transforms, lambda clip: clip.channels)
    copies: dict[tuple[int, bytes], Mesh] = {}
    replaced: dict[int, list[Mesh]] = {}  # id(mesh in the copy) -> the moved meshes standing in for it
    matrices: dict[int, np.ndarray] = {}  # id(moved mesh) -> how its points moved
    for (node, _), (new, world) in zip(scene.walk(), copy.walk()):
        if node in moves:
            key = (id(node.mesh), moves[node].tobytes())
            if key not in copies:
                moved = _transformed_mesh(new.mesh, moves[node])
                if rebind:
                    moved.skin = dataclasses.replace(moved.skin, geom_bind=world)
                copies[key] = moved
                replaced.setdefault(id(new.mesh), []).append(moved)
                matrices[id(moved)] = moves[node]
            new.mesh = copies[key]
    for clip in copy.animations:  # weight channels and point samples follow their mesh (to every copy of it)
        clip.channels = [dataclasses.replace(c, target=m) for c in clip.channels
                         for m in (replaced.get(id(c.target), [c.target]) if c.is_blend_shape else [c.target])]  # fmt: skip
        clip.point_samples = [
            dataclasses.replace(p, mesh=m, points=p.points @ matrices[id(m)][:3, :3].T + matrices[id(m)][:3, 3])
            if id(m) in matrices else p
            for p in clip.point_samples for m in replaced.get(id(p.mesh), [p.mesh])
        ]  # fmt: skip
    return copy


def _transformed_mesh(mesh: Mesh, matrix: NDArray) -> Mesh:
    """`mesh` with its points moved by `matrix` (normals by its inverse transpose, offsets by its linear part)."""
    linear = matrix[:3, :3]
    normal = np.linalg.inv(linear).T

    def directions(values):
        if values is None:
            return None
        out = values @ normal.T
        return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)

    def offsets(values):
        return None if values is None else values @ linear.T

    normals = mesh.normals
    if normals is not None:
        normals = Primvar(directions(normals.values), normals.interpolation, normals.indices)
    shapes = [
        dataclasses.replace(shape, offsets=offsets(shape.offsets), normal_offsets=offsets(shape.normal_offsets),
                            inbetweens=[Inbetween(i.weight, offsets(i.offsets), offsets(i.normal_offsets)) for i in shape.inbetweens])
        for shape in mesh.blend_shapes
    ]  # fmt: skip
    return dataclasses.replace(
        mesh, vertices=mesh.vertices @ linear.T + matrix[:3, 3], normals=normals, blend_shapes=shapes
    )



# -- armatures -----------------------------------------------------------------------------


@dataclass(eq=False)
class Bone:
    joint: Node
    parent: Node | None  # the parent joint (a bone of the same rig), or None for a root bone
    rest: NDArray[np.float64]  # (4, 4) rest frame in armature space: the bind pose, without scale
    pose: NDArray[np.float64]  # (4, 4) posed matrix in armature space (Blender's PoseBone.matrix)
    length: float
    bind: NDArray[np.float64] = field(default_factory=lambda: np.eye(4))  # (4, 4) the joint's bind world matrix


@dataclass(eq=False)
class Rig:
    """One armature, held by the plain node `node` (its object becomes the armature object)."""

    name: str
    node: Node
    world: NDArray[np.float64]  # (4, 4) the armature object's world matrix
    bones: list[Bone] = field(default_factory=list)  # parents before children

    def bone(self, joint: Node) -> Bone | None:
        return next((b for b in self.bones if b.joint is joint), None)


def rigs(scene: Scene) -> list[Rig]:
    """The armatures for a prepared scene's joints, with every bone's rest and pose matrices and length.

    With A the armature's world matrix, B a joint's bind world matrix (from its skin's inverse bind, else
    its current world matrix) and J its current world matrix: rest R = A^-1 @ B without scale, and pose
    P = A^-1 @ J @ B^-1 @ A @ R, so the armature modifier's P @ R^-1 is the skin's J @ B^-1 in armature
    space (P = A^-1 @ J when the bind has no scale).
    """
    world = scene.world_matrices()
    parents = scene.parents()
    bind = _bind_worlds(scene)
    roots: dict[Node | None, list[Node]] = {}
    for node, _ in scene.walk():
        if node.joint and not (parents[node] is not None and parents[node].joint):
            roots.setdefault(parents[node], []).append(node)
    out = []
    for parent, tree_roots in roots.items():
        if not _holds_armature(parent):
            raise ValueError(f"joints {[j.name for j in tree_roots]} have no plain parent node; run prepare() first")
        rig = Rig(parent.name, parent, world[parent])
        a_inv = np.linalg.inv(rig.world)
        stack = list(reversed(tree_roots))
        while stack:
            joint = stack.pop()
            b = bind.get(joint, world[joint])
            rest = _without_scale(a_inv @ b)
            pose = a_inv @ world[joint] @ np.linalg.inv(b) @ rig.world @ rest
            parent_joint = parents[joint] if parents[joint] is not None and parents[joint].joint else None
            rig.bones.append(Bone(joint, parent_joint, rest, pose, 0.0, b))
            stack.extend(c for c in reversed(joint.children) if c.joint)
        _bone_lengths(rig)
        out.append(rig)
    return out


def _bind_worlds(scene: Scene) -> dict[Node, NDArray]:
    """Each skinned joint's bind world matrix, from the first skin using it (warns when skins disagree)."""
    out: dict[Node, NDArray] = {}
    differ = set()
    for mesh in scene.meshes:
        if mesh.skin is None:
            continue
        for joint, b in zip(mesh.skin.joints, mesh.skin.bind_world()):
            if joint not in out:
                out[joint] = b
            elif not np.allclose(out[joint], b, atol=1e-6 * max(1.0, np.abs(b).max())):
                differ.add(joint.name)
    if differ:
        warnings.warn(f"joints {sorted(differ)[:3]}{'...' if len(differ) > 3 else ''} have different bind poses in "
                      "different skins; Blender has one rest pose per bone, so the first skin's is used")  # fmt: skip
    return out


def _without_scale(m: NDArray) -> NDArray:
    """`m` with its basis made a rotation (scale removed; a mirroring basis has its X axis flipped)."""
    basis = m[:3, :3]
    scale = np.linalg.norm(basis, axis=0)
    rotation = complete_rotation(basis / np.where(scale == 0, 1, scale), scale != 0)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, m[:3, 3]
    return out


def _bone_lengths(rig: Rig) -> None:
    """Bones reach to the child joint lying ahead along +Y (the closest-aligned one), else to their nearest
    child joint, else half their parent's length; never shorter than a small part of the rig's size."""
    heads = np.array([b.rest[:3, 3] for b in rig.bones])
    size = float(np.ptp(heads, axis=0).max()) if len(heads) > 1 else 0.0
    size = size if size > 1e-9 else 1.0
    lengths: dict[Node, float] = {}
    for bone in rig.bones:
        head, axis = bone.rest[:3, 3], bone.rest[:3, 1]
        offsets = np.array([c.rest[:3, 3] - head for c in rig.bones if c.parent is bone.joint]).reshape(-1, 3)
        distances = np.linalg.norm(offsets, axis=1)
        offsets, distances = offsets[distances > 1e-9 * size], distances[distances > 1e-9 * size]
        along = offsets @ axis
        ahead = along > 0.5 * distances
        if ahead.any():
            length = float(along[ahead][np.argmax(along[ahead] / distances[ahead])])
        elif len(distances):
            length = float(distances.min())
        elif bone.parent is not None:
            length = 0.5 * lengths[bone.parent]
        else:
            length = 0.1 * size
        lengths[bone.joint] = bone.length = max(length, 0.002 * size)


def pose_tracks(scene: Scene, rig: Rig, clip: Animation) -> tuple[NDArray, dict[Node, NDArray]] | None:
    """The rig's posed matrices over `clip`, sampled on every frame and key: (times, {joint: (F, 4, 4)}), with
    P(t) = A(t)^-1 @ J(t) @ B^-1 @ A @ R as in rigs() (A(t) the armature's world matrix then, A the one the
    rest pose is expressed in). None when nothing moves the rig in the clip."""
    parents = scene.parents()
    moving = {rig.node, *(b.joint for b in rig.bones)}
    node = rig.node
    while parents[node] is not None:
        node = parents[node]
        moving.add(node)
    channels = [c for c in clip.channels if c.is_transform and c.target in moving]
    if not channels:
        return None
    times = key_times(clip, channels)
    world = clip.world_matrices(scene, times)
    a_inv = np.linalg.inv(world[rig.node])
    return times, {b.joint: a_inv @ world[b.joint] @ np.linalg.inv(b.bind) @ rig.world @ b.rest for b in rig.bones}
