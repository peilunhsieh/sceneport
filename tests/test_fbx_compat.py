"""FBX export rewrites for Blender: split in-between shapes, unmirror mirrored joints (interop/fbx/compat.py)."""

import numpy as np
import pytest

from sceneport.interop.fbx import document_from_scene, scene_from_document
from sceneport.interop.fbx.binary import dumps, loads
from sceneport.interop.fbx.compat import hat_weights
from sceneport.interop.fbx.document import FBXDocument
from sceneport.scene import Animation, BlendShape, Channel, Inbetween, Node, Scene
from helpers import rig_scene, shaped_mesh


def fbx_roundtrip(scene: Scene, **options) -> Scene:
    return scene_from_document(FBXDocument.from_file(loads(dumps(document_from_scene(scene, **options).to_file()))))


# -- in-betweens -------------------------------------------------------------------


def test_hat_weights_match_inbetween_interpolation():
    shape = BlendShape("s", [[0, 0, 1]], inbetweens=[Inbetween(-0.5, [[1, 0, 0]]), Inbetween(0.4, [[0, 2, 0]])])
    keys = sorted([(-0.5, [1, 0, 0]), (0.4, [0, 2, 0]), (1.0, [0, 0, 1])])
    key_weights, key_offsets = np.array([k[0] for k in keys]), np.array([k[1] for k in keys], dtype=float)
    weights = np.array([-1.2, -0.5, -0.1, 0.0, 0.2, 0.4, 0.7, 1.0, 1.6])  # inside and outside the keys
    parts = hat_weights(weights, key_weights)  # (W, K)
    np.testing.assert_allclose(parts @ key_offsets, shape.offsets_at(weights)[:, 0], atol=1e-12)


def test_inbetweens_split_by_default():
    mesh = shaped_mesh()  # "Smile.L" has an in-between at 0.5 and default weight 0.25
    scene = Scene(nodes=[Node("Face", mesh=mesh)])
    scene.animations = [Animation("smile", [Channel(mesh, "Smile.L", 0, [0.0, 1.0, 2.0], [0.0, 0.8, -0.3])], fps=4)]
    back = fbx_roundtrip(scene)
    shapes = back.meshes[0].blend_shapes
    assert [s.name for s in shapes] == ["Smile.L_0.5", "Smile.L", "Wide"]
    assert not any(s.inbetweens for s in shapes)
    np.testing.assert_allclose(back.meshes[0].deformed_vertices(), mesh.deformed_vertices())  # default weights
    times = scene.animations[0].frame_times()
    expected = mesh.deformed_vertices(scene.animations[0].blend_weights(mesh, times))
    actual = back.meshes[0].deformed_vertices(back.animations[0].blend_weights(back.meshes[0], times))
    np.testing.assert_allclose(actual, expected, atol=1e-6)  # weights are float32 in FBX
    assert mesh.blend_shapes[0].inbetweens  # the scene itself is untouched

    kept = fbx_roundtrip(scene, split_inbetweens=False).meshes[0].blend_shapes
    assert [s.name for s in kept] == ["Smile.L", "Wide"] and kept[0].inbetweens


# -- mirrored joints -------------------------------------------------------------------


def mirrored_rig(scale) -> Scene:
    """rig_scene() with `mid` mirrored (and its child `tip.end` with it), animated."""
    scene = rig_scene()
    mid, root = scene.find("mid"), scene.find("root")
    mid.transform.scale = np.array(scale, dtype=float)
    skin = scene.meshes[0].skin  # bound in the mirrored pose, like a real mirrored rig
    skin.inverse_bind = np.linalg.inv(skin.joint_world(scene))
    scene.animations = [Animation("take", [
        Channel(root, "translation", 1, [0.0, 0.5, 1.0], [0.0, 2.0, 0.0], "cubic"),
        Channel(mid, "rotation", 2, [0.0, 0.5, 1.0], [0.0, 45.0, 90.0]),
        Channel(mid, "translation", 0, [0.0, 1.0], [1.0, 1.5]),
    ], fps=24)]  # fmt: skip
    return scene


def joint_worlds(scene: Scene, times) -> dict[str, np.ndarray]:
    return {node.name: m for node, m in scene.animations[0].world_matrices(scene, times).items() if node.joint}


@pytest.mark.parametrize("scale", [(-1, -1, -1), (-1, 1, 1), (2, -1, 1)])
def test_mirrored_joints_unmirrored_by_default(scale):
    scene = mirrored_rig(scale)
    times = scene.animations[0].frame_times()
    back = fbx_roundtrip(scene)

    worlds = joint_worlds(back, times)
    assert all((np.linalg.det(m[:, :3, :3]) > 0).all() for m in worlds.values())  # nothing mirrored is written
    for name, m in joint_worlds(scene, times).items():  # joints stay where they were
        np.testing.assert_allclose(worlds[name][:, :3, 3], m[:, :3, 3], atol=1e-5, err_msg=name)
    np.testing.assert_allclose(  # and the skin deforms the same
        back.animations[0].skinned_vertices(back, back.meshes[0], times),
        scene.animations[0].skinned_vertices(scene, scene.meshes[0], times),
        atol=1e-5,
    )
    helper_world = scene.animations[0].world_matrices(scene, times)[scene.find("helper")]
    back_helper = back.animations[0].world_matrices(back, times)[back.find("helper")]
    np.testing.assert_allclose(back_helper, helper_world, atol=1e-5)  # a non-joint child keeps its world matrix
    assert np.array_equal(scene.find("mid").transform.scale, scale)  # the scene itself is untouched


def test_mirrored_joints_kept_on_request():
    scene = mirrored_rig((-1, -1, -1))
    back = fbx_roundtrip(scene, unmirror_joints=False)
    assert np.linalg.det(back.find("mid").transform.matrix()[:3, :3]) < 0
