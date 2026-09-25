import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdSkel, Vt

import sceneport
from sceneport.interop.fbx import document_from_scene, scene_from_document
from sceneport.interop.fbx.binary import dumps, loads
from sceneport.interop.fbx.document import FBXDocument
from sceneport.interop.fbx.skin import matrix_from_fbx
from sceneport.interop.usd import scene_from_stage, stage_from_scene
from sceneport.scene import BlendShape, Mesh, Node, Scene, Skin, Transform


def rig_scene(shared_parent: bool = False) -> Scene:
    """A strip along X skinned to a 3-joint chain (root -> mid -> tip), with a helper
    attached to `mid` and a blend shape. The mesh node is offset, so geom_bind isn't identity."""
    xs = np.repeat([0.0, 1.0, 2.0], 2)
    vertices = np.stack([xs, np.tile([0.0, 1.0], 3), np.zeros(6)], axis=1)
    mesh = Mesh("Strip", vertices, [4, 4], [0, 2, 3, 1, 2, 4, 5, 3])
    mesh.blend_shapes = [BlendShape("Lift", [[0, 0, 1], [0, 0, 1]], indices=[4, 5], weight=0.5)]

    root = Node("root", Transform(translation=[0, 0, 0]), joint=True)
    mid = root.add(Node("mid", Transform(translation=[1, 0, 0]), joint=True))
    mid.add(Node("tip.end", Transform(translation=[1, 0, 0]), joint=True))
    mid.add(Node("helper", Transform(translation=[0, 2, 0], rotation=[0, 0, 30])))
    mesh_node = Node("strip", Transform(translation=[0, 0, 0.5]), mesh=mesh)

    # Bind = rest pose; the mesh node is offset by 0.5 in Z.
    joints = [mid, root]  # skin order differs from hierarchy order on purpose
    bind = np.stack([Transform(translation=[1, 0, 0]).matrix(), np.eye(4)])
    mesh.skin = Skin(
        joints,
        joint_indices=[[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]],
        joint_weights=[[1, 0], [1, 0], [0.5, 0.5], [0.5, 0.5], [1, 0], [1, 0]],
        inverse_bind=np.linalg.inv(bind),
        geom_bind=mesh_node.transform.matrix(),
    )
    if shared_parent:
        return Scene(nodes=[Node("character", children=[root, mesh_node])])
    return Scene(nodes=[root, mesh_node])


def pose(scene: Scene) -> None:
    scene.find("mid").transform.rotation = np.array([0.0, 0.0, 90.0])


def fbx_roundtrip(scene: Scene) -> Scene:
    return scene_from_document(FBXDocument.from_file(loads(dumps(document_from_scene(scene).to_file()))))


def usd_roundtrip(scene: Scene) -> Scene:
    return scene_from_stage(stage_from_scene(scene))


def assert_skins_equal(a: Skin, b: Skin, atol: float = 0.0) -> None:
    # Compare per-joint weights: FBX stores weights per joint, so per-vertex slot order may change.
    assert [j.name for j in a.joints] == [j.name for j in b.joints]
    assert all(j.joint for j in b.joints)
    np.testing.assert_allclose(a.dense_weights(), b.dense_weights(), atol=1e-7)
    np.testing.assert_allclose(a.inverse_bind, b.inverse_bind, atol=max(atol, 1e-12))
    np.testing.assert_allclose(a.geom_bind, b.geom_bind, atol=max(atol, 1e-12))


def world_by_name(scene: Scene) -> dict[str, np.ndarray]:
    return {node.name: world for node, world in scene.walk()}


# -- model --------------------------------------------------------------------


def test_skinning_at_bind_pose_is_mesh_in_world():
    scene = rig_scene()
    mesh = scene.meshes[0]
    mesh.validate()
    rest = mesh.skinned_vertices(mesh.skin.joint_world(scene), weights={"Lift": 0})
    np.testing.assert_allclose(rest, mesh.vertices + [0, 0, 0.5])  # geom_bind offset


def test_skinning_posed_and_frames():
    scene = rig_scene()
    mesh = scene.meshes[0]
    rest = mesh.skin.joint_world(scene)
    pose(scene)
    posed = mesh.skin.joint_world(scene)
    out = mesh.skinned_vertices(posed, weights={"Lift": 0})
    np.testing.assert_allclose(out[4], [1, 1, 0.5], atol=1e-12)  # (2, 0) rotated 90 degrees about mid at (1, 0)
    np.testing.assert_allclose(out[0], [0, 0, 0.5], atol=1e-12)  # root only: unchanged

    frames = mesh.skinned_vertices(np.stack([rest, posed]), weights=np.array([[0.0], [1.0]]))
    assert frames.shape == (2, 6, 3)
    np.testing.assert_allclose(frames[0], mesh.skinned_vertices(rest, {"Lift": 0}))
    np.testing.assert_allclose(frames[1], mesh.skinned_vertices(posed, {"Lift": 1}))
    # one pose, several blend shape frames
    assert mesh.skinned_vertices(posed, weights=np.array([[0.0], [1.0]])).shape == (2, 6, 3)


def test_dense_weights():
    skin = rig_scene().meshes[0].skin
    np.testing.assert_array_equal(skin.dense_weights()[2], [0.5, 0.5])
    np.testing.assert_array_equal(skin.dense_weights()[4], [1.0, 0.0])  # joint 0 = mid


def test_skinning_errors():
    scene = rig_scene()
    mesh = scene.meshes[0]
    with pytest.raises(ValueError, match="J = 2"):
        mesh.skinned_vertices(np.zeros((3, 4, 4)))
    mesh.skin.joint_indices[0, 0] = 5
    with pytest.raises(ValueError, match="joint index out of range"):
        mesh.validate()
    with pytest.raises(ValueError, match="no skin"):
        Mesh("m", np.zeros((3, 3)), [3], [0, 1, 2]).skinned_vertices(np.zeros((0, 4, 4)))


# -- FBX ----------------------------------------------------------------------


def test_fbx_roundtrip():
    scene = rig_scene()
    back = fbx_roundtrip(scene)
    assert_skins_equal(scene.meshes[0].skin, back.meshes[0].skin)
    assert back.find("helper") in back.find("mid").children and not back.find("helper").joint
    assert back.find("tip.end").joint
    for name, world in world_by_name(scene).items():
        np.testing.assert_allclose(world_by_name(back)[name], world, atol=1e-12, err_msg=name)
    pose(scene), pose(back)
    np.testing.assert_allclose(
        back.meshes[0].skinned_vertices(back.meshes[0].skin.joint_world(back)),
        scene.meshes[0].skinned_vertices(scene.meshes[0].skin.joint_world(scene)),
        atol=1e-12,
    )


def test_fbx_layout():
    scene = rig_scene()
    doc = document_from_scene(scene)
    assert [o.subtype for o in doc.iter_objects("Model")] == ["LimbNode", "LimbNode", "LimbNode", "Null", "Mesh"]
    assert len(list(doc.iter_objects("NodeAttribute", "LimbNode"))) == 3
    (skin,) = doc.iter_objects("Deformer", "Skin")
    clusters = doc.children_of(skin, "Deformer")
    assert [(c.name, c.subtype, c.cls) for c in clusters] == [("mid", "Cluster", "SubDeformer"), ("root", "Cluster", "SubDeformer")]
    assert [m.name for m in doc.children_of(clusters[0], "Model")] == ["mid"]
    link, transform = (matrix_from_fbx(clusters[0].child_value(k)) for k in ("TransformLink", "Transform"))
    np.testing.assert_allclose(link @ transform, scene.meshes[0].skin.geom_bind)  # Transform = inv(link) @ geom_bind
    (pose_obj,) = doc.iter_objects("Pose", "BindPose")
    assert pose_obj.child_value("NbPoseNodes") == 3  # mesh + 2 skin joints


def test_fbx_root_subtype_kept():
    scene = rig_scene()
    scene.find("root").extras["fbx"] = {"subtype": "Root"}
    doc = document_from_scene(scene)
    assert [o.subtype for o in doc.iter_objects("Model")][0] == "Root"
    assert fbx_roundtrip(scene).find("root").extras == {"fbx": {"subtype": "Root"}}


def test_fbx_file(tmp_path):
    scene = rig_scene()
    sceneport.save(scene, tmp_path / "rig.fbx")
    assert_skins_equal(scene.meshes[0].skin, sceneport.load(tmp_path / "rig.fbx").meshes[0].skin)


# -- USD ----------------------------------------------------------------------


def test_usd_roundtrip():
    scene = rig_scene()
    back = usd_roundtrip(scene)
    assert [n.name for n in back.nodes] == ["root", "strip"]  # the generated SkelRoot is dropped
    assert_skins_equal(scene.meshes[0].skin, back.meshes[0].skin, atol=1e-12)
    helper = back.find("helper")
    assert helper in back.find("mid").children and not helper.joint
    np.testing.assert_allclose(helper.transform.rotation, [0, 0, 30])  # exact, not decomposed
    for name, world in world_by_name(scene).items():
        np.testing.assert_allclose(world_by_name(back)[name], world, atol=1e-12, err_msg=name)
    assert back.meshes[0].blend_shapes[0].weight == 0.5  # from the real skeleton's animation


def test_usd_layout():
    stage = stage_from_scene(rig_scene())
    wrapper = stage.GetPrimAtPath("/SkelRoot")
    assert wrapper.GetTypeName() == "SkelRoot" and wrapper.GetCustomDataByKey("sceneport:generated")
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath("/SkelRoot/root"))
    assert list(skeleton.GetJointsAttr().Get()) == ["root", "root/mid", "root/mid/tip_end"]
    assert list(skeleton.GetJointNamesAttr().Get()) == ["root", "mid", "tip.end"]
    helper = stage.GetPrimAtPath("/SkelRoot/root/helper")
    assert helper.GetCustomDataByKey("sceneport:joint") == "root/mid"
    binding = UsdSkel.BindingAPI(stage.GetPrimAtPath("/SkelRoot/strip/Strip"))
    assert list(binding.GetJointsAttr().Get()) == ["root/mid", "root"]  # the skin's own order
    assert binding.GetJointIndicesPrimvar().GetElementSize() == 2


def test_usd_skelroot_at_common_ancestor():
    stage = stage_from_scene(rig_scene(shared_parent=True))
    assert stage.GetPrimAtPath("/character").GetTypeName() == "SkelRoot"
    assert not stage.GetPrimAtPath("/SkelRoot")
    assert_skins_equal(rig_scene().meshes[0].skin, scene_from_stage(stage).meshes[0].skin, atol=1e-12)


def test_usd_skinning_matches_usdskel():
    """Our linear blend skinning must match UsdSkel's evaluation of the written file."""
    scene = rig_scene()
    stage = stage_from_scene(scene)
    pose(scene)
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath("/SkelRoot/root"))
    animation = UsdSkel.Animation(stage.GetPrimAtPath("/SkelRoot/root/Anim"))  # holds the blend shape weight
    animation.CreateJointsAttr(skeleton.GetJointsAttr().Get())
    by_name = {n.name: n for n, _ in scene.walk()}
    local = np.stack([by_name[name].transform.matrix() for name in skeleton.GetJointNamesAttr().Get()])
    animation.SetTransforms(Vt.Matrix4dArray.FromNumpy(np.ascontiguousarray(local.transpose(0, 2, 1))), Usd.TimeCode.Default())
    assert UsdSkel.BakeSkinning(stage.Traverse())
    usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/SkelRoot/strip/Strip"))
    local = np.array(usd_mesh.GetPointsAttr().Get(Usd.TimeCode.EarliestTime()))  # baked points are mesh-local
    to_world = np.array(usd_mesh.ComputeLocalToWorldTransform(Usd.TimeCode.EarliestTime())).T
    baked = local @ to_world[:3, :3].T + to_world[:3, 3]
    mesh = scene.meshes[0]
    np.testing.assert_allclose(baked, mesh.skinned_vertices(mesh.skin.joint_world(scene)), atol=1e-6)


def test_usd_unskinned_skeleton():
    root = Node("hips", joint=True, children=[Node("spine", Transform(translation=[0, 1, 0]), joint=True)])
    back = usd_roundtrip(Scene(nodes=[root]))
    assert [n.name for n in back.nodes] == ["hips"] and back.nodes[0].joint
    np.testing.assert_allclose(back.find("spine").transform.translation, [0, 1, 0])


def test_usd_file(tmp_path):
    scene = rig_scene()
    sceneport.save(scene, tmp_path / "rig.usdc")
    assert_skins_equal(scene.meshes[0].skin, sceneport.load(tmp_path / "rig.usdc").meshes[0].skin, atol=1e-12)


def test_skin_joints_with_different_parents_rejected():
    scene = rig_scene()
    other = Node("other", joint=True)
    scene.nodes.append(Node("elsewhere", children=[other]))
    scene.meshes[0].skin.joints[0] = other
    with pytest.raises(ValueError, match="share one parent"):
        stage_from_scene(scene)
