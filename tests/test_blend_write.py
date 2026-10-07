"""Writing .blend files with Blender: hierarchy, meshes, normals, UVs and shape keys, in Blender's Z-up meters."""

import numpy as np
import pytest

import sceneport
from sceneport.interop.blend import run_in_blender
from sceneport.interop.reframe import basis_matrix
from sceneport.scene import Animation, BlendShape, Channel, Inbetween, Mesh, Node, Primvar, Scene, Transform


def face_scene() -> Scene:
    """A centimeter Y-up scene: a quad + triangle mesh with UVs, normals and shape keys (one sparse with
    a negative weight, one animated past 1, one with an in-between) under a parent, plus a plain empty."""
    vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 15, 2]], dtype=float)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0.6, 0.8], [0, 0.6, 0.8], [0, 0.8, 0.6]])
    mesh = Mesh("face", vertices, [4, 3], [0, 1, 2, 3, 3, 2, 4], normals=Primvar(normals, "vertex"),
                uvs={"UVMap": Primvar(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 1.5]]), "faceVarying",
                                      np.array([0, 1, 2, 3, 3, 2, 4]))},
                blend_shapes=[
                    BlendShape("brow", [[0, 2, 0]], indices=[4], weight=-2.5),
                    BlendShape("smile", np.tile([1.0, 0, 0], (5, 1))),
                    BlendShape("open", [[0, 0, 4]], indices=[2], inbetweens=[Inbetween(0.5, [[0, 0, 1]])]),
                ])  # fmt: skip
    head = Node("Head", Transform(translation=[0, 150, 0], rotation=[0, 30, 0], rotation_order=4), mesh=mesh)
    rig = Node("Rig", Transform(translation=[20, 0, 0]), children=[head])
    scene = Scene(nodes=[rig, Node("Marker", Transform(translation=[0, 0, 50]))], meters_per_unit=0.01)
    scene.animations = [Animation("talk", [Channel(mesh, "smile", 0, [0.0, 1.0], [0.0, 4.0])], fps=24)]
    return scene


def objects(path) -> dict:
    return {o["name"]: o for o in run_in_blender("inspect_blend", {"path": str(path)})["objects"]}


def test_write_mesh_and_shape_keys(blender, tmp_path):
    scene = face_scene()
    with pytest.warns(UserWarning, match="doesn't write animation"):
        sceneport.save(scene, tmp_path / "face.blend", blender=blender)
    written = objects(tmp_path / "face.blend")
    assert {(o["name"], o["type"], o["parent"]) for o in written.values()} == {
        ("Rig", "EMPTY", None), ("Head", "MESH", "Rig"), ("Marker", "EMPTY", None)}  # fmt: skip
    head = written["Head"]
    assert head["rotation_mode"] == "YXZ"  # ZXY with Y-up's y, z turned into Z-up's z, -y

    # world points: Blender's = M @ ours, with M the Y-up centimeter -> Z-up meter change
    m = 0.01 * basis_matrix(scene, "+Z", "-Y", "+X")
    ours = scene.world_matrices()[scene.find("Head")]
    mesh = scene.find("Head").mesh
    expected = (mesh.vertices @ ours[:3, :3].T + ours[:3, 3]) @ m.T
    world = head["matrix_world"]
    np.testing.assert_allclose(head["vertices"] @ world[:3, :3].T + world[:3, 3], expected, atol=1e-6)
    np.testing.assert_array_equal(head["face_vertex_counts"], [4, 3])
    np.testing.assert_array_equal(head["face_vertex_indices"], mesh.face_vertex_indices)
    np.testing.assert_allclose(head["uvs"]["UVMap"], mesh.uvs["UVMap"].expanded(), atol=1e-6)
    assert head["custom_normals"]
    world_normals = head["corner_normals"] @ world[:3, :3].T
    expected_normals = (mesh.normals.values[mesh.face_vertex_indices] @ ours[:3, :3].T) @ (m / 0.01).T
    np.testing.assert_allclose(world_normals, expected_normals, atol=1e-4)

    keys = {k["name"]: k for k in head["shape_keys"]}
    assert list(keys) == ["Basis", "brow", "smile", "open_0.5", "open"]  # the in-between split as for FBX
    assert (keys["brow"]["value"], keys["brow"]["slider_min"], keys["brow"]["slider_max"]) == (-2.5, -3.0, 1.0)
    assert (keys["smile"]["slider_min"], keys["smile"]["slider_max"]) == (0.0, 4.0)  # animated up to 4
    assert all(k["relative_key"] == "Basis" for k in keys.values())
    to_blender_mesh = np.linalg.inv(world[:3, :3]) @ m @ ours[:3, :3]  # offsets: our mesh space -> Blender's
    np.testing.assert_allclose(keys["brow"]["points"] - keys["Basis"]["points"],
                               np.array([[0, 0, 0]] * 4 + [[0, 2, 0]]) @ to_blender_mesh.T, atol=1e-6)


def test_shape_ranges_and_limit(blender, tmp_path):
    mesh = Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2], blend_shapes=[
        BlendShape("a", np.ones((3, 3)), range=(-3, 3)),
        BlendShape("narrow", np.ones((3, 3)), weight=0.2, range=(0.1, 0.5)),  # kept as given: no rounding
        BlendShape("big", np.ones((3, 3)), weight=12.0),
    ])  # fmt: skip
    scene = Scene(nodes=[Node("Tri", mesh=mesh)])
    with pytest.warns(UserWarning, match="beyond Blender's -10..10"):
        sceneport.save(scene, tmp_path / "tri.blend", blender=blender)
    keys = {k["name"]: k for k in objects(tmp_path / "tri.blend")["Tri"]["shape_keys"]}
    assert (keys["a"]["slider_min"], keys["a"]["slider_max"]) == (-3.0, 3.0)
    assert (keys["narrow"]["slider_min"], keys["narrow"]["slider_max"]) == pytest.approx((0.1, 0.5))
    assert (keys["big"]["slider_min"], keys["big"]["slider_max"], keys["big"]["value"]) == (0.0, 10.0, 10.0)


def test_blend_shape_range_validation():
    assert BlendShape("s", np.zeros((1, 3))).range == (0.0, 1.0)
    with pytest.raises(ValueError, match="min < max"):
        BlendShape("s", np.zeros((1, 3)), range=(1, 1))


def test_reading_blend_isnt_supported_yet(tmp_path):
    with pytest.raises(NotImplementedError):
        sceneport.load(tmp_path / "x.blend")


# -- armatures and skins ---------------------------------------------------------------------


def rig_scene() -> Scene:
    """A Y-up centimeter rig: Rig > Root > Spine > Head (+ a Hat under Head) and a mirrored LeftArm with a
    scaled bind, skinning a Body mesh (with a shape key) that moved after binding; posed away from the bind."""
    head = Node("Head", Transform(translation=[0, 30, 0]), joint=True,
                children=[Node("Hat", Transform(translation=[0, 12, 0], rotation=[0, 0, 10]))])  # fmt: skip
    spine = Node("Spine", Transform(translation=[0, 40, 0]), joint=True, children=[head])
    arm = Node("LeftArm", Transform(translation=[15, 30, 0], scale=[-1, 1, 1]), joint=True)
    root = Node("Root", Transform(translation=[0, 90, 0]), joint=True, children=[spine, arm])
    rig = Node("Rig", Transform(translation=[50, 0, 0], rotation=[0, 30, 0], scale=[2, 2, 2]), children=[root])
    y = np.linspace(80, 170, 10)
    vertices = np.column_stack([np.where(np.arange(10) % 2, 10.0, -10.0), y, np.zeros(10)])
    faces = np.array([[i, i + 1, i + 3, i + 2] for i in range(0, 8, 2)]).ravel()
    body = Mesh("body", vertices, [4] * 4, faces, blend_shapes=[BlendShape("bulge", np.tile([0, 0, 5.0], (10, 1)), weight=0.5)])
    body_node = Node("Body", Transform(translation=[50, 0, 0], rotation=[0, 30, 0], scale=[2, 2, 2]), mesh=body)
    scene = Scene(nodes=[rig, body_node], meters_per_unit=0.01)

    joints = [root, spine, head, arm]
    world = scene.world_matrices()
    inverse_bind = np.stack([np.linalg.inv(world[j]) for j in joints])
    inverse_bind[3] = np.linalg.inv(world[arm] @ np.diag([1.5, 1.5, 1.5, 1.0]))  # bound while scaled
    weights = np.clip(1 - np.abs(y[:, None] - np.array([90, 130, 160, 120])[None, :]) / 40, 0, None)
    weights /= weights.sum(axis=1, keepdims=True)
    from sceneport.scene import Skin

    body.skin = Skin(joints, np.tile(np.arange(4), (10, 1)), weights, inverse_bind, geom_bind=world[body_node])
    # the current pose differs from the bind pose, and the body moved after binding
    spine.transform.rotation = np.array([0, 0, 25.0])
    head.transform.rotation = np.array([20, 0, 0])
    root.transform.translation = np.array([0, 95, 5.0])
    body_node.transform.translation = np.array([60, 0, 0])
    return scene


def expected_world_points(scene: Scene, node: Node) -> np.ndarray:
    """Our deformed world-space points, in Blender's Z-up meters."""
    mesh = node.mesh
    if mesh.skin is not None:
        points = mesh.skinned_vertices(mesh.skin.joint_world(scene))
    else:
        w = scene.world_matrices()[node]
        points = mesh.deformed_vertices() @ w[:3, :3].T + w[:3, 3]
    return points @ (scene.meters_per_unit * basis_matrix(scene, "+Z", "-Y", "+X")).T


def test_write_armature_and_skin(blender, tmp_path):
    scene = rig_scene()
    sceneport.save(scene, tmp_path / "rig.blend", blender=blender)
    written = objects(tmp_path / "rig.blend")
    assert {(o["name"], o["type"], o["parent"], o["parent_bone"]) for o in written.values()} == {
        ("Rig", "ARMATURE", None, None), ("Body", "MESH", "Rig", None), ("Hat", "EMPTY", "Rig", "Head")}  # fmt: skip
    bones = {b["name"]: b for b in written["Rig"]["bones"]}
    assert {n: b["parent"] for n, b in bones.items()} == {"Root": None, "Spine": "Root", "Head": "Spine", "LeftArm": "Root"}
    assert all(b["length"] > 0 for b in bones.values())
    assert all(np.linalg.det(b["rest"][:3, :3]) > 0 for b in bones.values())  # the mirrored arm was unmirrored
    body = written["Body"]
    assert body["vertex_groups"] == ["Root", "Spine", "Head", "LeftArm"]
    assert body["modifiers"] == [("ARMATURE", "Rig")]
    np.testing.assert_allclose(body["world_vertices"], expected_world_points(scene, scene.find("Body")), atol=1e-5)

    m = np.eye(4)
    m[:3, :3] = 0.01 * basis_matrix(scene, "+Z", "-Y", "+X")
    hat = scene.world_matrices()[scene.find("Hat")]
    np.testing.assert_allclose(written["Hat"]["matrix_world"], m @ hat @ np.linalg.inv(m), atol=1e-6)
    head = scene.world_matrices()[scene.find("Head")]  # posed bone heads where our joints are
    np.testing.assert_allclose((written["Rig"]["matrix_world"] @ bones["Head"]["pose"])[:3, 3],
                               (m @ head)[:3, 3], atol=1e-6)  # fmt: skip


def test_write_human_female(blender, sample, tmp_path):
    path = sample("usdskel_examples") / "HumanFemale" / "HumanFemale.walk.usd"
    scene = sceneport.load(path)
    with pytest.warns(UserWarning):  # animation, materials ... aren't written yet
        sceneport.save(scene, tmp_path / "human.blend", blender=blender)
    written = objects(tmp_path / "human.blend")
    armatures = [o for o in written.values() if o["type"] == "ARMATURE"]
    assert len(armatures) == 1 and len(armatures[0]["bones"]) == 66
    checked, seen = 0, {}
    for node, _ in scene.walk():  # objects are created in this order; Blender renames repeats to name.001, ...
        if node.joint:
            continue
        count = seen[node.name] = seen.get(node.name, -1) + 1
        name = node.name if count == 0 else f"{node.name}.{count:03d}"
        if node.mesh is not None and node.mesh.skin is not None:
            got = written[name]["world_vertices"]
            np.testing.assert_allclose(got, expected_world_points(scene, node), atol=1e-5, err_msg=node.name)
            checked += 1
    assert checked == 89


def test_top_level_skeleton_gets_an_armature(blender, tmp_path):
    """A skeleton at the top level with its mesh beside it (e.g. MHR's FBX): both go under a new armature
    object named after the root joint, and the mesh still deforms the same."""
    scene = rig_scene()
    rig = scene.nodes[0]
    scene.nodes = [*rig.children, scene.find("Body")]  # Root joint and Body at the top level
    root = scene.nodes[0]
    root.transform.translation = np.array([50, 95, 5.0])
    sceneport.save(scene, tmp_path / "top.blend", blender=blender)
    written = objects(tmp_path / "top.blend")
    assert {(o["name"], o["type"], o["parent"]) for o in written.values()} == {
        ("Root", "ARMATURE", None), ("Body", "MESH", "Root"), ("Hat", "EMPTY", "Root")}  # fmt: skip
    assert written["Root"]["bones"][0]["name"] == "Root"  # the root bone keeps its name too
    np.testing.assert_allclose(written["Body"]["world_vertices"], expected_world_points(scene, scene.find("Body")), atol=1e-5)
