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
    return {o["name"]: o for o in run_in_blender("inspect_blend", {"path": str(path)})}


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


def test_slider_range_option_and_limit(blender, tmp_path):
    mesh = Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2],
                blend_shapes=[BlendShape("a", np.ones((3, 3))), BlendShape("big", np.ones((3, 3)), weight=12.0)])  # fmt: skip
    scene = Scene(nodes=[Node("Tri", mesh=mesh)])
    with pytest.warns(UserWarning, match="beyond Blender's -10..10"):
        sceneport.save(scene, tmp_path / "tri.blend", blender=blender, slider_range=(-3, 3))
    keys = {k["name"]: k for k in objects(tmp_path / "tri.blend")["Tri"]["shape_keys"]}
    assert (keys["a"]["slider_min"], keys["a"]["slider_max"]) == (-3.0, 3.0)
    assert (keys["big"]["slider_max"], keys["big"]["value"]) == (10.0, 10.0)


def test_reading_blend_isnt_supported_yet(tmp_path):
    with pytest.raises(NotImplementedError):
        sceneport.load(tmp_path / "x.blend")
