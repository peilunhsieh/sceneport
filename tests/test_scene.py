import numpy as np
import pytest

import sceneport
from sceneport.interop.fbx.binary import FBXNode, Prop, dumps, loads
from sceneport.interop.fbx.document import ROOT_ID, FBXDocument
from sceneport.math import euler_matrix
from sceneport.interop.fbx import document_from_scene, scene_from_document
from sceneport.scene import Mesh, Node, Primvar, Scene, Transform
from helpers import FIXTURES, mixed_mesh

CUBE = FIXTURES / "cube.fbx"


def roundtrip(scene: Scene) -> Scene:
    return scene_from_document(FBXDocument.from_file(loads(dumps(document_from_scene(scene).to_file()))))


def assert_primvars_equal(a: Primvar | None, b: Primvar | None) -> None:
    assert (a is None) == (b is None)
    if a is None:
        return
    assert a.interpolation == b.interpolation
    np.testing.assert_array_equal(a.values, b.values)
    assert (a.indices is None) == (b.indices is None)
    if a.indices is not None:
        np.testing.assert_array_equal(a.indices, b.indices)


def assert_meshes_equal(a: Mesh, b: Mesh) -> None:
    assert a.name == b.name
    np.testing.assert_array_equal(a.vertices, b.vertices)
    np.testing.assert_array_equal(a.face_vertex_counts, b.face_vertex_counts)
    np.testing.assert_array_equal(a.face_vertex_indices, b.face_vertex_indices)
    assert_primvars_equal(a.normals, b.normals)
    assert a.uvs.keys() == b.uvs.keys()
    for k in a.uvs:
        assert_primvars_equal(a.uvs[k], b.uvs[k])


def assert_transforms_equal(a: Transform, b: Transform) -> None:
    for name in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


# -- cube fixture -------------------------------------------------------------


def test_load_cube():
    scene = sceneport.load(CUBE)
    assert [n.name for n in scene.nodes] == ["Cube"]
    node = scene.nodes[0]
    mesh = node.mesh
    assert mesh is scene.meshes[0]
    assert (mesh.n_vertices, mesh.n_faces, mesh.n_corners) == (8, 6, 24)
    assert (mesh.face_vertex_counts == 4).all()
    assert mesh.normals.interpolation == "faceVarying"
    assert list(mesh.uvs) == ["UVMap"]
    np.testing.assert_allclose(node.transform.rotation, [-90, 0, 0], atol=1e-4)
    np.testing.assert_array_equal(node.transform.scale, [100, 100, 100])
    np.testing.assert_array_equal(node.transform.translation, [0, 0, 0])  # template default
    assert scene.meters_per_unit == 0.01
    assert (scene.up_axis, scene.front_axis, scene.right_axis) == ("+Y", "+Z", "+X")


def test_cube_normals_are_flat():
    mesh = sceneport.load(CUBE).meshes[0]
    normals = mesh.corner_values(mesh.normals).reshape(6, 4, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1)
    assert (normals == normals[:, :1]).all()  # one normal per face
    # each face normal is perpendicular to the face's edges
    face_verts = mesh.vertices[mesh.face_vertex_indices].reshape(6, 4, 3)
    edges = face_verts[:, 1] - face_verts[:, 0]
    np.testing.assert_allclose((edges * normals[:, 0]).sum(-1), 0, atol=1e-6)


def test_cube_roundtrip(tmp_path):
    scene = sceneport.load(CUBE)
    sceneport.save(scene, tmp_path / "cube.fbx")
    back = sceneport.load(tmp_path / "cube.fbx")
    assert_meshes_equal(scene.meshes[0], back.meshes[0])
    assert_transforms_equal(scene.nodes[0].transform, back.nodes[0].transform)
    assert back.meters_per_unit == scene.meters_per_unit


# -- mesh ---------------------------------------------------------------------


def test_triangulate_mixed():
    mesh = mixed_mesh()
    tris = mesh.triangles()
    assert tris.shape == (1 + 2 + 3, 3)
    np.testing.assert_array_equal(tris[:3], [[0, 1, 3], [1, 4, 5], [1, 5, 2]])
    corners = mesh.triangulate_corners()
    np.testing.assert_array_equal(mesh.face_vertex_indices[corners], tris)


def test_corner_values_interpolations():
    mesh = mixed_mesh()
    assert mesh.corner_values(mesh.normals).shape == (12, 3)
    np.testing.assert_array_equal(mesh.corner_values(mesh.uvs["UVMap"]), mesh.vertices[mesh.face_vertex_indices, :2] / 3)
    uniform = Primvar(np.array([[1.0], [2.0], [3.0]]), "uniform")
    np.testing.assert_array_equal(mesh.corner_values(uniform)[:, 0], [1] * 3 + [2] * 4 + [3] * 5)
    constant = Primvar(np.array([[7.0]]), "constant")
    assert (mesh.corner_values(constant) == 7).all()


def test_mixed_mesh_roundtrip():
    scene = Scene()
    scene.add(Node("Mixed", mesh=mixed_mesh()))
    back = roundtrip(scene)
    assert_meshes_equal(scene.meshes[0], back.meshes[0])
    assert back.nodes[0].transform.matrix().tolist() == np.eye(4).tolist()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda m: setattr(m, "face_vertex_counts", np.array([3, 4, 4], np.int32)), "add up"),
        (lambda m: setattr(m, "face_vertex_counts", np.array([2, 5, 5], np.int32)), "at least 3"),
        (lambda m: m.face_vertex_indices.__setitem__(0, 99), "out of range"),
        (lambda m: setattr(m, "normals", Primvar(np.zeros((3, 3)), "vertex")), "normals"),
    ],
)
def test_validate(change, message):
    mesh = mixed_mesh()
    mesh.validate()
    change(mesh)
    with pytest.raises(ValueError, match=message):
        mesh.validate()


# -- hierarchy and transforms ------------------------------------------------


def test_hierarchy_and_instancing():
    mesh = mixed_mesh()
    scene = Scene()
    root = scene.add(Node("Root", Transform(translation=[1, 0, 0])))
    root.add(Node("A", Transform(rotation=[0, 0, 90]), mesh=mesh))
    root.add(Node("B", Transform(scale=[2, 2, 2]), mesh=mesh))

    doc = document_from_scene(scene)
    assert len(list(doc.iter_objects("Geometry"))) == 1  # shared mesh written once
    assert [o.subtype for o in doc.iter_objects("Model")] == ["Null", "Mesh", "Mesh"]

    back = roundtrip(scene)
    root_b = back.nodes[0]
    assert [c.name for c in root_b.children] == ["A", "B"]
    assert root_b.children[0].mesh is root_b.children[1].mesh
    worlds = {n.name: w for n, w in back.walk()}
    np.testing.assert_allclose(worlds["A"] @ [1, 0, 0, 1], [1, 1, 0, 1], atol=1e-12)
    np.testing.assert_allclose(worlds["B"] @ [1, 0, 0, 1], [3, 0, 0, 1])


def test_euler_order():
    rx, rz = euler_matrix([90, 0, 0]), euler_matrix([0, 0, 90])
    np.testing.assert_allclose(euler_matrix([90, 0, 90], "XYZ"), rz @ rx, atol=1e-12)
    np.testing.assert_allclose(euler_matrix([90, 0, 90], 4), rx @ rz, atol=1e-12)  # 4 = ZXY: Z first


def test_rotation_pivot():
    t = Transform(rotation=[0, 0, 90], rotation_pivot=[1, 0, 0])
    np.testing.assert_allclose(t.matrix() @ [1, 0, 0, 1], [1, 0, 0, 1], atol=1e-12)  # pivot stays put
    np.testing.assert_allclose(t.matrix() @ [2, 0, 0, 1], [1, 1, 0, 1], atol=1e-12)


def test_rotation_active_roundtrip():
    t = Transform(rotation=[10, 20, 30], rotation_order=5, pre_rotation=[-90, 0, 0], scaling_pivot=[0, 1, 0])
    scene = Scene(nodes=[Node("N", t)])
    model = next(document_from_scene(scene).iter_objects("Model"))
    assert model.props.get("RotationActive") == 1
    assert_transforms_equal(roundtrip(scene).nodes[0].transform, t)


def test_pre_rotation_ignored_without_rotation_active():
    doc = FBXDocument()
    model = doc.add_object("Model", "N", "Null")
    model.props.set("PreRotation", "Vector3D", (-90, 0, 0))
    model.props.set("RotationOrder", "enum", 5)
    doc.connect(model, ROOT_ID)
    t = scene_from_document(doc).nodes[0].transform
    np.testing.assert_array_equal(t.pre_rotation, 0)
    assert t.rotation_order == 0


def test_unterminated_polygon_warns():
    doc = FBXDocument()
    model = doc.add_object("Model", "N", "Mesh")
    geom = doc.add_object(
        "Geometry",
        "G",
        "Mesh",
        children=[
            FBXNode("Vertices", [Prop("d", np.zeros(9))]),
            FBXNode("PolygonVertexIndex", [Prop("i", np.array([0, 1, 2], np.int32))]),
        ],
    )
    doc.connect(model, ROOT_ID)
    doc.connect(geom, model)
    with pytest.warns(UserWarning, match="doesn't end"):
        mesh = scene_from_document(doc).meshes[0]
    np.testing.assert_array_equal(mesh.face_vertex_counts, [3])
    np.testing.assert_array_equal(mesh.face_vertex_indices, [0, 1, 2])


@pytest.mark.parametrize("scale", [(0.0, 0.0, 2.0), (0.0, 1.5, 0.7), (3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 2.0)])
def test_decompose_zero_scale_recomposes(scale):
    """A node squashed to zero along some axes (e.g. hidden by animation) must recompose to the same matrix."""
    from sceneport.math import decompose_matrix, euler_matrix

    matrix = Transform(translation=[0.3, -0.5, 0.7], rotation=[25, -60, 140], scale=scale).matrix()
    translation, rotation, decomposed_scale = decompose_matrix(matrix)
    back = Transform(translation=translation, rotation=rotation, scale=decomposed_scale).matrix()
    np.testing.assert_allclose(back, matrix, atol=1e-12)
    batched = decompose_matrix(np.stack([matrix, np.eye(4)]))  # degenerate and regular matrices together
    np.testing.assert_allclose(batched[2], [decomposed_scale, [1, 1, 1]], atol=1e-12)
