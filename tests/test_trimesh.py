import numpy as np
import pytest
import trimesh

import fbxio
from fbxio.interop.trimesh import VERTEX_MAP_KEY, from_trimesh, from_trimesh_scene, to_trimesh, to_trimesh_scene
from fbxio.math import decompose_matrix, euler_matrix, scale_matrix, translation_matrix
from fbxio.scene import Node, Scene, Transform
from helpers import FIXTURES, mixed_mesh

CUBE = FIXTURES / "cube.fbx"


@pytest.fixture
def cube_mesh():
    return fbxio.load(CUBE).meshes[0]


def test_cube_split_seams(cube_mesh):
    tm = to_trimesh(cube_mesh)
    assert tm.vertices.shape == (24, 3)  # 4 per face: hard edges + UV seams
    assert tm.faces.shape == (12, 3)
    vmap = tm.metadata[VERTEX_MAP_KEY]
    np.testing.assert_array_equal(tm.vertices, cube_mesh.vertices[vmap])

    # every triangle corner carries the same normal and UV as the FBX corner it came from
    corners = cube_mesh.triangulate_corners()
    np.testing.assert_array_equal(tm.vertices[tm.faces], cube_mesh.vertices[cube_mesh.face_vertex_indices[corners]])
    np.testing.assert_allclose(tm.vertex_normals[tm.faces], cube_mesh.corner_values(cube_mesh.normals)[corners])
    np.testing.assert_array_equal(tm.visual.uv[tm.faces], cube_mesh.corner_values(cube_mesh.uvs["UVMap"])[corners])


def test_cube_keep_vertices(cube_mesh):
    with pytest.warns(UserWarning, match="UVs have seams"):
        tm = to_trimesh(cube_mesh, split_seams=False)
    np.testing.assert_array_equal(tm.vertices, cube_mesh.vertices)
    np.testing.assert_array_equal(tm.metadata[VERTEX_MAP_KEY], np.arange(8))
    assert tm.is_watertight
    assert getattr(tm.visual, "uv", None) is None
    # averaged corner normals point outwards from the cube's center
    np.testing.assert_allclose(np.linalg.norm(tm.vertex_normals, axis=1), 1)
    assert (np.sum(tm.vertex_normals * tm.vertices, axis=1) > 0).all()


def test_no_seams_keeps_order():
    mesh = mixed_mesh()
    mesh.uvs.pop("Second")  # the faceVarying set has seams
    tm = to_trimesh(mesh)
    np.testing.assert_array_equal(tm.vertices, mesh.vertices)
    np.testing.assert_array_equal(tm.faces, mesh.triangles())
    np.testing.assert_array_equal(tm.visual.uv, mesh.uvs["UVMap"].values)


def test_uv_set_choice():
    mesh = mixed_mesh()
    tm = to_trimesh(mesh, uv_set="Second")
    corners = mesh.triangulate_corners()
    np.testing.assert_array_equal(tm.visual.uv[tm.faces], mesh.corner_values(mesh.uvs["Second"])[corners])


def test_from_trimesh_fbx_roundtrip(tmp_path):
    box = trimesh.creation.box(extents=[1, 2, 3])
    mesh = from_trimesh(box, "Box")
    assert mesh.n_faces == len(box.faces) and (mesh.face_vertex_counts == 3).all()

    scene = Scene(nodes=[Node("Box", mesh=mesh)])
    fbxio.save(scene, tmp_path / "box.fbx")
    back = to_trimesh(fbxio.load(tmp_path / "box.fbx").meshes[0], split_seams=False)
    np.testing.assert_array_equal(back.vertices, box.vertices)
    np.testing.assert_array_equal(back.faces, box.faces)
    np.testing.assert_allclose(back.vertex_normals, box.vertex_normals)
    assert back.is_watertight and np.isclose(back.volume, 6)


def test_from_trimesh_uvs():
    tm = trimesh.creation.box()
    tm.visual = trimesh.visual.TextureVisuals(uv=tm.vertices[:, :2] + 0.5)
    mesh = from_trimesh(tm)
    np.testing.assert_array_equal(mesh.uvs["UVMap"].values, tm.visual.uv)
    assert mesh.uvs["UVMap"].interpolation == "vertex"


def test_scene_roundtrip_with_instancing():
    mesh = mixed_mesh()
    scene = Scene()
    parent = scene.add(Node("Parent", Transform(translation=[0, 0, 5], scale=[2, 2, 2])))
    parent.add(Node("A", Transform(rotation=[0, 0, 90]), mesh=mesh))
    parent.add(Node("B", Transform(translation=[1, 0, 0]), mesh=mesh))

    tscene = to_trimesh_scene(scene)
    assert len(tscene.geometry) == 1
    assert sorted(tscene.graph.nodes_geometry) == ["A", "B"]
    expected = {n.name: w for n, w in scene.walk()}
    for name in ("A", "B"):
        np.testing.assert_allclose(tscene.graph[name][0], expected[name], atol=1e-12)

    back = from_trimesh_scene(tscene)
    by_name = {n.name: n for n in back.nodes}
    assert by_name["A"].mesh is by_name["B"].mesh
    for name in ("A", "B"):
        np.testing.assert_allclose(by_name[name].transform.matrix(), expected[name], atol=1e-12)


def test_geometric_transform_applied():
    node = Node("G", Transform(geometric_translation=[0, 3, 0]), mesh=mixed_mesh())
    tscene = to_trimesh_scene(Scene(nodes=[node]))
    np.testing.assert_allclose(tscene.graph["G"][0][:3, 3], [0, 3, 0])


@pytest.mark.parametrize(
    "t, r, s",
    [
        ([1, 2, 3], [10, 20, 30], [1, 2, 3]),
        ([0, 0, 0], [-170, 80, 45], [0.5, 0.5, 0.5]),
        ([0, 0, 0], [30, 90, 0], [1, 1, 1]),  # gimbal lock
        ([5, 0, 0], [0, 0, 0], [-1, 1, 1]),  # mirrored
    ],
)
def test_decompose_matrix(t, r, s):
    m = translation_matrix(t) @ euler_matrix(r) @ scale_matrix(s)
    t2, r2, s2 = decompose_matrix(m)
    np.testing.assert_allclose(translation_matrix(t2) @ euler_matrix(r2) @ scale_matrix(s2), m, atol=1e-9)
