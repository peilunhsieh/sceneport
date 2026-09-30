import numpy as np
import pytest

import sceneport
from sceneport.scene import Mesh, Node, Primvar, Scene, Transform
from helpers import mixed_mesh, rig_scene, shaped_mesh


def one_mesh(scene: Scene) -> Mesh:
    (mesh,) = scene.meshes
    return mesh


def assert_same_geometry(a: Mesh, b: Mesh, atol: float = 1e-7) -> None:
    np.testing.assert_allclose(b.vertices, a.vertices, atol=atol)
    np.testing.assert_array_equal(b.face_vertex_counts, a.face_vertex_counts)
    np.testing.assert_array_equal(b.face_vertex_indices, a.face_vertex_indices)
    a_uv, b_uv = next(iter(a.uvs.values())), next(iter(b.uvs.values()))
    np.testing.assert_allclose(b.corner_values(b_uv), a.corner_values(a_uv), atol=atol)


def single_uv_mesh() -> Mesh:
    mesh = mixed_mesh()
    del mesh.uvs["UVMap"]  # keep the per-corner ("faceVarying", indexed) set only
    return mesh


# -- OBJ ----------------------------------------------------------------------


def test_obj_roundtrip(tmp_path):
    mesh = single_uv_mesh()
    sceneport.save(Scene(nodes=[Node("Mixed", mesh=mesh)]), tmp_path / "m.obj")
    back = one_mesh(sceneport.load(tmp_path / "m.obj"))
    assert back.name == "Mixed"
    assert_same_geometry(mesh, back)
    assert back.normals.interpolation == "vertex"  # vn indexed like v comes back per vertex
    np.testing.assert_allclose(back.normals.values, mesh.normals.values)
    np.testing.assert_array_equal(back.uvs["UVMap"].indices, mesh.uvs["Second"].indices)  # indexing kept


def test_obj_bakes_world_transforms_and_keeps_objects(tmp_path):
    a = Node("Left part", Transform(translation=[10, 0, 0]), mesh=single_uv_mesh())
    b = Node("B", Transform(scale=[2, 2, 2]), mesh=single_uv_mesh())
    sceneport.save(Scene(nodes=[Node("root", Transform(translation=[0, 1, 0]), children=[a, b])]), tmp_path / "s.obj")
    text = (tmp_path / "s.obj").read_text()
    assert "o Left_part\n" in text and "o B\n" in text  # OBJ names are single tokens
    back = sceneport.load(tmp_path / "s.obj")
    assert [n.name for n in back.nodes] == ["Left_part", "B"]
    np.testing.assert_allclose(back.nodes[0].mesh.vertices, a.mesh.vertices + [10, 1, 0], atol=1e-7)
    np.testing.assert_allclose(back.nodes[1].mesh.vertices, a.mesh.vertices * 2 + [0, 1, 0], atol=1e-7)
    np.testing.assert_array_equal(back.nodes[1].mesh.face_vertex_indices, a.mesh.face_vertex_indices)


def test_obj_parsing_variants(tmp_path):
    (tmp_path / "v.obj").write_text(
        "# comment\nmtllib x.mtl\ng first\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\n"
        "usemtl m\ns off\nf -4//1 -3//1 -2//1 -1//1  # relative indices\n"
        "g second\nv 5 5 5\nv 6 5 5\nv 6 6 5\nf 5 6 7\n"
    )
    with pytest.warns(UserWarning, match="not found|not in any material library"):  # x.mtl doesn't exist
        back = sceneport.load(tmp_path / "v.obj")
    assert [n.name for n in back.nodes] == ["first", "second"]  # g groups when there is no o
    assert [m.name for m in back.nodes[1].materials] == ["m"]  # usemtl carries over to the next group
    quad, tri = back.nodes[0].mesh, back.nodes[1].mesh
    np.testing.assert_array_equal(quad.face_vertex_indices, [0, 1, 2, 3])
    assert quad.normals.interpolation == "vertex"  # each vertex has a single normal, so per vertex is lossless
    np.testing.assert_allclose(quad.normals.values, [[0, 0, 1]] * 4)
    np.testing.assert_array_equal(tri.face_vertex_indices, [0, 1, 2])  # renumbered per group
    np.testing.assert_allclose(tri.vertices[0], [5, 5, 5])
    assert tri.normals is None and not tri.uvs


def test_obj_warns_about_dropped_data(tmp_path):
    with pytest.warns(UserWarning, match="blend shapes, skins"):
        sceneport.save(rig_scene(), tmp_path / "rig.obj")
    mesh = one_mesh(sceneport.load(tmp_path / "rig.obj"))
    expected = rig_scene().meshes[0].deformed_vertices() + [0, 0, 0.5]  # default blend weights, posed in world
    np.testing.assert_allclose(mesh.vertices, expected, atol=1e-7)
    with pytest.warns(UserWarning, match="one UV set"):
        sceneport.save(Scene(nodes=[Node("m", mesh=mixed_mesh())]), tmp_path / "two_uvs.obj")


# -- PLY ----------------------------------------------------------------------


@pytest.mark.parametrize("ascii", [False, True])
def test_ply_roundtrip_mixed_faces(tmp_path, ascii):
    mesh = single_uv_mesh()  # triangle + quad + pentagon: rows of different lengths
    sceneport.save(Scene(nodes=[Node("m", mesh=mesh)]), tmp_path / "m.ply", ascii=ascii)
    head = (tmp_path / "m.ply").read_bytes()[:40]
    assert (b"format ascii" in head) == ascii
    back = one_mesh(sceneport.load(tmp_path / "m.ply"))
    assert_same_geometry(mesh, back, atol=1e-6)
    np.testing.assert_allclose(back.normals.values, mesh.normals.values, atol=1e-7)


def test_ply_uniform_faces_and_double(tmp_path):
    quad = Mesh("q", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], [4, 4], [0, 1, 2, 3, 1, 4, 5, 2])
    quad.uvs["UVMap"] = Primvar(quad.vertices[:, :2] / 3 + 0.1, "vertex")
    sceneport.save(Scene(nodes=[Node("q", mesh=quad)]), tmp_path / "q.ply", double=True)
    assert b"property double x" in (tmp_path / "q.ply").read_bytes()[:200]
    back = one_mesh(sceneport.load(tmp_path / "q.ply"))
    assert_same_geometry(quad, back, atol=1e-15)


def test_ply_joins_meshes_and_averages_corner_normals(tmp_path):
    corner = single_uv_mesh()
    corner.normals = Primvar(np.array([[0, 0, 1.0], [0, 1.0, 0]]), "faceVarying", np.array([0] * 7 + [1] * 5))
    scene = Scene(nodes=[Node("a", mesh=single_uv_mesh()), Node("b", Transform(translation=[5, 0, 0]), mesh=corner)])
    with pytest.warns(UserWarning) as record:
        sceneport.save(scene, tmp_path / "j.ply")
    messages = " ".join(str(w.message) for w in record)
    assert "joined" in messages and "averaged" in messages
    back = one_mesh(sceneport.load(tmp_path / "j.ply"))
    assert back.n_vertices == 16 and back.n_faces == 6
    np.testing.assert_allclose(back.vertices[8:], corner.vertices + [5, 0, 0], atol=1e-6)


def test_ply_reads_other_writers(tmp_path):
    # Big-endian, per-vertex s/t UVs, an extra element to skip, int face counts.
    header = (
        "ply\nformat binary_big_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "property float s\nproperty float t\nelement material 1\nproperty uchar id\n"
        "element face 1\nproperty list int int vertex_index\nend_header\n"
    ).encode()
    vertices = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype=">f4").tobytes()
    face = np.array([3], ">i4").tobytes() + np.array([0, 1, 2], ">i4").tobytes()
    (tmp_path / "be.ply").write_bytes(header + vertices + bytes([7]) + face)
    mesh = one_mesh(sceneport.load(tmp_path / "be.ply"))
    np.testing.assert_array_equal(mesh.face_vertex_indices, [0, 1, 2])
    np.testing.assert_allclose(mesh.vertices[1], [1, 0, 0])
    assert mesh.uvs["UVMap"].interpolation == "vertex"
    np.testing.assert_allclose(mesh.uvs["UVMap"].values[2], [0, 1])


def test_ply_needs_a_mesh(tmp_path):
    with pytest.raises(ValueError, match="at least one mesh"):
        sceneport.save(Scene(nodes=[Node("empty")]), tmp_path / "e.ply")


def test_convert_between_all_formats(tmp_path):
    from sceneport.cli import main

    scene = Scene(nodes=[Node("m", mesh=single_uv_mesh())])
    sceneport.save(scene, tmp_path / "m.fbx")
    main(["convert", str(tmp_path / "m.fbx"), str(tmp_path / "m.ply")])
    main(["convert", str(tmp_path / "m.ply"), str(tmp_path / "m.obj")])
    main(["convert", str(tmp_path / "m.obj"), str(tmp_path / "m.usda")])
    assert_same_geometry(single_uv_mesh(), one_mesh(sceneport.load(tmp_path / "m.usda")), atol=1e-6)


def test_blend_shapes_are_baked_at_default_weights(tmp_path):
    mesh = shaped_mesh()
    del mesh.uvs["Second"]
    with pytest.warns(UserWarning, match="blend shapes"):
        sceneport.save(Scene(nodes=[Node("m", mesh=mesh)]), tmp_path / "s.obj")
    np.testing.assert_allclose(one_mesh(sceneport.load(tmp_path / "s.obj")).vertices, mesh.deformed_vertices(), atol=1e-7)


def test_ply_vertex_uvs_split_seams(tmp_path):
    mesh = single_uv_mesh()  # per-corner UVs with seams
    mesh.normals = Primvar(np.array([[0, 0, 1.0], [0, 1.0, 0]]), "faceVarying", np.array([0] * 7 + [1] * 5))
    with pytest.warns(UserWarning, match="vertices added"):
        sceneport.save(Scene(nodes=[Node("m", mesh=mesh)]), tmp_path / "v.ply", uvs="vertex")
    header = (tmp_path / "v.ply").read_bytes().split(b"end_header")[0]
    assert b"property float s" in header and b"texcoord" not in header
    back = one_mesh(sceneport.load(tmp_path / "v.ply"))
    assert back.n_vertices > mesh.n_vertices and back.uvs["UVMap"].interpolation == "vertex"
    # Every corner keeps its position, UV and normal exactly (up to float32).
    np.testing.assert_allclose(back.vertices[back.face_vertex_indices], mesh.vertices[mesh.face_vertex_indices], atol=1e-6)
    np.testing.assert_allclose(back.corner_values(back.uvs["UVMap"]), mesh.corner_values(mesh.uvs["Second"]), atol=1e-6)
    np.testing.assert_allclose(back.corner_values(back.normals), mesh.corner_values(mesh.normals), atol=1e-6)


def test_ply_vertex_uvs_without_seams_keep_vertices(tmp_path):
    import warnings

    mesh = mixed_mesh()
    del mesh.uvs["Second"]  # per-vertex UVs: nothing to split
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sceneport.save(Scene(nodes=[Node("m", mesh=mesh)]), tmp_path / "n.ply", uvs="vertex")
    back = one_mesh(sceneport.load(tmp_path / "n.ply"))
    np.testing.assert_allclose(back.vertices, mesh.vertices, atol=1e-6)  # same count and order
    np.testing.assert_array_equal(back.face_vertex_indices, mesh.face_vertex_indices)
    np.testing.assert_allclose(back.uvs["UVMap"].values, mesh.uvs["UVMap"].values, atol=1e-6)
    with pytest.raises(ValueError, match="corner"):
        sceneport.save(Scene(nodes=[Node("m", mesh=mesh)]), tmp_path / "x.ply", uvs="face")


def test_ply_without_uvs(tmp_path):
    mesh = single_uv_mesh()
    sceneport.save(Scene(nodes=[Node("m", mesh=mesh)]), tmp_path / "n.ply", uvs=None)
    header = (tmp_path / "n.ply").read_bytes().split(b"end_header")[0]
    assert b"texcoord" not in header and b" s\n" not in header
    back = one_mesh(sceneport.load(tmp_path / "n.ply"))
    assert not back.uvs
    np.testing.assert_array_equal(back.face_vertex_indices, mesh.face_vertex_indices)  # vertices untouched
