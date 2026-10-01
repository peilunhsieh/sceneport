import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdSkel

import sceneport
from sceneport.interop.fbx import document_from_scene, scene_from_document
from sceneport.interop.fbx.binary import FBXNode, Prop, dumps, loads
from sceneport.interop.fbx.document import ROOT_ID, FBXDocument
from sceneport.interop.usd import scene_from_stage, stage_from_scene
from sceneport.scene import BlendShape, Inbetween, Node, Scene
from helpers import assert_blend_shapes_close, mixed_mesh, shaped_mesh


# These tests check FBX's own in-between layout, so they write real in-betweens (FBX export splits them by default).
def fbx_roundtrip(scene: Scene) -> Scene:
    return scene_from_document(FBXDocument.from_file(loads(dumps(document_from_scene(scene, split_inbetweens=False).to_file()))))


# -- model --------------------------------------------------------------------


def test_offsets_at_interpolates_through_inbetweens():
    shape = shaped_mesh().blend_shapes[0]  # in-between at 0.5: z = 0.25, 0.5; full: z = 1, 2
    z = lambda w: shape.offsets_at(w)[:, 2]  # noqa: E731
    np.testing.assert_allclose(z(0.25), [0.125, 0.25])
    np.testing.assert_allclose(z(0.75), [0.625, 1.25])
    np.testing.assert_allclose(z(1.5), [1.75, 3.5])  # extrapolates the last segment
    np.testing.assert_allclose(z(-1.0), [-0.5, -1.0])  # and the first


def test_deformed_vertices():
    mesh = shaped_mesh()
    expected = mesh.vertices + 0.5  # "Wide" at its default weight 1
    expected[[1, 2], 2] += [0.125, 0.25]  # "Smile.L" at its default weight 0.25
    np.testing.assert_allclose(mesh.deformed_vertices(), expected)
    np.testing.assert_allclose(mesh.deformed_vertices({"Smile.L": 0, "Wide": 0}), mesh.vertices)
    dense = mesh.blend_shapes[0].dense_offsets(mesh.n_vertices)
    assert dense.shape == (8, 3) and dense[1, 2] == 1 and not dense[0].any()


def test_offsets_at_array_matches_scalar():
    shape = shaped_mesh().blend_shapes[0]
    weights = np.array([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    batched = shape.offsets_at(weights)
    assert batched.shape == (7, 2, 3)
    for w, offsets in zip(weights, batched):
        np.testing.assert_allclose(offsets, shape.offsets_at(w))


def test_deformed_vertices_array_and_frames():
    mesh = shaped_mesh()
    w = np.array([0.75, -0.5])
    by_name = mesh.deformed_vertices({"Smile.L": 0.75, "Wide": -0.5})
    np.testing.assert_allclose(mesh.deformed_vertices(w), by_name)

    frames = np.array([[0.0, 0.0], [0.75, -0.5], [1.5, 2.0]])
    out = mesh.deformed_vertices(frames)
    assert out.shape == (3, 8, 3)
    np.testing.assert_allclose(out[0], mesh.vertices)
    for row, verts in zip(frames, out):
        np.testing.assert_allclose(verts, mesh.deformed_vertices(row))


@pytest.mark.parametrize(
    "weights, error, message",
    [
        (np.zeros(3), ValueError, r"\(S,\) or \(F, S\) with S = 2, got \(3,\)"),
        (np.zeros((2, 3)), ValueError, "S = 2"),
        (np.zeros((1, 2, 2)), ValueError, r"got \(1, 2, 2\)"),
        ({"Nope": 1.0}, KeyError, "Nope"),
    ],
)
def test_deformed_vertices_bad_weights(weights, error, message):
    with pytest.raises(error, match=message):
        shaped_mesh().deformed_vertices(weights)


def test_blend_shape_matrix():
    mesh = shaped_mesh()
    with pytest.raises(ValueError, match="Smile.L"):
        mesh.blend_shape_matrix()  # has an in-between
    mesh.blend_shapes[0].inbetweens = []
    basis = mesh.blend_shape_matrix()
    assert basis.shape == (2, 8, 3)
    np.testing.assert_array_equal(basis[0], mesh.blend_shapes[0].dense_offsets(8))
    frames = np.array([[0.3, -1.0], [2.0, 0.5]])
    linear = mesh.vertices + np.einsum("fs,snk->fnk", frames, basis)
    np.testing.assert_allclose(linear, mesh.deformed_vertices(frames))
    assert mesh.blend_shape_matrix(dtype=np.float32).dtype == np.float32


@pytest.mark.parametrize(
    "shape, message",
    [
        (BlendShape("s", np.zeros((2, 3)), indices=[0, 99]), "out of range"),
        (BlendShape("s", np.zeros((2, 3)), indices=[4, 4]), "duplicate"),
        (BlendShape("s", np.zeros((3, 3))), "offsets has 3 rows, expected 8"),
        (BlendShape("s", np.zeros((8, 3)), normal_offsets=np.zeros((2, 3))), "normal_offsets"),
        (BlendShape("s", np.zeros((8, 3)), inbetweens=[Inbetween(1.0, np.zeros((8, 3)))]), "not be 0 or 1"),
        (BlendShape("s", np.zeros((8, 3)), inbetweens=[Inbetween(0.5, np.zeros((2, 3)))]), "in-between"),
    ],
)
def test_validate(shape, message):
    mesh = mixed_mesh()
    mesh.blend_shapes = [shape]
    with pytest.raises(ValueError, match=message):
        mesh.validate()


# -- FBX ----------------------------------------------------------------------


def test_fbx_roundtrip(tmp_path):
    mesh = shaped_mesh()
    sceneport.save(Scene(nodes=[Node("Face", mesh=mesh)]), tmp_path / "face.fbx", split_inbetweens=False)
    back = sceneport.load(tmp_path / "face.fbx").meshes[0]
    assert_blend_shapes_close(mesh.blend_shapes, back.blend_shapes)
    np.testing.assert_allclose(back.deformed_vertices(), mesh.deformed_vertices())


def test_fbx_layout():
    doc = document_from_scene(Scene(nodes=[Node("Face", mesh=shaped_mesh())]), split_inbetweens=False)
    (geometry,) = doc.iter_objects("Geometry", "Mesh")
    (deformer,) = doc.children_of(geometry, "Deformer")
    assert (deformer.subtype, deformer.cls) == ("BlendShape", "Deformer")
    channels = doc.children_of(deformer, "Deformer")
    assert [(c.name, c.subtype, c.cls) for c in channels] == [
        ("Smile.L", "BlendShapeChannel", "SubDeformer"),
        ("Wide", "BlendShapeChannel", "SubDeformer"),
    ]
    assert channels[0].child_value("FullWeights").tolist() == [50.0, 100.0]
    assert channels[0].child_value("DeformPercent") == 25.0
    assert doc.get_property(channels[0], "DeformPercent") == 25.0
    shapes = doc.children_of(channels[0], "Geometry")
    assert [(s.name, s.subtype) for s in shapes] == [("Smile.L_50", "Shape"), ("Smile.L", "Shape")]
    assert shapes[1].child_value("Indexes").tolist() == [1, 2]
    assert doc.children_of(channels[1], "Geometry")[0].child_value("Indexes").tolist() == list(range(8))


def test_fbx_shared_mesh_writes_shapes_once():
    mesh = shaped_mesh()
    scene = Scene(nodes=[Node("A", mesh=mesh), Node("B", mesh=mesh)])
    doc = document_from_scene(scene, split_inbetweens=False)
    assert len(list(doc.iter_objects("Deformer", "BlendShape"))) == 1
    back = fbx_roundtrip(scene)
    assert back.nodes[0].mesh is back.nodes[1].mesh
    assert len(back.meshes[0].blend_shapes) == 2


def test_fbx_inbetween_with_other_vertices():
    """An in-between touching different vertices than the full shape is merged onto their union."""
    doc = FBXDocument()
    model = doc.add_object("Model", "M", "Mesh")
    geometry = doc.add_object("Geometry", "M", "Mesh", children=[
        FBXNode("Vertices", [Prop("d", np.zeros(12))]),
        FBXNode("PolygonVertexIndex", [Prop("i", np.array([0, 1, 2, -4], np.int32))]),
    ])  # fmt: skip
    deformer = doc.add_object("Deformer", "M", "BlendShape", cls="Deformer")
    channel = doc.add_object("Deformer", "S", "BlendShapeChannel", cls="SubDeformer", children=[
        FBXNode("FullWeights", [Prop("d", np.array([100.0, 40.0]))]),  # out of order on purpose
    ])  # fmt: skip
    full = doc.add_object("Geometry", "S", "Shape", children=[
        FBXNode("Indexes", [Prop("i", np.array([2], np.int32))]),
        FBXNode("Vertices", [Prop("d", np.array([0.0, 0.0, 1.0]))]),
    ])  # fmt: skip
    partial = doc.add_object("Geometry", "S40", "Shape", children=[
        FBXNode("Indexes", [Prop("i", np.array([0], np.int32))]),
        FBXNode("Vertices", [Prop("d", np.array([0.0, 0.5, 0.0]))]),
    ])  # fmt: skip
    for src, dst in [(model, ROOT_ID), (geometry, model), (deformer, geometry), (channel, deformer), (full, channel), (partial, channel)]:
        doc.connect(src, dst)

    (shape,) = scene_from_document(doc).meshes[0].blend_shapes
    np.testing.assert_array_equal(shape.indices, [0, 2])
    np.testing.assert_array_equal(shape.offsets, [[0, 0, 0], [0, 0, 1]])
    (inbetween,) = shape.inbetweens
    assert inbetween.weight == 0.4
    np.testing.assert_array_equal(inbetween.offsets, [[0, 0.5, 0], [0, 0, 0]])
    assert shape.weight == 0.0


def test_fbx_duplicate_connections_read_once():
    """FBX SDK files can repeat a connection; shapes must not be read twice."""
    doc = document_from_scene(Scene(nodes=[Node("Face", mesh=shaped_mesh())]), split_inbetweens=False)
    doc.connections += [c for c in doc.connections if c.src in {o.id for o in doc.iter_objects("Deformer")}]
    assert len(fbx_roundtrip_doc(doc).meshes[0].blend_shapes) == 2


def test_fbx_shape_names_follow_blender():
    """Single-target channels take the Shape geometry's name (MHR: channel shape_c_0, Shape shape_0);
    channels with in-betweens keep the channel name."""
    doc = document_from_scene(Scene(nodes=[Node("Face", mesh=shaped_mesh())]), split_inbetweens=False)
    for i, channel in enumerate(doc.iter_objects("Deformer", "BlendShapeChannel")):
        channel.name = f"shape_c_{i}"
        for target in doc.children_of(channel, "Geometry"):
            target.name = f"shape_{i}" if target.name == shaped_mesh().blend_shapes[i].name else "partial"
    shapes = fbx_roundtrip_doc(doc).meshes[0].blend_shapes
    assert [s.name for s in shapes] == ["shape_c_0", "shape_1"]  # "Smile.L" has an in-between


def fbx_roundtrip_doc(doc: FBXDocument) -> Scene:
    return scene_from_document(FBXDocument.from_file(loads(dumps(doc.to_file()))))


# -- USD ----------------------------------------------------------------------


def test_usd_roundtrip(tmp_path):
    mesh = shaped_mesh()
    sceneport.save(Scene(nodes=[Node("Face", mesh=mesh)]), tmp_path / "face.usdc")
    back = sceneport.load(tmp_path / "face.usdc")
    assert [n.name for n in back.nodes] == ["Face"]  # the Skeleton doesn't become a node
    assert_blend_shapes_close(mesh.blend_shapes, back.meshes[0].blend_shapes, rtol=1e-6)


def test_usd_layout_and_usdskel_evaluation():
    """USD's own skinning evaluation must match Mesh.deformed_vertices, for every instance."""
    mesh = shaped_mesh()
    scene = Scene(nodes=[Node("A", mesh=mesh), Node("B", mesh=mesh), Node("Plain", mesh=mixed_mesh())])
    stage = stage_from_scene(scene)
    assert stage.GetPrimAtPath("/A").GetTypeName() == "SkelRoot"
    assert stage.GetPrimAtPath("/Plain").GetTypeName() == "Xform"  # no shapes: no skeleton
    binding = UsdSkel.BindingAPI(stage.GetPrimAtPath("/A/Mixed"))
    assert list(binding.GetBlendShapesAttr().Get()) == ["Smile_L", "Wide"]
    assert stage.GetPrimAtPath("/A/Mixed/Smile_L").GetDisplayName() == "Smile.L"

    assert UsdSkel.BakeSkinning(stage.Traverse())
    for path in ("/A/Mixed", "/B/Mixed"):
        baked = np.array(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get(Usd.TimeCode.EarliestTime()))
        np.testing.assert_allclose(baked, mesh.deformed_vertices(), atol=1e-6, err_msg=path)


def test_usd_shared_mesh():
    mesh = shaped_mesh()
    back = scene_from_stage(stage_from_scene(Scene(nodes=[Node("A", mesh=mesh), Node("B", mesh=mesh)])))
    assert back.nodes[0].mesh is back.nodes[1].mesh
    assert len(back.meshes[0].blend_shapes) == 2


def test_fbx_to_usd_to_fbx(tmp_path):
    mesh = shaped_mesh()
    sceneport.save(Scene(nodes=[Node("Face", mesh=mesh)]), tmp_path / "a.fbx", split_inbetweens=False)
    sceneport.save(sceneport.load(tmp_path / "a.fbx"), tmp_path / "b.usda")
    sceneport.save(sceneport.load(tmp_path / "b.usda"), tmp_path / "c.fbx", split_inbetweens=False)
    assert_blend_shapes_close(mesh.blend_shapes, sceneport.load(tmp_path / "c.fbx").meshes[0].blend_shapes, rtol=1e-6)
