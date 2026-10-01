import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import sceneport
from sceneport.interop.usd import scene_from_stage, stage_from_scene
from sceneport.scene import Mesh, Node, Primvar, Scene, Transform
from helpers import FIXTURES, mixed_mesh

CUBE = FIXTURES / "cube.fbx"


def assert_primvars_close(a: Primvar | None, b: Primvar | None) -> None:
    assert (a is None) == (b is None)
    if a is None:
        return
    assert a.interpolation == b.interpolation
    np.testing.assert_allclose(a.values, b.values, rtol=1e-6)  # USD stores float32
    assert (a.indices is None) == (b.indices is None)
    if a.indices is not None:
        np.testing.assert_array_equal(a.indices, b.indices)


def assert_meshes_close(a: Mesh, b: Mesh) -> None:
    assert a.name == b.name
    np.testing.assert_allclose(a.vertices, b.vertices, rtol=1e-6)
    np.testing.assert_array_equal(a.face_vertex_counts, b.face_vertex_counts)
    np.testing.assert_array_equal(a.face_vertex_indices, b.face_vertex_indices)
    assert_primvars_close(a.normals, b.normals)
    assert list(a.uvs) == list(b.uvs)
    for k in a.uvs:
        assert_primvars_close(a.uvs[k], b.uvs[k])


def assert_transforms_equal(a: Transform, b: Transform) -> None:
    for name in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def roundtrip(scene: Scene) -> Scene:
    return scene_from_stage(stage_from_scene(scene))


# -- files --------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".usda", ".usdc", ".usd", ".usdz"])
def test_cube_via_file(tmp_path, suffix):
    scene = sceneport.load(CUBE)
    path = tmp_path / f"cube{suffix}"
    sceneport.save(scene, path)
    back = sceneport.load(path)
    assert [n.name for n in back.nodes] == ["Cube"]
    assert_meshes_close(scene.meshes[0], back.meshes[0])
    assert_transforms_equal(scene.nodes[0].transform, back.nodes[0].transform)
    assert back.meters_per_unit == scene.meters_per_unit
    assert back.up_axis == "+Y"


def test_fbx_usd_fbx(tmp_path):
    scene = sceneport.load(CUBE)
    sceneport.save(scene, tmp_path / "cube.usdc")
    sceneport.save(sceneport.load(tmp_path / "cube.usdc"), tmp_path / "cube.fbx")
    back = sceneport.load(tmp_path / "cube.fbx")
    assert_meshes_close(scene.meshes[0], back.meshes[0])
    assert_transforms_equal(scene.nodes[0].transform, back.nodes[0].transform)


def test_overwrite_and_reload(tmp_path):
    path = tmp_path / "x.usda"
    sceneport.save(Scene(nodes=[Node("First")]), path)
    assert sceneport.load(path).nodes[0].name == "First"
    sceneport.save(Scene(nodes=[Node("Second")]), path)
    assert sceneport.load(path).nodes[0].name == "Second"


# -- writing details ----------------------------------------------------------


def test_mesh_prim_layout():
    stage = stage_from_scene(Scene(nodes=[Node("Cube.001", mesh=mixed_mesh())]))
    xform = stage.GetPrimAtPath("/Cube_001")
    assert xform.GetTypeName() == "Xform" and xform.GetDisplayName() == "Cube.001"
    assert stage.GetDefaultPrim() == xform
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Cube_001/Mixed"))
    assert mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none
    primvars = UsdGeom.PrimvarsAPI(mesh)
    assert primvars.GetPrimvar("st").HasValue()  # first UV set
    assert primvars.GetPrimvar("Second").HasValue()
    assert primvars.GetPrimvar("st").GetAttr().GetCustomDataByKey("sceneport:name") == "UVMap"


def test_mixed_mesh_roundtrip():
    scene = Scene(nodes=[Node("Mixed", mesh=mixed_mesh())])
    assert_meshes_close(scene.meshes[0], roundtrip(scene).meshes[0])


def test_hierarchy_instancing_and_names():
    mesh = mixed_mesh()
    scene = Scene()
    root = scene.add(Node("Root", Transform(translation=[1, 0, 0])))
    root.add(Node("A", Transform(rotation=[0, 0, 90], geometric_translation=[0, 2, 0]), mesh=mesh))
    root.add(Node("A", Transform(scale=[2, 2, 2]), mesh=mesh))  # duplicate name
    root.add(Node("3 bad name!"))

    stage = stage_from_scene(scene)
    assert [p.GetName() for p in stage.GetPrimAtPath("/Root").GetChildren()] == ["A", "A_1", "__bad_name_"]
    assert stage.GetPrimAtPath("/Root/A_1/Mixed").HasAuthoredReferences()  # shared mesh

    back = scene_from_stage(stage)
    kids = back.nodes[0].children
    assert [k.name for k in kids] == ["A", "A", "3 bad name!"]
    assert kids[0].mesh is kids[1].mesh
    np.testing.assert_array_equal(kids[0].transform.geometric_translation, [0, 2, 0])
    np.testing.assert_array_equal(kids[1].transform.geometric_translation, [0, 0, 0])
    expected = {n.name + str(i): w for i, (n, w) in enumerate(scene.walk())}
    actual = {n.name + str(i): w for i, (n, w) in enumerate(back.walk())}
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)


def test_full_transform_exact():
    t = Transform(
        translation=[1, 2, 3], rotation=[10, 20, 30], scale=[1, 2, 3], rotation_order=2,
        pre_rotation=[-90, 0, 0], post_rotation=[5, 6, 7], rotation_pivot=[0, 1, 0],
        rotation_offset=[0.5, 0, 0], scaling_pivot=[1, 0, 0], scaling_offset=[0, 0, 2],
    )  # fmt: skip
    stage = stage_from_scene(Scene(nodes=[Node("N", t)]))
    xform = UsdGeom.Xformable(stage.GetPrimAtPath("/N"))
    np.testing.assert_allclose(np.array(xform.GetLocalTransformation()).T, t.matrix(), atol=1e-12)
    assert_transforms_equal(scene_from_stage(stage).nodes[0].transform, t)


def test_axes_and_units():
    scene = Scene(meters_per_unit=1.0, up_axis="+Z", front_axis="-Y")
    back = roundtrip(scene)
    assert (back.meters_per_unit, back.up_axis, back.front_axis, back.right_axis) == (1.0, "+Z", "-Y", "+X")
    with pytest.raises(ValueError, match="only supports"):
        stage_from_scene(Scene(up_axis="+X"))
    with pytest.warns(UserWarning, match="front/right"):
        stage_from_scene(Scene(front_axis="-Z"))


# -- reading stages from other tools ------------------------------------------


def foreign_stage() -> Usd.Stage:
    """A Scope-wrapped, left-handed quad with plain `normals` and non-indexed float2 UVs."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(stage, "/Geo")
    mesh = UsdGeom.Mesh.Define(stage, "/Geo/Quad")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1])  # clockwise, as leftHanded expects
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("uv", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.faceVarying)
    uv.Set(Vt.Vec2fArray([(0, 0), (0, 1), (1, 1), (1, 0)]))
    mesh.AddRotateXOp().Set(90.0)
    mesh.AddTranslateOp().Set(Gf.Vec3d(0, 0, 5))
    return stage


def test_read_foreign_stage():
    stage = foreign_stage()
    scene = scene_from_stage(stage)
    (node,) = scene.nodes  # the Scope is flattened away
    assert node.name == "Quad"
    mesh = node.mesh
    np.testing.assert_array_equal(mesh.face_vertex_indices, [1, 2, 3, 0])  # winding flipped
    assert mesh.normals.interpolation == "vertex"
    np.testing.assert_array_equal(mesh.corner_values(mesh.uvs["uv"]), [[1, 0], [1, 1], [0, 1], [0, 0]])
    expected = UsdGeom.Xformable(stage.GetPrimAtPath("/Geo/Quad")).GetLocalTransformation()
    np.testing.assert_allclose(node.transform.matrix(), np.array(expected).T, atol=1e-12)
    assert scene.meters_per_unit == 0.01  # USD's fallback when unauthored


def test_several_top_level_nodes_share_a_default_root():
    """Viewers such as Apple's Quick Look only show the default prim, so everything must be under it."""
    scene = Scene(nodes=[Node("A", mesh=mixed_mesh()), Node("B"), Node("C")])
    stage = stage_from_scene(scene)
    assert stage.GetDefaultPrim().GetPath() == "/root"
    assert [p.GetName() for p in stage.GetPseudoRoot().GetChildren()] == ["root"]
    back = scene_from_stage(stage)
    assert [n.name for n in back.nodes] == ["A", "B", "C"]  # the generated root is dropped on load
    single = stage_from_scene(Scene(nodes=[Node("Only", mesh=mixed_mesh())]))
    assert single.GetDefaultPrim().GetPath() == "/Only"  # one top-level node is the default prim itself
