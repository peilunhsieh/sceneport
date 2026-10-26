"""Named vertex groups (per-vertex weights): USD primvars, Blender vertex groups, warnings elsewhere."""

import numpy as np
import pytest
from pxr import Sdf, UsdGeom

import sceneport
from sceneport.interop.blend import run_in_blender
from sceneport.interop.usd import scene_from_stage, stage_from_scene
from sceneport.scene import Mesh, Node, Scene

GROUPS = {"skin exterior": [1.0, 1.0, 0.0, 0.5], "lips": [0.0, 0.25, 1.0, 0.0]}


def quad() -> Scene:
    mesh = Mesh("quad", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [4], [0, 1, 2, 3], vertex_groups=GROUPS)
    return Scene(nodes=[Node("quad", mesh=mesh)], up_axis="+Y", meters_per_unit=1.0)


def test_weights_must_be_one_per_vertex():
    with pytest.raises(ValueError, match="vertex group 'lips'"):
        Mesh("quad", [[0, 0, 0], [1, 0, 0], [1, 1, 0]], [3], [0, 1, 2], vertex_groups={"lips": [1.0, 0.0]})
    assert quad().nodes[0].mesh.vertex_group("skin exterior").tolist() == [0, 1, 3]


def test_usd_float_primvars_roundtrip():
    stage = stage_from_scene(quad())
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("group:skin_exterior")
    assert primvar.GetTypeName() == Sdf.ValueTypeNames.FloatArray and primvar.GetInterpolation() == "vertex"
    mesh = scene_from_stage(stage).meshes[0]
    assert sorted(mesh.vertex_groups) == sorted(GROUPS) and not mesh.uvs  # USD lists properties by name
    for name, weights in GROUPS.items():
        np.testing.assert_array_equal(mesh.vertex_groups[name], weights)


@pytest.mark.parametrize("suffix", [".obj", ".ply", ".fbx"])
def test_formats_without_vertex_groups_warn(tmp_path, suffix):
    with pytest.warns(UserWarning, match="vertex groups"):
        sceneport.save(quad(), tmp_path / f"quad{suffix}")


def test_blender_vertex_groups(blender, tmp_path):
    sceneport.save(quad(), tmp_path / "quad.blend", blender=blender)
    (obj,) = run_in_blender("inspect_blend", {"path": str(tmp_path / "quad.blend")}, blender=blender)["objects"]
    assert obj["vertex_groups"] == list(GROUPS)
    for name, weights in GROUPS.items():
        np.testing.assert_array_equal(obj["vertex_group_weights"][name], weights)
