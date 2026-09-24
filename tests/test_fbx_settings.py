import pytest

import sceneport
from sceneport.interop.fbx import document_from_scene, scene_from_document
from sceneport.interop.fbx.binary import read
from sceneport.interop.fbx.document import FBXDocument
from sceneport.interop.fbx.settings import axis_name, axis_value
from sceneport.scene import Scene
from helpers import FIXTURES, assert_nodes_equal

CUBE = FIXTURES / "cube.fbx"


def written_settings(scene: Scene):
    return document_from_scene(scene).global_settings


def test_axis_names():
    assert axis_name(1, 1) == "+Y" and axis_name(2, -1) == "-Z"
    assert axis_value("-X") == (0, -1)


def test_cube_settings_kept_exactly():
    original = read(CUBE).find("GlobalSettings").find("Properties70")
    scene = sceneport.load(CUBE)
    assert_nodes_equal(written_settings(scene).to_node(), original)


def test_new_scene_defaults_to_meters():
    settings = written_settings(Scene())
    assert settings.get("UnitScaleFactor") == 100.0
    assert settings.get("OriginalUnitScaleFactor") == 100.0
    assert [settings.get(k) for k in ("UpAxis", "FrontAxis", "CoordAxis")] == [1, 2, 0]
    assert settings.get("CustomFrameRate") == 24.0  # other defaults still present


def test_axes_and_units_roundtrip():
    scene = Scene(meters_per_unit=0.0254, up_axis="+Z", front_axis="-Y", right_axis="+X")
    settings = written_settings(scene)
    assert (settings.get("UpAxis"), settings.get("UpAxisSign")) == (2, 1)
    assert (settings.get("FrontAxis"), settings.get("FrontAxisSign")) == (1, -1)
    back = scene_from_document(FBXDocument(global_settings=settings))
    assert back.meters_per_unit == pytest.approx(0.0254)
    assert (back.up_axis, back.front_axis, back.right_axis) == ("+Z", "-Y", "+X")


def test_changed_unit_keeps_original_unit():
    scene = sceneport.load(CUBE)
    scene.meters_per_unit = 1.0
    settings = written_settings(scene)
    assert settings.get("UnitScaleFactor") == 100.0
    assert settings.get("OriginalUnitScaleFactor") == 1.0  # from the file


def test_invalid_axis():
    with pytest.raises(ValueError, match="up_axis"):
        Scene(up_axis="Y")


def test_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match="unsupported file extension '.obj'"):
        sceneport.load(tmp_path / "x.obj")
    with pytest.raises(ValueError, match="unsupported"):
        sceneport.save(Scene(), tmp_path / "x.blend")
