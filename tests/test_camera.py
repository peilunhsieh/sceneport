"""Cameras: lens settings on a Node, through USD (tenths of a unit) and FBX (inches, +X view axis)."""

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from sceneport.interop.fbx import document_from_scene, scene_from_document
from sceneport.interop.fbx.binary import dumps, loads
from sceneport.interop.fbx.document import FBXDocument
from sceneport.interop.usd import scene_from_stage, stage_from_scene
from sceneport.scene import Animation, Camera, Channel, Node, Scene, Transform

LENS = dict(focal_length=35.0, horizontal_aperture=36.0, vertical_aperture=20.25, horizontal_aperture_offset=1.5,
            vertical_aperture_offset=-0.75, near=0.05, far=250.0, focus_distance=4.0)  # fmt: skip


def camera_scene(meters_per_unit: float = 1.0) -> Scene:
    """An animated camera with a child that must stay put, under a rotated parent."""
    camera = Node("Cam", Transform(translation=[1, 2, 5], rotation=[-10, 20, 0]), camera=Camera(**LENS))
    camera.add(Node("Light", Transform(translation=[0, 0.5, 0])))
    rig = Node("Rig", Transform(rotation=[0, 0, 15]), children=[camera])
    scene = Scene(nodes=[rig], meters_per_unit=meters_per_unit)
    scene.animations = [Animation("move", [
        Channel(camera, "translation", 0, [0.0, 1.0], [1.0, 3.0]),
        Channel(camera, "rotation", 1, [0.0, 1.0], [20.0, 60.0]),
    ], fps=24)]  # fmt: skip
    return scene


def assert_lens_equal(a: Camera, b: Camera, rtol: float = 1e-6) -> None:
    for name in (*LENS, "projection"):
        if name == "projection":
            assert a.projection == b.projection
        else:
            assert getattr(b, name) == pytest.approx(getattr(a, name), rel=rtol), name


def worlds(scene: Scene, times) -> dict[str, np.ndarray]:
    return {node.name: m for node, m in scene.animations[0].world_matrices(scene, times).items()}


@pytest.mark.parametrize("meters_per_unit", [1.0, 0.01])
def test_usd_camera(meters_per_unit):
    scene = camera_scene(meters_per_unit)
    stage = stage_from_scene(scene)
    usd_camera = UsdGeom.Camera(stage.GetPrimAtPath("/Rig/Cam"))
    assert usd_camera  # the node's own prim is a Camera
    # Lens values in tenths of a scene unit: mm in a centimeter scene.
    assert usd_camera.GetFocalLengthAttr().Get() == pytest.approx(35.0 / (100 * meters_per_unit))
    back = scene_from_stage(stage)
    assert_lens_equal(scene.find("Cam").camera, back.find("Cam").camera)
    times = scene.animations[0].frame_times()
    for name, m in worlds(scene, times).items():
        np.testing.assert_allclose(worlds(back, times)[name], m, atol=1e-5, err_msg=name)


def test_fbx_camera():
    scene = camera_scene()
    doc = document_from_scene(scene)
    (attribute,) = [o for o in doc.iter_objects("NodeAttribute") if o.subtype == "Camera"]
    assert attribute.props.get("FilmWidth") == pytest.approx(36.0 / 25.4)  # inches
    assert attribute.props.get("FocalLength") == pytest.approx(35.0)
    (model,) = [o for o in doc.iter_objects("Model") if o.subtype == "Camera"]
    assert model.name == "Cam"

    back = scene_from_document(FBXDocument.from_file(loads(dumps(doc.to_file()))))
    assert_lens_equal(scene.find("Cam").camera, back.find("Cam").camera)
    times = scene.animations[0].frame_times()
    for name, m in worlds(scene, times).items():  # the -Z <-> +X turn is undone on load; the child doesn't move
        np.testing.assert_allclose(worlds(back, times)[name], m, atol=1e-5, err_msg=name)


def test_fbx_camera_looks_down_x():
    """In the file, the camera Model's +X axis points where the camera's -Z did."""
    scene = camera_scene()
    scene.animations = []
    doc = document_from_scene(scene)
    raw = scene_from_document(FBXDocument.from_file(loads(dumps(doc.to_file()))))  # with the turn undone
    expected = scene.world_matrices()[scene.find("Cam")]
    from sceneport.interop.fbx.from_fbx import _read_transform

    rig, cam = next(o for o in doc.iter_objects("Model") if o.name == "Rig"), next(o for o in doc.iter_objects("Model") if o.name == "Cam")
    file_world = _read_transform(doc, rig).matrix() @ _read_transform(doc, cam).matrix()
    np.testing.assert_allclose(file_world[:3, 0], -expected[:3, 2], atol=1e-9)  # FBX +X = our -Z
    np.testing.assert_allclose(file_world[:3, 1], expected[:3, 1], atol=1e-9)  # up stays up
    assert raw.find("Cam").camera is not None


# -- exposure -----------------------------------------------------------------------------------

from sceneport.interop.usd import scene_from_stage as _from_stage  # noqa: E402


def exposed_scene() -> Scene:
    cam = Node("Cam", Transform(translation=[0, 0, 5]), camera=Camera(**LENS, exposure=2.0))
    scene = Scene(nodes=[cam])
    scene.animations = [Animation("expose", [Channel(cam, "camera:exposure", 0, [0.0, 1.0], [2.0, -1.0])], fps=24)]
    return scene


def test_usd_camera_exposure():
    scene = exposed_scene()
    stage = stage_from_scene(scene)
    attr = UsdGeom.Camera(stage.GetDefaultPrim()).GetExposureAttr()
    assert attr.Get(Usd.TimeCode.Default()) == pytest.approx(2.0) and attr.GetNumTimeSamples() == 25
    back = _from_stage(stage)
    assert back.find("Cam").camera.exposure == pytest.approx(2.0)
    values = back.animations[0].attribute_values(back.find("Cam"), "camera", [0.0, 0.5, 1.0])["exposure"]
    np.testing.assert_allclose(values, [2.0, 0.5, -1.0], atol=1e-6)
    assert _from_stage(stage_from_scene(Scene(nodes=[Node("C", camera=Camera())]))).find("C").camera.exposure == 0.0


def test_fbx_has_no_camera_exposure():
    scene = exposed_scene()
    with pytest.warns(UserWarning, match="no camera exposure"):
        document_from_scene(scene)
