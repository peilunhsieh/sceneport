"""Animated camera and light settings ("camera:<field>" / "light:<field>" channels) through USD and FBX."""

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdLux

from sceneport.interop.fbx import document_from_scene, scene_from_document
from sceneport.interop.fbx.binary import dumps, loads
from sceneport.interop.fbx.document import FBXDocument
from sceneport.interop.usd import scene_from_stage, stage_from_scene
from sceneport.scene import Animation, Camera, Channel, Light, Node, Scene, Transform

TIMES = np.linspace(0.0, 1.0, 25)  # every frame at 24 fps: both formats bake to frames


def animated_scene() -> Scene:
    cam = Node("Cam", Transform(translation=[0, 1, 5], rotation=[-10, 0, 0]),
               camera=Camera(focal_length=35.0, horizontal_aperture=36.0, vertical_aperture=24.0, focus_distance=5.0))  # fmt: skip
    spot = Node("Spot", Transform(translation=[2, 3, 1], rotation=[-60, 0, 0]),
                light=Light("sphere", intensity=40.0, radius=0.05, cone_angle=30.0, cone_softness=0.2))  # fmt: skip
    sun = Node("Sun", Transform(rotation=[-45, 30, 0]), light=Light("distant", intensity=3.0))
    scene = Scene(nodes=[cam, spot, sun])
    scene.animations = [Animation("anim", [
        Channel(cam, "camera:focal_length", 0, [0.0, 1.0], [35.0, 85.0]),
        Channel(cam, "camera:focus_distance", 0, [0.0, 0.5, 1.0], [5.0, 2.0, 8.0]),
        Channel(cam, "translation", 2, [0.0, 1.0], [5.0, 3.0]),
        Channel(spot, "light:intensity", 0, [0.0, 1.0], [40.0, 10.0]),
        Channel(spot, "light:cone_angle", 0, [0.0, 1.0], [30.0, 20.0]),
        Channel(spot, "light:cone_softness", 0, [0.0, 1.0], [0.2, 0.6]),
        *[Channel(sun, "light:color", i, [0.0, 1.0], [1.0, v]) for i, v in enumerate((1.0, 0.5, 0.2))],
    ], fps=24)]  # fmt: skip
    return scene


def assert_animated_like(scene: Scene, back: Scene, rtol: float) -> None:
    a, b = scene.animations[0], back.animations[0]
    for name, kind in (("Cam", "camera"), ("Spot", "light"), ("Sun", "light")):
        expected = a.attribute_values(scene.find(name), kind, TIMES)
        got = b.attribute_values(back.find(name), kind, TIMES)
        for field in {c.attribute[1] for c in a.channels if c.target is scene.find(name) and c.attribute}:
            np.testing.assert_allclose(got[field], expected[field], rtol=rtol, err_msg=f"{name}.{field}")
    worlds = a.world_matrices(scene, TIMES)
    back_worlds = b.world_matrices(back, TIMES)
    for name in ("Cam", "Spot", "Sun"):  # the camera / light turns don't drop the transform animation
        np.testing.assert_allclose(back_worlds[back.find(name)], worlds[scene.find(name)], atol=1e-5, err_msg=name)


def test_attribute_values_and_copies():
    scene = animated_scene()
    clip, cam = scene.animations[0], scene.find("Cam")
    values = clip.attribute_values(cam, "camera", [0.0, 0.5])
    np.testing.assert_allclose(values["focal_length"], [35.0, 60.0])
    np.testing.assert_allclose(values["horizontal_aperture"], [36.0, 36.0])  # unanimated: the stored value
    lenses = clip.camera_at(cam, [0.25])
    assert lenses[0].focal_length == pytest.approx(47.5) and cam.camera.focal_length == 35.0  # copies
    assert clip.light_at(scene.find("Sun"), [1.0])[0].color == pytest.approx([1.0, 0.5, 0.2])


def test_attribute_channel_validation():
    node = Node("Empty")
    with pytest.raises(ValueError, match="camera"):
        Animation("bad", [Channel(node, "camera:focal_length", 0, [0.0], [50.0])]).validate()
    cam = Node("Cam", camera=Camera())
    with pytest.raises(ValueError):
        Animation("bad", [Channel(cam, "camera:zoom", 0, [0.0], [1.0])]).validate()


def test_usd_attribute_animation_roundtrip():
    scene = animated_scene()
    stage = stage_from_scene(scene)
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/root/Cam"))
    assert camera.GetFocalLengthAttr().GetNumTimeSamples() == 25
    assert camera.GetFocalLengthAttr().Get(Usd.TimeCode(24)) == pytest.approx(0.85)  # tenths of a unit: 85 mm at 1 m per unit
    spot = UsdLux.SphereLight(stage.GetPrimAtPath("/root/Spot"))
    assert spot.GetIntensityAttr().Get(Usd.TimeCode(24)) == pytest.approx(10.0)
    assert not spot.GetRadiusAttr().GetNumTimeSamples()  # unanimated fields stay single values
    back = scene_from_stage(stage)
    assert_animated_like(scene, back, rtol=1e-6)


def test_fbx_attribute_animation_roundtrip():
    scene = animated_scene()
    doc = document_from_scene(scene)
    back = scene_from_document(FBXDocument.from_file(loads(dumps(doc.to_file()))))
    assert_animated_like(scene, back, rtol=1e-5)  # FBX keys are float32


def test_fbx_attribute_curves_in_file():
    """Curves on the NodeAttribute's properties, in Blender's units (watts x 100, the full spot cone)."""
    scene = animated_scene()
    doc = document_from_scene(scene)
    attributes = {o.name: o for o in doc.iter_objects("NodeAttribute")}
    curves = {}
    for connection in doc.connections:
        node = doc.objects.get(connection.src)
        if node is not None and node.kind == "AnimationCurveNode" and connection.dst in {a.id for a in attributes.values()}:
            owner = doc.objects[connection.dst].name
            curves[(owner, connection.prop)] = {
                c.prop: doc.objects[c.src] for c in doc.connections if c.dst == node.id and c.kind == "OP"
            }
    assert set(curves) == {("Cam", "FocalLength"), ("Cam", "FocusDistance"), ("Spot", "Intensity"),
                           ("Spot", "OuterAngle"), ("Spot", "InnerAngle"), ("Sun", "Color")}  # fmt: skip
    assert set(curves[("Sun", "Color")]) == {"d|X", "d|Y", "d|Z"}

    def last(owner, prop, component):
        return float(curves[(owner, prop)][component].child("KeyValueFloat").props[0].value[-1])

    assert last("Cam", "FocalLength", "d|FocalLength") == pytest.approx(85.0)
    assert last("Spot", "Intensity", "d|Intensity") == pytest.approx(10.0 * np.pi * 100.0, rel=1e-6)
    assert last("Spot", "OuterAngle", "d|OuterAngle") == pytest.approx(40.0)
    assert last("Spot", "InnerAngle", "d|InnerAngle") == pytest.approx(40.0 * (1 - 0.6))


def test_fbx_unsupported_attribute_animation_warns():
    scene = animated_scene()
    sun = scene.find("Sun")
    scene.animations[0].channels.append(Channel(sun, "light:angle", 0, [0.0, 1.0], [0.5, 2.0]))
    with pytest.warns(UserWarning, match="FBX can't animate its light's angle"):
        document_from_scene(scene)
