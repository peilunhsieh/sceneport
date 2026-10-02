"""Dome (environment) lights on nodes, the Scene.environment shortcut, and USD DomeLight round trips."""

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdLux

import sceneport
from sceneport.interop.usd import scene_from_stage, stage_from_scene
from sceneport.scene import Animation, AssetPath, Channel, Light, Mesh, Node, Scene, Transform


def test_environment_shortcut(tmp_path):
    scene = Scene(nodes=[Node("Box", mesh=Mesh("box", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2]))])
    assert scene.environment is None
    (tmp_path / "sky.exr").write_bytes(b"")
    node = scene.add_environment(tmp_path / "sky.exr", rotation=(90, 0, 0), intensity=2.0)
    assert scene.environment is node and node.light.type == "dome" and node.light.intensity == 2.0
    assert node.light.texture.resolved == str(tmp_path / "sky.exr")
    np.testing.assert_allclose(node.transform.rotation, [90, 0, 0])


def lit_scene(texture: AssetPath) -> Scene:
    scene = Scene(nodes=[Node("Box", mesh=Mesh("box", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2]))])
    env = scene.add_environment(texture, rotation=(90, 0, 30), intensity=1.5)
    env.light.exposure, env.light.color, env.light.texture_format = 0.5, np.array([1.0, 0.9, 0.8]), "latlong"
    scene.animations = [Animation("spin", [Channel(env, "rotation", 2, [0.0, 1.0], [30.0, 120.0])], fps=24)]
    return scene


def test_usd_dome_light_roundtrip(tmp_path):
    (tmp_path / "hdri").mkdir()
    (tmp_path / "hdri" / "sky.exr").write_bytes(b"")
    (tmp_path / "out").mkdir()
    scene = lit_scene(AssetPath("sky.exr", str(tmp_path / "hdri" / "sky.exr")))
    sceneport.save(scene, tmp_path / "out" / "lit.usda")

    stage = Usd.Stage.Open(str(tmp_path / "out" / "lit.usda"))
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/root/Environment"))
    assert dome and stage.GetPrimAtPath("/root/Environment").GetTypeName() == "DomeLight"  # the classic schema
    assert dome.GetTextureFileAttr().Get().path == "../hdri/sky.exr"  # relative to the written file

    back = sceneport.load(tmp_path / "out" / "lit.usda")
    light, original = back.environment.light, scene.environment.light
    assert (light.intensity, light.exposure, light.texture_format) == (1.5, 0.5, "latlong")
    np.testing.assert_allclose(light.color, original.color, rtol=1e-6)
    assert light.texture.resolved == str(tmp_path / "hdri" / "sky.exr")
    times = scene.animations[0].frame_times()
    np.testing.assert_allclose(  # the environment's orientation, animated
        back.animations[0].world_matrices(back, times)[back.environment],
        scene.animations[0].world_matrices(scene, times)[scene.environment],
        atol=1e-6,
    )


@pytest.mark.parametrize(("pole", "up"), [("Z", "Y"), ("scene", "Z"), ("Y", "Z"), ("scene", "Y")])
def test_usd_dome_light_1_pole_axis(pole, up):
    """A DomeLight_1 can put the image's pole on Z; sceneport turns the node so its +Y is that pole."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, up)
    dome = UsdLux.DomeLight_1.Define(stage, "/Sky")
    dome.CreatePoleAxisAttr(pole)
    dome.AddRotateXYZOp().Set((10.0, 20.0, 30.0))
    world = np.array(dome.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    pole_world = world[:3, 2] if pole == "Z" or (pole == "scene" and up == "Z") else world[:3, 1]

    scene = scene_from_stage(stage)
    node = scene.environment
    assert node is not None and node.light.type == "dome"
    np.testing.assert_allclose(scene.world_matrices()[node][:3, 1], pole_world, atol=1e-9)  # pole on +Y
    written = stage_from_scene(scene)  # keep the stage alive while its prim is used
    assert written.GetPrimAtPath("/Sky").GetTypeName() == "DomeLight"  # the classic schema, whose pole is +Y


def test_formats_without_dome_lights_warn(tmp_path):
    scene = lit_scene(AssetPath("sky.exr"))
    scene.animations = []
    for suffix in (".fbx", ".obj"):
        with pytest.warns(UserWarning, match="environment"):
            sceneport.save(scene, tmp_path / f"out{suffix}")


def test_hdri_sample_as_environment(sample, tmp_path):
    hdri = sample("blender_institute_hdri")
    scene = Scene(meters_per_unit=1.0, up_axis="+Z", front_axis="-Y", right_axis="+X")
    scene.add_environment(hdri, rotation=(90, 0, 90))  # Blender's Z-up World orientation
    sceneport.save(scene, tmp_path / "env.usda")
    light = sceneport.load(tmp_path / "env.usda").environment.light
    assert light.texture.resolved == str(hdri)
    assert not light.texture.path.startswith("/")  # written relative to the USD file
