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
        with pytest.warns(UserWarning, match="environment|lights"):
            sceneport.save(scene, tmp_path / f"out{suffix}")


def test_hdri_sample_as_environment(sample, tmp_path):
    hdri = sample("blender_institute_hdri")
    scene = Scene(meters_per_unit=1.0, up_axis="+Z", front_axis="-Y", right_axis="+X")
    scene.add_environment(hdri, rotation=(90, 0, 90))  # Blender's Z-up World orientation
    sceneport.save(scene, tmp_path / "env.usda")
    light = sceneport.load(tmp_path / "env.usda").environment.light
    assert light.texture.resolved == str(hdri)
    assert not light.texture.path.startswith("/")  # written relative to the USD file



# -- light types -------------------------------------------------------------------------

LIGHTS = {
    "Sun": Light("distant", intensity=3.0, angle=0.6, color=[1.0, 0.95, 0.9]),
    "Bulb": Light("sphere", intensity=20.0, radius=0.1, temperature=3200.0),
    "Point": Light("sphere", intensity=5.0, radius=0.0, cast_shadows=False),
    "Spot": Light("sphere", intensity=40.0, radius=0.05, cone_angle=30.0, cone_softness=0.25),
    "Panel": Light("rect", intensity=8.0, width=2.0, height=0.5, normalize=True, diffuse=0.5),
    "Disc": Light("disk", intensity=6.0, radius=0.3, shadow_color=[0.1, 0.0, 0.2]),
    "Tube": Light("cylinder", intensity=2.0, radius=0.05, length=1.2),
}


def lights_scene() -> Scene:
    """Each light on its own node, aimed somewhere; one animated."""
    scene = Scene()
    for i, (name, light) in enumerate(LIGHTS.items()):
        scene.add(Node(name, Transform(translation=[i, 2.0, -1.0], rotation=[-40.0, 15.0 * i, 0.0]), light=light))
    spot = scene.find("Spot")
    scene.animations = [Animation("sweep", [Channel(spot, "rotation", 1, [0.0, 1.0], [45.0, 135.0])], fps=24)]
    return scene


def assert_light_equal(a: Light, b: Light, fields, label: str) -> None:
    for name in fields:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, np.ndarray):
            np.testing.assert_allclose(y, x, rtol=1e-6, err_msg=f"{label}.{name}")
        elif isinstance(x, float):
            assert y == pytest.approx(x, rel=1e-6), f"{label}.{name}"
        else:
            assert y == x, f"{label}.{name}"


ALL_FIELDS = ("type", "color", "intensity", "exposure", "temperature", "diffuse", "specular", "normalize", "radius",
              "width", "height", "length", "angle", "cone_angle", "cone_softness", "cast_shadows", "shadow_color")  # fmt: skip


def test_usd_light_types_roundtrip():
    scene = lights_scene()
    stage = stage_from_scene(scene)
    types = {p.GetName(): p.GetTypeName() for p in stage.GetPrimAtPath("/root").GetChildren()}
    assert types == {"Sun": "DistantLight", "Bulb": "SphereLight", "Point": "SphereLight", "Spot": "SphereLight",
                     "Panel": "RectLight", "Disc": "DiskLight", "Tube": "CylinderLight"}  # fmt: skip
    assert stage.GetPrimAtPath("/root/Spot").HasAPI(UsdLux.ShapingAPI)
    assert stage.GetPrimAtPath("/root/Point").HasAPI(UsdLux.ShadowAPI)  # shadows off
    assert UsdLux.SphereLight(stage.GetPrimAtPath("/root/Point")).GetTreatAsPointAttr().Get()
    assert UsdGeom.Boundable(stage.GetPrimAtPath("/root/Panel")).GetExtentAttr().HasAuthoredValue()

    back = scene_from_stage(stage)
    for name, light in LIGHTS.items():
        assert_light_equal(light, back.find(name).light, ALL_FIELDS, name)
    times = scene.animations[0].frame_times()
    np.testing.assert_allclose(back.animations[0].world_matrices(back, times)[back.find("Spot")],
                               scene.animations[0].world_matrices(scene, times)[scene.find("Spot")], atol=1e-6)  # fmt: skip


def fbx_roundtrip(scene: Scene) -> Scene:
    from sceneport.interop.fbx import document_from_scene, scene_from_document
    from sceneport.interop.fbx.binary import dumps, loads
    from sceneport.interop.fbx.document import FBXDocument

    return scene_from_document(FBXDocument.from_file(loads(dumps(document_from_scene(scene).to_file()))))


def test_fbx_light_types_roundtrip():
    scene = lights_scene()
    with pytest.warns(UserWarning, match="FBX can't hold"):
        back = fbx_roundtrip(scene)
    kept = {"Sun": "distant", "Bulb": "sphere", "Point": "sphere", "Spot": "sphere", "Panel": "rect", "Disc": "rect",
            "Tube": "sphere"}  # fmt: skip
    for name, light in LIGHTS.items():
        read = back.find(name).light
        assert read.type == kept[name], name
        # intensity survives through Blender's watt conventions (exposure folded in)
        assert read.intensity * 2**read.exposure == pytest.approx(light.intensity * 2**light.exposure, rel=1e-6), name
        assert_light_equal(light, read, ("color", "cast_shadows", "shadow_color", "cone_angle", "cone_softness"), name)
    times = scene.animations[0].frame_times()
    for name in LIGHTS:  # the -Z <-> -Y turn is undone on load
        np.testing.assert_allclose(back.animations[0].world_matrices(back, times)[back.find(name)],
                                   scene.animations[0].world_matrices(scene, times)[scene.find(name)], atol=1e-5, err_msg=name)  # fmt: skip


def test_fbx_light_file_conventions():
    """Blender's conventions in the file: watts x 100, the full spot cone, lights shining down -Y."""
    from sceneport.interop.fbx import document_from_scene
    from sceneport.interop.fbx.from_fbx import _read_transform

    scene = Scene(nodes=[Node("Spot", Transform(rotation=[-30.0, 20.0, 0.0]), light=LIGHTS["Spot"]),
                         Node("Sun", light=LIGHTS["Sun"])])  # fmt: skip
    doc = document_from_scene(scene)
    attributes = {o.name: o for o in doc.iter_objects("NodeAttribute") if o.subtype == "Light"}
    spot = attributes["Spot"].props
    assert (spot.get("LightType"), attributes["Sun"].props.get("LightType")) == (2, 1)
    assert spot.get("Intensity") == pytest.approx(40.0 * np.pi * 100.0)  # USD intensity -> watts -> x100
    assert attributes["Sun"].props.get("Intensity") == pytest.approx(3.0 * 4.0 * 100.0)  # a sun: strength / 4
    assert (spot.get("OuterAngle"), spot.get("InnerAngle")) == (pytest.approx(60.0), pytest.approx(45.0))
    model = next(o for o in doc.iter_objects("Model") if o.name == "Spot")
    assert model.subtype == "Light"
    ours = scene.world_matrices()[scene.find("Spot")]
    in_file = _read_transform(doc, model).matrix()
    np.testing.assert_allclose(-in_file[:3, 1], -ours[:3, 2], atol=1e-9)  # FBX -Y = our -Z (the light direction)


def test_unsupported_usd_light_warns():
    stage = Usd.Stage.CreateInMemory()
    UsdLux.PortalLight.Define(stage, "/Portal")
    with pytest.warns(UserWarning, match="isn't supported"):
        scene = scene_from_stage(stage)
    assert scene.find("Portal").light is None
