"""Cameras, lights and the World in .blend files: values as Blender's USD importer sets them, checked in Blender."""

import math

import numpy as np
import pytest

import sceneport
from sceneport.interop.blend import run_in_blender
from sceneport.interop.reframe import basis_matrix
from sceneport.scene import Animation, AssetPath, Camera, Channel, Light, Mesh, Node, Scene, Transform
from test_light import LIGHTS, lights_scene


def inspect(blender, path, **payload):
    return run_in_blender("inspect_blend", {"path": str(path), **payload}, blender=blender)


def to_blender(scene: Scene) -> np.ndarray:
    """4x4 from scene space to Blender's Z-up meters."""
    m = np.eye(4)
    m[:3, :3] = scene.meters_per_unit * basis_matrix(scene, "+Z", "-Y", "+X")
    return m


def projected(camera: Camera, world: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Normalized frame coordinates (0..1, origin bottom left) of world `points` seen by a camera at `world`,
    with sceneport's lens model: film offset shifts the frame in mm."""
    local = (np.linalg.inv(world) @ np.column_stack([points, np.ones(len(points))]).T).T[:, :3]
    depth = -local[:, 2]
    film = camera.focal_length * local[:, :2] / depth[:, None]
    offset = np.array([camera.horizontal_aperture_offset, camera.vertical_aperture_offset])
    size = np.array([camera.horizontal_aperture, camera.vertical_aperture])
    return 0.5 + (film - offset) / size


def test_camera_view_and_animated_lens(blender, tmp_path):
    """A Y-up centimeter camera with film offsets and an animated move and lens: Blender's view of a few
    points matches sceneport's lens model at every checked frame."""
    cam = Node("Cam", Transform(translation=[20, 150, 400], rotation=[-10, 5, 0]),
               camera=Camera(focal_length=35.0, horizontal_aperture=36.0, vertical_aperture=20.25,
                             horizontal_aperture_offset=1.5, vertical_aperture_offset=-0.75, near=5.0, far=5000.0,
                             focus_distance=300.0, f_stop=2.8))  # fmt: skip
    scene = Scene(nodes=[Node("Rig", Transform(rotation=[0, 20, 0]), children=[cam])], meters_per_unit=0.01)
    scene.animations = [Animation("move", [
        Channel(cam, "translation", 2, [0.0, 1.0], [400.0, 250.0]),
        Channel(cam, "camera:focal_length", 0, [0.0, 0.5, 1.0], [35.0, 50.0, 85.0], "cubic"),
        Channel(cam, "camera:focus_distance", 0, [0.0, 1.0], [300.0, 150.0]),
    ], fps=24)]  # fmt: skip
    sceneport.save(scene, tmp_path / "cam.blend", blender=blender)

    points = np.array([[0, 140, 0], [30, 160, -50], [-40, 120, 30], [10, 180, 60]], dtype=float)
    m = to_blender(scene)
    blender_points = (points @ m[:3, :3].T).tolist()
    frames = [0, 6, 12, 18, 24]
    result = inspect(blender, tmp_path / "cam.blend", frames=frames, points=blender_points)
    assert result["scene_camera"] == "Cam" and result["render"] == (1920, 1080)  # 36 x 20.25 mm film is 16:9
    data = {o["name"]: o for o in result["objects"]}["Cam"]["camera"]
    assert (data["sensor_fit"], data["sensor_width"], data["sensor_height"]) == ("HORIZONTAL", 36.0, pytest.approx(20.25))
    assert (data["shift_x"], data["shift_y"]) == (pytest.approx(1.5 / 36), pytest.approx(-0.75 / 36))
    assert (data["clip_start"], data["clip_end"], data["use_dof"]) == (pytest.approx(0.05), pytest.approx(50.0), True)
    clip = scene.animations[0]
    for frame in frames:
        t = frame / 24
        lens = clip.camera_at(cam, [t])[0]
        world = clip.world_matrices(scene, [t])[cam][0]
        state = result["frames"][frame]
        assert state["Cam"]["camera"]["lens"] == pytest.approx(lens.focal_length, rel=1e-5), frame
        assert state["Cam"]["camera"]["focus_distance"] == pytest.approx(lens.focus_distance * 0.01, rel=1e-5), frame
        np.testing.assert_allclose(state["_view"][:, :2], projected(lens, world, points), atol=1e-5, err_msg=f"frame {frame}")


def test_light_types(blender, tmp_path):
    scene = lights_scene()
    with pytest.warns(UserWarning, match="cylinder shape"):
        sceneport.save(scene, tmp_path / "lights.blend", blender=blender)
    result = inspect(blender, tmp_path / "lights.blend", frames=[0, 24])
    written = {o["name"]: o for o in result["objects"]}
    got = {name: written[name]["light"] for name in LIGHTS}
    assert {n: (g["type"], g.get("shape")) for n, g in got.items()} == {
        "Sun": ("SUN", None), "Bulb": ("POINT", None), "Point": ("POINT", None), "Spot": ("SPOT", None),
        "Panel": ("AREA", "RECTANGLE"), "Disc": ("AREA", "DISK"), "Tube": ("POINT", None)}  # fmt: skip
    for name, light in LIGHTS.items():  # energy as Blender's USD importer: intensity x pi, a sun's x 4
        assert got[name]["energy"] == pytest.approx(light.intensity * (4.0 if light.type == "distant" else math.pi), rel=1e-6)
        assert got[name]["normalize"] == light.normalize and got[name]["use_shadow"] == light.cast_shadows
        np.testing.assert_allclose(got[name]["color"], light.color, rtol=1e-6)
    assert got["Bulb"]["use_temperature"] and got["Bulb"]["temperature"] == 3200.0
    assert got["Spot"]["spot_size"] == pytest.approx(math.radians(60)) and got["Spot"]["spot_blend"] == 0.25
    assert (got["Panel"]["size"], got["Panel"]["size_y"], got["Panel"]["diffuse_factor"]) == (2.0, 0.5, 0.5)
    assert got["Disc"]["size"] == pytest.approx(0.6) and got["Sun"]["angle"] == pytest.approx(math.radians(0.6))
    c = basis_matrix(scene, "+Z", "-Y", "+X")
    clip = scene.animations[0]
    for frame in (0, 24):  # shining the same way, down their own -Z (the Spot sweeping)
        worlds = clip.world_matrices(scene, [frame / 24])
        for name in LIGHTS:
            shine = c @ -worlds[scene.find(name)][0][:3, 2]
            np.testing.assert_allclose(-result["frames"][frame][name]["matrix_world"][:3, 2], shine, atol=1e-6, err_msg=name)


def test_animated_light(blender, tmp_path):
    scene = lights_scene()
    spot = scene.find("Spot")
    scene.animations[0].channels += [
        Channel(spot, "light:intensity", 0, [0.0, 1.0], [40.0, 10.0]),
        Channel(spot, "light:cone_angle", 0, [0.0, 1.0], [30.0, 15.0]),
        Channel(spot, "light:color", 1, [0.0, 1.0], [1.0, 0.2]),
    ]
    with pytest.warns(UserWarning):
        sceneport.save(scene, tmp_path / "lights.blend", blender=blender)
    result = inspect(blender, tmp_path / "lights.blend", frames=[0, 12, 24])
    for frame, (intensity, cone, green) in {0: (40, 30, 1.0), 12: (25, 22.5, 0.6), 24: (10, 15, 0.2)}.items():
        light = result["frames"][frame]["Spot"]["light"]
        assert light["energy"] == pytest.approx(intensity * math.pi, rel=1e-5), frame
        assert light["spot_size"] == pytest.approx(math.radians(2 * cone), rel=1e-5), frame
        assert light["color"][1] == pytest.approx(green, rel=1e-5), frame


@pytest.mark.parametrize(("up", "rotation", "mapping"), [
    ("+Z", (90, 0, 90), (0, 0, 0)), ("+Z", (90, 0, 120), (0, 0, -30)), ("+Z", (0, 0, 0), (90, 0, 90)),
    ("+Y", (0, 0, 0), (0, 0, 90)), ("+Y", (0, 30, 0), (0, 0, 60)),
])  # fmt: skip  # the mapping rotations Blender's USD importer gives these domes
def test_dome_as_world(blender, sample, tmp_path, up, rotation, mapping):
    hdri = sample("blender_institute_hdri")
    scene = Scene(up_axis=up, front_axis="-Y" if up == "+Z" else "+Z")
    scene.add(Node("Tri", mesh=Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2])))
    scene.add_environment(hdri, rotation=rotation, intensity=1.5, exposure=1.0)
    sceneport.save(scene, tmp_path / "sky" / "env.blend", blender=blender)
    world = inspect(blender, tmp_path / "sky" / "env.blend")["world"]
    assert world["strength"] == pytest.approx(3.0)
    assert world["mapping"][0] == "POINT"
    got = euler_xyz_matrix(world["mapping"][1])
    np.testing.assert_allclose(got, euler_xyz_matrix(np.radians(mapping)), atol=1e-6)  # same rotation (Euler may differ)
    assert world["image"].startswith("//") and world["image"].endswith("blender_institute_1k.exr")  # relative path


def euler_xyz_matrix(radians) -> np.ndarray:
    from sceneport.math import euler_matrix

    return euler_matrix(np.degrees(radians), "XYZ")[:3, :3]


def test_turning_dome(blender, sample, tmp_path):
    scene = Scene(up_axis="+Z", front_axis="-Y")
    env = scene.add_environment(sample("blender_institute_hdri"), rotation=(90, 0, 90))
    scene.animations = [Animation("spin", [Channel(env, "rotation", 2, [0.0, 1.0], [90.0, 180.0]),
                                           Channel(env, "light:intensity", 0, [0.0, 1.0], [1.0, 3.0])], fps=24)]  # fmt: skip
    sceneport.save(scene, tmp_path / "spin.blend", blender=blender)
    result = inspect(blender, tmp_path / "spin.blend", frames=[0, 12, 24])
    for frame, (z, strength) in {0: (0, 1.0), 12: (-45, 2.0), 24: (-90, 3.0)}.items():
        state = result["frames"][frame]["_world"]
        np.testing.assert_allclose(euler_xyz_matrix(state["mapping"][1]), euler_xyz_matrix(np.radians([0, 0, z])), atol=1e-5)
        assert state["strength"] == pytest.approx(strength, rel=1e-5)


def eevee_reach(light: dict, threshold: float) -> float:
    """How far EEVEE lets a light reach: sqrt(power / threshold), power = max color x energy x 2^exposure / 100."""
    return math.sqrt(max(light["color"]) * light["energy"] * 2.0 ** light["exposure"] / 100.0 / threshold)


def stage_scene(intensity: float) -> Scene:
    """Spot lights 1 m around a small card at the center, aimed at it (a light stage in miniature)."""
    from sceneport.math import euler_from_matrix

    card = Node("Card", mesh=Mesh("card", [[-0.05, 0, -0.05], [0.05, 0, -0.05], [0.05, 0, 0.05], [-0.05, 0, 0.05]], [4], [0, 1, 2, 3]))
    scene = Scene(nodes=[card], up_axis="+Z", front_axis="-Y")
    for i, d in enumerate([(0, -1, 0), (0.6, -0.8, 0), (-0.6, -0.8, 0), (0, -0.6, 0.8)]):
        z = np.array(d, dtype=float)
        x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        rotation = euler_from_matrix(np.column_stack([x, np.cross(z, x), z]), "XYZ")
        light = Light("sphere", intensity=intensity, radius=0.025, cone_angle=17.0, cone_softness=1.0, normalize=True)
        scene.add(Node(f"Lamp{i}", Transform(translation=z, rotation=rotation), light=light))
    return scene


def test_dim_lights_lower_the_eevee_threshold(blender, tmp_path):
    """Physically scaled lamps (0.37 W, 1 m away) fall below EEVEE's default threshold (0.01) and would render
    black: the threshold is lowered until every light reaches 4x the scene's size."""
    sceneport.save(stage_scene(0.117), tmp_path / "dim.blend", blender=blender)
    result = inspect(blender, tmp_path / "dim.blend")
    threshold = result["scene"]["eevee_light_threshold"]
    assert threshold < 0.01
    lights = [o for o in result["objects"] if o["type"] == "LIGHT"]
    corners = np.array([o["matrix_world"][:3, 3] for o in result["objects"]] + [[-0.05, 0, -0.05], [0.05, 0, 0.05]])
    size = np.linalg.norm(corners.max(0) - corners.min(0))
    for light in lights:
        assert eevee_reach(light["light"], threshold) == pytest.approx(4 * size, rel=1e-3) or \
            eevee_reach(light["light"], threshold) > 4 * size  # fmt: skip
    rendered = run_in_blender("render_eevee_center", {"path": str(tmp_path / "dim.blend")}, blender=blender)
    # As physics has it (black at the default threshold): each lamp gives P / (4 pi) W/m^2 at 1 m on its axis,
    # times the cosine to the card's normal; Blender's default surface (albedo 0.8) returns albedo / pi of it.
    watts = lights[0]["light"]["energy"]
    cosines = sum(max(0.0, -o["matrix_world"][1, 3]) for o in lights)  # the card faces -Y; lamps are 1 m out
    assert rendered == pytest.approx(0.8 / math.pi * watts / (4 * math.pi) * cosines, rel=0.1)


def test_bright_lights_keep_the_default_threshold(blender, tmp_path):
    sceneport.save(stage_scene(100.0), tmp_path / "bright.blend", blender=blender)
    assert inspect(blender, tmp_path / "bright.blend")["scene"]["eevee_light_threshold"] == pytest.approx(0.01)


def test_lights_switched_off_still_get_their_reach(blender, tmp_path):
    """A light that's off part of the time (energy 0, e.g. one light at a time) needs its reach when it's on:
    its dimmest non-zero energy sets the threshold, not the 0."""
    scene = stage_scene(0.117)
    lamps = [n for n, _ in scene.walk() if n.light is not None]
    scene.animations = [Animation("olat", [
        Channel(lamp, "light:intensity", 0, [0.0, i / 24, (i + 1) / 24], [0.0, 0.117, 0.0], "constant")
        for i, lamp in enumerate(lamps)
    ], fps=24)]  # fmt: skip
    sceneport.save(scene, tmp_path / "olat.blend", blender=blender)
    lit = inspect(blender, tmp_path / "olat.blend")
    always_on = tmp_path / "on.blend"
    sceneport.save(stage_scene(0.117), always_on, blender=blender)
    assert lit["scene"]["eevee_light_threshold"] == pytest.approx(inspect(blender, always_on)["scene"]["eevee_light_threshold"])


def test_unassigned_clips_count_for_the_eevee_threshold(blender, tmp_path):
    """A later clip (an action kept with a fake user) that dims the lights still sets the threshold, so switching
    to it in Blender doesn't cull them."""
    bright, dim = stage_scene(100.0), stage_scene(0.117)
    lamps = [n for n, _ in bright.walk() if n.light is not None]
    bright.animations = [
        Animation("bright", [Channel(lamp, "light:intensity", 0, [0.0, 1.0], [100.0, 90.0]) for lamp in lamps], fps=24),
        Animation("dim", [Channel(lamp, "light:intensity", 0, [0.0, 1.0], [0.117, 0.2]) for lamp in lamps], fps=24),
    ]  # fmt: skip
    sceneport.save(bright, tmp_path / "two.blend", blender=blender)
    sceneport.save(dim, tmp_path / "dim.blend", blender=blender)
    assert inspect(blender, tmp_path / "two.blend")["scene"]["eevee_light_threshold"] == pytest.approx(
        inspect(blender, tmp_path / "dim.blend")["scene"]["eevee_light_threshold"], rel=1e-3)


def test_scene_camera_exposure_is_the_view_exposure(blender, tmp_path):
    cam = Node("Cam", Transform(translation=[0, -2, 0], rotation=[90, 0, 0]), camera=Camera(exposure=2.0))
    other = Node("Other", Transform(translation=[2, 0, 0]), camera=Camera(exposure=0.5))
    scene = Scene(nodes=[cam, other], up_axis="+Z", front_axis="-Y")
    scene.animations = [Animation("expose", [Channel(cam, "camera:exposure", 0, [0.0, 1.0], [2.0, -1.0])], fps=24)]
    with pytest.warns(UserWarning, match="one exposure for the view"):
        sceneport.save(scene, tmp_path / "exposure.blend", blender=blender)
    result = inspect(blender, tmp_path / "exposure.blend", frames=[0, 12, 24])
    assert result["scene_camera"] == "Cam"
    assert [result["frames"][f]["_exposure"] for f in (0, 12, 24)] == pytest.approx([2.0, 0.5, -1.0], abs=1e-5)
    assert "view_settings.exposure" in result["actions"][0]["fcurves"]
