import numpy as np
import pytest

from sceneport.math import euler_matrix, quaternion_continuity, quaternion_from_matrix
from sceneport.scene import Animation, Channel, Scene
from helpers import rig_scene


def spin_mid(scene: Scene, orientation: bool = False) -> Animation:
    """`mid` rotates 0 -> 90 degrees about Z over one second (linear); "Lift" goes 0 -> 1."""
    mid, mesh = scene.find("mid"), scene.meshes[0]
    if orientation:
        q = quaternion_from_matrix(euler_matrix([[0, 0, 0], [0, 0, 90]]))
        channels = [Channel(mid, "orientation", c, [0.0, 1.0], q[:, c]) for c in range(4)]
    else:
        channels = [Channel(mid, "rotation", 2, [0.0, 1.0], [0.0, 90.0])]
    channels.append(Channel(mesh, "Lift", 0, [0.0, 1.0], [0.0, 1.0]))
    return Animation("spin", channels, fps=4)


# -- channels -----------------------------------------------------------------


def test_channel_interpolation_modes():
    times, values = [0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 10.0, 0.0]
    linear = Channel(None, "translation", 0, times, values, "linear")
    np.testing.assert_allclose(linear.evaluate([0.5, 1.5, 2.5]), [5, 10, 5])
    constant = Channel(None, "translation", 0, times, values, "constant")
    np.testing.assert_allclose(constant.evaluate([0.5, 2.99]), [0, 10])
    mixed = Channel(None, "translation", 0, times, values, ["constant", "linear", "linear", "linear"])
    np.testing.assert_allclose(mixed.evaluate([0.5, 2.5]), [0, 5])
    cubic = Channel(None, "translation", 0, times, values, "cubic")
    out = cubic.evaluate(np.linspace(0, 3, 31))
    np.testing.assert_allclose(cubic.evaluate(times), values)  # passes through the keys
    assert out[5] > 5  # eases in faster than linear between (0, 0) and (1, 10)
    assert np.all(np.diff(cubic.evaluate(np.linspace(0, 1, 11))) > 0)


def test_channel_holds_outside_keys():
    channel = Channel(None, "translation", 0, [1.0, 2.0], [3.0, 5.0])
    np.testing.assert_allclose(channel.evaluate([-1.0, 1.0, 2.0, 9.0]), [3, 3, 5, 5])
    np.testing.assert_allclose(Channel(None, "scale", 1, [0.5], [2.0]).evaluate([0, 1, 2]), [2, 2, 2])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(times=[1.0, 0.0], values=[0.0, 1.0]), "strictly increasing"),
        (dict(times=[0.0, 1.0], values=[0.0]), "one value"),
        (dict(times=[0.0], values=[0.0], interpolation="bezier"), "interpolation"),
        (dict(times=[0.0], values=[0.0], component=3), "component"),
    ],
)
def test_channel_validate(kwargs, message):
    args = dict(target=None, property="rotation", component=0, times=[0.0], values=[0.0]) | kwargs
    with pytest.raises(ValueError, match=message):
        Channel(**args).validate()


# -- clips --------------------------------------------------------------------


def test_frame_times_and_range():
    clip = spin_mid(rig_scene())
    assert clip.time_range() == (0.0, 1.0)
    np.testing.assert_allclose(clip.frame_times(), [0, 0.25, 0.5, 0.75, 1.0])
    clip.start, clip.end = 0.5, 1.0
    np.testing.assert_allclose(clip.frame_times(), [0.5, 0.75, 1.0])


def test_world_matrices_match_posing_by_hand():
    scene = rig_scene()
    clip = spin_mid(scene)
    times = clip.frame_times()
    world = clip.world_matrices(scene, times)
    assert world[scene.find("tip.end")].shape == (5, 4, 4)
    for f, t in enumerate(times):
        scene.find("mid").transform.rotation = np.array([0.0, 0.0, 90.0 * t])
        by_hand = scene.world_matrices()
        for node in by_hand:
            np.testing.assert_allclose(world[node][f], by_hand[node], atol=1e-12)


def test_orientation_channels_match_euler():
    euler_scene, quat_scene = rig_scene(), rig_scene()
    a, b = spin_mid(euler_scene), spin_mid(quat_scene, orientation=True)
    for t in (0.0, 1.0):  # keys agree exactly; in between, component-wise quaternion lerp differs slightly
        np.testing.assert_allclose(
            a.world_matrices(euler_scene, [t])[euler_scene.find("tip.end")],
            b.world_matrices(quat_scene, [t])[quat_scene.find("tip.end")],
            atol=1e-12,
        )
    mid_q = b.node_values([0.5])[quat_scene.find("mid")]["orientation"]
    np.testing.assert_allclose(np.linalg.norm(mid_q, axis=1), 1)  # normalized after interpolation


def test_blend_weights_and_skinned_vertices():
    scene = rig_scene()
    mesh = scene.meshes[0]
    clip = spin_mid(scene)
    clip.channels = clip.channels[:1]  # rotation only: "Lift" keeps its default weight 0.5
    np.testing.assert_allclose(clip.blend_weights(mesh, [0.0, 1.0]), [[0.5], [0.5]])

    clip = spin_mid(scene)
    times = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(clip.blend_weights(mesh, times), [[0.0], [0.5], [1.0]])
    frames = clip.skinned_vertices(scene, mesh, times)
    assert frames.shape == (3, 6, 3)
    for f, t in enumerate(times):
        scene.find("mid").transform.rotation = np.array([0.0, 0.0, 90.0 * t])
        by_hand = mesh.skinned_vertices(mesh.skin.joint_world(scene), {"Lift": t})
        np.testing.assert_allclose(frames[f], by_hand, atol=1e-12)


# -- FBX ----------------------------------------------------------------------


def fbx_roundtrip(scene: Scene) -> Scene:
    from sceneport.interop.fbx import document_from_scene, scene_from_document
    from sceneport.interop.fbx.binary import dumps, loads
    from sceneport.interop.fbx.document import FBXDocument

    return scene_from_document(FBXDocument.from_file(loads(dumps(document_from_scene(scene).to_file()))))


def keys(clip: Animation) -> dict:
    return {(c.target.name, c.property, c.component): (c.times, c.values, list(c.interpolation)) for c in clip.channels}


def mixed_clip(scene: Scene) -> Animation:
    """Translation (cubic), rotation (constant then linear) and a blend shape weight."""
    mid, root, mesh = scene.find("mid"), scene.find("root"), scene.meshes[0]
    return Animation("take", [
        Channel(root, "translation", 1, [0.0, 0.5, 1.0], [0.0, 2.0, 0.0], "cubic"),
        Channel(mid, "rotation", 2, [0.0, 0.5, 1.0], [0.0, 45.0, 90.0], ["constant", "linear", "linear"]),
        Channel(mesh, "Lift", 0, [0.0, 1.0], [0.25, 1.0]),
    ], fps=24)  # fmt: skip


def test_fbx_roundtrip_keys_and_evaluation():
    scene = rig_scene()
    scene.animations = [mixed_clip(scene)]
    back = fbx_roundtrip(scene)
    (clip,) = back.animations
    assert (clip.name, clip.fps, clip.time_range()) == ("take", 24.0, (0.0, 1.0))
    expected, actual = keys(scene.animations[0]), keys(clip)
    assert expected.keys() == actual.keys()
    for k in expected:
        np.testing.assert_allclose(actual[k][0], expected[k][0], atol=1e-9)  # int64 ticks
        np.testing.assert_allclose(actual[k][1], expected[k][1], rtol=1e-6)  # float32 values
        assert actual[k][2] == expected[k][2]
    times = np.linspace(0, 1, 13)
    mesh, mesh2 = scene.meshes[0], back.meshes[0]
    np.testing.assert_allclose(
        clip.skinned_vertices(back, mesh2, times), scene.animations[0].skinned_vertices(scene, mesh, times), atol=1e-5
    )


def test_fbx_layout():
    from sceneport.interop.fbx import document_from_scene

    scene = rig_scene()
    scene.animations = [mixed_clip(scene)]
    doc = document_from_scene(scene)
    (stack,) = doc.iter_objects("AnimationStack")
    assert (stack.name, stack.cls) == ("take", "AnimStack")
    (layer,) = doc.children_of(stack, "AnimationLayer")
    targets = {}
    for curve_node in doc.children_of(layer, "AnimationCurveNode"):
        (link,) = [c for c in doc.connections if c.src == curve_node.id and c.kind == "OP"]
        targets[curve_node.name] = (doc.objects[link.dst].name, link.prop)
    assert targets == {"T": ("root", "Lcl Translation"), "R": ("mid", "Lcl Rotation"), "DeformPercent": ("Lift", "DeformPercent")}
    percent = next(c for c in doc.iter_objects("AnimationCurve") if c.child_value("KeyValueFloat")[-1] == 100.0)
    assert percent.child_value("KeyValueFloat").tolist() == [25.0, 100.0]  # weights in percent
    assert percent.child_value("KeyTime").tolist() == [0, 46186158000]
    settings = doc.global_settings
    assert (settings.get("TimeMode"), settings.get("TimeSpanStop")) == (11, 46186158000)  # 24 fps


def test_fbx_orientation_becomes_euler():
    scene = rig_scene()
    # Keys must be sign-continuous, or interpolation takes the long way round.
    q = quaternion_continuity(quaternion_from_matrix(euler_matrix([[0, 0, 0], [0, 0, 170], [0, 0, 340]], 0)))
    mid = scene.find("mid")
    scene.animations = [Animation("q", [Channel(mid, "orientation", c, [0.0, 1.0, 2.0], q[:, c]) for c in range(4)], fps=10)]
    back_scene = fbx_roundtrip(scene)
    (clip,) = back_scene.animations
    assert {c.property for c in clip.channels} == {"rotation"}
    z = next(c for c in clip.channels if c.component == 2)
    assert len(z.times) == 21  # a key per frame
    assert np.all(np.diff(z.values) > 0)  # unwrapped: no jump from 180 to -180
    source = scene.animations[0].world_matrices(scene, z.times)[mid]
    back = clip.world_matrices(back_scene, z.times)[back_scene.find("mid")]
    np.testing.assert_allclose(back, source, atol=1e-6)  # float32 Euler keys


def test_fbx_several_clips_and_custom_rate():
    scene = rig_scene()
    mid = scene.find("mid")
    scene.animations = [
        Animation("a", [Channel(mid, "rotation", 0, [0.0, 1.0], [0.0, 10.0])], fps=12.5),
        Animation("b", [Channel(mid, "rotation", 1, [0.0, 2.0], [0.0, 20.0])], fps=12.5),
    ]
    back = fbx_roundtrip(scene)
    assert [(c.name, c.fps) for c in back.animations] == [("a", 12.5), ("b", 12.5)]
    assert back.animations[1].time_range() == (0.0, 2.0)


# -- USD ----------------------------------------------------------------------


def usd_roundtrip(scene: Scene) -> Scene:
    from sceneport.interop.usd import scene_from_stage, stage_from_scene

    return scene_from_stage(stage_from_scene(scene))


def by_name(scene: Scene, name: str):
    return next(n for n, _ in scene.walk() if n.name == name)


def test_usd_roundtrip_matches_evaluation():
    scene = rig_scene()
    scene.animations = [mixed_clip(scene)]
    back = usd_roundtrip(scene)
    (clip,) = back.animations
    assert (clip.name, clip.fps, clip.time_range()) == ("take", 24.0, (0.0, 1.0))
    times = scene.animations[0].frame_times()  # USD holds a sample per frame
    np.testing.assert_allclose(
        clip.skinned_vertices(back, back.meshes[0], times),
        scene.animations[0].skinned_vertices(scene, scene.meshes[0], times),
        atol=1e-5,
    )
    # Rest pose and default weight are not overwritten by the animation's first frame.
    np.testing.assert_allclose(back.find("root").transform.translation, [0, 0, 0], atol=1e-12)
    assert back.meshes[0].blend_shapes[0].weight == 0.5


def test_usd_skel_animation_matches_usdskel():
    from pxr import Usd, UsdGeom, UsdSkel

    from sceneport.interop.usd import stage_from_scene

    scene = rig_scene()
    clip = mixed_clip(scene)
    scene.animations = [clip]
    stage = stage_from_scene(scene)
    assert (stage.GetTimeCodesPerSecond(), stage.GetStartTimeCode(), stage.GetEndTimeCode()) == (24.0, 0.0, 24.0)
    skeleton = stage.GetPrimAtPath("/SkelRoot/root")
    assert [str(t) for t in UsdSkel.BindingAPI(skeleton).GetAnimationSourceRel().GetTargets()] == ["/SkelRoot/root/take"]
    assert UsdSkel.BakeSkinning(stage.Traverse())
    usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/SkelRoot/strip/Strip"))
    times = clip.frame_times()
    ours = clip.skinned_vertices(scene, scene.meshes[0], times)
    for f in range(0, len(times), 6):
        to_world = np.array(usd_mesh.ComputeLocalToWorldTransform(f)).T
        baked = np.array(usd_mesh.GetPointsAttr().Get(f)) @ to_world[:3, :3].T + to_world[:3, 3]
        np.testing.assert_allclose(baked, ours[f], atol=1e-5, err_msg=f"frame {f}")


def test_usd_node_animation_and_several_clips():
    scene = rig_scene()
    strip, helper, mid = scene.find("strip"), scene.find("helper"), scene.find("mid")
    first = Animation("first", [
        Channel(strip, "translation", 0, [0.0, 1.0], [0.0, 3.0]),  # a regular node: xformOp time samples
        Channel(helper, "rotation", 2, [0.0, 1.0], [30.0, 120.0]),  # an attachment under a joint
        Channel(mid, "rotation", 2, [0.0, 1.0], [0.0, 45.0]),
    ], fps=10)  # fmt: skip
    second = Animation("second", [Channel(mid, "rotation", 0, [0.0, 0.5], [0.0, 20.0])], fps=10)
    scene.animations = [first, second]
    back = usd_roundtrip(scene)
    assert [c.name for c in back.animations] == ["first", "second"]  # the timeline's clip first
    times = first.frame_times()
    expected = first.world_matrices(scene, times)
    actual = back.animations[0].world_matrices(back, times)
    for name in ("strip", "helper", "tip.end"):
        np.testing.assert_allclose(actual[by_name(back, name)], expected[by_name(scene, name)], atol=1e-5, err_msg=name)
    np.testing.assert_allclose(
        back.animations[1].world_matrices(back, [0.5])[by_name(back, "tip.end")],
        second.world_matrices(scene, [0.5])[by_name(scene, "tip.end")],
        atol=1e-5,
    )
    np.testing.assert_allclose(back.find("strip").transform.translation, [0, 0, 0.5])  # rest kept


def test_usd_unskinned_blend_shape_animation():
    from helpers import shaped_mesh
    from sceneport.scene import Node

    mesh = shaped_mesh()
    scene = Scene(nodes=[Node("Face", mesh=mesh)])
    scene.animations = [Animation("talk", [Channel(mesh, "Wide", 0, [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])], fps=2)]
    back = usd_roundtrip(scene)
    (clip,) = back.animations
    (channel,) = clip.channels
    assert (channel.target is back.meshes[0], channel.property) == (True, "Wide")
    np.testing.assert_allclose(channel.values, [0, 0.5, 1, 0.5, 0], atol=1e-6)  # baked at 2 fps
    assert back.meshes[0].blend_shapes[1].weight == 1.0  # default kept
