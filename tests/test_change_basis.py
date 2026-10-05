"""reframe.change_basis: a scene moved to other axes and units looks the same, with exact channels."""

import numpy as np
import pytest

from sceneport.interop.reframe import basis_matrix, change_basis
from sceneport.scene import Animation, BlendShape, Camera, Channel, Light, Mesh, Node, Primvar, Scene, Transform

TIMES = np.linspace(0.0, 1.0, 13)


def y_up_scene(**axes) -> Scene:
    """A centimeter Y-up scene: a parent with a mesh child (rotation order YZX, animated), a node with
    pivots (animated), a quaternion-animated node, a camera and a light."""
    quad = Mesh("quad", [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 5]], [4, 3], [0, 1, 2, 3, 0, 1, 4],
                normals=Primvar(np.tile([0.0, 0.0, 1.0], (7, 1)), "faceVarying"),
                uvs={"UVMap": Primvar(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]), "faceVarying",
                                      np.array([0, 1, 2, 3, 0, 1, 4]))},
                blend_shapes=[BlendShape("up", [[0, 0, 3]], indices=[4], weight=0.5)])  # fmt: skip
    child = Node("Child", Transform(translation=[1, 2, 3], rotation=[10, 20, 30], scale=[1, 2, 3], rotation_order=2,
                                    geometric_translation=[0, 1, 0]), mesh=quad)  # fmt: skip
    pivoted = Node("Pivoted", Transform(translation=[0, 5, 0], rotation=[0, 45, 0], rotation_pivot=[2, 0, 0],
                                        pre_rotation=[0, 0, 15]))  # fmt: skip
    spun = Node("Spun", Transform(translation=[3, 0, 0]))
    cam = Node("Cam", Transform(translation=[0, 150, 500], rotation=[-10, 5, 0]),
               camera=Camera(near=5.0, far=5000.0, focus_distance=300.0))  # fmt: skip
    cam.add(Node("CamChild", Transform(translation=[0, 0, -20])))
    lamp = Node("Lamp", Transform(translation=[50, 300, 0], rotation=[-90, 0, 0]),
                light=Light("sphere", radius=10.0, cone_angle=30.0))  # fmt: skip
    parent = Node("Parent", Transform(translation=[100, 0, 0], rotation=[0, 90, 0]), children=[child, pivoted, spun])
    scene = Scene(nodes=[parent, cam, lamp], meters_per_unit=0.01, **axes)
    q = np.array([[1, 0, 0, 0], [np.cos(0.5), np.sin(0.5), 0, 0], [np.cos(1.0), 0, np.sin(1.0) * 0.6, np.sin(1.0) * 0.8]])
    scene.animations = [Animation("anim", [
        Channel(child, "rotation", 1, [0.0, 1.0], [20.0, 200.0]),
        Channel(child, "translation", 0, [0.0, 0.5, 1.0], [1.0, -4.0, 6.0], "cubic"),
        Channel(child, "scale", 2, [0.0, 1.0], [3.0, 0.5]),
        Channel(pivoted, "rotation", 1, [0.0, 1.0], [45.0, 90.0]),
        Channel(cam, "translation", 2, [0.0, 1.0], [500.0, 200.0]),
        Channel(cam, "camera:focus_distance", 0, [0.0, 1.0], [300.0, 100.0]),
        Channel(quad, "up", 0, [0.0, 1.0], [0.0, 2.0]),
        *[Channel(spun, "orientation", i, [0.0, 0.5, 1.0], q[:, i]) for i in range(4)],
    ], fps=24)]  # fmt: skip
    return scene


def worlds(scene: Scene) -> dict[str, np.ndarray]:
    return {n.name: m for n, m in scene.animations[0].world_matrices(scene, TIMES).items()}


@pytest.mark.parametrize("axes", [{}, {"up_axis": "+Y", "front_axis": "-Z", "right_axis": "+X"}])  # right- / left-handed
def test_scene_looks_the_same(axes):
    scene = y_up_scene(**axes)
    out = change_basis(scene, up="+Z", front="-Y", right="+X", meters_per_unit=1.0)
    assert (out.up_axis, out.front_axis, out.right_axis, out.meters_per_unit) == ("+Z", "-Y", "+X", 1.0)
    c = basis_matrix(scene, "+Z", "-Y", "+X")
    m = np.eye(4)
    m[:3, :3] = 0.01 * c
    before, after = worlds(scene), worlds(out)
    for name in ("Parent", "Child", "Pivoted", "Spun", "CamChild"):
        np.testing.assert_allclose(after[name], m @ before[name] @ np.linalg.inv(m), atol=1e-9, err_msg=name)
    for name in ("Cam", "Lamp"):  # still looking down -Z, from the mapped place, in the mapped direction
        np.testing.assert_allclose(after[name][:, :3, 3], (m @ before[name])[:, :3, 3], atol=1e-9)
        np.testing.assert_allclose(after[name][:, :3, 2], before[name][:, :3, 2] @ c.T, atol=1e-9)
        np.testing.assert_allclose(after[name][:, :3, 1], before[name][:, :3, 1] @ c.T, atol=1e-9)  # and up stays up

    # mesh points in world space (with blend shapes and the geometric offset) map by M
    old, new = scene.find("Child"), out.find("Child")
    old_points = old.mesh.deformed_vertices({"up": 0.5}) @ old.transform.geometric_matrix()[:3, :3].T + old.transform.geometric_matrix()[:3, 3]
    new_points = new.mesh.deformed_vertices({"up": 0.5}) @ new.transform.geometric_matrix()[:3, :3].T + new.transform.geometric_matrix()[:3, 3]
    old_world = old_points @ before["Child"][0, :3, :3].T + before["Child"][0, :3, 3]
    new_world = new_points @ after["Child"][0, :3, :3].T + after["Child"][0, :3, 3]
    np.testing.assert_allclose(new_world, old_world @ (0.01 * c).T, atol=1e-9)
    np.testing.assert_allclose(new.mesh.normals.values, old.mesh.normals.values @ c.T)

    # faces keep facing the mapped normal direction (reversed winding when the change mirrors)
    def face_normal(mesh):
        a, b, c_ = mesh.vertices[mesh.face_vertex_indices[:3]]
        return np.cross(b - a, c_ - a)

    np.testing.assert_allclose(face_normal(new.mesh) / np.linalg.norm(face_normal(new.mesh)),
                               face_normal(old.mesh) @ c.T / np.linalg.norm(face_normal(old.mesh)), atol=1e-9)  # fmt: skip
    uv = lambda mesh: mesh.uvs["UVMap"].expanded()  # noqa: E731
    corner_pairs = {tuple(p) for p in zip(old.mesh.face_vertex_indices, map(tuple, uv(old.mesh)))}
    assert {tuple(p) for p in zip(new.mesh.face_vertex_indices, map(tuple, uv(new.mesh)))} == corner_pairs

    # distances scale with the units
    cam = out.find("Cam").camera
    assert (cam.near, cam.far, cam.focus_distance) == pytest.approx((0.05, 50.0, 3.0))
    assert out.find("Lamp").light.radius == pytest.approx(0.1)
    focus = out.animations[0].attribute_values(out.find("Cam"), "camera", [1.0])["focus_distance"]
    assert focus == pytest.approx([1.0])


def test_plain_channels_map_one_to_one():
    """A plain node keeps its keys (times, interpolation) on permuted, sign-flipped channels: Y-up -> Z-up
    turns x, y, z into x, -z, y, and the Euler order YZX into ZYX."""
    scene = y_up_scene()
    out = change_basis(scene, up="+Z", front="-Y", right="+X", meters_per_unit=1.0)
    child = out.find("Child")
    assert child.transform.rotation_order == 5  # ZYX
    np.testing.assert_allclose(child.transform.translation, [0.01, -0.03, 0.02])
    np.testing.assert_allclose(child.transform.rotation, [10, -30, 20])
    np.testing.assert_allclose(child.transform.scale, [1, 3, 2])
    channels = {(c.property, c.component): c for c in out.animations[0].channels if c.target is child}
    assert set(channels) == {("rotation", 2), ("translation", 0), ("scale", 1)}
    np.testing.assert_allclose(channels[("rotation", 2)].values, [20.0, 200.0])
    np.testing.assert_allclose(channels[("translation", 0)].values, [0.01, -0.04, 0.06])
    assert list(channels[("translation", 0)].interpolation) == ["cubic"] * 3
    spun = {c.component: c.values for c in out.animations[0].channels if c.target is out.find("Spun")}
    assert set(spun) == {0, 1, 2, 3}  # quaternion (w, x, y, z) -> (w, x, -z, y)
    np.testing.assert_allclose(spun[2], -np.array([0, 0, np.sin(1.0) * 0.8]))


def test_same_axes_and_units_is_a_no_op():
    scene = y_up_scene()
    out = change_basis(scene, up="+Y", front="+Z", right="+X", meters_per_unit=0.01)
    assert out.nodes is scene.nodes
