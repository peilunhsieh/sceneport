"""The Gaussian splat scene type."""

import numpy as np
import pytest

import sceneport
from sceneport.scene import GaussianSplats, Node, Scene
from sceneport.scene.splats import SH_C0


def test_defaults_and_shapes():
    splats = GaussianSplats("s", np.zeros((2, 3)), np.full((2, 3), 0.1))
    assert len(splats) == 2 and splats.sh_degree == 0
    np.testing.assert_array_equal(splats.orientations, [[1, 0, 0, 0], [1, 0, 0, 0]])  # identity
    np.testing.assert_array_equal(splats.opacities, [1, 1])
    np.testing.assert_allclose(splats.base_colors(), 0.5)  # zero SH is gray in the 3DGS convention


def test_sh_degree_and_base_colors():
    sh = np.zeros((1, 16, 3))
    sh[0, 0] = [1.0, 0.0, -1.0] / np.array(SH_C0) * 0.25
    splats = GaussianSplats("s", np.zeros((1, 3)), np.ones((1, 3)), sh=sh)
    assert splats.sh_degree == 3
    np.testing.assert_allclose(splats.base_colors(), [[0.75, 0.5, 0.25]])


@pytest.mark.parametrize("kwargs", [{"sh": np.zeros((2, 5, 3))}, {"attributes": {"binding": np.zeros(3)}}])
def test_rejects_mismatched_data(kwargs):
    with pytest.raises(ValueError):
        GaussianSplats("s", np.zeros((2, 3)), np.ones((2, 3)), **kwargs)


def gaussian_ply(path, props: dict, ascii: bool = False) -> None:
    """A vertex-only PLY with float properties in the given order (the 3DGS reference layout)."""
    n = len(next(iter(props.values())))
    header = f"ply\nformat {'ascii' if ascii else 'binary_little_endian'} 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in props) + "end_header\n"
    table = np.stack([np.asarray(v, dtype=np.float32) for v in props.values()], axis=1)
    body = ("\n".join(" ".join(repr(float(x)) for x in row) for row in table) + "\n").encode() if ascii else table.tobytes()
    path.write_bytes(header.encode() + body)


def reference_props(n=3, degree=3, rng=np.random.default_rng(0)):
    props = {a: rng.normal(size=n) for a in ("x", "y", "z")} | {a: np.zeros(n) for a in ("nx", "ny", "nz")}
    props |= {f"f_dc_{i}": rng.normal(size=n) for i in range(3)}
    props |= {f"f_rest_{i}": rng.normal(size=n) for i in range(3 * ((degree + 1) ** 2 - 1))}
    props |= {"opacity": rng.normal(size=n) * 3}
    props |= {f"scale_{i}": rng.normal(size=n) - 4 for i in range(3)}
    props |= {f"rot_{i}": rng.normal(size=n) for i in range(4)}
    return props


@pytest.mark.parametrize("ascii", [False, True])
def test_reads_3dgs_ply(tmp_path, ascii):
    props = reference_props() | {"binding_0": np.array([7.0, 0.0, 10143.0])}  # GaussianAvatars' parent triangle
    gaussian_ply(tmp_path / "avatar.ply", props, ascii=ascii)
    (node,) = sceneport.load(tmp_path / "avatar.ply").nodes
    splats = node.splats
    assert node.mesh is None and splats.name == "avatar" and len(splats) == 3 and splats.sh_degree == 3
    f32 = {k: np.asarray(v, dtype=np.float32).astype(float) for k, v in props.items()}
    np.testing.assert_allclose(splats.positions, np.stack([f32[a] for a in "xyz"], axis=1))
    np.testing.assert_allclose(splats.scales, np.exp(np.stack([f32[f"scale_{i}"] for i in range(3)], axis=1)))
    np.testing.assert_allclose(splats.opacities, 1 / (1 + np.exp(-f32["opacity"])))
    rot = np.stack([f32[f"rot_{i}"] for i in range(4)], axis=1)
    np.testing.assert_allclose(splats.orientations, rot / np.linalg.norm(rot, axis=1, keepdims=True))
    np.testing.assert_allclose(splats.sh[:, 0], np.stack([f32[f"f_dc_{i}"] for i in range(3)], axis=1))
    # f_rest holds red's 15 higher-order coefficients, then green's, then blue's.
    np.testing.assert_allclose(splats.sh[:, 1:, 1], np.stack([f32[f"f_rest_{15 + k}"] for k in range(15)], axis=1))
    assert set(splats.attributes) == {"binding_0"}
    np.testing.assert_array_equal(splats.attributes["binding_0"], [7, 0, 10143])


def test_reads_2dgs_and_color_only_splats(tmp_path):
    props = {k: v for k, v in reference_props(degree=0).items() if k not in ("scale_2", "f_dc_0", "f_dc_1", "f_dc_2")}
    props |= {"red": np.array([255.0, 0.0, 128.0]), "green": np.zeros(3), "blue": np.full(3, 255.0)}
    gaussian_ply(tmp_path / "surfels.ply", props)
    splats = sceneport.load(tmp_path / "surfels.ply").nodes[0].splats
    assert splats.sh_degree == 0 and not splats.attributes
    np.testing.assert_array_equal(splats.scales[:, 2], 0)  # flat Gaussians
    np.testing.assert_allclose(splats.base_colors(), [[1, 0, 1], [0, 0, 1], [128 / 255, 0, 1]], atol=1e-6)


def test_point_cloud_without_splat_properties_is_still_a_mesh(tmp_path):
    gaussian_ply(tmp_path / "points.ply", {a: np.arange(3.0) for a in "xyz"})
    (node,) = sceneport.load(tmp_path / "points.ply").nodes
    assert node.splats is None and node.mesh.n_vertices == 3


def assert_same_splats(a, b, rtol=1e-6):
    for field in ("positions", "scales", "orientations", "opacities", "sh"):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), rtol=rtol, atol=1e-7, err_msg=field)
    assert set(a.attributes) == set(b.attributes)
    for name in a.attributes:
        np.testing.assert_array_equal(a.attributes[name], b.attributes[name])


@pytest.mark.parametrize("ascii", [False, True])
def test_3dgs_ply_round_trip(tmp_path, ascii):
    gaussian_ply(tmp_path / "in.ply", reference_props() | {"binding_0": np.array([7.0, 0.0, 10143.0])})
    scene = sceneport.load(tmp_path / "in.ply")
    sceneport.save(scene, tmp_path / "out.ply", ascii=ascii)
    back = sceneport.load(tmp_path / "out.ply").nodes[0].splats
    assert_same_splats(scene.nodes[0].splats, back)
    header = (tmp_path / "out.ply").read_bytes().split(b"end_header")[0].decode()
    names = [line.split()[-1] for line in header.splitlines() if line.startswith("property")]
    assert names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]  # the reference order
    assert names[-6:] == ["scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "binding_0"]


def test_flat_splats_keep_two_scales(tmp_path):
    props = {k: v for k, v in reference_props(degree=1).items() if k != "scale_2"}
    gaussian_ply(tmp_path / "in.ply", props)
    sceneport.save(sceneport.load(tmp_path / "in.ply"), tmp_path / "out.ply")
    header = (tmp_path / "out.ply").read_bytes().split(b"end_header")[0].decode()
    assert "scale_1" in header and "scale_2" not in header
    np.testing.assert_array_equal(sceneport.load(tmp_path / "out.ply").nodes[0].splats.scales[:, 2], 0)


def covariances(splats):
    from sceneport.math import quaternion_matrix

    axes = quaternion_matrix(splats.orientations)[:, :3, :3] * splats.scales[:, None, :]
    return axes @ axes.transpose(0, 2, 1)


@pytest.mark.parametrize("linear", [
    2.0 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # rotation x uniform scale: axes kept
    np.array([[1.0, 0.5, 0], [0, 2, 0], [0, 0, -0.5]]),  # shear, non-uniform scale and a mirror
])  # fmt: skip
def test_transformed_maps_covariances(linear):
    rng = np.random.default_rng(1)
    q = rng.normal(size=(5, 4))
    splats = GaussianSplats("s", rng.normal(size=(5, 3)), np.exp(rng.normal(size=(5, 3))), q / np.linalg.norm(q, axis=1)[:, None])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = linear, [1, 2, 3]
    moved = splats.transformed(matrix)
    np.testing.assert_allclose(moved.positions, splats.positions @ linear.T + [1, 2, 3])
    np.testing.assert_allclose(covariances(moved), linear @ covariances(splats) @ linear.T, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(moved.orientations, axis=1), 1)


def test_node_transform_and_units_applied_on_write(tmp_path):
    splats = GaussianSplats("s", [[1.0, 0, 0]], [[0.1, 0.2, 0.3]], sh=np.zeros((1, 4, 3)))
    rotate = sceneport.Transform(rotation=np.array([0.0, 0.0, 90.0]))
    scene = Scene(nodes=[Node("splats", rotate, splats=splats)], meters_per_unit=0.01)  # centimeters
    with pytest.warns(UserWarning, match="view-dependent"):
        sceneport.save(scene, tmp_path / "out.ply")
    back = sceneport.load(tmp_path / "out.ply").nodes[0].splats
    np.testing.assert_allclose(back.positions, [[0, 0.01, 0]], atol=1e-9)
    np.testing.assert_allclose(back.scales, [[0.001, 0.002, 0.003]], rtol=1e-6)


@pytest.mark.parametrize("suffix", [".obj", ".ply", ".usda", ".fbx"])
def test_writers_warn_instead_of_dropping_silently(tmp_path, suffix):
    triangle = sceneport.Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2])  # OBJ and PLY need a mesh
    scene = Scene(nodes=[Node("tri", mesh=triangle), Node("splats", splats=GaussianSplats("s", np.zeros((1, 3)), np.ones((1, 3))))])
    with pytest.warns(UserWarning, match="Gaussian splats"):
        sceneport.save(scene, tmp_path / f"out{suffix}")
