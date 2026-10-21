"""Gaussian splats in .blend files (point clouds with Blender 5.3's splat attributes) and SH rotation."""

import subprocess

import numpy as np
import pytest

import sceneport
from sceneport.interop.blend import BlenderError, run_in_blender
from sceneport.math import quaternion_matrix
from sceneport.scene import GaussianSplats, Node, Scene
from sceneport.scene.splats import rotate_sh, sh_basis


def random_splats(n=6, degree=3, rng=np.random.default_rng(3)):
    q = rng.normal(size=(n, 4))
    return GaussianSplats("splats", rng.normal(size=(n, 3)), np.exp(rng.normal(size=(n, 3)) - 3),
                          q / np.linalg.norm(q, axis=1)[:, None], rng.uniform(0.1, 0.9, n),
                          rng.normal(size=(n, (degree + 1) ** 2, 3)), {"binding_0": np.arange(n, dtype=np.int64) * 7})  # fmt: skip


@pytest.mark.parametrize("rotation", [
    np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),  # Y-up to Z-up
    np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0],  # any rotation or reflection
])  # fmt: skip
def test_rotated_sh_gives_the_same_colors_along_rotated_directions(rotation):
    sh = random_splats().sh
    directions = np.random.default_rng(1).normal(size=(50, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    before = np.einsum("dk,nkc->ndc", sh_basis(directions, 3), sh)
    after = np.einsum("dk,nkc->ndc", sh_basis(directions @ rotation.T, 3), rotate_sh(sh, rotation))
    np.testing.assert_allclose(after, before, atol=1e-10)


def test_splats_written_as_a_blender_point_cloud(blender, tmp_path):
    splats = random_splats()
    scene = Scene(nodes=[Node("avatar", splats=splats)], up_axis="+Y", meters_per_unit=1.0)
    sceneport.save(scene, tmp_path / "splats.blend", blender=blender)
    (obj,) = run_in_blender("inspect_blend", {"path": str(tmp_path / "splats.blend")}, blender=blender)["objects"]
    assert obj["type"] == "POINTCLOUD"
    attributes = {name: values for name, (_, values) in obj["attributes"].items()}

    # Blender is Z-up: (x, y, z) Y-up is (x, -z, y), positions and splat axes alike, and SH colors turn with them.
    to_z_up = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(attributes["position"], splats.positions @ to_z_up.T, atol=1e-6)
    np.testing.assert_allclose(attributes["scale"], splats.scales, rtol=1e-6)
    axes = quaternion_matrix(attributes["rotation"].astype(float))[:, :3, :3]
    np.testing.assert_allclose(axes, to_z_up @ quaternion_matrix(splats.orientations)[:, :3, :3], atol=1e-6)
    np.testing.assert_allclose(attributes["radiance:base"], np.column_stack([splats.sh[:, 0], splats.opacities]), atol=1e-6)
    turned = rotate_sh(splats.sh, to_z_up)
    np.testing.assert_allclose(np.stack([attributes[f"radiance:sh_{k}"] for k in range(15)], axis=1), turned[:, 1:], atol=1e-5)
    np.testing.assert_array_equal(attributes["binding_0"], splats.attributes["binding_0"])
    if obj["pointcloud_type"] is not None:  # Blender 5.3+
        assert obj["pointcloud_type"] == "GAUSSIAN_SPLAT"


def test_scripts_run_before_saving(blender, tmp_path):
    triangle = sceneport.Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2])
    script = tmp_path / "setup.py"
    script.write_text("import bpy\nbpy.data.objects['tri'].add_rest_position_attribute = True\n"
                      "bpy.data.texts.new('ran_' + __name__ + ('_saved' if bpy.data.filepath else '_unsaved'))\n")  # fmt: skip
    sceneport.save(Scene(nodes=[Node("tri", mesh=triangle)]), tmp_path / "tri.blend", blender=blender, scripts=[script])
    check = "import bpy; print('REST', bpy.data.objects['tri'].add_rest_position_attribute, [t.name for t in bpy.data.texts])"
    run = subprocess.run([str(blender.path), "-b", "--factory-startup", str(tmp_path / "tri.blend"), "--python-expr", check],
                         capture_output=True, text=True)  # fmt: skip
    assert "REST True ['ran___main___unsaved']" in run.stdout

    (tmp_path / "broken.py").write_text("raise ValueError('setup failed')\n")
    with pytest.raises(BlenderError, match="setup failed"):
        sceneport.save(Scene(nodes=[Node("tri", mesh=triangle)]), tmp_path / "tri.blend", blender=blender,
                       scripts=[tmp_path / "broken.py"])  # fmt: skip
