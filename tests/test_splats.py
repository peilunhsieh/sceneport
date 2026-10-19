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


@pytest.mark.parametrize("suffix", [".obj", ".ply", ".usda", ".fbx"])
def test_writers_warn_instead_of_dropping_silently(tmp_path, suffix):
    triangle = sceneport.Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2])  # OBJ and PLY need a mesh
    scene = Scene(nodes=[Node("tri", mesh=triangle), Node("splats", splats=GaussianSplats("s", np.zeros((1, 3)), np.ones((1, 3))))])
    with pytest.warns(UserWarning, match="Gaussian splats"):
        sceneport.save(scene, tmp_path / f"out{suffix}")
