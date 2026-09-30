"""Tests on downloaded sample files (see fetch_samples.py); each skips until its sample is fetched."""

import numpy as np

import sceneport


def world_by_path(scene, clip, times) -> dict:
    """World matrices per frame, keyed by each node's name path from the root ("Spring_high/GEO_spring_scarf")."""
    parents = scene.parents()

    def path(node):
        return node.name if parents[node] is None else f"{path(parents[node])}/{node.name}"

    matrices = {path(node): m for node, m in clip.world_matrices(scene, times).items()}
    assert len(matrices) == len(parents), "node paths must be unique to compare scenes"
    return matrices


def test_spring_runcycle(sample, tmp_path):
    scene = sceneport.load(sample("spring_runcycle"))
    assert (scene.meters_per_unit, scene.up_axis) == (1.0, "+Z")  # metersPerUnit is stored as a float here
    assert len(scene.meshes) == 165
    assert len(scene.materials) == 21
    assert all(m.pbr() is not None and m.pbr().source == "UsdPreviewSurface" for m in scene.materials)
    (clip,) = scene.animations
    assert clip.fps == 24
    assert len({c.target for c in clip.channels}) == 211  # animated transforms

    times = clip.frame_times()[::10]
    expected = world_by_path(scene, clip, times)
    for suffix, atol in ((".usdc", 1e-4), (".fbx", 1e-3)):  # FBX keys are float32
        sceneport.save(scene, tmp_path / f"spring{suffix}")
        back = sceneport.load(tmp_path / f"spring{suffix}")
        assert len(back.meshes) == 165, suffix
        assert sorted(m.name for m in back.materials) == sorted(m.name for m in scene.materials), suffix
        slots = sorted(len(n.materials) for n, _ in back.walk() if n.mesh is not None)
        assert slots == sorted(len(n.materials) for n, _ in scene.walk() if n.mesh is not None), suffix
        actual = world_by_path(back, back.animations[0], times)
        for name, matrices in expected.items():
            np.testing.assert_allclose(actual[name], matrices, atol=atol, err_msg=f"{suffix} {name}")
