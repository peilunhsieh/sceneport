"""Tests on downloaded sample files (see fetch_samples.py); each skips until its sample is fetched."""

import warnings
from pathlib import Path

import numpy as np
import pytest

import sceneport
from sceneport.interop.usd import scene_from_stage


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
    assert len(clip.point_samples) == 148  # vertex-animated meshes (time-sampled points)

    times = clip.frame_times()[::10]
    expected = world_by_path(scene, clip, times)
    for suffix, atol in ((".usdc", 1e-4), (".fbx", 1e-3)):  # FBX keys are float32
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "FBX can't hold per-vertex animation")
            sceneport.save(scene, tmp_path / f"spring{suffix}")
        back = sceneport.load(tmp_path / f"spring{suffix}")
        assert len(back.meshes) == 165, suffix
        assert sorted(m.name for m in back.materials) == sorted(m.name for m in scene.materials), suffix
        slots = sorted(len(n.materials) for n, _ in back.walk() if n.mesh is not None)
        assert slots == sorted(len(n.materials) for n, _ in scene.walk() if n.mesh is not None), suffix
        actual = world_by_path(back, back.animations[0], times)
        for name, matrices in expected.items():
            np.testing.assert_allclose(actual[name], matrices, atol=atol, err_msg=f"{suffix} {name}")
    # The vertex animation survives a trip through USD (FBX can't hold it).
    back = sceneport.load(tmp_path / "spring.usdc")
    (back_clip,) = back.animations
    assert len(back_clip.point_samples) == 148
    for samples, back_samples in zip(clip.point_samples[:10], back_clip.point_samples[:10]):
        np.testing.assert_allclose(back_clip.vertices(back_samples.mesh, times), clip.vertices(samples.mesh, times), atol=1e-4)


# -- Pixar's UsdSkel example: HumanFemale -------------------------------------------------

HUMAN_FEMALE = "HumanFemale"  # folder in the usdskel_examples sample


def open_stage(path, variants: dict[str, str]):
    """The stage with `variants` selected on its default prim, edits going to the session layer
    (the sample's layers are cached by USD and shared between stages, so they must stay untouched)."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    stage.SetEditTarget(stage.GetSessionLayer())
    for name, value in variants.items():
        stage.GetDefaultPrim().GetVariantSet(name).SetVariantSelection(value)
    return stage


def usdskel_points(stage, codes) -> list:
    """World-space points of every mesh (in traversal order) at `codes`, skinned by UsdSkel itself."""
    from pxr import Gf, UsdGeom, UsdSkel

    assert UsdSkel.BakeSkinning(stage.Traverse(), Gf.Interval(codes[0], codes[-1]))
    out = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            frames = []
            for code in codes:
                to_world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(code)).T
                points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(code), dtype=np.float64)
                frames.append(points @ to_world[:3, :3].T + to_world[:3, 3])
            out.append(np.stack(frames))
    return out


def test_human_female_structure(sample):
    scene = sceneport.load(sample("usdskel_examples") / HUMAN_FEMALE / "HumanFemale.usd")
    assert (scene.meters_per_unit, scene.up_axis) == (0.01, "+Z")
    assert len(scene.meshes) == 89 and all(mesh.skin is not None for mesh in scene.meshes)  # composed from payloads
    assert sum(1 for node, _ in scene.walk() if node.joint) == 66
    assert sum(len(mesh.blend_shapes) for mesh in scene.meshes) == 1229
    assert len(scene.materials) == 14
    textures = []
    for material in scene.materials:
        pbr = material.pbr()
        assert pbr is not None and pbr.source == "UsdPreviewSurface"
        textures += [t for t in (pbr.base_color_texture, pbr.roughness_texture, pbr.specular_texture) if t is not None]
    assert textures and all(t.file.resolved and Path(t.file.resolved).is_file() for t in textures)  # "../textures/..."


def test_human_female_rest_pose_matches_usdskel(sample):
    path = sample("usdskel_examples") / HUMAN_FEMALE / "HumanFemale.usd"
    scene = scene_from_stage(open_stage(path, {}))
    expected = usdskel_points(open_stage(path, {}), [0])
    assert len(expected) == len(scene.meshes)
    for mesh, points in zip(scene.meshes, expected):
        np.testing.assert_allclose(mesh.skinned_vertices(mesh.skin.joint_world(scene)), points[0], atol=1e-4)


@pytest.mark.parametrize("variants", [
    {},  # full geometry, high rig (blend shapes)
    {"geometricLOD": "low"},
    {"geometricLOD": "medium", "rigComplexity": "reduced"},  # no fingers, face joints or blend shapes
    {"rigComplexity": "faceBones"},  # face joints instead of blend shapes
])  # fmt: skip
def test_human_female_walk_matches_usdskel(sample, variants):
    """The walk clip's SkelAnimation is bound on the SkelRoot (an inherited animation source)."""
    path = sample("usdskel_examples") / HUMAN_FEMALE / "HumanFemale.walk.usd"
    scene = scene_from_stage(open_stage(path, variants))
    (clip,) = scene.animations
    codes = [101, 115, 129]
    expected = usdskel_points(open_stage(path, variants), codes)
    assert len(expected) == len(scene.meshes)
    times = np.array(codes) / 24.0
    for mesh, points in zip(scene.meshes, expected):
        # cm; the file stores joint rotations as float32 quaternions and UsdSkel skins in float32
        np.testing.assert_allclose(clip.skinned_vertices(scene, mesh, times), points, atol=5e-3, err_msg=mesh.name)


def test_human_female_walk_roundtrip(sample, tmp_path):
    """The walk (skin, 1229 blend shapes, mirrored leg joints, root motion, textures) through USD and FBX."""
    scene = sceneport.load(sample("usdskel_examples") / HUMAN_FEMALE / "HumanFemale.walk.usd")
    (clip,) = scene.animations
    times = clip.frame_times()
    expected = [clip.skinned_vertices(scene, mesh, times) for mesh in scene.meshes]
    for suffix, atol in ((".usdc", 1e-4), (".fbx", 1e-3)):  # cm; FBX stores key values as float32
        path = tmp_path / f"walk{suffix}"
        sceneport.save(scene, path)
        back = sceneport.load(path)
        (back_clip,) = back.animations
        assert [m.n_vertices for m in back.meshes] == [m.n_vertices for m in scene.meshes], suffix
        # FBX export splits in-betweens into plain shapes (Blender can't import them)
        shapes = sum(1 + (len(b.inbetweens) if suffix == ".fbx" else 0) for m in scene.meshes for b in m.blend_shapes)
        assert sum(len(m.blend_shapes) for m in back.meshes) == shapes, suffix
        assert len(back_clip.frame_times()) == len(times), suffix
        for mesh, points in zip(back.meshes, expected):
            np.testing.assert_allclose(back_clip.skinned_vertices(back, mesh, times), points, atol=atol, err_msg=f"{suffix} {mesh.name}")
        assert len(back.materials) == 14, suffix
        textures = [m.pbr().base_color_texture for m in back.materials if m.pbr().base_color_texture is not None]
        assert textures and all(t.file.resolved and Path(t.file.resolved).is_file() for t in textures), suffix
