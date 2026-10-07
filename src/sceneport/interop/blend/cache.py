"""Point caches for Blender's Mesh Sequence Cache modifier: per-vertex animation in a sidecar USD file.

A .blend can't hold time-sampled points itself (a CacheFile only references a
file, and can't be packed), so the meshes whose point samples move are written to
`<name>_cache.usdc` next to the .blend: one Mesh prim each (topology and
time-sampled points, in the prepared scene's Z-up meters and mesh space), one
time code per frame at the scene's fps, so frame f reads time code f. Uses only
pxr and numpy: it runs inside Blender on Blender's own USD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...scene.animation import PointSamples


def moving(samples: PointSamples, tolerance: float = 1e-7) -> bool:
    """Whether the samples ever leave the mesh's rest points (exporters often sample static meshes too)."""
    rest = samples.mesh.vertices
    if not samples.points.size:
        return False
    size = max(float(np.ptp(rest, axis=0).max()) if len(rest) else 0.0, 1.0)
    return bool(np.abs(samples.points - rest[None]).max() > tolerance * size)


def write_point_cache(path: str | Path, samples: list[PointSamples], fps: float) -> list[str]:
    """Write `samples` (one PointSamples per mesh) to the USD file `path`; returns each one's prim path."""
    from pxr import Sdf, Tf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    root = UsdGeom.Scope.Define(stage, "/cache").GetPrim()
    stage.SetDefaultPrim(root)
    taken: set[str] = set()
    paths, first, last = [], None, None
    for sample in samples:
        mesh = sample.mesh
        name = Tf.MakeValidIdentifier(mesh.name or "mesh")
        candidate, i = name, 1
        while candidate in taken:
            candidate, i = f"{name}_{i}", i + 1
        taken.add(candidate)
        prim_path = Sdf.Path("/cache").AppendChild(candidate)
        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
        usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(mesh.face_vertex_counts.astype(np.int32)))
        usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.face_vertex_indices.astype(np.int32)))
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(mesh.vertices, dtype=np.float32)))
        points_attr, extent_attr = usd_mesh.GetPointsAttr(), usd_mesh.CreateExtentAttr()
        for t, points in zip(sample.times, sample.points):
            code = float(np.round(t * fps, 6))
            points = np.ascontiguousarray(points, dtype=np.float32)
            points_attr.Set(Vt.Vec3fArray.FromNumpy(points), code)
            extent_attr.Set(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0), points.max(0)])), code)
            first = code if first is None else min(first, code)
            last = code if last is None else max(last, code)
        paths.append(str(prim_path))
    if first is not None:
        stage.SetStartTimeCode(first)
        stage.SetEndTimeCode(last)
    stage.GetRootLayer().Save()
    return paths
