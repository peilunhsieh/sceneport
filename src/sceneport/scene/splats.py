"""3D Gaussian splats attached to a Node (the node's transform places them).

Each splat is a 3D Gaussian: a center, a per-axis size and an orientation, an
opacity, and a view-dependent color as real spherical harmonics (SH). Values are
the physical ones that OpenUSD's ParticleField3DGaussianSplat uses, not the
training encoding of the 3DGS reference PLY (which stores log scales, logit
opacities and unnormalized quaternions; interop/ply.py converts).

  scales: standard deviations along the splat's local axes, in scene units.
      A 2D Gaussian (2DGS surfel) has a zero third scale.
  orientations: unit quaternions (w, x, y, z) rotating the local axes into the node.
  opacities: 0 (transparent) to 1 at the center of the Gaussian.
  sh: (N, K, 3) coefficients per color channel, K = (degree + 1)^2, in the 3DGS
      order (band by band, m = -l..l). Kept as stored: the 3DGS convention adds 0.5
      to the evaluated color, so the view-independent color is 0.5 + SH_C0 * sh[:, 0].
  attributes: any other per-splat values a file carried, by name (e.g. a parent
      triangle index), kept so they can be written back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..math import quaternion_from_matrix, quaternion_matrix, quaternion_multiply

SH_C0 = 0.28209479177387814  # the degree-0 real SH basis value, 1 / (2 sqrt(pi))
# The 3DGS reference renderer's real SH constants for bands 1 to 3 (utils/sh_utils.py), sign
# conventions included: sh_basis evaluates exactly what it renders.
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)  # fmt: skip


def sh_basis(directions: NDArray, degree: int) -> NDArray[np.float64]:
    """(K, (degree + 1)^2) real SH basis values at unit `directions` (K, 3), in the 3DGS order and signs:
    a splat's color toward direction d (camera to splat) is 0.5 + sh_basis(d) @ sh (per channel)."""
    x, y, z = np.asarray(directions, dtype=np.float64).T
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    bands = [[np.full_like(x, SH_C0)], [-SH_C1 * y, SH_C1 * z, -SH_C1 * x],
             [SH_C2[0] * xy, SH_C2[1] * yz, SH_C2[2] * (2 * zz - xx - yy), SH_C2[3] * xz, SH_C2[4] * (xx - yy)],
             [SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * xy * z, SH_C3[2] * y * (4 * zz - xx - yy),
              SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy), SH_C3[4] * x * (4 * zz - xx - yy), SH_C3[5] * z * (xx - yy),
              SH_C3[6] * x * (xx - 3 * yy)]]  # fmt: skip
    if degree > 3:
        raise ValueError(f"SH degree {degree} isn't supported (3DGS uses at most 3)")
    return np.stack([value for band in bands[: degree + 1] for value in band], axis=1)


def rotate_sh(sh: NDArray, rotation: NDArray) -> NDArray[np.float64]:
    """(N, K, 3) SH coefficients for splats turned by the orthogonal 3x3 `rotation` (a reflection
    works too): the new color toward rotation @ d is the old color toward d. Each band maps onto
    itself linearly; its matrix is fitted from the basis at sample directions, which is exact."""
    sh = np.asarray(sh, dtype=np.float64)
    degree = int(round(np.sqrt(sh.shape[1]))) - 1
    if degree == 0 or np.allclose(rotation, np.eye(3)):
        return sh.copy()
    index = np.arange(64) + 0.5  # Fibonacci sphere: well spread directions
    polar, azimuth = np.arccos(1 - 2 * index / 64), np.pi * (1 + 5**0.5) * index
    directions = np.stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], axis=1)
    basis, turned = sh_basis(directions, degree), sh_basis(directions @ rotation, degree)  # at d and at rotation.T @ d
    out = sh.copy()
    for band in range(1, degree + 1):
        k = slice(band * band, (band + 1) ** 2)
        mapping = np.linalg.lstsq(basis[:, k], turned[:, k], rcond=None)[0]
        out[:, k] = np.einsum("ij,njc->nic", mapping, sh[:, k])
    return out


@dataclass
class GaussianSplats:
    name: str
    positions: NDArray[np.float64]  # (N, 3)
    scales: NDArray[np.float64]  # (N, 3)
    orientations: NDArray[np.float64] | None = None  # (N, 4) w, x, y, z; None = identity
    opacities: NDArray[np.float64] | None = None  # (N,); None = 1
    sh: NDArray[np.float64] | None = None  # (N, K, 3); None = degree 0 with zero coefficients (gray 0.5)
    attributes: dict[str, NDArray] = field(default_factory=dict)  # name -> (N, ...)

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=np.float64).reshape(-1, 3)
        n = len(self.positions)
        self.scales = np.asarray(self.scales, dtype=np.float64).reshape(n, 3)
        if self.orientations is None:
            self.orientations = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        self.orientations = np.asarray(self.orientations, dtype=np.float64).reshape(n, 4)
        self.opacities = np.ones(n) if self.opacities is None else np.asarray(self.opacities, dtype=np.float64).reshape(n)
        self.sh = np.zeros((n, 1, 3)) if self.sh is None else np.asarray(self.sh, dtype=np.float64).reshape(n, -1, 3)
        degree = int(round(np.sqrt(self.sh.shape[1]))) - 1
        if (degree + 1) ** 2 != self.sh.shape[1]:
            raise ValueError(f"sh must have (degree + 1)^2 coefficients per channel, got {self.sh.shape[1]}")
        self.attributes = {name: np.asarray(values) for name, values in self.attributes.items()}
        for name, values in self.attributes.items():
            if len(values) != n:
                raise ValueError(f"attribute {name!r} has {len(values)} values for {n} splats")

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def sh_degree(self) -> int:
        return int(round(np.sqrt(self.sh.shape[1]))) - 1

    def base_colors(self) -> NDArray[np.float64]:
        """(N, 3) view-independent RGB (the degree-0 term), clipped to 0..1."""
        return np.clip(0.5 + SH_C0 * self.sh[:, 0], 0.0, 1.0)

    def transformed(self, matrix: NDArray) -> GaussianSplats:
        """A copy with every Gaussian mapped by the 4x4 `matrix` (positions moved, covariances A Σ Aᵀ).

        A rotation times a uniform scale keeps each splat's axes (scales scale, orientations rotate);
        anything else (non-uniform scale, shear, mirroring) re-derives the axes from the transformed
        covariance, largest first, so a flat (2DGS) splat keeps its zero third scale. SH coefficients
        are rotated with the matrix's rotation (or reflection) part, exactly for a rotation times a
        uniform scale; for other matrices view directions don't map rigidly, so that is approximate."""
        matrix = np.asarray(matrix, dtype=np.float64)
        linear = matrix[:3, :3]
        positions = self.positions @ linear.T + matrix[:3, 3]
        determinant = np.linalg.det(linear)
        uniform = np.cbrt(determinant)
        if determinant > 0 and np.allclose(linear @ linear.T, uniform**2 * np.eye(3), rtol=1e-9, atol=1e-12 * uniform**2):
            rotation = quaternion_from_matrix(linear / uniform)
            orientations = quaternion_multiply(rotation, self.orientations)
            scales = self.scales * uniform
        else:
            axes = linear @ quaternion_matrix(self.orientations)[:, :3, :3] * self.scales[:, None, :]  # A R S, per axis
            variances, vectors = np.linalg.eigh(axes @ axes.transpose(0, 2, 1))
            variances, vectors = variances[:, ::-1], vectors[:, :, ::-1]  # largest first
            vectors[:, :, 2] *= np.sign(np.linalg.det(vectors))[:, None]  # a proper rotation
            scales = np.sqrt(np.maximum(variances, 0.0))
            orientations = quaternion_from_matrix(vectors)
        u, _, vt = np.linalg.svd(linear)
        sh = rotate_sh(self.sh, u @ vt)  # the closest rotation or reflection
        return GaussianSplats(self.name, positions, scales, orientations, self.opacities.copy(), sh,
                              {name: values.copy() for name, values in self.attributes.items()})  # fmt: skip
