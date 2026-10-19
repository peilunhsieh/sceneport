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

SH_C0 = 0.28209479177387814  # the degree-0 real SH basis value, 1 / (2 sqrt(pi))


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
