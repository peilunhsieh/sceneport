"""Keyframe animation: clips of channels, each animating one scalar.

A Channel animates one component of a node transform property, or one blend
shape weight. Node properties:

    "translation" (x, y, z)      "scale" (x, y, z)
    "rotation" (x, y, z)         Euler degrees in the node's rotation order (FBX, USD xformOps)
    "orientation" (w, x, y, z)   unit quaternion replacing the Euler rotation (UsdSkel, Blender bones)

Keys hold an interpolation mode for the segment that starts at them: "constant",
"linear" or "cubic" (Hermite with automatic, Catmull-Rom style tangents). Before
the first and after the last key the value holds. Quaternion components are
interpolated one by one and normalized, as Blender does.

PointSamples animate a whole mesh's vertex positions (a point cache, e.g. USD
time-sampled points): linear between samples, held outside them. They replace
the mesh's rest positions before blend shapes and skinning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .types import Mesh, Node, Scene, Skin

INTERPOLATIONS = ("constant", "linear", "cubic")
NODE_PROPERTIES = {"translation": 3, "rotation": 3, "scale": 3, "orientation": 4}


@dataclass(eq=False)
class Channel:
    """Keys (`times` in seconds, `values`) for one scalar of a Node property or a Mesh blend shape.

    `target` is a Node (with `property` in NODE_PROPERTIES and `component` the axis or
    quaternion component) or a Mesh (with `property` a blend shape name, component 0).
    """

    target: Union[Node, Mesh]
    property: str
    component: int
    times: NDArray[np.float64]
    values: NDArray[np.float64]
    interpolation: NDArray = "linear"  # one mode per key, or a single mode for all
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        self.values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        modes = np.asarray(self.interpolation, dtype=object)
        self.interpolation = np.full(len(self.times), modes.item(), dtype=object) if modes.ndim == 0 else modes

    @property
    def is_blend_shape(self) -> bool:
        return self.property not in NODE_PROPERTIES

    def validate(self) -> None:
        label = f"channel {self.property}[{self.component}]"
        if len(self.times) == 0 or len(self.values) != len(self.times) or len(self.interpolation) != len(self.times):
            raise ValueError(f"{label}: needs one value and one interpolation per key, and at least one key")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError(f"{label}: key times must be strictly increasing")
        if not set(self.interpolation) <= set(INTERPOLATIONS):
            raise ValueError(f"{label}: interpolation must be one of {INTERPOLATIONS}")
        size = NODE_PROPERTIES.get(self.property, 1)
        if not 0 <= self.component < size:
            raise ValueError(f"{label}: component must be in [0, {size})")

    def evaluate(self, times: ArrayLike) -> NDArray[np.float64]:
        """Values at `times` (seconds), same shape as `times`."""
        t = np.asarray(times, dtype=np.float64)
        keys, values = self.times, self.values
        if len(keys) == 1:
            return np.full(t.shape, values[0])
        i = np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 2)
        t0, t1, v0, v1 = keys[i], keys[i + 1], values[i], values[i + 1]
        u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        out = v0 + (v1 - v0) * u
        mode = self.interpolation[i]
        out = np.where(mode == "constant", v0, out)
        cubic = mode == "cubic"
        if np.any(cubic):
            slopes = _auto_slopes(keys, values)
            dt = t1 - t0
            h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
            hermite = h00 * v0 + h10 * dt * slopes[i] + h01 * v1 + h11 * dt * slopes[i + 1]
            out = np.where(cubic, hermite, out)
        return np.where(t <= keys[0], values[0], np.where(t >= keys[-1], values[-1], out))


def _auto_slopes(keys: NDArray, values: NDArray) -> NDArray:
    """Catmull-Rom style tangents: central differences inside, one-sided at the ends."""
    slopes = np.empty_like(values)
    slopes[1:-1] = (values[2:] - values[:-2]) / (keys[2:] - keys[:-2])
    slopes[0] = (values[1] - values[0]) / (keys[1] - keys[0])
    slopes[-1] = (values[-1] - values[-2]) / (keys[-1] - keys[-2])
    return slopes


@dataclass(eq=False)
class PointSamples:
    """Vertex positions of `mesh` (in the mesh's own space) at `times` (seconds): (T, N, 3)."""

    mesh: Mesh
    times: NDArray[np.float64]
    points: NDArray[np.float64]

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        self.points = np.asarray(self.points, dtype=np.float64).reshape(len(self.times), -1, 3)

    def validate(self) -> None:
        if len(self.times) == 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError(f"point samples of {self.mesh.name!r}: times must be strictly increasing, at least one")
        if self.points.shape[1] != self.mesh.n_vertices:
            raise ValueError(f"point samples of {self.mesh.name!r}: {self.points.shape[1]} points, mesh has {self.mesh.n_vertices}")

    def evaluate(self, times: ArrayLike) -> NDArray[np.float64]:
        """(F, N, 3) positions at `times`: linear between samples, held before the first and after the last."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if len(self.times) == 1:
            return np.repeat(self.points[:1], len(t), axis=0)
        i = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2)
        u = np.clip((t - self.times[i]) / (self.times[i + 1] - self.times[i]), 0.0, 1.0)[:, None, None]
        return self.points[i] * (1 - u) + self.points[i + 1] * u


@dataclass
class Animation:
    """A clip: channels (and point samples) sharing a timeline. `start`/`end` (seconds) default to the key range."""

    name: str
    channels: list[Channel] = field(default_factory=list)
    fps: float = 30.0
    start: float | None = None
    end: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    point_samples: list[PointSamples] = field(default_factory=list)  # animated vertex positions, one per mesh

    def time_range(self) -> tuple[float, float]:
        keys = [c.times for c in self.channels if len(c.times)] + [p.times for p in self.point_samples if len(p.times)]
        first = min((k[0] for k in keys), default=0.0)
        last = max((k[-1] for k in keys), default=0.0)
        return (first if self.start is None else self.start, last if self.end is None else self.end)

    def frame_times(self) -> NDArray[np.float64]:
        """One time per frame at `fps`, from start to end inclusive."""
        start, end = self.time_range()
        count = int(np.floor((end - start) * self.fps + 1e-6)) + 1
        return start + np.arange(count) / self.fps

    def validate(self) -> None:
        for channel in self.channels:
            channel.validate()
        for samples in self.point_samples:
            samples.validate()

    # -- evaluation ----------------------------------------------------------

    def node_values(self, times: ArrayLike) -> dict[Node, dict[str, NDArray]]:
        """Animated transform values per node: {node: {property: (F, 3) or (F, 4)}}.

        Components without a channel keep the node's stored value; quaternions are normalized.
        """
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        out: dict = {}
        for channel in self.channels:
            if channel.is_blend_shape:
                continue
            node = channel.target
            props = out.setdefault(node, {})
            if channel.property not in props:
                props[channel.property] = np.tile(rest_value(node, channel.property), (len(times), 1))
            props[channel.property][:, channel.component] = channel.evaluate(times)
        for props in out.values():
            if "orientation" in props:
                props["orientation"] /= np.linalg.norm(props["orientation"], axis=1, keepdims=True)
        return out

    def world_matrices(self, scene: Scene, times: ArrayLike) -> dict[Node, NDArray]:
        """World matrix of every node at `times`: (F, 4, 4) each."""
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        animated = self.node_values(times)
        frames = len(times)
        out: dict = {}
        stack = [(node, np.eye(4)) for node in reversed(scene.nodes)]
        while stack:
            node, parent = stack.pop()
            world = parent @ node.transform.matrix(**animated.get(node, {}))
            out[node] = np.broadcast_to(world, (frames, 4, 4))
            stack.extend((child, world) for child in reversed(node.children))
        return out

    def joint_world(self, scene: Scene, skin: Skin, times: ArrayLike) -> NDArray[np.float64]:
        """(F, J, 4, 4) world matrices of `skin.joints` at `times`, for Mesh.skinned_vertices."""
        world = self.world_matrices(scene, times)
        return np.stack([world[joint] for joint in skin.joints], axis=1)

    def blend_weights(self, mesh: Mesh, times: ArrayLike) -> NDArray[np.float64]:
        """(F, S) blend shape weights of `mesh` at `times`; unanimated shapes keep their `weight`."""
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        weights = np.tile([shape.weight for shape in mesh.blend_shapes], (len(times), 1)).astype(np.float64)
        index = {shape.name: s for s, shape in enumerate(mesh.blend_shapes)}
        for channel in self.channels:
            if channel.target is mesh and channel.property in index:
                weights[:, index[channel.property]] = channel.evaluate(times)
        return weights

    def points(self, mesh: Mesh, times: ArrayLike) -> NDArray[np.float64] | None:
        """(F, N, 3) animated rest positions of `mesh` at `times`, or None if the clip has no point samples for it."""
        samples = next((p for p in self.point_samples if p.mesh is mesh), None)
        return None if samples is None else samples.evaluate(times)

    def vertices(self, mesh: Mesh, times: ArrayLike) -> NDArray[np.float64]:
        """(F, N, 3) vertices of `mesh` at `times` in its own space: point samples, then blend shapes (no skinning)."""
        return mesh.deformed_vertices(self.blend_weights(mesh, times), self.points(mesh, times))

    def skinned_vertices(self, scene: Scene, mesh: Mesh, times: ArrayLike) -> NDArray[np.float64]:
        """(F, N, 3) world-space vertices of a skinned `mesh` at `times`."""
        return mesh.skinned_vertices(
            self.joint_world(scene, mesh.skin, times), self.blend_weights(mesh, times), self.points(mesh, times)
        )


def rest_value(node: Node, prop: str) -> NDArray[np.float64]:
    """The node's stored (unanimated) value for a node property; quaternion for \"orientation\"."""
    if prop == "orientation":
        from ..math import euler_matrix, quaternion_from_matrix

        return quaternion_from_matrix(euler_matrix(node.transform.rotation, node.transform.rotation_order))
    return np.asarray(getattr(node.transform, prop), dtype=np.float64)
