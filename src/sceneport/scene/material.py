"""Materials as generic shading node graphs (the UsdShade model).

A Material holds shader nodes and node graphs, keyed by their path relative to
the material ("Surface", "Graph/Image"). Nodes are identified by an id from a
node library ("UsdPreviewSurface", "ND_standard_surface_surfaceshader",
"PxrSurface", ...) or by a source asset/code, and are kept as they are: nothing
here knows what a node means. Ports hold a value, connections to other ports,
or both (a connection wins). The material's outputs are keyed by render context
("surface" for every renderer, "mtlx:surface", "ri:surface", ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class AssetPath:
    """A file reference (e.g. a texture): the path as authored, and where it resolved to if known."""

    path: str
    resolved: str | None = None


@dataclass(frozen=True)
class Connection:
    """The source of a connected port.

    `node` is the source's path relative to the material ("" = the material
    itself, an absolute "/..." path = a prim outside it); `port` is the full port
    name, "outputs:<name>" or "inputs:<name>" (the interface input of a node
    graph or of the material).
    """

    node: str
    port: str


@dataclass
class Port:
    """A node input or output.

    `type_name` is the USD value type ("float", "color3f", "float2", "asset",
    "token", ...). Values are Python scalars, str, AssetPath, or float64 numpy
    arrays for vectors, colors and matrices.
    """

    type_name: str
    value: Any = None
    connections: list[Connection] = field(default_factory=list)
    color_space: str | None = None  # e.g. "raw" or "sRGB" on a texture's file input


@dataclass(eq=False)
class ShaderNode:
    """One node of a shading network."""

    path: str
    shader_id: str | None = None  # info:id, the node's identifier in its library
    source_assets: dict[str, AssetPath] = field(default_factory=dict)  # source type ("osl", "mtlx", ...) -> file
    source_code: dict[str, str] = field(default_factory=dict)  # source type -> inline code
    inputs: dict[str, Port] = field(default_factory=dict)
    outputs: dict[str, Port] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class NodeGraph:
    """A group of nodes with its own interface: inputs feed nodes inside, outputs connect to nodes inside."""

    path: str
    inputs: dict[str, Port] = field(default_factory=dict)
    outputs: dict[str, Port] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class Material:
    """A shading network: nodes and node graphs, an interface (`inputs`) and outputs per render context."""

    name: str
    nodes: dict[str, ShaderNode | NodeGraph] = field(default_factory=dict)  # path relative to the material -> node
    inputs: dict[str, Port] = field(default_factory=dict)  # interface, e.g. for variants to override
    outputs: dict[str, Port] = field(default_factory=dict)  # "surface", "mtlx:surface", "displacement", ...
    extras: dict[str, Any] = field(default_factory=dict)

    def shaders(self) -> Iterator[ShaderNode]:
        return (n for n in self.nodes.values() if isinstance(n, ShaderNode))

    def trace(self, port: Port) -> tuple[ShaderNode | NodeGraph | Material | None, str | None, Port]:
        """Follow `port`'s connections to what produces its value.

        Connections are followed through node graph outputs and the interface
        inputs of node graphs and the material. Returns (node, name, port): a
        shader node's output, or the first unconnected port on the way (its value
        is the value, `name` is "inputs:<name>"). A connection to something that
        doesn't exist ends the trace at (None, None, last port). Only the first of
        several connections is followed.
        """
        seen = set()
        owner, name = None, None
        while port.connections:
            if id(port) in seen:
                raise ValueError(f"material {self.name!r}: connection cycle")
            seen.add(id(port))
            connection = port.connections[0]
            source = self if connection.node == "" else self.nodes.get(connection.node)
            kind, _, port_name = connection.port.partition(":")
            ports = None if source is None else {"inputs": source.inputs, "outputs": source.outputs}.get(kind)
            if ports is None or port_name not in ports:
                return None, None, port
            owner, name, port = source, connection.port, ports[port_name]
            if isinstance(source, ShaderNode) and kind == "outputs":
                return owner, name, port
        return owner, name, port

    def surface(self, context: str | None = None) -> ShaderNode | None:
        """The shader node driving the surface output of `context` (e.g. "mtlx"), else the universal "surface"."""
        for key in ([f"{context}:surface"] if context else []) + ["surface"]:
            if key in self.outputs:
                node, name, _ = self.trace(self.outputs[key])
                if isinstance(node, ShaderNode) and name and name.startswith("outputs:"):
                    return node
        return None
