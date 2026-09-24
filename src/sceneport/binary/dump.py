"""Human-readable text and JSON views of an FBXNode tree, for inspection and debugging."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import types as T
from .node import FBXFile, FBXNode, Prop


def format_prop(prop: Prop, max_items: int | None = 8) -> str:
    code, value = prop
    if code in T.ARRAY_DTYPES:
        shown = value if max_items is None else value[:max_items]
        items = ", ".join(str(x) for x in shown)
        more = ", ..." if len(shown) < len(value) else ""
        return f"{code}[{len(value)}] [{items}{more}]"
    if code in (T.STRING, T.RAW):
        return f"{code} {value!r}"
    return f"{code} {value}"


def format_tree(
    fbx: FBXFile, *, max_items: int | None = 8, max_depth: int | None = None, indent: str = "  "
) -> str:
    """Render `fbx` as an indented tree. `None` disables the item or depth limit."""
    lines = [f"FBX version {fbx.version}"]

    def visit(node: FBXNode, depth: int) -> None:
        pad = indent * depth
        props = ", ".join(format_prop(p, max_items) for p in node.props)
        lines.append(f"{pad}{node.name}: {props}" if props else f"{pad}{node.name}")
        if not node.children:
            return
        if max_depth is not None and depth + 1 >= max_depth:
            count = len(node.children)
            lines.append(f"{pad}{indent}... ({count} {'child' if count == 1 else 'children'})")
            return
        for child in node.children:
            visit(child, depth + 1)

    for node in fbx.nodes:
        visit(node, 0)
    return "\n".join(lines)


def to_json(fbx: FBXFile, *, max_depth: int | None = None) -> dict[str, Any]:
    """Convert `fbx` to JSON-compatible data. Arrays are always complete; raw bytes become hex."""

    def prop(p: Prop) -> list[Any]:
        code, value = p
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, bytes):
            value = value.hex()
        return [code, value]

    def node(n: FBXNode, depth: int) -> dict[str, Any]:
        out: dict[str, Any] = {"name": n.name, "props": [prop(p) for p in n.props]}
        if n.children:
            if max_depth is not None and depth + 1 >= max_depth:
                out["children_omitted"] = len(n.children)
            else:
                out["children"] = [node(c, depth + 1) for c in n.children]
        return out

    return {"version": fbx.version, "nodes": [node(n, 0) for n in fbx.nodes]}
