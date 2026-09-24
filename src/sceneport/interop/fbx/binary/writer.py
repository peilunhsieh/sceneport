"""Encode an FBXNode tree as binary FBX, including header and footer."""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np

from . import types as T
from .node import FBXFile, FBXNode, Prop
from .reader import HEADER_MAGIC, uses_64bit_offsets

FOOTER_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"

_UINT32 = struct.Struct("<I")
_ARRAY_HEADER = struct.Struct("<III")


def write(fbx: FBXFile, path: str | os.PathLike, **kwargs) -> None:
    with open(path, "wb") as f:
        f.write(dumps(fbx, **kwargs))


def dumps(fbx: FBXFile, *, compress_threshold: int = 128, compress_level: int = 1) -> bytes:
    """Encode `fbx`. Arrays larger than `compress_threshold` bytes are zlib compressed
    (defaults match Blender and the FBX converter)."""
    writer = _Writer(fbx.version, compress_threshold, compress_level)
    out = writer.out
    out += HEADER_MAGIC
    out += _UINT32.pack(fbx.version)
    for node in fbx.nodes:
        writer.write_node(node)
    out += writer.null_record
    _write_footer(out, fbx)
    return bytes(out)


def _write_footer(out: bytearray, fbx: FBXFile) -> None:
    out += fbx.footer_id
    out += b"\x00" * 4
    pad = -len(out) % 16 or 16
    out += b"\x00" * pad
    out += _UINT32.pack(fbx.version)
    out += b"\x00" * 120
    out += FOOTER_MAGIC


class _Writer:
    def __init__(self, version: int, compress_threshold: int, compress_level: int):
        self.out = bytearray()
        wide = uses_64bit_offsets(version)
        self.node_header = struct.Struct("<QQQ" if wide else "<III")
        self.null_record = b"\x00" * (self.node_header.size + 1)
        self.compress_threshold = compress_threshold
        self.compress_level = compress_level

    def write_node(self, node: FBXNode) -> None:
        out = self.out
        name = T.encode_string(node.name)
        if len(name) > 255:
            raise ValueError(f"node name too long: {node.name!r}")

        header_pos = len(out)
        out += self.node_header.pack(0, 0, 0)  # patched below
        out += bytes([len(name)])
        out += name

        props_pos = len(out)
        for prop in node.props:
            self._write_prop(prop)
        props_size = len(out) - props_pos

        sentinel = node.sentinel
        if sentinel is None:
            sentinel = bool(node.children) or not node.props
        elif node.children and not sentinel:
            raise ValueError(f"node {node.name!r} has children but sentinel=False")
        for child in node.children:
            self.write_node(child)
        if sentinel:
            out += self.null_record

        self.node_header.pack_into(out, header_pos, len(out), len(node.props), props_size)

    def _write_prop(self, prop: Prop) -> None:
        out = self.out
        code, value = prop
        if (s := T.SCALAR_STRUCTS.get(code)) is not None:
            out += code.encode()
            out += s.pack(value)
        elif (dtype := T.ARRAY_DTYPES.get(code)) is not None:
            arr = np.ascontiguousarray(value, dtype=dtype).reshape(-1)
            payload = arr.tobytes()
            encoding = T.ENCODING_RAW
            if len(payload) > self.compress_threshold:
                payload = zlib.compress(payload, self.compress_level)
                encoding = T.ENCODING_ZLIB
            out += code.encode()
            out += _ARRAY_HEADER.pack(arr.size, encoding, len(payload))
            out += payload
        elif code in (T.STRING, T.RAW):
            data = T.encode_string(value) if code == T.STRING else bytes(value)
            out += code.encode()
            out += _UINT32.pack(len(data))
            out += data
        else:
            raise ValueError(f"unknown property type {code!r}")
