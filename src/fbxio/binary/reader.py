"""Parse binary FBX (7.4 with 32-bit offsets, 7.5+ with 64-bit offsets) into an FBXNode tree."""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np

from . import types as T
from .node import FBXFile, FBXNode, Prop

HEADER_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
HEADER_SIZE = len(HEADER_MAGIC) + 4  # magic + uint32 version

_UINT32 = struct.Struct("<I")
_ARRAY_HEADER = struct.Struct("<III")  # length, encoding, byte size


class FBXParseError(ValueError):
    pass


def uses_64bit_offsets(version: int) -> bool:
    return version >= 7500


def read(path: str | os.PathLike) -> FBXFile:
    with open(path, "rb") as f:
        return loads(f.read())


def loads(data: bytes) -> FBXFile:
    if not data.startswith(HEADER_MAGIC):
        raise FBXParseError("not a binary FBX file")
    (version,) = _UINT32.unpack_from(data, len(HEADER_MAGIC))
    reader = _Reader(data, version)
    nodes = []
    pos = HEADER_SIZE
    try:
        while True:
            node, pos = reader.read_node(pos)
            if node is None:
                break
            nodes.append(node)
    except (struct.error, IndexError, zlib.error) as e:
        raise FBXParseError(f"truncated or corrupt FBX: {e}") from e
    footer_id = data[pos : pos + 16]
    return FBXFile(version=version, nodes=nodes, footer_id=footer_id)


class _Reader:
    def __init__(self, data: bytes, version: int):
        self.data = data
        wide = uses_64bit_offsets(version)
        # end_offset, num_props, props_byte_size
        self.node_header = struct.Struct("<QQQ" if wide else "<III")
        self.null_size = self.node_header.size + 1  # header + name length byte

    def read_node(self, pos: int) -> tuple[FBXNode | None, int]:
        data = self.data
        end, num_props, _ = self.node_header.unpack_from(data, pos)
        pos += self.node_header.size
        name_len = data[pos]
        pos += 1
        if end == 0:
            return None, pos  # null record
        if end > len(data):
            raise FBXParseError(f"node at {pos} ends past end of file")
        name = T.decode_string(data[pos : pos + name_len])
        pos += name_len

        props = []
        for _ in range(num_props):
            prop, pos = self._read_prop(pos)
            props.append(prop)

        node = FBXNode(name, props, sentinel=pos < end)
        if node.sentinel:
            while pos < end - self.null_size:
                child, pos = self.read_node(pos)
                if child is None:
                    raise FBXParseError(f"unexpected null record inside {name!r} at {pos}")
                node.children.append(child)
            if any(data[pos:end]):
                raise FBXParseError(f"bad null record at end of {name!r} at {pos}")
        return node, end

    def _read_prop(self, pos: int) -> tuple[Prop, int]:
        data = self.data
        code = chr(data[pos])
        pos += 1
        if (s := T.SCALAR_STRUCTS.get(code)) is not None:
            (value,) = s.unpack_from(data, pos)
            return Prop(code, value), pos + s.size
        if (dtype := T.ARRAY_DTYPES.get(code)) is not None:
            length, encoding, size = _ARRAY_HEADER.unpack_from(data, pos)
            pos += _ARRAY_HEADER.size
            payload = data[pos : pos + size]
            if encoding == T.ENCODING_ZLIB:
                payload = zlib.decompress(payload)
            elif encoding != T.ENCODING_RAW:
                raise FBXParseError(f"unknown array encoding {encoding} at {pos}")
            if len(payload) != length * dtype.itemsize:
                raise FBXParseError(f"array size mismatch at {pos}")
            return Prop(code, np.frombuffer(payload, dtype).copy()), pos + size
        if code in (T.STRING, T.RAW):
            (size,) = _UINT32.unpack_from(data, pos)
            pos += 4
            raw = data[pos : pos + size]
            value = T.decode_string(raw) if code == T.STRING else bytes(raw)
            return Prop(code, value), pos + size
        raise FBXParseError(f"unknown property type {code!r} at {pos - 1}")
