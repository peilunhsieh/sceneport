"""FBX property type codes (Y C I F D L S R, arrays f d i l b) and their numpy dtypes."""

import struct

import numpy as np

# Scalar types: code -> little-endian struct format.
SCALAR_FORMATS: dict[str, str] = {
    "Y": "<h",  # int16
    "C": "<?",  # bool (1 byte)
    "I": "<i",  # int32
    "F": "<f",  # float32
    "D": "<d",  # float64
    "L": "<q",  # int64
}
SCALAR_STRUCTS: dict[str, struct.Struct] = {k: struct.Struct(v) for k, v in SCALAR_FORMATS.items()}

# Array types: code -> little-endian numpy dtype.
ARRAY_DTYPES: dict[str, np.dtype] = {
    "f": np.dtype("<f4"),
    "d": np.dtype("<f8"),
    "i": np.dtype("<i4"),
    "l": np.dtype("<i8"),
    "b": np.dtype("?"),
}

# Length-prefixed types.
STRING = "S"  # stored as str (utf-8, surrogateescape keeps invalid bytes round-trippable)
RAW = "R"  # stored as bytes

ALL_CODES = frozenset(SCALAR_FORMATS) | frozenset(ARRAY_DTYPES) | {STRING, RAW}

# Array payload encodings.
ENCODING_RAW = 0
ENCODING_ZLIB = 1


def decode_string(data: bytes) -> str:
    return data.decode("utf-8", "surrogateescape")


def encode_string(value: str) -> bytes:
    return value.encode("utf-8", "surrogateescape")
