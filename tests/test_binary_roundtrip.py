import numpy as np
import pytest

from sceneport.interop.fbx.binary import FBXFile, FBXNode, FBXParseError, Prop, dumps, loads
from helpers import FIXTURES, assert_files_equal


def all_types_node() -> FBXNode:
    return FBXNode(
        "AllTypes",
        [
            Prop("Y", -12345),
            Prop("C", True),
            Prop("I", -(2**31)),
            Prop("F", 1.5),
            Prop("D", np.pi),
            Prop("L", 2**63 - 1),
            Prop("S", "Cube\x00\x01Model"),
            Prop("S", "ünïcode"),
            Prop("R", b"\x00\xff raw"),
            Prop("f", np.arange(3, dtype=np.float32)),
            Prop("d", np.linspace(0, 1, 100)),  # 800 bytes: compressed
            Prop("i", np.array([0, 1, -3], dtype=np.int32)),
            Prop("l", np.array([2**40], dtype=np.int64)),
            Prop("b", np.array([True, False, True])),
            Prop("i", np.zeros(0, dtype=np.int32)),
        ],
        [FBXNode("Leaf", [Prop("I", 1)]), FBXNode("EmptyLeaf")],
    )


@pytest.mark.parametrize("name", ["cube.fbx"])
def test_fixture_byte_exact(name):
    data = (FIXTURES / name).read_bytes()
    assert dumps(loads(data)) == data


def test_cube_structure():
    fbx = loads((FIXTURES / "cube.fbx").read_bytes())
    assert fbx.version == 7400
    assert fbx.find("Objects") is not None
    geom = fbx.find("Objects").find("Geometry")
    verts = geom.find("Vertices").props[0].value
    assert verts.dtype == np.float64 and verts.size == 24  # 8 vertices * xyz


@pytest.mark.parametrize("version", [7400, 7500])
def test_all_types_roundtrip(version):
    fbx = FBXFile(version=version, nodes=[all_types_node()])
    data = dumps(fbx)
    back = loads(data)
    assert_files_equal(fbx, back)
    assert dumps(back) == data


def test_version_7500_from_fixture():
    fbx = loads((FIXTURES / "cube.fbx").read_bytes())
    fbx.version = 7500
    back = loads(dumps(fbx))
    assert_files_equal(fbx, back)


def test_compression_is_optional():
    fbx = FBXFile(nodes=[all_types_node()])
    raw = dumps(fbx, compress_threshold=1 << 30)
    assert len(raw) > len(dumps(fbx))
    assert_files_equal(fbx, loads(raw))


def test_bad_magic():
    with pytest.raises(FBXParseError):
        loads(b"; FBX 7.4.0 project file")


def test_truncated():
    data = (FIXTURES / "cube.fbx").read_bytes()
    for cut in (30, 500, len(data) // 2):
        with pytest.raises(FBXParseError):
            loads(data[:cut])
