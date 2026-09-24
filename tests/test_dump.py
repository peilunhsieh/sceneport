import json
from pathlib import Path

import numpy as np
import pytest

from sceneport.binary import FBXFile, FBXNode, Prop, read
from sceneport.binary.dump import format_prop, format_tree, to_json
from sceneport.cli import main

CUBE = Path(__file__).parent / "fixtures" / "cube.fbx"


def test_format_prop():
    assert format_prop(Prop("I", 3)) == "I 3"
    assert format_prop(Prop("S", "a\x00\x01b")) == r"S 'a\x00\x01b'"
    assert format_prop(Prop("R", b"\x01")) == r"R b'\x01'"
    arr = Prop("i", np.arange(5, dtype=np.int32))
    assert format_prop(arr, max_items=2) == "i[5] [0, 1, ...]"
    assert format_prop(arr, max_items=None) == "i[5] [0, 1, 2, 3, 4]"
    assert format_prop(Prop("f", np.array([0.1], dtype=np.float32))) == "f[1] [0.1]"


def test_format_tree_depth():
    root = FBXNode("A", children=[FBXNode("B", children=[FBXNode("C")])])
    text = format_tree(FBXFile(nodes=[root]), max_depth=2)
    assert text.splitlines() == ["FBX version 7400", "A", "  B", "    ... (1 child)"]


def test_to_json_cube():
    data = to_json(read(CUBE))
    json.dumps(data)  # serializable
    objects = next(n for n in data["nodes"] if n["name"] == "Objects")
    geom = objects["children"][0]
    verts = next(c for c in geom["children"] if c["name"] == "Vertices")
    assert verts["props"][0][0] == "d" and len(verts["props"][0][1]) == 24


def test_cli_dump(capsys):
    main(["dump", str(CUBE), "-d", "1"])
    out = capsys.readouterr().out
    assert out.startswith("FBX version 7400\nFBXHeaderExtension\n")
    assert "Objects" in out


def test_cli_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.fbx"
    bad.write_bytes(b"not fbx")
    with pytest.raises(SystemExit) as e:
        main(["dump", str(bad)])
    assert "not a binary FBX file" in str(e.value)
