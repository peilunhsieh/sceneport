import json
from pathlib import Path

import numpy as np
import pytest

from sceneport.interop.fbx.binary import FBXFile, FBXNode, Prop, read
from sceneport.interop.fbx.binary.dump import format_prop, format_tree, to_json
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


def test_cli_convert(tmp_path, capsys):
    import sceneport
    from sceneport.scene import Animation, Channel
    from helpers import rig_scene

    scene = rig_scene()
    scene.animations = [Animation("wave", [Channel(scene.find("mid"), "rotation", 2, [0.0, 1.0], [0.0, 90.0])], fps=4)]
    sceneport.save(scene, tmp_path / "rig.fbx")

    main(["convert", str(tmp_path / "rig.fbx"), str(tmp_path / "out" / "rig.usda")])  # creates out/
    main(["convert", str(tmp_path / "out" / "rig.usda"), str(tmp_path / "back.fbx")])
    out = capsys.readouterr().out
    assert "rig.usda: 1 meshes, 3 joints, animations: wave (5 frames)" in out
    back = sceneport.load(tmp_path / "back.fbx")
    assert [c.name for c in back.animations] == ["wave"] and back.meshes[0].skin is not None

    with pytest.raises(SystemExit) as e:
        main(["convert", str(tmp_path / "rig.fbx"), str(tmp_path / "rig.stl")])
    assert "unsupported file extension '.stl'" in str(e.value)
