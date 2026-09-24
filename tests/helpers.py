from pathlib import Path

import numpy as np

from sceneport.binary import FBXFile, FBXNode
from sceneport.scene import Mesh, Primvar

FIXTURES = Path(__file__).parent / "fixtures"


def assert_nodes_equal(a: FBXNode, b: FBXNode, path: str = "") -> None:
    path = f"{path}/{a.name}"
    assert a.name == b.name, path
    assert len(a.props) == len(b.props), path
    for i, (pa, pb) in enumerate(zip(a.props, b.props)):
        assert pa.type == pb.type, f"{path}[{i}]"
        if isinstance(pa.value, np.ndarray):
            assert pa.value.dtype == pb.value.dtype, f"{path}[{i}]"
            np.testing.assert_array_equal(pa.value, pb.value, err_msg=f"{path}[{i}]")
        else:
            assert pa.value == pb.value, f"{path}[{i}]"
    assert [c.name for c in a.children] == [c.name for c in b.children], path
    for ca, cb in zip(a.children, b.children):
        assert_nodes_equal(ca, cb, path)


def assert_files_equal(a: FBXFile, b: FBXFile) -> None:
    assert a.version == b.version
    assert a.footer_id == b.footer_id
    assert len(a.nodes) == len(b.nodes)
    for na, nb in zip(a.nodes, b.nodes):
        assert_nodes_equal(na, nb)


def mixed_mesh() -> Mesh:
    """A triangle, a quad and a pentagon sharing vertices."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0], [3, 0.5, 0], [2.5, 2, 0]])
    counts = [3, 4, 5]
    indices = [0, 1, 3, 1, 4, 5, 2, 4, 6, 7, 5, 2]
    return Mesh(
        "Mixed",
        vertices,
        counts,
        indices,
        normals=Primvar(np.tile([0.0, 0.0, 1.0], (8, 1)), "vertex"),
        uvs={
            "UVMap": Primvar(vertices[:, :2] / 3, "vertex"),
            "Second": Primvar(np.array([[0.0, 0.0], [1.0, 1.0]]), "faceVarying", np.array([0, 1] * 6)),
        },
    )
