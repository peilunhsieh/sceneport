import datetime as dt

import numpy as np
import pytest

from sceneport.binary import FBXNode, Prop, dumps, loads, read
from sceneport.document import ROOT_ID, FBXDocument, Properties70, Property
from sceneport.document.templates import default_global_settings, default_template
from helpers import FIXTURES, assert_nodes_equal

CUBE = FIXTURES / "cube.fbx"


@pytest.fixture
def cube_file():
    return read(CUBE)


@pytest.fixture
def cube(cube_file):
    return FBXDocument.from_file(cube_file)


# -- Properties70 -------------------------------------------------------------


def test_property_make():
    p = Property.make("Lcl Scaling", "Lcl Scaling", (1, 2, 3), flags="A")
    assert p.value == (1.0, 2.0, 3.0)
    assert [q.type for q in p.raw] == ["D", "D", "D"]
    assert Property.make("Show", "bool", True).raw == (Prop("I", 1),)
    assert Property.make("UpAxis", "int", 1).subtype == "Integer"
    assert Property.make("SourceObject", "object").value is None
    with pytest.raises(ValueError):
        Property.make("Lcl Scaling", "Lcl Scaling", (1, 2))
    with pytest.raises(ValueError):
        Property.make("X", "NoSuchType", 1)


def test_property_node_roundtrip():
    p = Property.make("Lcl Rotation", "Lcl Rotation", (-90, 0, 0), flags="A")
    node = p.to_node()
    assert node.values == ["Lcl Rotation", "Lcl Rotation", "", "A", -90.0, 0.0, 0.0]
    assert Property.from_node(node) == p


def test_properties70_access(cube):
    gs = cube.global_settings
    assert gs.get("UpAxis") == 1
    assert gs.get("UnitScaleFactor") == 1.0
    assert gs.get("Missing", "x") == "x"
    assert "TimeSpanStop" in gs and len(gs) == 16
    gs.set("UnitScaleFactor", "double", 100.0)
    assert gs.get("UnitScaleFactor") == 100.0
    assert len(gs) == 16  # replaced in place


def test_default_templates_match_blender(cube_file):
    """Our built-in defaults must equal what Blender writes."""
    definitions = cube_file.find("Definitions")
    for obj_type in definitions.iter("ObjectType"):
        tmpl = obj_type.find("PropertyTemplate")
        if tmpl is None:
            continue
        name, props = default_template(obj_type.props[0].value)
        assert name == tmpl.props[0].value
        assert_nodes_equal(props.to_node(), tmpl.find("Properties70"))
    ours = default_global_settings().to_node()
    assert_nodes_equal(ours, cube_file.find("GlobalSettings").find("Properties70"))


# -- reading ------------------------------------------------------------------


def test_cube_objects(cube):
    assert cube.version == 7400
    assert cube.creator.startswith("Blender")
    (model,) = cube.iter_objects("Model")
    (geom,) = cube.iter_objects("Geometry")
    assert (model.name, model.subtype) == ("Cube", "Mesh")
    assert (geom.name, geom.subtype) == ("Cube.001", "Mesh")
    assert geom.child_value("Vertices").shape == (24,)
    assert geom.child_value("Missing", 0) == 0

    assert cube.children_of(ROOT_ID) == [model]
    assert cube.children_of(model) == [geom]
    assert cube.parents_of(geom) == [model]
    assert cube.children_of(model, kind="Model") == []


def test_template_fallback(cube):
    (model,) = cube.iter_objects("Model")
    assert cube.get_property(model, "Lcl Scaling") == (100.0, 100.0, 100.0)  # set on the object
    assert cube.get_property(model, "Lcl Translation") == (0.0, 0.0, 0.0)  # from template
    assert cube.get_property(model, "NotAProperty", 5) == 5


# -- writing ------------------------------------------------------------------


def test_cube_rewrite_matches_blender(cube_file, cube):
    """Everything except the header and the Document id is reproduced exactly."""
    ours = cube.to_file()
    assert [n.name for n in ours.nodes] == [n.name for n in cube_file.nodes]
    for a, b in zip(ours.nodes, cube_file.nodes):
        if a.name in ("FBXHeaderExtension", "Documents"):
            continue
        assert_nodes_equal(a, b)


def test_build_from_scratch():
    doc = FBXDocument(timestamp=dt.datetime(2026, 1, 2, 3, 4, 5))
    model = doc.add_object("Model", "Tri", "Mesh")
    model.props.set("Lcl Translation", "Lcl Translation", (1, 2, 3), flags="A")
    geom = doc.add_object(
        "Geometry",
        "Tri",
        "Mesh",
        children=[
            FBXNode("Vertices", [Prop("d", np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float))]),
            FBXNode("PolygonVertexIndex", [Prop("i", np.array([0, 1, -3], dtype=np.int32))]),
        ],
    )
    doc.connect(model, ROOT_ID)
    doc.connect(geom, model)
    assert model.id != geom.id

    data = dumps(doc.to_file())
    assert dumps(doc.to_file()) == data  # deterministic

    back = FBXDocument.from_file(loads(data))
    assert [o.name for o in back.objects.values()] == ["Tri", "Tri"]
    assert back.connections == doc.connections
    (m,) = back.iter_objects("Model")
    assert back.get_property(m, "Lcl Translation") == (1.0, 2.0, 3.0)
    assert back.get_property(m, "Lcl Scaling") == (1.0, 1.0, 1.0)  # default template
    defs = loads(data).find("Definitions")
    assert defs.find("Count").values == [3]
    assert [t.props[0].value for t in defs.iter("ObjectType")] == ["GlobalSettings", "Model", "Geometry"]


def test_duplicate_id_rejected():
    doc = FBXDocument()
    obj = doc.add_object("Model", "A")
    with pytest.raises(ValueError):
        doc.add_object("Model", "B", id=obj.id)
    with pytest.raises(ValueError):
        doc.add_object("Model", "C", id=ROOT_ID)


def test_properties70_empty_node():
    assert len(Properties70.from_node(None)) == 0
    assert Properties70().to_node().children == []
