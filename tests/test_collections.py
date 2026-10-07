"""Collections: named, nestable groups of nodes, kept through scene rewrites."""

import numpy as np
import pytest

from sceneport.interop.reframe import change_basis, unmirror
from sceneport.scene import Collection, Mesh, Node, Scene, Transform


def furniture() -> Scene:
    """Table > Chair (parented), Shared in two collections, Lamp in none; Props > Chairs nested."""
    tri = Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2])
    chair = Node("Chair", Transform(translation=[1, 0, 0]), mesh=tri)
    table = Node("Table", mesh=tri, children=[chair])
    shared, lamp = Node("Shared", mesh=tri), Node("Lamp")
    chairs = Collection("Chairs", [chair])
    props = Collection("Props", [table, shared], children=[chairs])
    lighting = Collection("Lighting", [shared], extras={"blender": {"hide_render": True}})
    return Scene(nodes=[table, shared, lamp], collections=[props, lighting])


def test_walk_and_members():
    scene = furniture()
    assert [c.name for c in scene.all_collections()] == ["Props", "Chairs", "Lighting"]
    props = scene.find_collection("Props")
    assert [n.name for n in props.nodes] == ["Table", "Shared"]  # explicit: Table's child Chair isn't implied
    assert [n.name for n in props.all_nodes()] == ["Table", "Shared", "Chair"]  # with nested collections
    assert scene.find_collection("Missing") is None
    scene.validate_collections()


def test_validation():
    scene = furniture()
    scene.find_collection("Chairs").nodes.append(Node("Stranger"))
    with pytest.raises(ValueError, match="not in the scene"):
        scene.validate_collections()
    scene = furniture()
    scene.find_collection("Chairs").children.append(scene.find_collection("Props"))
    with pytest.raises(ValueError, match="contains itself"):
        scene.validate_collections()


@pytest.mark.parametrize("rewrite", [
    lambda s: change_basis(s, up="+Z", front="-Y", right="+X", meters_per_unit=0.01),
    lambda s: unmirror(s, {s.find("Chair")}),
])  # fmt: skip
def test_rewrites_keep_collections(rewrite):
    scene = furniture()
    out = rewrite(scene)
    assert out.find("Table") is not scene.find("Table")  # new nodes ...
    out.validate_collections()  # ... and the collections list them
    assert [n.name for n in out.find_collection("Props").all_nodes()] == ["Table", "Shared", "Chair"]
    assert out.find_collection("Lighting").extras == {"blender": {"hide_render": True}}
    assert scene.find_collection("Props").nodes[0] is scene.find("Table")  # the input is untouched
    assert np.allclose(scene.find("Chair").transform.translation, [1, 0, 0])


# -- USD --------------------------------------------------------------------------------------

from pxr import Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

from sceneport.interop.usd import scene_from_stage, stage_from_scene  # noqa: E402


def names(collection: Collection) -> list[str]:
    return [n.name for n in collection.nodes]


def test_usd_roundtrip():
    scene = furniture()
    scene.collections.append(Collection("My Props!", [scene.find("Lamp")], extras={"blender": {"color_tag": "COLOR_01"}, "x": object()}))
    stage = stage_from_scene(scene)
    root = stage.GetDefaultPrim()
    api = Usd.CollectionAPI(root, "Props")
    assert api.GetExpansionRuleAttr().Get() == Usd.Tokens.explicitOnly
    assert api.GetIncludesRel().GetTargets() == [Sdf.Path("/root/Table"), Sdf.Path("/root/Shared"),
                                                 Sdf.Path("/root.collection:Chairs")]  # fmt: skip
    query = Usd.CollectionAPI(root, "Chairs").ComputeMembershipQuery()
    assert query.IsPathIncluded("/root/Table/Chair") and not query.IsPathIncluded("/root/Table")

    back = scene_from_stage(stage)
    assert [c.name for c in back.collections] == ["Props", "Lighting", "My Props!"]  # order and names kept
    props = back.find_collection("Props")
    assert names(props) == ["Table", "Shared"] and [c.name for c in props.children] == ["Chairs"]
    assert names(back.find_collection("Chairs")) == ["Chair"]
    assert back.find_collection("Shared") is None and names(back.find_collection("Lighting")) == ["Shared"]
    assert back.find_collection("Lighting").extras == {"blender": {"hide_render": True}}
    assert back.find_collection("My Props!").extras == {"blender": {"color_tag": "COLOR_01"}}  # only what USD can hold
    back.validate_collections()


def test_usd_joints_are_written_as_their_skeleton():
    from sceneport.scene import Skin

    hip = Node("Hip", joint=True, children=[Node("Knee", Transform(translation=[0, -1, 0]), joint=True)])
    tri = Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2])
    tri.skin = Skin([hip], np.zeros((3, 1)), np.ones((3, 1)), np.eye(4)[None])
    rig = Node("Rig", children=[hip, Node("Body", mesh=tri)])
    scene = Scene(nodes=[rig], collections=[Collection("Bones", [hip.children[0]])])
    with pytest.warns(UserWarning, match="Skeleton is included"):
        stage = stage_from_scene(scene)
    assert [str(p) for p in Usd.CollectionAPI(stage.GetDefaultPrim(), "Bones").GetIncludesRel().GetTargets()] == ["/Rig/Hip"]
    assert names(scene_from_stage(stage).find_collection("Bones")) == ["Hip"]  # the Skeleton reads as its root joint


def test_usd_collections_from_other_tools():
    """Default expandPrims rule (descendants included), excludes, and light linking (not organizational)."""
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    for path in ("/World/Set", "/World/Set/Table", "/World/Set/Table/Cup", "/World/Set/Chair", "/World/Other"):
        UsdGeom.Xform.Define(stage, path)
    UsdLux.SphereLight.Define(stage, "/World/Lamp")
    stuff = Usd.CollectionAPI.Apply(world, "stuff")
    stuff.CreateIncludesRel().SetTargets(["/World/Set"])
    stuff.CreateExcludesRel().SetTargets(["/World/Set/Table/Cup"])
    Usd.CollectionAPI.Apply(world, "lightLink").CreateIncludesRel().SetTargets(["/World/Other"])
    scene = scene_from_stage(stage)
    assert [c.name for c in scene.collections] == ["stuff"]
    assert names(scene.collections[0]) == ["Set", "Table", "Chair"]  # every included descendant, Cup excluded


# -- .blend --------------------------------------------------------------------------------------

import sceneport  # noqa: E402


def blend_collections(blender, scene, path):
    from sceneport.interop.blend import run_in_blender

    sceneport.save(scene, path, blender=blender)
    return run_in_blender("inspect_blend", {"path": str(path)}, blender=blender)["collections"]


def test_blend_collections(blender, tmp_path):
    scene = furniture()
    scene.find_collection("Props").extras["blender"] = {"color_tag": "COLOR_04", "not_a_property": 1}
    tree = blend_collections(blender, scene, tmp_path / "furniture.blend")
    assert tree["objects"] == ["Lamp"]  # in no collection: stays in the scene's own collection
    props, lighting = tree["children"]
    assert (props["name"], props["objects"], props["color_tag"]) == ("Props", ["Table", "Shared"], "COLOR_04")
    assert [(c["name"], c["objects"]) for c in props["children"]] == [("Chairs", ["Chair"])]
    assert (lighting["name"], lighting["objects"], lighting["hide_render"]) == ("Lighting", ["Shared"], True)


def test_collections_through_usd_to_blend(blender, tmp_path):
    """USD -> sceneport -> .blend keeps the collections sceneport wrote to USD."""
    sceneport.save(furniture(), tmp_path / "furniture.usda")
    tree = blend_collections(blender, sceneport.load(tmp_path / "furniture.usda"), tmp_path / "furniture.blend")
    assert [(c["name"], c["objects"]) for c in tree["children"]] == [("Props", ["Table", "Shared"]), ("Lighting", ["Shared"])]
