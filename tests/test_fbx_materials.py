import numpy as np
import pytest

from sceneport.interop.fbx import scene_from_document
from sceneport.interop.fbx.binary import FBXNode, Prop, dumps, loads
from sceneport.interop.fbx.document import ROOT_ID, FBXDocument, Properties70
from sceneport.scene import AssetPath, Connection, Material


def layer_element_material(mapping: str, indices) -> FBXNode:
    return FBXNode("LayerElementMaterial", [Prop("I", 0)], [
        FBXNode("Version", [Prop("I", 101)]),
        FBXNode("Name", [Prop("S", "")]),
        FBXNode("MappingInformationType", [Prop("S", mapping)]),
        FBXNode("ReferenceInformationType", [Prop("S", "IndexToDirect")]),
        FBXNode("Materials", [Prop("i", np.array(indices, dtype=np.int32))]),
    ])  # fmt: skip


def material_document(mapping: str = "ByPolygon", indices=(1, 0, 1)) -> FBXDocument:
    """A 3-triangle mesh on Model "Tri" with slots [Skin, Metal]:

    Skin (phong): DiffuseColor <- diffuse.png (embedded, via a Video), NormalMap <-
    a layered texture of two file textures; Metal (lambert) has no textures.
    Unused is connected to nothing.
    """
    doc = FBXDocument()
    model = doc.add_object("Model", "Tri", "Mesh")
    vertices = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0], dtype=float)
    geom = doc.add_object("Geometry", "Tri", "Mesh", children=[
        FBXNode("Vertices", [Prop("d", vertices)]),
        FBXNode("PolygonVertexIndex", [Prop("i", np.array([0, 1, -3, 1, 3, -3, 1, 4, -4], dtype=np.int32))]),
        layer_element_material(mapping, indices),
    ])  # fmt: skip
    doc.connect(model, ROOT_ID)
    doc.connect(geom, model)

    skin_props = Properties70()
    skin_props.set("DiffuseColor", "Color", (0.8, 0.5, 0.4), flags="A")
    skin_props.set("ShininessExponent", "Number", 12.0, flags="A")
    skin = doc.add_object("Material", "Skin", children=[FBXNode("ShadingModel", [Prop("S", "phong")])], props=skin_props)
    metal = doc.add_object("Material", "Metal", children=[FBXNode("ShadingModel", [Prop("S", "lambert")])])
    doc.add_object("Material", "Unused", children=[FBXNode("ShadingModel", [Prop("S", "unknown")])])
    doc.connect(skin, model)
    doc.connect(metal, model)

    diffuse = doc.add_object("Texture", "diffuse", children=[
        FBXNode("FileName", [Prop("S", "C:/nowhere/diffuse.png")]),
        FBXNode("RelativeFilename", [Prop("S", "textures\\diffuse.png")]),
    ])  # fmt: skip
    diffuse.props.set("UVSet", "KString", "UVMap")
    video = doc.add_object("Video", "diffuse", "Clip", children=[
        FBXNode("Filename", [Prop("S", "C:/nowhere/diffuse.png")]),
        FBXNode("Content", [Prop("R", b"\x89PNG fake")]),
    ])  # fmt: skip
    doc.connect(video, diffuse)
    doc.connect(diffuse, skin, "DiffuseColor")

    layered = doc.add_object("LayeredTexture", "normals")
    for name in ("base_n", "detail_n"):
        texture = doc.add_object("Texture", name, children=[FBXNode("RelativeFilename", [Prop("S", f"{name}.png")])])
        doc.connect(texture, layered)
    doc.connect(layered, skin, "NormalMap")
    return doc


def read_back(doc: FBXDocument, directory=None):
    return scene_from_document(FBXDocument.from_file(loads(dumps(doc.to_file()))), directory)


def by_name(scene) -> dict[str, Material]:
    return {m.name: m for m in scene.materials}


def test_slots_and_polygon_indices():
    scene = read_back(material_document())
    assert [m.name for m in scene.materials] == ["Skin", "Metal", "Unused"]
    node = scene.find("Tri")
    assert [m.name for m in node.materials] == ["Skin", "Metal"]
    assert node.materials[0] is by_name(scene)["Skin"]
    np.testing.assert_array_equal(node.mesh.material_indices, [1, 0, 1])
    node.mesh.validate()


@pytest.mark.parametrize(("mapping", "indices", "expected"), [
    ("AllSame", [0], None),
    ("AllSame", [1], [1, 1, 1]),
    ("ByPolygon", [0, 0], None),  # wrong length: ignored with a warning
])  # fmt: skip
def test_polygon_index_mappings(mapping, indices, expected):
    doc = material_document(mapping, indices)
    if expected is None and len(indices) == 2:
        with pytest.warns(UserWarning, match="ignored"):
            scene = read_back(doc)
    else:
        scene = read_back(doc)
    result = scene.find("Tri").mesh.material_indices
    assert result is None if expected is None else list(result) == expected


def test_surface_properties():
    materials = by_name(read_back(material_document()))
    skin = materials["Skin"].surface()
    assert skin.shader_id == "FbxSurfacePhong"
    assert skin.inputs["DiffuseColor"].type_name == "color3d"
    np.testing.assert_array_equal(skin.inputs["DiffuseColor"].value, [0.8, 0.5, 0.4])
    assert skin.inputs["ShininessExponent"].value == 12.0
    assert materials["Metal"].surface().shader_id == "FbxSurfaceLambert"
    assert materials["Unused"].surface().shader_id == "FbxSurfaceMaterial"
    assert materials["Unused"].surface().extras["fbx"]["shading_model"] == "unknown"


def test_textures(tmp_path):
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"")
    skin = by_name(read_back(material_document(), str(tmp_path)))["Skin"]
    surface = skin.surface()

    texture, output, _ = skin.trace(surface.inputs["DiffuseColor"])
    assert (texture.shader_id, output) == ("FbxFileTexture", "outputs:color")
    assert texture.inputs["file"].value == AssetPath("textures\\diffuse.png", str(tmp_path / "textures" / "diffuse.png"))
    assert texture.inputs["FileName"].value == "C:/nowhere/diffuse.png"
    assert texture.inputs["UVSet"].value == "UVMap"
    assert texture.extras["fbx"]["content"] == b"\x89PNG fake"
    np.testing.assert_array_equal(surface.inputs["DiffuseColor"].value, [0.8, 0.5, 0.4])  # value kept next to the connection

    layered, _, _ = skin.trace(surface.inputs["NormalMap"])
    assert layered.shader_id == "FbxLayeredTexture"
    layers = [skin.trace(layered.inputs[f"layer{i}"])[0] for i in range(2)]
    assert [layer.path for layer in layers] == ["base_n", "detail_n"]
    assert layers[0].inputs["file"].value == AssetPath("base_n.png", None)  # not on disk
    assert layered.inputs["layer1"].connections == [Connection("detail_n", "outputs:color")]


def test_scene_without_materials():
    doc = material_document()
    for obj in list(doc.objects.values()):
        if obj.kind in ("Material", "Texture", "LayeredTexture", "Video"):
            del doc.objects[obj.id]
    doc.connections = [c for c in doc.connections if c.src in doc.objects and (c.dst in doc.objects or c.dst == ROOT_ID)]
    scene = read_back(doc)
    assert scene.materials == [] and scene.find("Tri").materials == []
