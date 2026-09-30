import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

import sceneport
from sceneport.interop.usd import scene_from_stage
from sceneport.scene import AssetPath, Connection, Material, NodeGraph, Port, Scene, ShaderNode


def material_stage() -> Usd.Stage:
    """/World binds Wood (inherited); /World/Mesh has a "top" subset bound to Metal; Unused is bound nowhere.

    Wood: UsdPreviewSurface (textured diffuse, metallic from a node graph fed by
    the material's interface), a MaterialX surface and an OSL source-asset node.
    """
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0), (3, 0, 0), (3, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 1, 4, 5, 2, 4, 6, 7, 5])
    plain = UsdGeom.Mesh.Define(stage, "/Other/Plain")
    plain.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    plain.CreateFaceVertexCountsAttr([3])
    plain.CreateFaceVertexIndicesAttr([0, 1, 2])

    wood = UsdShade.Material.Define(stage, "/World/Looks/Wood")
    gloss = wood.CreateInput("gloss", Sdf.ValueTypeNames.Float)
    gloss.Set(0.25)

    reader = UsdShade.Shader.Define(stage, "/World/Looks/Wood/Reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, "/World/Looks/Wood/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    file_input = texture.CreateInput("file", Sdf.ValueTypeNames.Asset)
    file_input.Set(Sdf.AssetPath("textures/wood.png"))
    file_input.GetAttr().SetColorSpace("sRGB")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 0.5, 0.5, 1))

    graph = UsdShade.NodeGraph.Define(stage, "/World/Looks/Wood/Graph")
    amount = graph.CreateInput("amount", Sdf.ValueTypeNames.Float)
    amount.ConnectToSource(gloss)
    inner = UsdShade.Shader.Define(stage, "/World/Looks/Wood/Graph/Double")
    inner.CreateIdAttr("ND_multiply_float")
    inner.CreateInput("in1", Sdf.ValueTypeNames.Float).ConnectToSource(amount)
    inner.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(2.0)
    graph.CreateOutput("value", Sdf.ValueTypeNames.Float).ConnectToSource(inner.ConnectableAPI(), "out")

    surface = UsdShade.Shader.Define(stage, "/World/Looks/Wood/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(graph.ConnectableAPI(), "value")
    wood.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    mtlx = UsdShade.Shader.Define(stage, "/World/Looks/Wood/MtlxSurface")
    mtlx.CreateIdAttr("ND_standard_surface_surfaceshader")
    mtlx.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.5, 0.2))
    wood.CreateSurfaceOutput("mtlx").ConnectToSource(mtlx.ConnectableAPI(), "out")

    osl = UsdShade.Shader.Define(stage, "/World/Looks/Wood/Noise")
    osl.SetSourceAsset(Sdf.AssetPath("noise.osl"), "osl")

    metal = UsdShade.Material.Define(stage, "/World/Looks/Metal")
    UsdShade.Material.Define(stage, "/World/Looks/Unused")

    UsdShade.MaterialBindingAPI.Apply(world.GetPrim()).Bind(wood)
    top = UsdShade.MaterialBindingAPI(mesh.GetPrim()).CreateMaterialBindSubset("top", [0])
    UsdShade.MaterialBindingAPI.Apply(top.GetPrim()).Bind(metal)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    return stage


@pytest.fixture
def scene() -> Scene:
    return scene_from_stage(material_stage())


def materials_by_name(scene: Scene) -> dict[str, Material]:
    return {m.name: m for m in scene.materials}


def test_every_material_is_read(scene):
    assert [m.name for m in scene.materials] == ["Wood", "Metal", "Unused"]


def test_subsets_become_slots(scene):
    by_name = materials_by_name(scene)
    node = scene.find("World")  # a leaf Mesh is read as its parent Xform's geometry
    assert node.materials == [by_name["Metal"], by_name["Wood"]]  # subset first, then the inherited binding
    assert node.materials[0] is by_name["Metal"]
    np.testing.assert_array_equal(node.mesh.material_indices, [0, 1, 1])
    node.mesh.validate()
    plain = scene.find("Plain")
    assert plain.materials == [] and plain.mesh.material_indices is None


def test_graph_structure(scene):
    wood = materials_by_name(scene)["Wood"]
    assert set(wood.nodes) == {"Reader", "Texture", "Graph", "Graph/Double", "Surface", "MtlxSurface", "Noise"}
    assert isinstance(wood.nodes["Graph"], NodeGraph)
    assert all(isinstance(n, ShaderNode) for path, n in wood.nodes.items() if path != "Graph")
    assert wood.inputs["gloss"].value == pytest.approx(0.25)
    assert wood.outputs["surface"].connections == [Connection("Surface", "outputs:surface")]

    texture = wood.nodes["Texture"]
    assert texture.shader_id == "UsdUVTexture"
    assert texture.inputs["file"].type_name == "asset"
    assert texture.inputs["file"].value.path == "textures/wood.png"
    assert isinstance(texture.inputs["file"].value, AssetPath)
    assert texture.inputs["file"].color_space == "sRGB"
    assert texture.inputs["st"] == Port("float2", None, [Connection("Reader", "outputs:result")])
    np.testing.assert_allclose(texture.inputs["scale"].value, [1, 0.5, 0.5, 1])
    assert texture.inputs["scale"].value.dtype == np.float64
    assert wood.nodes["Reader"].inputs["varname"].value == "st"

    noise = wood.nodes["Noise"]
    assert noise.shader_id is None
    assert noise.source_assets["osl"].path == "noise.osl"
    assert noise.extras["usd"]["implementationSource"] == "sourceAsset"


def test_surfaces_by_render_context(scene):
    wood = materials_by_name(scene)["Wood"]
    assert wood.surface() is wood.nodes["Surface"]
    assert wood.surface("mtlx") is wood.nodes["MtlxSurface"]
    assert wood.surface("ri") is wood.nodes["Surface"]  # falls back to the universal output
    assert materials_by_name(scene)["Metal"].surface() is None


def test_trace_through_graphs_and_interface(scene):
    wood = materials_by_name(scene)["Wood"]
    surface = wood.nodes["Surface"]
    node, name, _ = wood.trace(surface.inputs["diffuseColor"])
    assert (node, name) == (wood.nodes["Texture"], "outputs:rgb")
    node, name, _ = wood.trace(surface.inputs["metallic"])  # through the node graph's output
    assert (node, name) == (wood.nodes["Graph/Double"], "outputs:out")
    node, name, port = wood.trace(wood.nodes["Graph/Double"].inputs["in1"])  # graph input -> material input
    assert (node, name) == (wood, "inputs:gloss")
    assert port.value == pytest.approx(0.25)
    node, name, port = wood.trace(surface.inputs["roughness"])  # unconnected: the port itself
    assert (node, name) == (None, None) and port.value == pytest.approx(0.4)


def test_trace_missing_source_and_cycle():
    material = Material("m", nodes={"A": ShaderNode("A", inputs={"x": Port("float", connections=[Connection("B", "outputs:out")])})})
    node, name, port = material.trace(material.nodes["A"].inputs["x"])
    assert (node, name, port) == (None, None, material.nodes["A"].inputs["x"])
    graph = NodeGraph("G")
    graph.inputs["a"] = Port("float", connections=[Connection("G", "inputs:b")])
    graph.inputs["b"] = Port("float", connections=[Connection("G", "inputs:a")])
    material.nodes["G"] = graph
    with pytest.raises(ValueError, match="cycle"):
        material.trace(graph.inputs["a"])


def test_geometry_formats_warn_that_materials_are_dropped(scene, tmp_path):
    for suffix in (".obj", ".ply"):
        with pytest.warns(UserWarning, match="materials"):
            sceneport.save(scene, tmp_path / f"out{suffix}")


def test_material_file_resolves_relative_to_layer(tmp_path):
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "wood.png").write_bytes(b"")
    stage = material_stage()  # keep the stage alive while its layer is exported
    stage.GetRootLayer().Export(str(tmp_path / "scene.usda"))
    wood = materials_by_name(sceneport.load(tmp_path / "scene.usda"))["Wood"]
    value = wood.nodes["Texture"].inputs["file"].value
    assert value.path == "textures/wood.png"
    assert value.resolved is not None and value.resolved.endswith("wood.png")
