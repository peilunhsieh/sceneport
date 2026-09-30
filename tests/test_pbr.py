from math import sqrt

import numpy as np
import pytest

from sceneport.interop.usd import scene_from_stage
from sceneport.scene import AssetPath, Connection, Material, Port, ShaderNode, pbr_material
from test_fbx_materials import material_document, read_back
from test_usd_materials import material_stage


def connected(node: str, output: str = "out", type_name: str = "color3f") -> Port:
    return Port(type_name, connections=[Connection(node, f"outputs:{output}")])


def by_name(scene) -> dict[str, Material]:
    return {m.name: m for m in scene.materials}


def test_preview_surface():
    wood = by_name(scene_from_stage(material_stage()))["Wood"]
    pbr = wood.pbr()
    assert pbr.source == "UsdPreviewSurface"
    assert pbr.roughness == pytest.approx(0.4)
    texture = pbr.base_color_texture
    assert texture.file.path == "textures/wood.png"
    assert (texture.channels, texture.uv_set, texture.color_space) == ("rgb", "st", "sRGB")
    np.testing.assert_allclose(texture.scale, [1, 0.5, 0.5, 1])
    np.testing.assert_allclose(pbr.base_color, 0.18)  # the spec default; the texture replaces it
    assert pbr.metallic_texture is None and pbr.metallic == 0.0  # fed by a math node
    assert any("ND_multiply_float" in note for note in pbr.notes)


def test_render_context_choice():
    wood = by_name(scene_from_stage(material_stage()))["Wood"]
    mtlx = wood.pbr("mtlx")
    assert mtlx.source == "ND_standard_surface_surfaceshader"
    np.testing.assert_allclose(mtlx.base_color, [0.8, 0.5, 0.2], rtol=1e-6)
    assert wood.pbr("ri").source == "UsdPreviewSurface"  # no ri surface: the universal one
    assert by_name(scene_from_stage(material_stage()))["Metal"].pbr() is None  # no surface at all


def materialx_material() -> Material:
    nodes = {
        "Surface": ShaderNode("Surface", "ND_standard_surface_surfaceshader", inputs={
            "base": Port("float", 0.5),
            "base_color": connected("Albedo"),
            "specular_roughness": connected("Green", type_name="float"),
            "normal": connected("NormalMap", type_name="vector3f"),
            "emission": Port("float", 2.0),
            "emission_color": Port("color3f", np.array([1.0, 0.5, 0.0])),
        }),
        "Albedo": ShaderNode("Albedo", "ND_image_color3", inputs={
            "file": Port("asset", AssetPath("albedo.png"), color_space="srgb_texture"),
            "texcoord": connected("UV", type_name="float2"),
            "uaddressmode": Port("string", "periodic"),
            "vaddressmode": Port("string", "clamp"),
        }),
        "UV": ShaderNode("UV", "ND_geompropvalue_vector2", inputs={"geomprop": Port("string", "uv1")}),
        "Packed": ShaderNode("Packed", "ND_image_color3", inputs={"file": Port("asset", AssetPath("orm.png"))}),
        "Green": ShaderNode("Green", "ND_extract_color3", inputs={"in": connected("Packed"), "index": Port("int", 1)}),
        "NormalImage": ShaderNode("NormalImage", "ND_tiledimage_vector3", inputs={
            "file": Port("asset", AssetPath("normal.png")), "uvtiling": Port("float2", np.array([4.0, 2.0])),
        }),
        "NormalMap": ShaderNode("NormalMap", "ND_normalmap_float", inputs={"in": connected("NormalImage", type_name="vector3f")}),
    }
    return Material("mx", nodes=nodes, outputs={"mtlx:surface": connected("Surface", type_name="token")})


def test_materialx_standard_surface():
    pbr = materialx_material().pbr()  # only an mtlx surface: found without naming the context
    assert pbr.source == "ND_standard_surface_surfaceshader"
    np.testing.assert_allclose(pbr.base_color, 0.4)  # default base_color 0.8 * base 0.5
    albedo = pbr.base_color_texture
    assert (albedo.file.path, albedo.uv_set, albedo.wrap, albedo.color_space) == ("albedo.png", "uv1", ("repeat", "clamp"), "srgb_texture")
    assert (pbr.roughness_texture.file.path, pbr.roughness_texture.channels) == ("orm.png", "g")
    assert pbr.normal_texture.file.path == "normal.png"
    np.testing.assert_allclose(pbr.normal_texture.uv_scale, [4, 2])
    np.testing.assert_allclose(pbr.emissive, [2.0, 1.0, 0.0])
    assert pbr.notes == []


def test_fbx_phong_like_blender(tmp_path):
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"")
    materials = by_name(read_back(material_document(), str(tmp_path)))
    skin = materials["Skin"]
    pbr = skin.pbr()
    assert pbr.source == "FbxSurfacePhong"
    surface = skin.surface()
    shininess = surface.inputs["Shininess"].value if "Shininess" in surface.inputs else 12.0
    assert pbr.roughness == pytest.approx(1 - sqrt(float(np.ravel(shininess)[0])) / 10)
    np.testing.assert_allclose(pbr.base_color, [0.8, 0.5, 0.4])
    texture = pbr.base_color_texture
    assert (texture.file.path, texture.uv_set, texture.wrap) == ("textures\\diffuse.png", "UVMap", ("repeat", "repeat"))
    assert texture.content == b"\x89PNG fake"
    assert pbr.normal_texture.file.path == "base_n.png"  # the layered texture's first layer
    assert any("layered" in note for note in pbr.notes)
    assert materials["Metal"].pbr().source == "FbxSurfaceLambert"


@pytest.mark.parametrize(("props", "opacity"), [
    ({"TransparencyFactor": 0.25}, 0.75),
    ({"TransparencyFactor": 1.0, "Opacity": 0.6}, 0.6),  # Maya/Mixamo: factor 1, real value in Opacity
    ({"TransparencyFactor": 0.0, "TransparentColor": np.array([0.3, 0.3, 0.3])}, 0.7),
])  # fmt: skip
def test_fbx_opacity_rules(props, opacity):
    inputs = {k: Port("double" if np.isscalar(v) else "color3d", v) for k, v in props.items()}
    material = Material("m", nodes={"Surface": ShaderNode("Surface", "FbxSurfacePhong", inputs=inputs)},
                        outputs={"surface": connected("Surface", "surface", "token")})  # fmt: skip
    assert pbr_material(material).opacity == pytest.approx(opacity)
