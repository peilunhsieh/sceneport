"""OBJ export with MTL materials (Blender's OBJ exporter conventions)."""

import numpy as np
import pytest

import sceneport
from sceneport.interop.fbx import scene_from_document
from sceneport.interop.fbx.binary import dumps, loads
from sceneport.interop.fbx.document import FBXDocument
from sceneport.scene import AssetPath, Material, Mesh, Node, Scene
from sceneport.scene.pbr import PBRMaterial, TextureRef, preview_surface_nodes
from test_fbx_materials import material_document
from test_material_writing import slot_scene


def pbr_material(pbr: PBRMaterial) -> Material:
    nodes, surface = preview_surface_nodes(pbr)
    return Material(pbr.name, nodes=nodes, outputs={"surface": surface})


def single(material: Material) -> Scene:
    mesh = Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2])
    return Scene(nodes=[Node("tri", mesh=mesh, materials=[material])])


def mtl_blocks(path) -> dict[str, dict[str, str]]:
    """newmtl name -> {key: rest of the line}."""
    blocks, current = {}, None
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        if key == "newmtl":
            current = blocks[rest] = {}
        else:
            current[key] = rest
    return blocks


def floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def test_slots_become_usemtl_groups(tmp_path):
    sceneport.save(slot_scene(), tmp_path / "out.obj")
    lines = (tmp_path / "out.obj").read_text().splitlines()
    assert lines[1] == "mtllib out.mtl"
    groups = [line for line in lines if line.startswith(("o ", "usemtl"))]
    assert groups == ["o A", "usemtl red", "usemtl blue", "usemtl red", "o B", "usemtl green", "usemtl", "usemtl green"]
    assert list(mtl_blocks(tmp_path / "out.mtl")) == ["blue", "green", "red"]  # sorted, like Blender
    with pytest.warns(UserWarning, match="materials"):
        sceneport.save(slot_scene(), tmp_path / "plain.obj", materials=False)
    assert not (tmp_path / "plain.mtl").exists()
    assert "usemtl" not in (tmp_path / "plain.obj").read_text()


def test_no_materials_no_mtl(tmp_path):
    scene = single(pbr_material(PBRMaterial("m")))
    scene.nodes[0].materials = []
    sceneport.save(scene, tmp_path / "out.obj")
    assert not (tmp_path / "out.mtl").exists() and "mtllib" not in (tmp_path / "out.obj").read_text()


def test_blender_values(tmp_path):
    materials = [
        pbr_material(PBRMaterial("plain", base_color=np.array([0.2, 0.4, 0.6]), roughness=0.3, ior=1.45)),
        pbr_material(PBRMaterial("metal", metallic=0.8)),
        pbr_material(PBRMaterial("glass", opacity=0.25)),
        pbr_material(PBRMaterial("tinted_metal", metallic=1.0, opacity=0.5, emissive=np.array([1.0, 0.5, 0.0]))),
    ]
    scene = single(materials[0])
    scene.nodes += [Node(m.name, mesh=scene.nodes[0].mesh, materials=[m]) for m in materials[1:]]
    sceneport.save(scene, tmp_path / "out.obj")
    blocks = mtl_blocks(tmp_path / "out.mtl")
    plain = blocks["plain"]
    assert floats(plain["Ns"]) == pytest.approx([(1 - 0.3) ** 2 * 1000])
    assert floats(plain["Kd"]) == pytest.approx([0.2, 0.4, 0.6])
    assert floats(plain["Ka"]) == [1, 1, 1] and floats(plain["Ks"]) == [0.5, 0.5, 0.5]
    assert floats(plain["Ni"]) == pytest.approx([1.45]) and plain["illum"] == "2"
    assert floats(blocks["metal"]["Ka"]) == pytest.approx([0.8] * 3) and blocks["metal"]["illum"] == "3"
    assert floats(blocks["glass"]["d"]) == [0.25] and blocks["glass"]["illum"] == "9"
    assert blocks["tinted_metal"]["illum"] == "6"
    assert floats(blocks["tinted_metal"]["Ke"]) == [1.0, 0.5, 0.0]


def test_textures_from_fbx(tmp_path):
    scene = scene_from_document(FBXDocument.from_file(loads(dumps(material_document().to_file()))))
    sceneport.save(scene, tmp_path / "out.obj")
    assert (tmp_path / "textures" / "diffuse.png").read_bytes() == b"\x89PNG fake"  # embedded in the FBX
    skin = mtl_blocks(tmp_path / "out.mtl")["Skin"]
    assert "Kd" not in skin  # the texture replaces it
    assert skin["map_Kd"] == "textures/diffuse.png"
    assert skin["map_Bump"] == "-bm 1.000000 base_n.png"  # a layered texture's first layer


def test_texture_options_and_pbr_extensions(tmp_path):
    (tmp_path / "tex").mkdir()
    for name in ("albedo.png", "rough.png"):
        (tmp_path / "tex" / name).write_bytes(b"")
    albedo = TextureRef(AssetPath("albedo.png", str(tmp_path / "tex" / "albedo.png")),
                        uv_translation=np.array([0.5, 0.25]), uv_scale=np.array([2.0, 3.0]))  # fmt: skip
    rough = TextureRef(AssetPath("rough.png", str(tmp_path / "tex" / "rough.png")), channels="g")
    material = pbr_material(PBRMaterial("m", base_color_texture=albedo, roughness_texture=rough, metallic=0.5, clearcoat=0.25))

    sceneport.save(single(material), tmp_path / "classic.obj")
    classic = mtl_blocks(tmp_path / "classic.mtl")["m"]
    assert classic["map_Kd"] == "-o 0.500000 0.250000 0 -s 2.000000 3.000000 1 tex/albedo.png"
    assert classic["map_Ns"] == "tex/rough.png" and "Ns" not in classic
    assert not {"Pr", "Pm", "map_Pr"} & set(classic)

    sceneport.save(single(material), tmp_path / "pbr.obj", pbr_extensions=True)
    pbr = mtl_blocks(tmp_path / "pbr.mtl")["m"]
    assert pbr["map_Pr"] == "tex/rough.png" and "Pr" not in pbr
    assert floats(pbr["Pm"]) == [0.5] and floats(pbr["Pc"]) == [1.0]  # Blender's 4x clearcoat factor
    assert not {"Ns", "Ka", "map_Ns"} & set(pbr)


def test_uv_rotation_warns(tmp_path):
    texture = TextureRef(AssetPath("a.png"), uv_rotation=30.0)
    with pytest.warns(UserWarning, match="rotation"):
        sceneport.save(single(pbr_material(PBRMaterial("m", base_color_texture=texture))), tmp_path / "out.obj")


def test_unique_names(tmp_path):
    a, b = pbr_material(PBRMaterial("my mat")), pbr_material(PBRMaterial("my mat"))
    scene = single(a)
    scene.nodes.append(Node("other", mesh=scene.nodes[0].mesh, materials=[b]))
    sceneport.save(scene, tmp_path / "out.obj")
    assert list(mtl_blocks(tmp_path / "out.mtl")) == ["my_mat", "my_mat.001"]


# -- reading -------------------------------------------------------------------------


def test_obj_mtl_roundtrip(tmp_path):
    scene = scene_from_document(FBXDocument.from_file(loads(dumps(material_document().to_file()))))
    sceneport.save(scene, tmp_path / "a.obj")
    back = sceneport.load(tmp_path / "a.obj")
    node = back.nodes[0]
    assert [m.name for m in node.materials] == ["Metal", "Skin"]  # order of first use
    np.testing.assert_array_equal(node.mesh.material_indices, [0, 1, 0])
    skin = {m.name: m for m in back.materials}["Skin"]
    assert skin.surface().shader_id == "MtlSurface"
    assert skin.pbr().base_color_texture.file.resolved == str(tmp_path / "textures" / "diffuse.png")
    sceneport.save(back, tmp_path / "b.obj")
    assert (tmp_path / "b.mtl").read_text() == (tmp_path / "a.mtl").read_text()  # MTL materials written as read

    sceneport.save(slot_scene(), tmp_path / "slots.obj")
    slots = sceneport.load(tmp_path / "slots.obj")
    assert [[m.name if m else None for m in n.materials] for n in slots.nodes] == [["red", "blue"], ["green", None]]


MTL = """\
# comment
newmtl shiny
Ns 250.0
Ka 0.3 0.3 0.3
Kd 0.1 0.2 0.3
illum 3.0
Tr 0.25
custom_key some value
map_Kd -o 0.5 0.25 -s 2 -clamp on textures/my albedo.png
bump -bm 0.4 normal.png
refl metal.png

newmtl highlight
illum 2

newmtl flat
Kd 1 1 1
illum 1
Ns 900
Ka 1 1 1

newmtl pbr
illum 2
Ns 10
Pr 0.3
Pm 0.9
Pc 0.5
"""


def write_obj_with_mtl(tmp_path, obj_text: str, mtl_name: str = "lib.mtl") -> None:
    (tmp_path / mtl_name).write_text(MTL)
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "my albedo.png").write_bytes(b"")
    (tmp_path / "scene.obj").write_text(obj_text)


QUADS = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"


def test_parse_mtl(tmp_path):
    write_obj_with_mtl(tmp_path, f"mtllib lib.mtl\n{QUADS}usemtl shiny\nf 1 2 3\n")
    shiny = {m.name: m for m in sceneport.load(tmp_path / "scene.obj").materials}["shiny"]
    surface = shiny.surface()
    assert surface.inputs["illum"].value == 3 and surface.inputs["illum"].type_name == "int"
    np.testing.assert_allclose(surface.inputs["Kd"].value, [0.1, 0.2, 0.3])
    assert surface.inputs["custom_key"].value == "some value"
    albedo = shiny.nodes["map_Kd"]
    assert albedo.inputs["file"].value.path == "textures/my albedo.png"  # paths may contain spaces
    assert albedo.inputs["file"].value.resolved == str(tmp_path / "textures" / "my albedo.png")
    np.testing.assert_allclose(albedo.inputs["o"].value, [0.5, 0.25, 0.0])
    np.testing.assert_allclose(albedo.inputs["s"].value, [2.0, 1.0, 1.0])  # missing components: defaults
    assert albedo.inputs["clamp"].value == "on"
    assert shiny.nodes["bump"].inputs["bm"].value == pytest.approx(0.4)

    pbr = shiny.pbr()
    assert pbr.roughness == pytest.approx(1 - np.sqrt(250 / 1000))
    assert pbr.metallic == pytest.approx(0.3)  # illum 3: reflection, metallic = mean(Ka)
    assert pbr.opacity == pytest.approx(0.75)  # from Tr, as there's no d
    texture = pbr.base_color_texture
    np.testing.assert_allclose(texture.uv_translation, [0.5, 0.25])
    np.testing.assert_allclose(texture.uv_scale, [2.0, 1.0])
    assert texture.wrap == ("clamp", "clamp")
    assert pbr.normal_texture.file.path == "normal.png" and pbr.normal_scale == pytest.approx(0.4)
    assert pbr.metallic_texture.file.path == "metal.png"


@pytest.mark.parametrize(("name", "roughness", "metallic"), [
    ("highlight", 0.0, 0.0),  # illum 2 without Ns
    ("flat", 1 - np.sqrt(0.9), 0.0),  # illum 1: no reflection, Ka ignored
    ("pbr", 0.3, 0.9),  # Pr / Pm override
])  # fmt: skip
def test_blender_import_rules(tmp_path, name, roughness, metallic):
    write_obj_with_mtl(tmp_path, f"mtllib lib.mtl\n{QUADS}usemtl {name}\nf 1 2 3\n")
    pbr = {m.name: m for m in sceneport.load(tmp_path / "scene.obj").materials}[name].pbr()
    assert (pbr.roughness, pbr.metallic) == (pytest.approx(roughness), pytest.approx(metallic))
    if name == "pbr":
        assert pbr.clearcoat == pytest.approx(0.125)  # Blender's 4x factor undone


def test_usemtl_state_and_default_library(tmp_path):
    obj = f"{QUADS}o first\nf 1 2 3\nusemtl flat\nf 1 3 4\no second\nf 1 2 4\nusemtl\nf 2 3 4\no third\nusemtl nowhere\nf 1 2 3\n"
    write_obj_with_mtl(tmp_path, obj, mtl_name="scene.mtl")  # no mtllib: the OBJ's own name is tried
    with pytest.warns(UserWarning, match="nowhere"):
        scene = sceneport.load(tmp_path / "scene.obj")
    first, second, third = scene.nodes
    assert [m.name if m else None for m in first.materials] == [None, "flat"]
    np.testing.assert_array_equal(first.mesh.material_indices, [0, 1])
    assert [m.name if m else None for m in second.materials] == ["flat", None]  # carried over from "first"
    assert third.materials[0].name == "nowhere" and third.materials[0].pbr() is None  # empty material
    assert [m.name for m in scene.materials] == ["shiny", "highlight", "flat", "pbr", "nowhere"]


def test_mtl_through_usd_and_fbx(tmp_path):
    write_obj_with_mtl(tmp_path, f"mtllib lib.mtl\n{QUADS}usemtl shiny\nf 1 2 3\n")
    scene = sceneport.load(tmp_path / "scene.obj")
    sceneport.save(scene, tmp_path / "out.usda")
    shiny = {m.name: m for m in sceneport.load(tmp_path / "out.usda").materials}["shiny"]
    assert {"surface", "mtl:surface"} <= set(shiny.outputs)
    assert shiny.pbr().source == "UsdPreviewSurface"  # generated for USD renderers
    assert shiny.pbr().metallic == pytest.approx(0.3)
    sceneport.save(sceneport.load(tmp_path / "out.usda"), tmp_path / "again.obj")
    assert "custom_key some value" in (tmp_path / "again.mtl").read_text()  # the MTL surface is written back

    sceneport.save(scene, tmp_path / "out.fbx")
    fbx = {m.name: m for m in sceneport.load(tmp_path / "out.fbx").materials}["shiny"].pbr()
    assert fbx.source == "FbxSurfacePhong" and fbx.metallic == pytest.approx(0.3)
