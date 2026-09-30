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
