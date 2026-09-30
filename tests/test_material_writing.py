"""Writing materials: FBX and USD round trips, and conversions between them."""

import numpy as np
import pytest

import sceneport
from sceneport.interop.fbx import document_from_scene, scene_from_document
from sceneport.interop.fbx.binary import dumps, loads
from sceneport.interop.fbx.document import FBXDocument
from sceneport.interop.usd import scene_from_stage
from sceneport.scene import Connection, Material, Mesh, Node, Port, Scene, ShaderNode, pbr_material
from sceneport.scene.pbr import PBRMaterial, preview_surface_nodes
from test_fbx_materials import material_document
from test_usd_materials import material_stage


def fbx_roundtrip(scene: Scene) -> Scene:
    return scene_from_document(FBXDocument.from_file(loads(dumps(document_from_scene(scene).to_file()))))


def by_name(scene) -> dict[str, Material]:
    return {m.name: m for m in scene.materials}


def assert_ports_equal(a: dict, b: dict, label: str) -> None:
    assert set(a) == set(b), label
    for name, port in a.items():
        other = b[name]
        assert (other.type_name, other.connections, other.color_space) == (port.type_name, port.connections, port.color_space), f"{label}.{name}"
        if isinstance(port.value, (float, np.ndarray)):
            np.testing.assert_allclose(other.value, port.value, rtol=1e-6, err_msg=f"{label}.{name}")
        else:
            assert other.value == port.value, f"{label}.{name}"


def assert_graphs_equal(a: Material, b: Material) -> None:
    assert a.name == b.name
    assert_ports_equal(a.inputs, b.inputs, "inputs")
    assert_ports_equal(a.outputs, b.outputs, "outputs")
    assert set(a.nodes) == set(b.nodes)
    for path, node in a.nodes.items():
        other = b.nodes[path]
        assert type(other) is type(node), path
        assert_ports_equal(node.inputs, other.inputs, path)
        assert_ports_equal(node.outputs, other.outputs, path)
        if hasattr(node, "shader_id"):
            assert (other.shader_id, other.source_code) == (node.shader_id, node.source_code), path
            assert {k: v.path for k, v in other.source_assets.items()} == {k: v.path for k, v in node.source_assets.items()}


# -- USD -> USD -------------------------------------------------------------------


def test_usd_graphs_roundtrip(tmp_path):
    scene = scene_from_stage(material_stage())
    sceneport.save(scene, tmp_path / "out.usda")
    back = sceneport.load(tmp_path / "out.usda")
    assert [m.name for m in back.materials] == ["Wood", "Metal", "Unused"]
    for a, b in zip(scene.materials, back.materials):
        assert_graphs_equal(a, b)
    node = back.find("World")
    assert [m.name for m in node.materials] == ["Metal", "Wood"]
    np.testing.assert_array_equal(node.mesh.material_indices, [0, 1, 1])


def test_usd_invalid_names_are_kept(tmp_path):
    surface = ShaderNode("Tex.001", "UsdPreviewSurface", inputs={"my input|weird": Port("float", 0.5)},
                         outputs={"surface": Port("token")})  # fmt: skip
    material = Material("Mat.001", nodes={"Tex.001": surface},
                        outputs={"surface": Port("token", connections=[Connection("Tex.001", "outputs:surface")])})  # fmt: skip
    sceneport.save(Scene(nodes=[Node("n")], materials=[material]), tmp_path / "out.usda")
    back = sceneport.load(tmp_path / "out.usda").materials[0]
    assert back.name == "Mat.001"
    assert set(back.nodes) == {"Tex.001"}
    assert back.nodes["Tex.001"].inputs["my input|weird"].value == pytest.approx(0.5)
    assert back.surface() is back.nodes["Tex.001"]


# -- FBX -> FBX -------------------------------------------------------------------


def test_fbx_materials_roundtrip():
    scene = scene_from_document(FBXDocument.from_file(loads(dumps(material_document().to_file()))))
    back = fbx_roundtrip(scene)
    assert [m.name for m in back.materials] == ["Skin", "Metal", "Unused"]
    for a, b in zip(scene.materials, back.materials):
        assert set(a.nodes) == set(b.nodes)
        for path, node in a.nodes.items():
            assert b.nodes[path].shader_id == node.shader_id
            written = dict(b.nodes[path].inputs)
            if "FileName" not in node.inputs:  # always written (from the relative path), as Blender does
                written.pop("FileName", None)
            assert_ports_equal(node.inputs, written, path)
    skin = by_name(back)["Skin"]
    assert skin.nodes["diffuse"].extras["fbx"]["content"] == b"\x89PNG fake"
    node = back.find("Tri")
    assert [m.name for m in node.materials] == ["Skin", "Metal"]
    np.testing.assert_array_equal(node.mesh.material_indices, [1, 0, 1])


def test_fbx_property_types_roundtrip():
    doc = material_document()
    back = document_from_scene(scene_from_document(FBXDocument.from_file(loads(dumps(doc.to_file())))))
    for a, b in zip(doc.iter_objects("Material"), back.iter_objects("Material")):
        written = {p.name: (p.type, p.subtype, p.flags, p.value) for p in b.props}
        assert all(written[p.name] == (p.type, p.subtype, p.flags, p.value) for p in a.props), a.name


# -- FBX <-> USD -----------------------------------------------------------------


def test_fbx_to_usd_adds_preview_surface_and_writes_embedded_textures(tmp_path):
    scene = scene_from_document(FBXDocument.from_file(loads(dumps(material_document().to_file()))))
    sceneport.save(scene, tmp_path / "out.usda")
    assert (tmp_path / "textures" / "diffuse.png").read_bytes() == b"\x89PNG fake"  # only embedded in the FBX

    skin = by_name(sceneport.load(tmp_path / "out.usda"))["Skin"]
    assert {"surface", "fbx:surface"} <= set(skin.outputs)
    pbr = skin.pbr()
    assert pbr.source == "UsdPreviewSurface"
    assert pbr.base_color_texture.file.path == "textures/diffuse.png"
    assert pbr.base_color_texture.file.resolved is not None
    assert pbr.base_color_texture.color_space == "sRGB"
    assert pbr.base_color_texture.uv_set == "UVMap"  # the mesh has no UV set called that, so it keeps its name
    np.testing.assert_allclose(pbr.normal_texture.scale, [2, 2, 2, 1])  # 0..1 normal map decoded
    assert pbr.roughness == pytest.approx(pbr_material(by_name(scene)["Skin"]).roughness, rel=1e-6)

    back = by_name(fbx_roundtrip(sceneport.load(tmp_path / "out.usda")))["Skin"]  # the FBX surface is written back
    assert back.surface().shader_id == "FbxSurfacePhong"
    np.testing.assert_allclose(back.surface().inputs["DiffuseColor"].value, [0.8, 0.5, 0.4])
    assert set(back.nodes) == {"Surface", "diffuse", "normals", "base_n", "detail_n"}


def test_usd_preview_surface_to_fbx_phong():
    scene = scene_from_stage(material_stage())
    doc = document_from_scene(scene)
    wood = next(m for m in doc.iter_objects("Material") if m.name == "Wood")
    shininess = ((1 - 0.4) * 10) ** 2  # Blender's exporter: roughness 0.4
    assert wood.props.get("ShininessExponent") == pytest.approx(shininess)
    assert wood.props.get("DiffuseColor") == pytest.approx((0.18, 0.18, 0.18))
    assert wood.child_value("ShadingModel") == "phong"
    textures = {c.prop: doc.objects[c.src] for c in doc.connections if c.dst == wood.id and c.prop}
    assert textures["DiffuseColor"].child_value("RelativeFilename") == "textures/wood.png"
    assert "UVSet" not in textures["DiffuseColor"].props  # "st" is the first UV set

    back = by_name(scene_from_document(FBXDocument.from_file(loads(dumps(doc.to_file())))))["Wood"].pbr()
    assert back.roughness == pytest.approx(0.4)
    assert back.base_color_texture.file.path == "textures/wood.png"
    assert by_name(fbx_roundtrip(scene))["Unused"].surface().shader_id == "FbxSurfacePhong"  # empty: default Phong


# -- slots ---------------------------------------------------------------------------


def slot_scene() -> Scene:
    """Two nodes share a 3-face mesh with per-face slots; B has an empty slot."""
    def material(name, color):
        nodes, surface = preview_surface_nodes(PBRMaterial(name, base_color=np.array(color, dtype=float)))
        return Material(name, nodes=nodes, outputs={"surface": surface})

    red, blue, green = material("red", (1, 0, 0)), material("blue", (0, 0, 1)), material("green", (0, 1, 0))
    mesh = Mesh("tris", np.eye(3).tolist() + [[1, 1, 0]], [3, 3, 3], [0, 1, 2, 1, 3, 2, 0, 2, 3], material_indices=[0, 1, 0])
    a, b = Node("A", mesh=mesh, materials=[red, blue]), Node("B", mesh=mesh, materials=[green, None])
    b.transform.translation[:] = (2, 0, 0)
    return Scene(nodes=[Node("root", children=[a, b])], materials=[red, blue, green])


def test_usd_slots_on_shared_mesh(tmp_path):
    scene = slot_scene()
    scene.find("root").add(Node("C", mesh=scene.find("A").mesh))  # no materials
    sceneport.save(scene, tmp_path / "out.usda")
    back = sceneport.load(tmp_path / "out.usda")
    a, b = back.find("A"), back.find("B")
    assert back.find("C").materials == []  # A's bindings, brought by the mesh reference, are blocked
    assert a.mesh is b.mesh
    assert [m.name for m in a.materials] == ["red", "blue"]
    assert [m.name if m else None for m in b.materials] == ["green", None]
    np.testing.assert_array_equal(a.mesh.material_indices, [0, 1, 0])
    np.testing.assert_allclose(by_name(back)["green"].pbr().base_color, [0, 1, 0])


def test_fbx_slots_on_shared_mesh():
    back = fbx_roundtrip(slot_scene())
    a, b = back.find("A"), back.find("B")
    assert [m.name for m in a.materials] == ["red", "blue"]
    assert [m.name for m in b.materials] == ["green", "default"]  # FBX has no empty slots
    np.testing.assert_array_equal(a.mesh.material_indices, [0, 1, 0])
    np.testing.assert_allclose(by_name(back)["blue"].pbr().base_color, [0, 0, 1])


def test_single_material_mesh(tmp_path):
    scene = slot_scene()
    node = scene.find("A")
    node.mesh.material_indices = None
    node.materials = node.materials[:1]
    scene.find("root").children.remove(scene.find("B"))
    for suffix in (".usda", ".fbx"):
        sceneport.save(scene, tmp_path / f"out{suffix}")
        back = sceneport.load(tmp_path / f"out{suffix}").find("A")
        assert [m.name for m in back.materials] == ["red"] and back.mesh.material_indices is None, suffix



def test_usdz_packages_embedded_textures(tmp_path):
    import zipfile

    scene = scene_from_document(FBXDocument.from_file(loads(dumps(material_document().to_file()))))
    with pytest.warns(UserWarning, match="base_n.png' not found"):  # not on disk, not embedded
        sceneport.save(scene, tmp_path / "out.usdz")
    assert any(name.endswith("diffuse.png") for name in zipfile.ZipFile(tmp_path / "out.usdz").namelist())
    texture = by_name(sceneport.load(tmp_path / "out.usdz"))["Skin"].pbr().base_color_texture
    assert texture.file.resolved is not None
