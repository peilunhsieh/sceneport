"""PBR materials in .blend files, compared with what Blender's own USD importer builds for the same scene."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

import sceneport
from sceneport.interop.blend import run_in_blender
from sceneport.scene import AssetPath, Material, Mesh, Node, Primvar, Scene
from sceneport.scene.pbr import PBRMaterial, TextureRef, preview_surface_nodes
from test_material_writing import slot_scene


def png(path: Path, rgb) -> None:
    """A tiny solid-color 4x4 PNG."""
    raw = b"".join(b"\x00" + bytes(rgb) * 4 for _ in range(4))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))  # fmt: skip


def material_scene(root: Path, threshold: float = 0.0) -> Scene:
    """Textured (UV transform, wrap, channels of one ORM image, normal map, emission, coat, opacity), Plain,
    Specular-workflow and an unused material, on a two-slot mesh and a second object."""
    for name, rgb in (("albedo", (200, 120, 60)), ("orm", (255, 128, 64)), ("normal", (128, 128, 255)), ("emit", (255, 0, 0))):
        png(root / "tex" / f"{name}.png", rgb)

    def tex(name, **kw):
        return TextureRef(AssetPath(f"tex/{name}.png", str(root / "tex" / f"{name}.png")), **kw)

    pbrs = [
        PBRMaterial("Textured", base_color_texture=tex("albedo", uv_translation=np.array([0.25, 0.5]), uv_rotation=30.0,
                                                       uv_scale=np.array([2.0, 3.0]), wrap=("mirror", "clamp")),
                    roughness_texture=tex("orm", channels="g"), metallic_texture=tex("orm", channels="b"),
                    normal_texture=tex("normal"), normal_scale=0.5, emissive_texture=tex("emit"), opacity=0.5,
                    opacity_threshold=threshold, clearcoat=0.4, clearcoat_roughness=0.2, ior=1.45),
        PBRMaterial("Plain", base_color=np.array([0.1, 0.5, 0.9]), metallic=0.7, roughness=0.25, emissive=np.array([0.0, 0.2, 0.0])),
        PBRMaterial("Specular", base_color=np.array([0.5, 0.5, 0.5]), specular_color=np.array([1.0, 0.8, 0.2]), roughness=0.4),
        PBRMaterial("Unused", base_color=np.array([1.0, 1.0, 0.0])),
    ]  # fmt: skip
    materials = []
    for pbr in pbrs:
        nodes, surface = preview_surface_nodes(pbr)
        materials.append(Material(pbr.name, nodes=nodes, outputs={"surface": surface}))
    quad = Mesh("quad", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [3, 3], [0, 1, 2, 0, 2, 3], material_indices=[0, 1],
                uvs={"st": Primvar(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float), "vertex")})  # fmt: skip
    tri = Mesh("tri", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [3], [0, 1, 2])
    scene = Scene(nodes=[Node("Quad", mesh=quad, materials=materials[:2]), Node("Other", mesh=tri, materials=[materials[2]])],
                  up_axis="+Z", front_axis="-Y", materials=materials)  # fmt: skip
    return scene


def effective(summary: dict) -> dict:
    """What each Principled input gets, independent of node names and layout: constants for unlinked inputs,
    and for linked ones the traced chain (image, color space, extension, channel, UV map, mapping, combined
    multiply-add, normal map strength, cutout)."""
    nodes = {n["name"]: n for n in summary["nodes"]}
    into = {(to, socket): (frm, out) for frm, out, to, socket in summary["links"]}

    def value(node, name):
        return next((v for k, v in node["inputs"].items() if k.split(":", 1)[1] == name), None)

    def trace(name: str, socket: str) -> dict:
        node = nodes[name]
        kind = node["type"]
        if kind == "ShaderNodeTexImage":
            out = {"image": Path(node["image"][0]).name, "color_space": node["image"][1], "extension": node["extension"],
                   "output": socket}  # fmt: skip
            if (name, "Vector") in into:
                out.update(trace(*into[(name, "Vector")]))
            return out
        if kind == "ShaderNodeSeparateColor":
            return {**trace(*into[(name, "Color")]), "channel": socket}
        if kind == "ShaderNodeUVMap":
            return {"uv_map": node["uv_map"]}
        if kind == "ShaderNodeMapping":
            m = (value(node, "Location"), value(node, "Rotation"), value(node, "Scale"))
            return {**trace(*into[(name, "Vector")]), "mapping": tuple(tuple(round(x, 4) for x in v) for v in m)}
        if kind == "ShaderNodeNormalMap":
            return {**trace(*into[(name, "Color")]), "normal_strength": round(value(node, "Strength"), 4)}
        if kind in ("ShaderNodeVectorMath", "ShaderNodeMath") and node["operation"] == "MULTIPLY_ADD":
            inner = trace(*into[(name, "Vector" if kind == "ShaderNodeVectorMath" else "Value")])
            scale = np.array([v for k, v in node["inputs"].items() if k.startswith("1:")][0], dtype=float)
            bias = np.array([v for k, v in node["inputs"].items() if k.startswith("2:")][0], dtype=float)
            s0, b0 = inner.pop("madd", (np.ones_like(scale), np.zeros_like(bias)))
            inner["madd"] = (np.asarray(s0) * scale, np.asarray(b0) * scale + bias)
            return inner
        if kind == "ShaderNodeMath":
            return {**trace(*into[(name, "Value")]), "math": (node["operation"], round(value(node, "Value"), 4))}
        return {"node": kind}

    principled = next(n for n in summary["nodes"] if n["type"] == "ShaderNodeBsdfPrincipled")
    out = {"render_method": summary["render_method"]}
    for key, constant in principled["inputs"].items():
        name = key.split(":", 1)[1]
        if name in ("Base Color", "Metallic", "Roughness", "IOR", "Alpha", "Emission Color", "Emission Strength",
                    "Coat Weight", "Coat Roughness", "Specular Tint"):  # fmt: skip
            out[name] = tuple(round(v, 4) for v in constant) if isinstance(constant, tuple) else round(constant, 4)
    for (to, socket), source in into.items():
        if to == principled["name"]:
            chain = trace(*source)
            if "madd" in chain:
                chain["madd"] = tuple(tuple(np.round(np.ravel(v), 4)) for v in chain["madd"])
            out[socket] = chain
    return out


def test_materials_match_blenders_usd_import(blender, tmp_path):
    scene = material_scene(tmp_path)
    sceneport.save(scene, tmp_path / "scene.usda")
    sceneport.save(scene, tmp_path / "ours.blend", blender=blender)
    run_in_blender("import_usd", {"usd": str(tmp_path / "scene.usda"), "path": str(tmp_path / "ref.blend")}, blender=blender)
    ours = run_in_blender("inspect_blend", {"path": str(tmp_path / "ours.blend")}, blender=blender)
    ref = run_in_blender("inspect_blend", {"path": str(tmp_path / "ref.blend")}, blender=blender)
    for name in ("Textured", "Plain", "Specular"):
        a, b = effective(ours["materials"][name]), effective(ref["materials"][name])
        assert a.pop("Alpha") == pytest.approx(0.5 if name == "Textured" else 1.0)  # opacity is coverage (Alpha) ...
        b.pop("Alpha")  # ... where Blender's USD importer reads it as glass (Transmission Weight)
        assert a == b, name
    assert ours["materials"]["Unused"]["fake_user"]  # kept though nothing uses it
    objects = {o["name"]: o for o in ours["objects"]}
    ref_objects = {o["name"]: o for o in ref["objects"]}
    for name in ("Quad", "Other"):
        assert objects[name]["material_slots"] == ref_objects[name]["material_slots"], name
        np.testing.assert_array_equal(objects[name]["material_indices"], ref_objects[name]["material_indices"])
    textured = {n["name"]: n for n in ours["materials"]["Textured"]["nodes"]}
    assert all(n["image"][0].startswith("//tex/") for n in textured.values() if n["type"] == "ShaderNodeTexImage")


def test_opacity_threshold_is_a_cutout(blender, tmp_path):
    scene = material_scene(tmp_path, threshold=0.3)
    scene.materials[0].pbr()  # the threshold survives the graph
    pbr = PBRMaterial("Cut", opacity_texture=TextureRef(AssetPath("tex/albedo.png", str(tmp_path / "tex" / "albedo.png")), channels="a"),
                      opacity_threshold=0.5)  # fmt: skip
    nodes, surface = preview_surface_nodes(pbr)
    cut = Material("Cut", nodes=nodes, outputs={"surface": surface})
    scene.nodes[1].materials = [cut]
    scene.materials.append(cut)
    sceneport.save(scene, tmp_path / "cut.blend", blender=blender)
    materials = run_in_blender("inspect_blend", {"path": str(tmp_path / "cut.blend")}, blender=blender)["materials"]
    textured, cut = effective(materials["Textured"]), effective(materials["Cut"])
    assert (textured["Alpha"], textured["render_method"]) == (1.0, "DITHERED")  # constant 0.5 passes the 0.3 threshold
    assert cut["Alpha"]["math"] == ("GREATER_THAN", 0.5) and cut["Alpha"]["output"] == "Alpha"
    assert cut["render_method"] == "DITHERED"


def test_shared_mesh_slots(blender, tmp_path):
    """Two nodes share a mesh with different materials: slots on the mesh for A, per object for B."""
    sceneport.save(slot_scene(), tmp_path / "slots.blend", blender=blender)
    objects = {o["name"]: o for o in run_in_blender("inspect_blend", {"path": str(tmp_path / "slots.blend")}, blender=blender)["objects"]}
    assert objects["A"]["material_slots"] == [("red", "DATA"), ("blue", "DATA")]
    assert objects["B"]["material_slots"] == [("green", "OBJECT"), (None, "OBJECT")]
    np.testing.assert_array_equal(objects["A"]["material_indices"], [0, 1, 0])


def test_pack_images(blender, tmp_path):
    def images(name, **kwargs):
        sceneport.save(material_scene(tmp_path), tmp_path / name, blender=blender, **kwargs)
        materials = run_in_blender("inspect_blend", {"path": str(tmp_path / name)}, blender=blender)["materials"]
        return {n["image"] for m in materials.values() for n in m["nodes"] if n["type"] == "ShaderNodeTexImage"}

    linked, packed = images("linked.blend"), images("packed.blend", pack_images=True)
    assert len(packed) == 4 and all(path.startswith("//tex/") and is_packed for path, _, is_packed in packed)
    assert {(path, space) for path, space, _ in packed} == {(path, space) for path, space, _ in linked}  # same paths
    assert not any(is_packed for _, _, is_packed in linked)  # linked by default


def test_embedded_fbx_texture_is_packed(blender, tmp_path):
    from sceneport.interop.fbx import scene_from_document
    from sceneport.interop.fbx.binary import dumps, loads
    from sceneport.interop.fbx.document import FBXDocument
    from test_fbx_materials import material_document

    scene = scene_from_document(FBXDocument.from_file(loads(dumps(material_document().to_file()))))
    with pytest.warns(UserWarning):  # the fake normal-map files aren't found
        sceneport.save(scene, tmp_path / "skin.blend", blender=blender)
    skin = run_in_blender("inspect_blend", {"path": str(tmp_path / "skin.blend")}, blender=blender)["materials"]["Skin"]
    images = [n["image"] for n in skin["nodes"] if n["type"] == "ShaderNodeTexImage"]
    assert any(Path(path).name == "diffuse.png" and packed for path, _, packed in images)
    assert list((tmp_path).iterdir()) == [tmp_path / "skin.blend"]  # nothing written beside it
