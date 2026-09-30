"""FBX materials and textures <-> generic Material graphs.

An FBX Material is one surface with a fixed set of Properties70 (Phong or
Lambert: DiffuseColor, SpecularColor, NormalMap, ...; exporters add their own,
e.g. "3dsMax|Parameters|base_color"). It becomes a Material with one ShaderNode
"Surface" (shader_id "FbxSurfacePhong" / "FbxSurfaceLambert" /
"FbxSurfaceMaterial") whose inputs are the effective properties (the object's
values over its template's), and outputs["surface"] connected to it.

Textures connect to a material property (an OP connection, e.g. to
"DiffuseColor"); each becomes a ShaderNode (shader_id "FbxFileTexture", or
"FbxLayeredTexture" with inputs layer0, layer1, ...) connected to that input
through "outputs:color". The file comes from the texture's (or its Video's)
RelativeFilename / FileName; embedded Video content is kept as bytes in the
node's extras["fbx"]["content"].

Slots: the materials connected to a Model, in connection order, are the node's
slots; the geometry's LayerElementMaterial gives the slot of each polygon.

Writing: a graph whose surface is one of the FBX surfaces above (e.g. read from
FBX, or from USD written by sceneport, where it is kept as "fbx:surface") is
written back as it was, property types included (kept in the node's
extras["fbx"]["types"]). Any other graph goes through Material.pbr() and is
converted to Phong properties and textures like Blender's FBX exporter does.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from numpy.typing import NDArray

from ...scene.material import AssetPath, Connection, Material, Port, ShaderNode
from ...scene.pbr import FBX_SURFACES, PBRMaterial, TextureRef
from ...scene.types import Node
from .binary.node import FBXNode, Prop
from .document import NAME_SEP, FBXDocument, FBXObject, Properties70, Property
from .layers import _value

SHADING_MODELS = {"phong": "FbxSurfacePhong", "lambert": "FbxSurfaceLambert"}
TEXTURE_KINDS = {"Texture": "FbxFileTexture", "LayeredTexture": "FbxLayeredTexture"}
COLOR_TYPES = ("Color", "ColorRGB", "ColorAndAlpha")


def read_materials(doc: FBXDocument, directory: str | None = None) -> dict[int, Material]:
    """Every Material object, keyed by FBX id, in file order. `directory` resolves relative texture paths."""
    return {obj.id: _read_material(doc, obj, directory) for obj in doc.iter_objects("Material")}


def read_material_slots(doc: FBXDocument, model: FBXObject, materials: dict[int, Material]) -> list[Material]:
    return [materials[m.id] for m in doc.children_of(model, "Material") if m.id in materials]


def read_material_indices(geom: FBXObject, n_faces: int) -> NDArray[np.int32] | None:
    """(F,) slot of each polygon from the first LayerElementMaterial; None when every polygon uses slot 0."""
    layer = geom.child("LayerElementMaterial")
    if layer is None or (values := _value(layer, "Materials")) is None:
        return None
    values = np.asarray(values, dtype=np.int32).reshape(-1)
    mapping = _value(layer, "MappingInformationType", "AllSame")
    if mapping == "AllSame" or len(values) == 1:
        return None if not len(values) or values[0] == 0 else np.full(n_faces, max(values[0], 0), dtype=np.int32)
    if mapping != "ByPolygon" or len(values) != n_faces:
        warnings.warn(f"{geom.name}: material layer with mapping {mapping!r} and {len(values)} values for {n_faces} faces ignored")
        return None
    if (values < 0).any():
        warnings.warn(f"{geom.name}: negative material indices read as slot 0")
        values = np.maximum(values, 0)
    return values


def _read_material(doc: FBXDocument, obj: FBXObject, directory: str | None) -> Material:
    shading = str(obj.child_value("ShadingModel", doc.get_property(obj, "ShadingModel", "")) or "")
    types: dict[str, list[str]] = {}
    surface = ShaderNode(
        "Surface",
        SHADING_MODELS.get(shading.lower(), "FbxSurfaceMaterial"),
        inputs=_ports(_effective(doc, obj), types),
        outputs={"surface": Port("token")},
        extras={"fbx": {"shading_model": shading, "types": types}},
    )
    material = Material(obj.name, nodes={"Surface": surface})
    material.outputs["surface"] = Port("token", connections=[Connection("Surface", "outputs:surface")])
    for connection in doc.connections:
        if connection.dst != obj.id or connection.prop is None:
            continue
        texture = doc.objects.get(connection.src)
        if texture is None or texture.kind not in TEXTURE_KINDS:
            continue
        path = _add_texture(doc, texture, material, directory)
        port = surface.inputs.setdefault(connection.prop, Port("color3d"))
        port.connections.append(Connection(path, "outputs:color"))
    return material


def _add_texture(doc: FBXDocument, obj: FBXObject, material: Material, directory: str | None) -> str:
    """Add `obj` (and, for layered textures, its layers) as nodes; returns its path. Textures used twice are added once."""
    for path, node in material.nodes.items():
        if node.extras.get("fbx", {}).get("id") == obj.id:
            return path
    path = obj.name or obj.kind
    while path in material.nodes:
        path += "_"
    types: dict[str, list[str]] = {}
    node = ShaderNode(path, TEXTURE_KINDS[obj.kind], inputs=_ports(_effective(doc, obj), types), extras={"fbx": {"id": obj.id, "types": types}})
    node.outputs["color"] = Port("color3d")
    material.nodes[path] = node
    if obj.kind == "LayeredTexture":
        for i, layer in enumerate(doc.children_of(obj)):
            if layer.kind in TEXTURE_KINDS:
                node.inputs[f"layer{i}"] = Port("color3d", connections=[Connection(_add_texture(doc, layer, material, directory), "outputs:color")])
        return path

    videos = doc.children_of(obj, "Video")
    sources = [obj] + videos
    relative = next((s.child_value("RelativeFilename") for s in sources if s.child_value("RelativeFilename")), "")
    absolute = next((s.child_value("FileName") or s.child_value("Filename") for s in sources
                     if s.child_value("FileName") or s.child_value("Filename")), "")  # fmt: skip
    for key, value in (("RelativeFilename", relative), ("FileName", absolute)):
        if value:
            node.inputs[key] = Port("string", value)
    if relative or absolute:
        node.inputs["file"] = Port("asset", AssetPath(relative or absolute, _resolve(relative, absolute, directory)))
    content = next((v.child_value("Content") for v in videos if v.child_value("Content")), None)
    if content:
        node.extras["fbx"]["content"] = bytes(content)
    return path


def _resolve(relative: str, absolute: str, directory: str | None) -> str | None:
    """The texture file on disk: the relative path next to the FBX file, else the absolute path, if it exists."""
    candidates = []
    if relative and directory is not None:
        candidates.append(os.path.join(directory, relative.replace("\\", "/")))
    if absolute:
        candidates.append(absolute)
    return next((os.path.abspath(c) for c in candidates if os.path.isfile(c)), None)


def _effective(doc: FBXDocument, obj: FBXObject) -> Properties70:
    """The object's properties over its kind's template (what the FBX SDK sees)."""
    merged = doc.template(obj.kind).copy()
    for entry in obj.props:
        merged.add(entry)
    return merged


def _ports(props: Properties70, types: dict[str, list[str]]) -> dict[str, Port]:
    """Ports from properties; `types` receives each property's [type, subtype, flags] for writing back."""
    ports = {}
    for entry in props:
        value = entry.value
        if value is None or entry.type in ("object", "Compound"):
            continue
        if isinstance(value, tuple):
            value = np.array(value, dtype=np.float64)
            type_name = "color3d" if entry.type in COLOR_TYPES and len(value) == 3 else f"double{len(value)}"
        elif isinstance(value, bool) or entry.type == "bool":
            value, type_name = bool(value), "bool"
        elif isinstance(value, int):
            type_name = "int"
        elif isinstance(value, float):
            type_name = "double"
        elif isinstance(value, (bytes, bytearray)):
            continue
        else:
            value, type_name = str(value), "string"
        ports[entry.name] = Port(type_name, value)
        types[entry.name] = [entry.type, entry.subtype, entry.flags]
    return ports


# -- writing ----------------------------------------------------------------------

FILE_INPUTS = ("file", "FileName", "RelativeFilename")


def fbx_surface(material: Material) -> ShaderNode | None:
    """The material's FBX surface node, if it has one (under "fbx:surface" or any other surface output)."""
    keys = sorted((k for k in material.outputs if k.endswith("surface")), key=lambda k: k != "fbx:surface")
    for key in keys:
        node, name, _ = material.trace(material.outputs[key])
        if isinstance(node, ShaderNode) and node.shader_id in FBX_SURFACES:
            return node
    return None


class MaterialWriter:
    """Writes Materials once each and connects them to Models as slots."""

    def __init__(self, doc: FBXDocument):
        self.doc = doc
        self.written: dict[int, FBXObject] = {}  # id(Material) -> Material object
        self.default: FBXObject | None = None  # stands in for empty (None) slots

    def connect_slots(self, node: Node, model: FBXObject) -> None:
        for material in node.materials:
            self.doc.connect(self.material(material) if material is not None else self.default_material(), model)

    def default_material(self) -> FBXObject:
        if self.default is None:
            self.default = self._material_object("default", "Phong", _phong_props(PBRMaterial("default")))
        return self.default

    def material(self, material: Material) -> FBXObject:
        if id(material) in self.written:
            return self.written[id(material)]
        surface = fbx_surface(material)
        if surface is not None:
            shading = surface.extras.get("fbx", {}).get("shading_model") or _shading_name(surface.shader_id)
            props = _props(surface, skip=())
            obj = self._material_object(material.name, shading, props)
            for prop, port in surface.inputs.items():
                for connection in port.connections:
                    node = material.nodes.get(connection.node)
                    if isinstance(node, ShaderNode) and node.shader_id in TEXTURE_KINDS.values():
                        self.doc.connect(self._native_texture(node, material, {}), obj, prop)
        else:
            pbr = material.pbr() or PBRMaterial(material.name)
            obj = self._material_object(material.name, "Phong", _phong_props(pbr))
            textures: dict[int, FBXObject] = {}
            for field_name, props in PBR_TEXTURE_PROPS:
                texture = getattr(pbr, field_name)
                if texture is None:
                    continue
                if id(texture) not in textures:
                    textures[id(texture)] = self._texture_object(_texture_name(texture, field_name), _texture_props(texture),
                                                                 texture.file, texture.content)  # fmt: skip
                for prop in props:
                    self.doc.connect(textures[id(texture)], obj, prop)
        self.written[id(material)] = obj
        return obj

    def _material_object(self, name: str, shading: str, props: Properties70) -> FBXObject:
        obj = self.doc.add_object("Material", name, props=props, children=[
            FBXNode("Version", [Prop("I", 102)]),
            FBXNode("ShadingModel", [Prop("S", shading.lower())]),
            FBXNode("MultiLayer", [Prop("I", 0)]),
        ])  # fmt: skip
        obj.props_index = 3
        return obj

    def _native_texture(self, node: ShaderNode, material: Material, done: dict[int, FBXObject]) -> FBXObject:
        if id(node) in done:
            return done[id(node)]
        if node.shader_id == "FbxLayeredTexture":
            layers = sorted((k for k in node.inputs if k.startswith("layer") and k[5:].isdigit()), key=lambda k: int(k[5:]))
            obj = done[id(node)] = self.doc.add_object("LayeredTexture", node.path, props=_props(node, skip=tuple(layers)),
                                                       children=[FBXNode("Version", [Prop("I", 101)])])  # fmt: skip
            obj.props_index = 1
            for key in layers:
                for connection in node.inputs[key].connections:
                    layer = material.nodes.get(connection.node)
                    if isinstance(layer, ShaderNode) and layer.shader_id in TEXTURE_KINDS.values():
                        self.doc.connect(self._native_texture(layer, material, done), obj)
            return obj
        file = node.inputs["file"].value if "file" in node.inputs else None
        relative = node.inputs["RelativeFilename"].value if "RelativeFilename" in node.inputs else None
        absolute = node.inputs["FileName"].value if "FileName" in node.inputs else None
        obj = done[id(node)] = self._texture_object(node.path, _props(node, skip=FILE_INPUTS), file,
                                                    node.extras.get("fbx", {}).get("content"), relative, absolute)  # fmt: skip
        return obj

    def _texture_object(self, name: str, props: Properties70, file: AssetPath | None, content: bytes | None,
                        relative: str | None = None, absolute: str | None = None) -> FBXObject:  # fmt: skip
        """A Texture (and its Video) laid out as Blender writes them."""
        relative = relative or (file.path if file is not None else "")
        absolute = absolute or (file.resolved or file.path if file is not None else "")
        texture = self.doc.add_object("Texture", name, children=[
            FBXNode("Type", [Prop("S", "TextureVideoClip")]),
            FBXNode("Version", [Prop("I", 202)]),
            FBXNode("TextureName", [Prop("S", f"{name}{NAME_SEP}Texture")]),
            FBXNode("Media", [Prop("S", f"{name}{NAME_SEP}Video")]),
            FBXNode("FileName", [Prop("S", absolute)]),
            FBXNode("RelativeFilename", [Prop("S", relative)]),
            FBXNode("ModelUVTranslation", [Prop("D", 0.0), Prop("D", 0.0)]),
            FBXNode("ModelUVScaling", [Prop("D", 1.0), Prop("D", 1.0)]),
            FBXNode("Texture_Alpha_Source", [Prop("S", "None")]),
            FBXNode("Cropping", [Prop("I", 0)] * 4),
        ], props=props)  # fmt: skip
        texture.props_index = 3
        video_props = Properties70([Property("Path", "KString", "XRefUrl", "", (Prop("S", absolute),))])
        children = [
            FBXNode("Type", [Prop("S", "Clip")]),
            FBXNode("UseMipMap", [Prop("I", 0)]),
            FBXNode("Filename", [Prop("S", absolute)]),
            FBXNode("RelativeFilename", [Prop("S", relative)]),
        ]
        if content:
            children.append(FBXNode("Content", [Prop("R", bytes(content))]))
        video = self.doc.add_object("Video", name, "Clip", props=video_props, children=children)
        video.props_index = 1
        self.doc.connect(video, texture)
        return texture


def material_layer(indices: NDArray[np.int32] | None) -> FBXNode:
    """LayerElementMaterial: one slot for every polygon, or one per polygon."""
    values = np.zeros(1, dtype=np.int32) if indices is None else np.asarray(indices, dtype=np.int32)
    return FBXNode("LayerElementMaterial", [Prop("I", 0)], [
        FBXNode("Version", [Prop("I", 101)]),
        FBXNode("Name", [Prop("S", "")]),
        FBXNode("MappingInformationType", [Prop("S", "AllSame" if indices is None else "ByPolygon")]),
        FBXNode("ReferenceInformationType", [Prop("S", "IndexToDirect")]),
        FBXNode("Materials", [Prop("i", values)]),
    ])  # fmt: skip


# PBRMaterial texture -> the Phong properties it connects to (Blender's PRINCIPLED_TEXTURE_SOCKETS_TO_FBX).
PBR_TEXTURE_PROPS = (
    ("base_color_texture", ("DiffuseColor",)),
    ("opacity_texture", ("TransparencyFactor",)),
    ("emissive_texture", ("EmissiveColor",)),
    ("normal_texture", ("NormalMap",)),
    ("specular_texture", ("SpecularFactor",)),
    ("roughness_texture", ("Shininess", "ShininessExponent")),
    ("metallic_texture", ("ReflectionFactor",)),
)


def _phong_props(pbr: PBRMaterial) -> Properties70:
    """Phong properties for a metallic-roughness material, as Blender's exporter computes them."""
    props = Properties70()
    base = [float(v) for v in pbr.base_color]
    alpha = float(pbr.opacity)
    shininess = ((1.0 - pbr.roughness) * 10) ** 2
    props.set("ShadingModel", "KString", "Phong")
    props.set("DiffuseColor", "Color", base, flags="A")
    props.set("DiffuseFactor", "Number", 1.0, flags="A")
    props.set("EmissiveColor", "Color", [float(v) for v in pbr.emissive], flags="A")
    props.set("EmissiveFactor", "Number", 1.0, flags="A")
    props.set("AmbientColor", "Color", (0.0, 0.0, 0.0), flags="A")
    props.set("AmbientFactor", "Number", 0.0, flags="A")
    transparent = (1.0 - alpha,) * 3 if alpha < 1e-5 or alpha > 1 - 1e-5 else base
    props.set("TransparentColor", "Color", transparent, flags="A")
    props.set("TransparencyFactor", "Number", 1.0 - alpha, flags="A")
    props.set("Opacity", "Number", alpha, flags="A")
    props.set("NormalMap", "Vector3D", (0.0, 0.0, 0.0))
    props.set("BumpFactor", "double", float(pbr.normal_scale))
    props.set("SpecularColor", "Color", base, flags="A")
    props.set("SpecularFactor", "Number", 0.25, flags="A")  # Blender: specular / 2 with the default 0.5
    props.set("Shininess", "Number", shininess, flags="A")
    props.set("ShininessExponent", "Number", shininess, flags="A")
    props.set("ReflectionColor", "Color", base, flags="A")
    props.set("ReflectionFactor", "Number", float(pbr.metallic), flags="A")
    return props


def _texture_props(texture: TextureRef) -> Properties70:
    props = Properties70()
    props.set("CurrentMappingType", "enum", 0)  # UV
    if texture.uv_set not in (None, "st"):  # "st" is USD's name for the first UV set
        props.set("UVSet", "KString", texture.uv_set)
    for axis, mode in zip("UV", texture.wrap):
        props.set(f"WrapMode{axis}", "enum", 1 if mode == "clamp" else 0)
    props.set("Translation", "Vector3D", (*texture.uv_translation[:2], 0.0))
    props.set("Rotation", "Vector3D", (0.0, 0.0, texture.uv_rotation))
    props.set("Scaling", "Vector3D", (*texture.uv_scale[:2], 1.0))
    props.set("UseMaterial", "bool", 1)
    props.set("UseMipMap", "bool", 0)
    return props


def _texture_name(texture: TextureRef, field_name: str) -> str:
    if texture.node is not None:
        return texture.node.path.rsplit("/", 1)[-1]
    if texture.file is not None and texture.file.path:
        return os.path.splitext(os.path.basename(texture.file.path.replace("\\", "/")))[0]
    return field_name


def _shading_name(shader_id: str | None) -> str:
    return {v: k for k, v in SHADING_MODELS.items()}.get(shader_id, "phong")


def _props(node: ShaderNode, skip: tuple[str, ...]) -> Properties70:
    """A node's input values as Properties70, with their original FBX types when known."""
    types = node.extras.get("fbx", {}).get("types", {})
    props = Properties70()
    for name, port in node.inputs.items():
        if name in skip or port.value is None:
            continue
        props.add(_property(name, port, types.get(name)))
    return props


def inferred_fbx_type(port: Port) -> tuple[str, str, str]:
    """(FBX type, subtype, flags) written for a port whose original FBX type isn't known."""
    type_name, value = port.type_name, port.value
    if isinstance(value, (str, AssetPath)):
        return "KString", "", ""
    if type_name.startswith("color3"):
        return "Color", "", "A"
    if type_name == "bool":
        return "bool", "", ""
    if type_name == "int":
        return "int", "Integer", ""
    if type_name[-1:] == "3" or type_name[-2:-1] == "3":  # double3, float3, vector3f, normal3f, ...
        return "Vector3D", "Vector", ""
    return "Number", "", "A"


def _property(name: str, port: Port, fbx_type: list[str] | None) -> Property:
    value = port.value
    if isinstance(value, AssetPath):
        value = value.path
    values = list(np.ravel(value)) if isinstance(value, np.ndarray) else [value]
    type_, subtype, flags = fbx_type if fbx_type is not None else inferred_fbx_type(port)
    raw = []
    for v in values:
        if isinstance(v, str):
            raw.append(Prop("S", v))
        elif isinstance(v, (bool, np.bool_, int, np.integer)):
            raw.append(Prop("L" if type_ in ("KTime", "ULongLong") else "I", int(v)))
        else:
            raw.append(Prop("D", float(v)))
    return Property(name, type_, subtype, flags, tuple(raw))
