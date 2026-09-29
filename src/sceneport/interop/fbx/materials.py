"""FBX materials and textures as generic Material graphs (reading only, for now).

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
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from numpy.typing import NDArray

from ...scene.material import AssetPath, Connection, Material, Port, ShaderNode
from .document import FBXDocument, FBXObject, Properties70
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
    surface = ShaderNode(
        "Surface",
        SHADING_MODELS.get(shading.lower(), "FbxSurfaceMaterial"),
        inputs=_ports(_effective(doc, obj)),
        outputs={"surface": Port("token")},
        extras={"fbx": {"shading_model": shading}},
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
    node = ShaderNode(path, TEXTURE_KINDS[obj.kind], inputs=_ports(_effective(doc, obj)), extras={"fbx": {"id": obj.id}})
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


def _ports(props: Properties70) -> dict[str, Port]:
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
    return ports
