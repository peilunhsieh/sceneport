"""Read and write Wavefront MTL material libraries (for OBJ), following Blender's OBJ importer and exporter.

Reading: each `newmtl` becomes a Material with one ShaderNode "Surface"
(shader_id "MtlSurface") whose inputs are the MTL keys as written (Kd, Ns, d,
illum, Pr, ...; colors as color3d, numbers as double, illum as int). Each texture
map line (map_Kd, map_Bump / bump, refl, ...) becomes a ShaderNode (shader_id
"MtlTexture") with the image as its `file` input (resolved next to the MTL file)
and its options as inputs ("o", "s", "bm", "clamp", ...), connected to the
surface input it drives (map_Kd -> Kd, bump -> bump, refl -> refl). pbr()
interprets it like Blender's importer.

Writing: a material read from MTL (with an MtlSurface) is written back as it
was. Any other Material is interpreted with Material.pbr() and written as:

    Ns  (1 - roughness)^2 * 1000       Ka  metallic (white when not metallic)
    Kd  base color                     Ks  specular color (0.5 grey by default)
    Ke  emissive color                 Ni  ior
    d   opacity                        illum 2 (3 metallic, 6 metallic + transparent, 9 transparent)

plus, with `pbr_extensions`, Pr (roughness), Pm (metallic), Pc / Pcr (clearcoat,
Blender's 4x scale) instead of Ns and Ka. Textures become map_Kd, map_Ks, map_Ke,
map_d, map_Bump (with -bm for the normal strength) and map_Ns / map_refl
(classic) or map_Pr / map_Pm (PBR), with -o / -s for the UV offset and scale. A
value is left out when a texture replaces it. UV rotation can't be written.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from ..scene.material import AssetPath, Connection, Material, Port, ShaderNode
from ..scene.pbr import MTL_SURFACE, PBRMaterial, TextureRef, native_surface
from .export import texture_path

# Map keyword -> the surface input it drives (Blender's importer accepts these spellings).
MAP_INPUTS = {"map_Bump": "bump", "map_bump": "bump", "bump": "bump", "refl": "refl", "map_refl": "refl"}
# Texture options and how many arguments they take ("floats" = up to 3 numbers).
_OPTIONS = {
    "o": "floats", "s": "floats", "t": "floats", "bm": 1, "mm": 2, "boost": 1, "texres": 1,
    "imfchan": 1, "type": 1, "blendu": 1, "blendv": 1, "clamp": 1, "cc": 1,
}  # fmt: skip
_COLOR_KEYS = ("Ka", "Kd", "Ks", "Ke", "Kt", "Tf")


def load_library(path: str | os.PathLike, materials: dict[str, Material] | None = None) -> dict[str, Material]:
    """Materials of an MTL file by name (added to `materials`; a repeated name updates that material)."""
    materials = {} if materials is None else materials
    directory = os.path.dirname(os.path.abspath(path))
    material = None
    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.strip()  # comments are whole lines ("#" can appear in texture paths)
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(" ")
            rest = rest.strip()
            if key == "newmtl":
                name = rest or "material"
                material = materials.get(name) or _new_material(name)
                materials[name] = material
            elif material is None:
                continue
            elif key.startswith("map_") or key in ("bump", "refl", "disp", "decal"):
                _add_map(material, key, rest, directory)
            else:
                material.nodes["Surface"].inputs[key] = _value_port(key, rest)
    return materials


def _new_material(name: str) -> Material:
    surface = ShaderNode("Surface", MTL_SURFACE, outputs={"surface": Port("token")})
    return Material(name, nodes={"Surface": surface},
                    outputs={"surface": Port("token", connections=[Connection("Surface", "outputs:surface")])})  # fmt: skip


def _value_port(key: str, text: str) -> Port:
    tokens = text.split()
    try:
        numbers = [float(t) for t in tokens]
    except ValueError:
        return Port("string", text)
    if key == "illum" and numbers:
        return Port("int", int(numbers[0]))  # some files write it as a float
    if len(numbers) == 1:
        return Port("double", numbers[0])
    if len(numbers) == 3 and key in _COLOR_KEYS:
        return Port("color3d", np.array(numbers))
    if 2 <= len(numbers) <= 4:
        return Port(f"double{len(numbers)}", np.array(numbers))
    return Port("string", text)


def _add_map(material: Material, key: str, text: str, directory: str) -> None:
    """A texture map line: options, then the image path (which may contain spaces)."""
    tokens = text.split()
    inputs: dict[str, Port] = {}
    i = 0
    while i < len(tokens) and tokens[i].startswith("-") and tokens[i][1:] in _OPTIONS:
        option, arity = tokens[i][1:], _OPTIONS[tokens[i][1:]]
        i += 1
        if arity == "floats":  # u [v [w]]; missing components get the default, as in Blender
            numbers = []
            while len(numbers) < 3 and i < len(tokens) - 1 and _is_number(tokens[i]):  # the last token is the file
                numbers.append(float(tokens[i]))
                i += 1
            default = 1.0 if option == "s" else 0.0
            inputs[option] = Port("double3", np.array(numbers + [default] * (3 - len(numbers))))
        else:
            args = tokens[i : i + arity]
            i += arity
            numbers = [float(a) for a in args if _is_number(a)]
            inputs[option] = Port("double", numbers[0]) if arity == 1 and len(numbers) == 1 else Port("string", " ".join(args))
    file = " ".join(tokens[i:]).strip().strip('"')
    if file:
        resolved = os.path.join(directory, file.replace("\\", "/"))
        inputs = {"file": Port("asset", AssetPath(file, os.path.abspath(resolved) if os.path.isfile(resolved) else None)), **inputs}
    target = MAP_INPUTS.get(key, key[4:] if key.startswith("map_") else key)
    path = key
    while path in material.nodes:
        path += "_"
    material.nodes[path] = ShaderNode(path, "MtlTexture", inputs=inputs, outputs={"color": Port("color3d")},
                                      extras={"mtl": {"key": key}})  # fmt: skip
    port = material.nodes["Surface"].inputs.setdefault(target, Port("color3d" if target in _COLOR_KEYS else "double"))
    port.connections.append(Connection(path, "outputs:color"))


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _native_block(name: str, material: Material, surface: ShaderNode, directory) -> list[str]:
    """A material read from MTL, written back key by key and map by map."""
    lines = [f"newmtl {name}"]
    for key, port in surface.inputs.items():
        if port.value is not None:
            lines.append(f"{key} {_format(port.value)}")
    for key, port in surface.inputs.items():
        for connection in port.connections:
            node = material.nodes.get(connection.node)
            if not isinstance(node, ShaderNode) or node.shader_id != "MtlTexture" or "file" not in node.inputs:
                continue
            path = texture_path(node.inputs["file"].value, directory)
            options = "".join(f" -{option} {_format(p.value)}" for option, p in node.inputs.items()
                              if option != "file" and p.value is not None)  # fmt: skip
            map_key = node.extras.get("mtl", {}).get("key") or (key if key in ("bump", "refl") else f"map_{key}")
            lines.append(f"{map_key}{options} {path}")
    return lines


def _format(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, np.ndarray):
        return " ".join(_format(v) for v in value.ravel())
    return str(value)

# (PBRMaterial texture field, MTL map, "classic" | "pbr" | "both"), in Blender's order.
_MAPS = (
    ("base_color_texture", "map_Kd", "both"),
    ("metallic_texture", "map_Pm", "pbr"),
    ("specular_texture", "map_Ks", "both"),
    ("roughness_texture", "map_Ns", "classic"),
    ("roughness_texture", "map_Pr", "pbr"),
    ("metallic_texture", "map_refl", "classic"),
    ("emissive_texture", "map_Ke", "both"),
    ("opacity_texture", "map_d", "both"),
    ("normal_texture", "map_Bump", "both"),
)


def material_block(
    name: str, material: Material, directory: str | os.PathLike | None, *, pbr_extensions: bool = False
) -> list[str]:
    """The `newmtl` block for `material`; texture paths are made relative to `directory`."""
    surface = native_surface(material, (MTL_SURFACE,))
    if surface is not None:
        return _native_block(name, material, surface, directory)
    pbr = material.pbr() or PBRMaterial(material.name)
    textures = {field: getattr(pbr, field) for field, _, _ in _MAPS}
    lines = [f"newmtl {name}"]
    roughness, metallic, alpha = float(pbr.roughness), float(pbr.metallic), float(pbr.opacity)
    specular = pbr.specular_color if pbr.specular_color is not None else np.full(3, 0.5)
    if not pbr_extensions:
        if textures["roughness_texture"] is None:
            lines.append(f"Ns {(1.0 - roughness) ** 2 * 1000.0:.6f}")
        lines.append(_float3("Ka", np.full(3, metallic) if metallic != 0.0 else np.ones(3)))
    if textures["base_color_texture"] is None:
        lines.append(_float3("Kd", pbr.base_color))
    if textures["specular_texture"] is None:
        lines.append(_float3("Ks", specular))
    if textures["emissive_texture"] is None:
        lines.append(_float3("Ke", pbr.emissive))
    lines.append(f"Ni {float(pbr.ior):.6f}")
    if textures["opacity_texture"] is None:
        lines.append(f"d {alpha:.6f}")
    lines.append(f"illum {_illum(float(np.mean(specular)), metallic, alpha != 1.0)}")
    if pbr_extensions:
        if textures["roughness_texture"] is None:
            lines.append(f"Pr {roughness:.6f}")
        if textures["metallic_texture"] is None:
            lines.append(f"Pm {metallic:.6f}")
        if pbr.clearcoat > 0:
            lines.append(f"Pc {pbr.clearcoat * 4.0:.6f}")  # Blender's exporter keeps an old 4x factor
            lines.append(f"Pcr {float(pbr.clearcoat_roughness):.6f}")

    for field, key, mode in _MAPS:
        texture: TextureRef | None = textures[field]
        if texture is None or mode == ("pbr" if not pbr_extensions else "classic"):
            continue
        path = _map_path(texture, directory)
        if path is None:
            continue
        options = ""
        if np.any(texture.uv_translation[:2]):
            options += f" -o {texture.uv_translation[0]:.6f} {texture.uv_translation[1]:.6f} 0"
        if np.any(np.asarray(texture.uv_scale[:2]) != 1):
            options += f" -s {texture.uv_scale[0]:.6f} {texture.uv_scale[1]:.6f} 1"
        if key == "map_Bump" and pbr.normal_scale > 0.0001:
            options += f" -bm {float(pbr.normal_scale):.6f}"
        if texture.uv_rotation:
            warnings.warn(f"MTL {name!r}: {key} UV rotation ({texture.uv_rotation} degrees) can't be written")
        lines.append(f"{key}{options} {path}")
    return lines


def _illum(specular: float, metallic: float, transparent: bool) -> int:
    """Blender's illumination model choice (see the Wavefront MTL illum list)."""
    if specular == 0.0:
        return 1  # color and ambient on
    if metallic > 0.0:
        return 6 if transparent else 3  # reflection (with refraction when transparent)
    return 9 if transparent else 2  # glass, else highlight on


def _float3(key: str, values) -> str:
    x, y, z = (float(v) for v in np.ravel(values)[:3])
    return f"{key} {x:.6f} {y:.6f} {z:.6f}"


def _map_path(texture: TextureRef, directory) -> str | None:
    if texture.file is None or not texture.file.path:
        return None
    fallback = f"{texture.node.path.rsplit('/', 1)[-1]}.png" if texture.node is not None else "texture.png"
    return texture_path(texture.file, directory, content=texture.content, fallback_name=fallback)
