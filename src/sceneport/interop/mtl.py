"""Write Wavefront MTL material libraries (for OBJ), the way Blender's OBJ exporter does.

Each Material is interpreted with Material.pbr() and written as:

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

from ..scene.material import Material
from ..scene.pbr import PBRMaterial, TextureRef
from .export import texture_path

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
