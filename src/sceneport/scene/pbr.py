"""A simple metallic-roughness description of a Material graph.

`pbr_material(material)` interprets the surface shader of a generic Material
graph (see material.py) into a PBRMaterial: constant values plus, per
parameter, an optional texture (file, channels, UV set, UV transform, wrap).
It understands these surface nodes:

- UsdPreviewSurface (textures: UsdUVTexture, UV sets from UsdPrimvarReader,
  UsdTransform2d)
- MaterialX ND_standard_surface_surfaceshader and ND_open_pbr_surface_surfaceshader
  (textures: ND_image_* / ND_tiledimage_*, through ND_normalmap_*, ND_extract_*,
  ND_separate*_* and ND_convert_* nodes)
- FBX FbxSurfacePhong / FbxSurfaceLambert (textures: FbxFileTexture), converted
  like Blender's FBX importer does (approximate; Phong isn't physically based)

A texture replaces its parameter's value (as in UsdPreviewSurface and Blender).
Anything that can't be expressed is skipped and described in `notes`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from .material import AssetPath, Material, Port, ShaderNode

PREVIEW_SURFACE = "UsdPreviewSurface"
STANDARD_SURFACE = "ND_standard_surface_surfaceshader"
OPEN_PBR = "ND_open_pbr_surface_surfaceshader"
FBX_SURFACES = ("FbxSurfacePhong", "FbxSurfaceLambert", "FbxSurfaceMaterial")
SUPPORTED_SURFACES = (PREVIEW_SURFACE, STANDARD_SURFACE, OPEN_PBR, *FBX_SURFACES)

_WRAP = {  # UsdUVTexture / MaterialX address modes -> common names
    "repeat": "repeat", "periodic": "repeat", "clamp": "clamp", "mirror": "mirror",
    "black": "black", "constant": "black",
}  # fmt: skip
_CHANNELS = {"r": "r", "g": "g", "b": "b", "a": "a", "rgb": "rgb", "rgba": "rgba", "x": "r", "y": "g", "z": "b", "w": "a"}


@dataclass
class TextureRef:
    """An image feeding one parameter: parameter = texel[channels] * scale + bias."""

    file: AssetPath | None
    channels: str = "rgb"  # "rgb", "rgba", or one of "r", "g", "b", "a"
    uv_set: str | None = None  # UV set / primvar name; None = the mesh's first
    scale: NDArray[np.float64] = field(default_factory=lambda: np.ones(4))
    bias: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    uv_translation: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    uv_rotation: float = 0.0  # degrees
    uv_scale: NDArray[np.float64] = field(default_factory=lambda: np.ones(2))
    wrap: tuple[str | None, str | None] = (None, None)  # (u, v): "repeat", "clamp", "mirror", "black"; None = unspecified
    color_space: str | None = None  # "sRGB", "raw", ...; None = unspecified
    node: ShaderNode | None = field(default=None, repr=False)  # the texture node it came from

    @property
    def content(self) -> bytes | None:
        """Image bytes embedded in the file (FBX), if any."""
        return None if self.node is None else self.node.extras.get("fbx", {}).get("content")


@dataclass
class PBRMaterial:
    """Metallic-roughness parameters (the UsdPreviewSurface / glTF set). Colors are linear RGB."""

    name: str
    base_color: NDArray[np.float64] = field(default_factory=lambda: np.full(3, 0.8))
    base_color_texture: TextureRef | None = None
    metallic: float = 0.0
    metallic_texture: TextureRef | None = None
    roughness: float = 0.5
    roughness_texture: TextureRef | None = None
    normal_texture: TextureRef | None = None  # tangent space
    normal_scale: float = 1.0
    emissive: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    emissive_texture: TextureRef | None = None
    opacity: float = 1.0
    opacity_texture: TextureRef | None = None
    opacity_threshold: float = 0.0  # > 0: alpha cutout at this value
    occlusion_texture: TextureRef | None = None
    ior: float = 1.5
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.01
    specular_color: NDArray[np.float64] | None = None  # set when the source uses a specular workflow
    specular_texture: TextureRef | None = None
    displacement_texture: TextureRef | None = None
    source: str | None = None  # the surface node's shader_id
    notes: list[str] = field(default_factory=list)  # what was approximated or skipped


def pbr_material(material: Material, context: str | None = None) -> PBRMaterial | None:
    """Interpret `material`'s surface shader, or None if it has no surface node this module understands.

    The surface for `context` (e.g. "mtlx") is tried first, then the universal
    "surface", then every other render context's surface.
    """
    contexts = [context] if context else []
    contexts += [None] + [key.rpartition(":")[0] for key in material.outputs if key.endswith(":surface")]
    for ctx in contexts:
        surface = material.surface(ctx)
        if surface is not None and surface.shader_id in SUPPORTED_SURFACES:
            out = PBRMaterial(material.name, source=surface.shader_id)
            reader = _Reader(material, surface, out)
            if surface.shader_id == PREVIEW_SURFACE:
                reader.preview_surface()
            elif surface.shader_id in (STANDARD_SURFACE, OPEN_PBR):
                reader.materialx(surface.shader_id == OPEN_PBR)
            else:
                reader.fbx()
            return out
    return None


class _Reader:
    def __init__(self, material: Material, surface: ShaderNode, out: PBRMaterial):
        self.material, self.surface, self.out = material, surface, out

    # -- surfaces -------------------------------------------------------------

    def preview_surface(self) -> None:
        o = self.out
        o.base_color, o.base_color_texture = self.color("diffuseColor", (0.18, 0.18, 0.18))
        o.emissive, o.emissive_texture = self.color("emissiveColor", (0, 0, 0))
        o.metallic, o.metallic_texture = self.number("metallic", 0.0)
        o.roughness, o.roughness_texture = self.number("roughness", 0.5)
        o.opacity, o.opacity_texture = self.number("opacity", 1.0)
        o.opacity_threshold, _ = self.number("opacityThreshold", 0.0)
        o.ior, _ = self.number("ior", 1.5)
        o.clearcoat, _ = self.number("clearcoat", 0.0)
        o.clearcoat_roughness, _ = self.number("clearcoatRoughness", 0.01)
        o.normal_texture = self.texture("normal")
        o.occlusion_texture = self.texture("occlusion")
        o.displacement_texture = self.texture("displacement")
        if self.number("useSpecularWorkflow", 0)[0]:
            o.specular_color, o.specular_texture = self.color("specularColor", (0, 0, 0))

    def materialx(self, open_pbr: bool) -> None:
        o = self.out
        if open_pbr:
            weight, _ = self.number("base_weight", 1.0)
            o.base_color, o.base_color_texture = self.color("base_color", (0.8, 0.8, 0.8))
            o.metallic, o.metallic_texture = self.number("base_metalness", 0.0)
            o.roughness, o.roughness_texture = self.number("specular_roughness", 0.3)
            o.ior, _ = self.number("specular_ior", 1.5)
            o.clearcoat, _ = self.number("coat_weight", 0.0)
            o.clearcoat_roughness, _ = self.number("coat_roughness", 0.0)
            o.opacity, o.opacity_texture = self.number("geometry_opacity", 1.0)
            o.normal_texture = self.texture("geometry_normal")
            emission, _ = self.number("emission_luminance", 0.0)
            if emission > 0:
                self.out.notes.append(f"emission_luminance ({emission} nits) dropped; emission_color used as is")
        else:
            weight, _ = self.number("base", 1.0)
            o.base_color, o.base_color_texture = self.color("base_color", (0.8, 0.8, 0.8))
            o.metallic, o.metallic_texture = self.number("metalness", 0.0)
            o.roughness, o.roughness_texture = self.number("specular_roughness", 0.2)
            o.ior, _ = self.number("specular_IOR", 1.5)
            o.clearcoat, _ = self.number("coat", 0.0)
            o.clearcoat_roughness, _ = self.number("coat_roughness", 0.1)
            opacity, o.opacity_texture = self.color("opacity", (1, 1, 1))
            o.opacity = float(np.mean(opacity))
            o.normal_texture = self.texture("normal")
            emission, _ = self.number("emission", 0.0)
        o.base_color = o.base_color * weight
        if emission > 0:
            color, o.emissive_texture = self.color("emission_color", (1, 1, 1))
            o.emissive = color if open_pbr else color * emission

    def fbx(self) -> None:
        """Blender's FBX importer conversion (io_scene_fbx: blen_read_material and texture links)."""
        o = self.out
        o.base_color, o.base_color_texture = self.color("DiffuseColor", (1, 1, 1))
        o.base_color_texture = self.texture("DiffuseColor") or self.texture("3dsMax|maps|texmap_diffuse")
        shininess = max(self.first("Shininess", self.first("ShininessExponent", 20.0)), 0.0)
        o.roughness = float(np.clip(1.0 - sqrt(shininess) / 10.0, 0.0, 1.0))
        if self.texture("ShininessExponent") is not None:
            self.out.notes.append("ShininessExponent texture skipped (it is inverse roughness)")
        alpha = 1.0 - self.first("TransparencyFactor", 0.0)
        if alpha in (0.0, 1.0):
            opacity = self.surface.inputs.get("Opacity")
            alpha = self.first("Opacity", 1.0) if opacity is not None else 1.0 - self.first("TransparentColor", 0.0)
        o.opacity = alpha
        o.opacity_texture = self.texture("TransparentColor") or self.texture("TransparencyFactor")
        if o.opacity_texture is not None:
            o.opacity_texture.channels = "a"
            self.out.notes.append("opacity texture read from the image's alpha")
        o.metallic = self.first("ReflectionFactor", 0.0)
        o.metallic_texture = next(filter(None, map(self.texture, ("ReflectionColor", "ReflectionFactor", "3dsMax|maps|texmap_reflection"))), None)
        o.normal_texture = next(filter(None, map(self.texture, ("NormalMap", "Bump", "3dsMax|maps|texmap_bump"))), None)
        o.normal_scale = self.first("BumpFactor", 1.0)
        emissive, _ = self.color("EmissiveColor", (0, 0, 0))
        o.emissive = emissive * self.first("EmissiveFactor", 1.0)
        o.emissive_texture = self.texture("EmissiveColor")
        if self.texture("SpecularColor") or self.texture("SpecularFactor"):
            self.out.notes.append("specular texture skipped (metallic-roughness has no specular level)")

    # -- inputs ---------------------------------------------------------------

    def resolve(self, name: str) -> tuple[Port | None, TextureRef | None]:
        """(port holding the value or None, texture or None); the value is a fallback when there is a texture."""
        port = self.surface.inputs.get(name)
        if port is None:
            return None, None
        if not port.connections:
            return port, None
        node, output, source = self.material.trace(port)
        if not isinstance(node, ShaderNode) or output is None or not output.startswith("outputs:"):
            return (source if source.value is not None else None), None
        texture = self.texture_from(node, output[len("outputs:") :], name)
        return (port if port.value is not None else None), texture  # FBX keeps the value next to the texture

    def number(self, name: str, default: float) -> tuple[float, TextureRef | None]:
        port, texture = self.resolve(name)
        value = default if port is None or port.value is None else float(np.ravel(port.value)[0])
        if texture is not None and texture.channels in ("rgb", "rgba"):
            texture.channels = "r"
        return value, texture

    def color(self, name: str, default) -> tuple[NDArray[np.float64], TextureRef | None]:
        port, texture = self.resolve(name)
        value = default if port is None or port.value is None else port.value
        value = np.broadcast_to(np.asarray(value, dtype=np.float64).ravel()[:3], 3).copy()
        return value, texture

    def first(self, name: str, default: float) -> float:
        """First component of an FBX property (legacy properties like Shininess are triples)."""
        port = self.surface.inputs.get(name)
        return default if port is None or port.value is None else float(np.ravel(port.value)[0])

    def texture(self, name: str) -> TextureRef | None:
        return self.resolve(name)[1]

    # -- textures -------------------------------------------------------------

    def texture_from(self, node: ShaderNode, output: str, param: str) -> TextureRef | None:
        channels = _CHANNELS.get(output, "rgb")
        for _ in range(32):  # follow pass-through nodes up to the image
            shader_id = node.shader_id or ""
            if shader_id == "UsdUVTexture":
                return self.usd_texture(node, channels)
            if shader_id.startswith(("ND_image_", "ND_tiledimage_")):
                return self.mtlx_texture(node, channels)
            if shader_id == "FbxFileTexture":
                return self.fbx_texture(node)
            if shader_id == "FbxLayeredTexture":
                self.out.notes.append(f"{param}: layered texture {node.path!r} read as its first layer")
                name = "in" if "in" in node.inputs else min((k for k in node.inputs if k.startswith("layer")), default=None)
            elif shader_id.startswith(("ND_normalmap", "ND_convert_")):
                name = "in"
            elif shader_id.startswith("ND_extract_"):
                name = "in"
                channels = "rgba"[int(self.value(node, "index", 0))]
            elif shader_id.startswith("ND_separate"):
                name = "in"
                channels = _CHANNELS.get(output[-1:], "rgb") if output.startswith("out") else channels
            else:
                self.out.notes.append(f"{param}: node {node.path!r} ({shader_id or 'no id'}) not supported")
                return None
            port = node.inputs.get(name) if name else None
            if port is None or not port.connections:
                self.out.notes.append(f"{param}: {node.path!r} has no connected input")
                return None
            source, source_output, _ = self.material.trace(port)
            if not isinstance(source, ShaderNode) or not (source_output or "").startswith("outputs:"):
                self.out.notes.append(f"{param}: {node.path!r} input is not a node output")
                return None
            node, output = source, source_output[len("outputs:") :]
        return None

    def value(self, node: ShaderNode, name: str, default=None):
        """A node input's value, following connections to interface values."""
        port = node.inputs.get(name)
        if port is None:
            return default
        _, _, source = self.material.trace(port)
        return default if source.value is None else source.value

    def usd_texture(self, node: ShaderNode, channels: str) -> TextureRef:
        file_port = node.inputs.get("file")
        texture = TextureRef(self.value(node, "file"), channels, node=node)
        texture.scale = _vec(self.value(node, "scale"), np.ones(4))
        texture.bias = _vec(self.value(node, "bias"), np.zeros(4))
        texture.wrap = tuple(_WRAP.get(str(self.value(node, key, ""))) for key in ("wrapS", "wrapT"))
        space = str(self.value(node, "sourceColorSpace", "auto"))
        texture.color_space = space if space != "auto" else (file_port.color_space if file_port else None)
        st = node.inputs.get("st")
        for _ in range(8):  # st <- [UsdTransform2d <-] UsdPrimvarReader_float2
            if st is None or not st.connections:
                break
            source, _, _ = self.material.trace(st)
            if not isinstance(source, ShaderNode):
                break
            if source.shader_id == "UsdTransform2d":
                texture.uv_translation = _vec(self.value(source, "translation"), np.zeros(2))
                texture.uv_rotation = float(self.value(source, "rotation", 0.0))
                texture.uv_scale = _vec(self.value(source, "scale"), np.ones(2))
                st = source.inputs.get("in")
            elif (source.shader_id or "").startswith("UsdPrimvarReader"):
                name = self.value(source, "varname")
                texture.uv_set = str(name) if name else None
                break
            else:
                break
        return texture

    def mtlx_texture(self, node: ShaderNode, channels: str) -> TextureRef:
        file_port = node.inputs.get("file")
        texture = TextureRef(self.value(node, "file"), channels, node=node)
        texture.color_space = file_port.color_space if file_port else None
        texture.wrap = tuple(_WRAP.get(str(self.value(node, key, ""))) for key in ("uaddressmode", "vaddressmode"))
        if node.shader_id.startswith("ND_tiledimage_"):
            texture.uv_scale = _vec(self.value(node, "uvtiling"), np.ones(2))
            texture.uv_translation = _vec(self.value(node, "uvoffset"), np.zeros(2))
        coord = node.inputs.get("texcoord")
        if coord is not None and coord.connections:
            source, _, _ = self.material.trace(coord)
            if isinstance(source, ShaderNode) and (source.shader_id or "").startswith("ND_geompropvalue"):
                name = self.value(source, "geomprop")
                texture.uv_set = str(name) if name else None
            elif isinstance(source, ShaderNode) and (source.shader_id or "").startswith("ND_texcoord"):
                index = int(self.value(source, "index", 0))
                texture.uv_set = None if index == 0 else f"index {index}"
        return texture

    def fbx_texture(self, node: ShaderNode) -> TextureRef:
        texture = TextureRef(self.value(node, "file"), "rgb", node=node)
        uv_set = self.value(node, "UVSet", "")
        texture.uv_set = uv_set if uv_set and uv_set != "default" else None
        texture.uv_translation = _vec(self.value(node, "Translation"), np.zeros(3))[:2]
        texture.uv_rotation = float(_vec(self.value(node, "Rotation"), np.zeros(3))[2])
        texture.uv_scale = _vec(self.value(node, "Scaling"), np.ones(3))[:2]
        texture.wrap = tuple("clamp" if self.value(node, key, 0) else "repeat" for key in ("WrapModeU", "WrapModeV"))
        return texture


def _vec(value, default: NDArray) -> NDArray[np.float64]:
    if value is None:
        return default
    return np.asarray(value, dtype=np.float64).ravel()
