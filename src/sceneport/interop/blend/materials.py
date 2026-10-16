"""PBR materials for the .blend writer, inside Blender (imports bpy).

Each sceneport Material becomes one Blender material built from its PBR reading
(`Material.pbr()`, whatever its source: UsdPreviewSurface, MaterialX, FBX Phong,
MTL) as a Principled BSDF, laid out the way Blender's own USD importer builds
UsdPreviewSurface materials, so both routes into Blender agree:

- Constants go on the Principled BSDF's inputs (Base Color, Metallic, Roughness,
  IOR, Alpha, Emission Color with strength 1, Coat Weight / Roughness, Specular
  Tint for the specular workflow).
- A texture is an Image Texture node (sRGB for colors, Non-Color for data),
  reading its UV set through a UV Map node and its UV transform through a
  Mapping node (point type: translation, rotation, scale as UsdTransform2d);
  wrap is the image's extension. One channel comes out of a Separate Color node
  (alpha from the image's Alpha). A scale / bias other than 1 / 0 is a
  multiply-add node.
- A normal texture feeds a Normal Map node (tangent space, strength =
  normal_scale), re-encoded to 0..1 by a multiply-add when its scale / bias
  decode it to -1..1 (USD's convention).
- Opacity is coverage, Principled Alpha (as Blender's FBX and glTF importers
  read transparency): below 1 (or a texture) renders blended; an opacity
  threshold cuts out instead (alpha = opacity > threshold, dithered). Blender's
  USD importer differs here: without a threshold it reads opacity as glass
  (Transmission Weight = 1 - opacity, Alpha 1), and it doesn't threshold a
  constant opacity.
- Displacement goes through a Displacement node to the output; occlusion has no
  Principled input and is kept as an unconnected image node.

Images keep paths relative to the .blend; images embedded in the source (FBX)
are packed into the .blend, and with `pack_images` the ones found on disk are
too. Materials nothing uses keep a fake user.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import bpy
import numpy as np

from ...scene.material import Material
from ...scene.pbr import PBRMaterial, TextureRef

WRAP = {"repeat": "REPEAT", "clamp": "EXTEND", "mirror": "MIRROR", "black": "CLIP"}
COLOR_INPUTS = {"base_color", "emissive", "specular_color"}  # sRGB textures; the rest are data


class MaterialWriter:
    def __init__(self, blend_dir: Path | None, pack_images: bool = False):
        self.blend_dir = blend_dir
        self.pack_images = pack_images  # also pack images found on disk (embedded ones always are)
        self.materials: dict[int, bpy.types.Material] = {}  # id(Material) -> Blender material
        self.images: dict[tuple, bpy.types.Image] = {}  # (file or content, color space) -> image

    def material(self, material: Material | None):
        if material is None:
            return None
        if id(material) not in self.materials:
            self.materials[id(material)] = self._build(material)
        return self.materials[id(material)]

    # building

    def _build(self, material: Material):
        data = bpy.data.materials.new(material.name)
        if data.node_tree is None:  # older Blender: materials start without nodes
            data.use_nodes = True
        tree = data.node_tree
        principled = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        output = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
        if principled is None or output is None:
            tree.nodes.clear()
            principled, output = tree.nodes.new("ShaderNodeBsdfPrincipled"), tree.nodes.new("ShaderNodeOutputMaterial")
            tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
        pbr = material.pbr()
        if pbr is None:
            warnings.warn(f"material {material.name!r}: no surface sceneport can read as PBR; written as a default Principled BSDF")
            return data
        self.tree, self.principled, self.uv_maps = tree, principled, {}
        inputs = principled.inputs
        inputs["Base Color"].default_value = (*pbr.base_color, 1.0)
        inputs["Metallic"].default_value = pbr.metallic
        inputs["Roughness"].default_value = pbr.roughness
        inputs["IOR"].default_value = pbr.ior
        inputs["Emission Color"].default_value = (*pbr.emissive, 1.0)
        inputs["Emission Strength"].default_value = 1.0
        inputs["Coat Weight"].default_value = pbr.clearcoat
        inputs["Coat Roughness"].default_value = pbr.clearcoat_roughness
        if pbr.specular_color is not None:
            inputs["Specular Tint"].default_value = (*pbr.specular_color, 1.0)
        for field, socket in (("base_color", "Base Color"), ("metallic", "Metallic"), ("roughness", "Roughness"),
                              ("emissive", "Emission Color"), ("specular_color", "Specular Tint")):  # fmt: skip
            texture = getattr(pbr, f"{field}_texture" if field != "specular_color" else "specular_texture")
            if texture is not None:
                self.tree.links.new(self._texture(texture, field), inputs[socket])
        self._opacity(data, pbr)
        if pbr.normal_texture is not None:
            self._normal(pbr)
        if pbr.displacement_texture is not None:
            displacement = tree.nodes.new("ShaderNodeDisplacement")
            displacement.inputs["Midlevel"].default_value = 0.0
            tree.links.new(self._texture(pbr.displacement_texture, "displacement"), displacement.inputs["Height"])
            tree.links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
        if pbr.occlusion_texture is not None:  # no Principled input: kept, unconnected
            self._image_node(pbr.occlusion_texture, "occlusion").label = "Occlusion"
        return data

    def _opacity(self, data, pbr: PBRMaterial) -> None:
        alpha_socket = self.principled.inputs["Alpha"]
        source = self._texture(pbr.opacity_texture, "opacity") if pbr.opacity_texture is not None else None
        if pbr.opacity_threshold > 0.0:  # a cutout: opaque where opacity passes the threshold
            if source is None:
                alpha_socket.default_value = 1.0 if pbr.opacity >= pbr.opacity_threshold else 0.0
                return
            cut = self.tree.nodes.new("ShaderNodeMath")
            cut.operation = "GREATER_THAN"
            cut.inputs[1].default_value = pbr.opacity_threshold
            self.tree.links.new(source, cut.inputs[0])
            self.tree.links.new(cut.outputs["Value"], alpha_socket)
            data.surface_render_method = "DITHERED"
            return
        alpha_socket.default_value = pbr.opacity
        if source is not None:
            self.tree.links.new(source, alpha_socket)
        if source is not None or pbr.opacity < 1.0:
            data.surface_render_method = "BLENDED"

    def _normal(self, pbr: PBRMaterial) -> None:
        texture = pbr.normal_texture
        normal_map = self.tree.nodes.new("ShaderNodeNormalMap")
        normal_map.space = "TANGENT"
        normal_map.inputs["Strength"].default_value = pbr.normal_scale
        if texture.uv_set:
            normal_map.uv_map = texture.uv_set
        color = self._image_output(texture, "normal")
        scale, bias = texture.scale[:3], texture.bias[:3]
        if not (np.allclose(scale, 1.0) and np.allclose(bias, 0.0)):  # decoded to -1..1: back to Blender's 0..1
            color = self._multiply_add(color, 0.5 * scale, 0.5 * bias + 0.5, vector=True)
        self.tree.links.new(color, normal_map.inputs["Color"])
        self.tree.links.new(normal_map.outputs["Normal"], self.principled.inputs["Normal"])

    def _texture(self, texture: TextureRef, field: str):
        """The socket giving `texture`'s value for `field` (channel picked, scale / bias applied)."""
        socket = self._image_output(texture, field)
        channels = "rgb" if texture.channels in ("rgb", "rgba") else texture.channels
        index = [("rgba".index(c)) for c in channels]
        scale, bias = texture.scale[index], texture.bias[index]
        if not (np.allclose(scale, 1.0) and np.allclose(bias, 0.0)):
            vector = len(index) > 1
            socket = self._multiply_add(socket, scale if vector else float(scale[0]), bias if vector else float(bias[0]), vector)
        return socket

    def _image_output(self, texture: TextureRef, field: str):
        node = self._image_node(texture, field)
        channels = texture.channels
        if channels in ("rgb", "rgba"):
            return node.outputs["Color"]
        if channels == "a":
            return node.outputs["Alpha"]
        separate = self.tree.nodes.new("ShaderNodeSeparateColor")
        separate.mode = "RGB"
        self.tree.links.new(node.outputs["Color"], separate.inputs["Color"])
        return separate.outputs[{"r": "Red", "g": "Green", "b": "Blue"}[channels]]

    def _image_node(self, texture: TextureRef, field: str):
        node = self.tree.nodes.new("ShaderNodeTexImage")
        color = texture.color_space in ("sRGB", "srgb_texture") or (texture.color_space is None and field in COLOR_INPUTS)
        node.image = self._image(texture, "sRGB" if color else "Non-Color")
        node.extension = WRAP.get(texture.wrap[0] or texture.wrap[1] or "repeat", "REPEAT")
        transformed = not (np.allclose(texture.uv_translation, 0.0) and texture.uv_rotation == 0.0 and np.allclose(texture.uv_scale, 1.0))
        if texture.uv_set or transformed:
            uv = self._uv_map(texture.uv_set or "")
            if transformed:
                mapping = self.tree.nodes.new("ShaderNodeMapping")
                mapping.vector_type = "POINT"
                mapping.inputs["Location"].default_value = (*texture.uv_translation, 0.0)
                mapping.inputs["Rotation"].default_value = (0.0, 0.0, np.radians(texture.uv_rotation))
                mapping.inputs["Scale"].default_value = (*texture.uv_scale, 1.0)
                self.tree.links.new(uv, mapping.inputs["Vector"])
                uv = mapping.outputs["Vector"]
            self.tree.links.new(uv, node.inputs["Vector"])
        return node

    def _uv_map(self, name: str):
        if name not in self.uv_maps:
            node = self.tree.nodes.new("ShaderNodeUVMap")
            node.uv_map = name  # "" = the mesh's active UV map
            self.uv_maps[name] = node.outputs["UV"]
        return self.uv_maps[name]

    def _multiply_add(self, socket, scale, bias, vector: bool):
        node = self.tree.nodes.new("ShaderNodeVectorMath" if vector else "ShaderNodeMath")
        node.operation = "MULTIPLY_ADD"
        self.tree.links.new(socket, node.inputs[0])
        node.inputs[1].default_value = tuple(float(v) for v in scale) if vector else scale
        node.inputs[2].default_value = tuple(float(v) for v in bias) if vector else bias
        return node.outputs[0]

    # images

    def _image(self, texture: TextureRef, color_space: str):
        content = texture.content
        asset = texture.file
        key = (id(content) if content is not None else (asset.resolved or asset.path if asset else None), color_space)
        if key in self.images:
            return self.images[key]
        source = asset.path.replace("\\", "/") if asset is not None else "texture"  # FBX may hold Windows paths
        name = source.rsplit("/", 1)[-1] or "texture"
        if content is not None:  # embedded in the source file: packed into the .blend
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                path.write_bytes(content)
                image = bpy.data.images.load(str(path))
                image.pack()
            image.filepath = "//" + source.lstrip("/")  # where unpacking would put it
        else:
            path = (asset.resolved or asset.path) if asset is not None else ""
            try:
                image = bpy.data.images.load(path, check_existing=False)
                if self.pack_images:  # packed while its path is still absolute; a missing file isn't
                    image.pack()
            except RuntimeError:
                warnings.warn(f"texture {path!r} not found; the material references it anyway")
                image = bpy.data.images.new(name, 1, 1)
                image.source, image.filepath = "FILE", path
            if self.blend_dir is not None and os.path.isabs(image.filepath):
                try:
                    image.filepath = "//" + os.path.relpath(image.filepath, self.blend_dir).replace(os.sep, "/")
                except ValueError:  # another drive (Windows): keep it absolute
                    pass
        image.colorspace_settings.name = color_space
        self.images[key] = image
        return image


def assign(obj, node_materials: list, writer: MaterialWriter, first_slots: dict) -> None:
    """Fill `obj`'s material slots: on the mesh data for its first user, per object where a later user of
    the same mesh has other materials (Blender's per-object slots)."""
    mats = [writer.material(m) for m in node_materials]
    mesh = obj.data
    if id(mesh) not in first_slots:
        for mat in mats:
            mesh.materials.append(mat)
        first_slots[id(mesh)] = mats
        return
    if mats == first_slots[id(mesh)]:
        return
    if len(mats) > len(obj.material_slots):
        warnings.warn(f"object {obj.name!r}: more material slots than its shared mesh has; the extra ones are dropped")
    for slot, mat in zip(obj.material_slots, mats):
        slot.link = "OBJECT"
        slot.material = mat
