"""Animation: FBX animation stacks <-> Scene.animations.

FBX layout (as the FBX SDK and Blender write it):

    AnimationStack (one clip)  <-  AnimationLayer
        <-  AnimationCurveNode "T" / "R" / "S" / "DeformPercent"  --OP-->  Model "Lcl Translation" /
                                                                          "Lcl Rotation" / "Lcl Scaling",
                                                                          or BlendShapeChannel "DeformPercent"
                <-  AnimationCurve  (OP "d|X" / "d|Y" / "d|Z" / "d|DeformPercent")
                        KeyTime (int64 ticks), KeyValueFloat (float32),
                        KeyAttrFlags / KeyAttrDataFloat / KeyAttrRefCount (interpolation and
                        tangents, run-length encoded: flags[i] covers refcount[i] keys)

Rotation curves are Euler degrees in the node's rotation order; blend shape
curves are percent. Quaternion ("orientation") channels have no FBX form and
are converted to Euler rotation keys at every frame (and every key).
"""

from __future__ import annotations

import warnings

import numpy as np

from ...math import TICKS_PER_SECOND, euler_from_matrix, quaternion_matrix
from ...scene.animation import Animation, Channel
from ...scene.types import Mesh, Node, Scene
from .binary.node import FBXNode, Prop
from . import camera as camera_fbx, light as light_fbx
from .document import FBXDocument, FBXObject

# FBX TimeMode enum (FbxTime::EMode) -> frames per second; 14 = custom (CustomFrameRate).
TIME_MODES = {
    0: 30.0, 1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97,
    10: 25.0, 11: 24.0, 12: 1000.0, 13: 23.976, 15: 96.0, 16: 72.0, 17: 59.94, 18: 119.88,
}  # fmt: skip
PREFERRED_TIME_MODES = {30.0: 6, 24.0: 11, 60.0: 3, 25.0: 10, 50.0: 4, 120.0: 1, 48.0: 5, 100.0: 2, 96.0: 15, 72.0: 16}
CUSTOM_TIME_MODE = 14

NODE_PROPERTIES = {"Lcl Translation": "translation", "Lcl Rotation": "rotation", "Lcl Scaling": "scale"}
CURVE_NODE_NAMES = {"translation": "T", "rotation": "R", "scale": "S"}
COMPONENTS = {"d|X": 0, "d|Y": 1, "d|Z": 2, "d|DeformPercent": 0}

# KeyAttrFlags interpolation bits, and what we write for each mode.
_CONSTANT, _LINEAR, _CUBIC = 0x02, 0x04, 0x08
_TANGENT_AUTO = 0x100
WRITE_FLAGS = {"constant": _CONSTANT, "linear": _LINEAR, "cubic": _CUBIC | _TANGENT_AUTO}
# Default KeyAttrDataFloat: slopes 0, packed tangent weights 0.3333 / 0.3333 (0x0D05, 0x0D05), velocity 0.
DEFAULT_KEY_DATA = np.array([0, 0, 0x0D050D05, 0], dtype=np.int32).view(np.float32)


def fps_from_settings(settings) -> float:
    mode = settings.get("TimeMode", 0)
    if mode == CUSTOM_TIME_MODE:
        return float(settings.get("CustomFrameRate", 30.0))
    return TIME_MODES.get(mode, 30.0)


def time_mode_for(fps: float) -> tuple[int, float]:
    """(TimeMode, CustomFrameRate) for `fps`."""
    for rate, mode in PREFERRED_TIME_MODES.items():
        if abs(rate - fps) < 1e-6:
            return mode, -1.0
    return CUSTOM_TIME_MODE, float(fps)


def _ticks_to_seconds(ticks) -> np.ndarray:
    return np.asarray(ticks, dtype=np.float64) / TICKS_PER_SECOND


def _seconds_to_ticks(seconds) -> np.ndarray:
    return np.round(np.asarray(seconds, dtype=np.float64) * TICKS_PER_SECOND).astype(np.int64)


# -- reading --------------------------------------------------------------------


def read_animations(
    doc: FBXDocument,
    nodes: dict[int, Node],
    shapes: dict[int, tuple[Mesh, str]],
    fps: float,
    attributes: dict[int, tuple[Node, str]] | None = None,
) -> list[Animation]:
    """One Animation per non-empty stack. `shapes` maps BlendShapeChannel ids to (mesh, shape name);
    `attributes` maps camera / light NodeAttribute ids to (node, "camera" | "light")."""
    attributes = attributes or {}
    clips = []
    for stack in doc.iter_objects("AnimationStack"):
        layers = [layer for layer in doc.children_of(stack, "AnimationLayer") if doc.children_of(layer, "AnimationCurveNode")]
        if not layers:
            continue
        if len(layers) > 1:
            warnings.warn(f"animation {stack.name!r}: {len(layers)} layers; only the first is read")
        channels = []
        for curve_node in doc.children_of(layers[0], "AnimationCurveNode"):
            channels += _read_curve_node(doc, curve_node, nodes, shapes, attributes)
        channels = _spot_softness(channels)
        if not channels:
            continue
        start, stop = stack.props.get("LocalStart"), stack.props.get("LocalStop")
        clips.append(Animation(
            stack.name,
            channels,
            fps=fps,
            start=None if start is None else float(_ticks_to_seconds(start)),
            end=None if stop is None else float(_ticks_to_seconds(stop)),
            extras={"fbx": {"layer": layers[0].name}},
        ))  # fmt: skip
    return clips


def _read_curve_node(doc, curve_node: FBXObject, nodes, shapes, attributes) -> list[Channel]:
    targets = [c for c in doc.connections if c.src == curve_node.id and c.kind == "OP" and c.dst in doc.objects]
    if not targets:
        return []
    link = targets[0]
    components = COMPONENTS
    if link.prop in NODE_PROPERTIES and link.dst in nodes:
        target, prop, scale = nodes[link.dst], NODE_PROPERTIES[link.prop], 1.0
    elif link.prop == "DeformPercent" and link.dst in shapes:
        (target, prop), scale = shapes[link.dst], 0.01
    elif link.dst in attributes and (found := _attribute_target(link.prop, *attributes[link.dst])) is not None:
        target, prop, scale = found
        components = {**COMPONENTS, f"d|{link.prop}": 0}
    else:
        return []  # other animated properties (visibility, materials, ...) aren't supported yet

    channels = []
    for connection in doc.connections:
        if connection.dst != curve_node.id or connection.kind != "OP" or connection.prop not in components:
            continue
        curve = doc.objects.get(connection.src)
        if curve is None or curve.kind != "AnimationCurve":
            continue
        ticks = curve.child_value("KeyTime")
        values = curve.child_value("KeyValueFloat")
        if ticks is None or values is None or len(ticks) == 0:
            continue
        flags = curve.child_value("KeyAttrFlags", np.array([_LINEAR], np.int32))
        refcount = curve.child_value("KeyAttrRefCount", np.array([len(ticks)], np.int32))
        channels.append(Channel(
            target,
            prop,
            components[connection.prop],
            _ticks_to_seconds(ticks),
            np.asarray(values, dtype=np.float64) * scale,
            _interpolation(flags, refcount, len(ticks)),
            extras={"fbx": {
                "flags": np.asarray(flags), "data": curve.child_value("KeyAttrDataFloat"),
                "refcount": np.asarray(refcount), "default": curve.child_value("Default"),
            }},
        ))  # fmt: skip
    return channels


_INNER_ANGLE = "light:_inner_angle"  # a spot's InnerAngle while reading; turned into cone_softness


def _attribute_target(prop: str, node: Node, kind: str) -> tuple[Node, str, float] | None:
    """(node, channel property, factor from FBX's value) for an animated camera / light NodeAttribute property."""
    if kind == "camera":
        found = camera_fbx.from_fbx_curve(prop)
        return None if found is None else (node, f"camera:{found[0]}", found[1])
    table = {
        "Intensity": ("light:intensity", 1.0 / (100.0 * light_fbx._watts_per_unit(node.light.type))),
        "Color": ("light:color", 1.0), "ShadowColor": ("light:shadow_color", 1.0),
        "OuterAngle": ("light:cone_angle", 0.5), "InnerAngle": (_INNER_ANGLE, 1.0),
    }  # fmt: skip
    return None if prop not in table else (node, *table[prop])


def _spot_softness(channels: list[Channel]) -> list[Channel]:
    """Turn read InnerAngle curves into cone_softness = 1 - inner / outer (outer from the cone_angle curve or rest)."""
    out = [c for c in channels if c.property != _INNER_ANGLE]
    for inner in (c for c in channels if c.property == _INNER_ANGLE):
        node = inner.target
        cone = next((c for c in channels if c.target is node and c.property == "light:cone_angle"), None)
        rest = node.light.cone_angle if node.light.cone_angle is not None else 45.0
        outer = 2.0 * (cone.evaluate(inner.times) if cone is not None else np.full(len(inner.times), rest))
        softness = 1.0 - inner.values / np.where(outer == 0, 1.0, outer)
        out.append(Channel(node, "light:cone_softness", 0, inner.times, softness, inner.interpolation))
    return out


def _interpolation(flags, refcount, count: int) -> np.ndarray:
    modes = []
    for flag, run in zip(flags, refcount):
        mode = "constant" if flag & _CONSTANT else "cubic" if flag & _CUBIC else "linear"
        modes += [mode] * int(run)
    modes = (modes + ["linear"] * count)[:count]
    return np.array(modes, dtype=object)


# -- writing --------------------------------------------------------------------


def write_animations(
    doc: FBXDocument,
    scene: Scene,
    models: dict[Node, FBXObject],
    shape_channels: dict[tuple[int, str], FBXObject],
    attributes: dict[Node, FBXObject] | None = None,
) -> None:
    """Stacks for scene.animations. `shape_channels` maps (id(mesh), shape name) to BlendShapeChannels;
    `attributes` maps camera / light nodes to their NodeAttributes (for animated camera / light settings)."""
    attributes = attributes or {}
    for clip in scene.animations:
        clip.validate()
        start, end = clip.time_range()
        stack = doc.add_object("AnimationStack", clip.name, cls="AnimStack")
        for name in ("LocalStart", "ReferenceStart"):
            stack.props.set(name, "KTime", int(_seconds_to_ticks(start)))
        for name in ("LocalStop", "ReferenceStop"):
            stack.props.set(name, "KTime", int(_seconds_to_ticks(end)))
        layer = doc.add_object("AnimationLayer", clip.extras.get("fbx", {}).get("layer", "BaseLayer"), cls="AnimLayer")
        doc.connect(layer, stack)

        groups: dict[tuple, list[Channel]] = {}
        for channel in _euler_channels(clip):
            if channel.attribute is None:
                groups.setdefault((id(channel.target), channel.property), []).append(channel)
        for channels in groups.values():
            _write_curve_node(doc, layer, channels, models, shape_channels)
        for node, kind in dict.fromkeys((c.target, c.attribute[0]) for c in clip.channels if c.attribute):
            _write_attribute_curves(doc, layer, clip, node, kind, attributes.get(node))


def _write_attribute_curves(doc, layer, clip: Animation, node: Node, kind: str, owner: FBXObject | None) -> None:
    """Curves on a camera / light NodeAttribute's properties, baked at every frame (FBX's properties combine
    several fields, e.g. Intensity from intensity and exposure)."""
    channels = [c for c in clip.channels if c.target is node and c.attribute and c.attribute[0] == kind]
    fields = {c.attribute[1] for c in channels}
    if owner is None:
        warnings.warn(f"{node.name!r}: its {kind} isn't written to FBX; its animation is skipped")
        return
    times = _ticks_to_seconds(np.unique(_seconds_to_ticks(np.concatenate([clip.frame_times(), *(c.times for c in channels)]))))
    values = clip.attribute_values(node, kind, times)
    if kind == "camera":
        props, handled = camera_fbx.fbx_values(fields, values), set(camera_fbx.ANIMATED_PROPERTIES)
    else:
        props, handled = light_fbx.fbx_values(node.light, fields, values), set(light_fbx.ANIMATED_PROPERTIES)
    if fields - handled:
        warnings.warn(f"{node.name!r}: FBX can't animate its {kind}'s {', '.join(sorted(fields - handled))}; not written")
    for prop, data in props.items():
        components = ["d|X", "d|Y", "d|Z"] if np.ndim(data) == 2 else [f"d|{prop}"]
        data = np.reshape(data, (len(times), len(components)))
        curve_node = doc.add_object("AnimationCurveNode", prop, cls="AnimCurveNode")
        for i, component in enumerate(components):
            curve_node.props.set(component, "Number", float(data[0, i]), flags="A")
        doc.connect(curve_node, layer)
        doc.connect(curve_node, owner, prop)
        for i, component in enumerate(components):
            curve = _curve_object(doc, Channel(node, f"{kind}:{prop}", i, times, data[:, i]), 1.0, float(data[0, i]))
            doc.connect(curve, curve_node, component)


def _euler_channels(clip: Animation) -> list[Channel]:
    """The clip's channels with quaternion "orientation" channels turned into Euler "rotation" keys."""
    out = [c for c in clip.channels if c.property != "orientation"]
    by_node: dict[Node, list[Channel]] = {}
    for channel in clip.channels:
        if channel.property == "orientation":
            by_node.setdefault(channel.target, []).append(channel)
    for node, channels in by_node.items():
        if any(c.target is node and c.property == "rotation" for c in out):
            warnings.warn(f"{node.name!r} has rotation and orientation channels; FBX keeps the orientation")
            out = [c for c in out if not (c.target is node and c.property == "rotation")]
        # Merged on whole ticks: the same frame computed two ways (e.g. 103 / 24) differs in the last float bit
        # and would otherwise become two keys on one tick, which Blender's importer misreads.
        times = _ticks_to_seconds(np.unique(_seconds_to_ticks(np.concatenate([clip.frame_times(), *(c.times for c in channels)]))))
        quaternions = clip.node_values(times)[node]["orientation"]
        euler = euler_from_matrix(quaternion_matrix(quaternions), node.transform.rotation_order)
        euler = np.degrees(np.unwrap(np.radians(euler), axis=0))  # no 360-degree jumps between keys
        out += [Channel(node, "rotation", axis, times, euler[:, axis]) for axis in range(3)]
    return out


def _write_curve_node(doc, layer, channels: list[Channel], models, shape_channels) -> None:
    target, prop = channels[0].target, channels[0].property
    if channels[0].is_blend_shape:
        owner = shape_channels.get((id(target), prop))
        if owner is None:
            warnings.warn(f"no blend shape {prop!r} on the animated mesh; its channel is skipped")
            return
        name, owner_prop, scale, components = "DeformPercent", "DeformPercent", 100.0, ["d|DeformPercent"]
        defaults = [next(s.weight for s in target.blend_shapes if s.name == prop) * 100.0]
    else:
        owner = models.get(target)
        if owner is None:
            warnings.warn(f"animated node {target.name!r} is not in the scene; its channels are skipped")
            return
        owner_prop = {v: k for k, v in NODE_PROPERTIES.items()}[prop]
        name, scale, components = CURVE_NODE_NAMES[prop], 1.0, ["d|X", "d|Y", "d|Z"]
        defaults = list(getattr(target.transform, prop))

    curve_node = doc.add_object("AnimationCurveNode", name, cls="AnimCurveNode")
    for component, default in zip(components, defaults):
        curve_node.props.set(component, "Number", float(default), flags="A")
    doc.connect(curve_node, layer)
    doc.connect(curve_node, owner, owner_prop)
    for channel in channels:
        curve = _curve_object(doc, channel, scale, defaults[channel.component])
        doc.connect(curve, curve_node, components[channel.component])


def _curve_object(doc, channel: Channel, scale: float, default: float) -> FBXObject:
    ticks = _seconds_to_ticks(channel.times)
    keep = np.concatenate([[True], np.diff(ticks) != 0])
    if not keep.all():  # keys closer than a tick: keep the first (FBX readers expect increasing times)
        channel = Channel(channel.target, channel.property, channel.component, channel.times[keep], channel.values[keep],
                          np.asarray(channel.interpolation)[keep])  # fmt: skip
        ticks = ticks[keep]
    kept = channel.extras.get("fbx", {})
    flags, data, refcount = kept.get("flags"), kept.get("data"), kept.get("refcount")
    keys = len(channel.times)
    same_modes = (
        flags is not None and refcount is not None and int(np.sum(refcount)) == keys
        and np.array_equal(_interpolation(flags, refcount, keys), channel.interpolation)
    )  # fmt: skip
    if not same_modes:  # run-length encode our interpolation modes
        runs = []
        for mode in channel.interpolation:
            if runs and runs[-1][0] == mode:
                runs[-1][1] += 1
            else:
                runs.append([mode, 1])
        flags = np.array([WRITE_FLAGS[mode] for mode, _ in runs], dtype=np.int32)
        refcount = np.array([count for _, count in runs], dtype=np.int32)
        data = np.tile(DEFAULT_KEY_DATA, len(runs))
    curve = doc.add_object("AnimationCurve", "", cls="AnimCurve", children=[
        FBXNode("Default", [Prop("D", float(default))]),
        FBXNode("KeyVer", [Prop("I", 4009)]),
        FBXNode("KeyTime", [Prop("l", ticks)]),
        FBXNode("KeyValueFloat", [Prop("f", (channel.values * scale).astype(np.float32))]),
        FBXNode("KeyAttrFlags", [Prop("i", np.asarray(flags, dtype=np.int32))]),
        FBXNode("KeyAttrDataFloat", [Prop("f", np.asarray(data, dtype=np.float32))]),
        FBXNode("KeyAttrRefCount", [Prop("i", np.asarray(refcount, dtype=np.int32))]),
    ])  # fmt: skip
    curve.props_index = -1
    return curve
