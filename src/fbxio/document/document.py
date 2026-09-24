"""FBXDocument: header, GlobalSettings, Definitions, Objects and Connections, plus object ID allocation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterator

from .. import __version__
from ..binary.node import DEFAULT_FOOTER_ID, FBXFile, FBXNode, Prop
from .properties import Properties70
from .templates import default_global_settings, default_template

NAME_SEP = "\x00\x01"  # between object name and class in "Cube\x00\x01Model"
ROOT_ID = 0  # connecting to 0 means "child of the scene root"

# FileId, CreationTime and the footer id are checked against each other by the FBX SDK;
# this is the known-good triple Blender writes.
FILE_ID = b"(\xb3*\xeb\xb6$\xcc\xc2\xbf\xc8\xb0*\xa9+\xfc\xf1"
CREATION_TIME = "1970-01-01 10:00:00:000"
_DOCUMENT_ID = 1
_FIRST_OBJECT_ID = 1_000_000


@dataclass
class FBXObject:
    """An entry under Objects, e.g. `Model: 123, "Cube\\x00\\x01Model", "Mesh"`."""

    kind: str  # node name: "Model", "Geometry", "Deformer", ...
    id: int
    name: str
    subtype: str = ""  # third value, e.g. "Mesh", "LimbNode", "BlendShape"
    cls: str | None = None  # class after the name separator; None means same as `kind`
    props: Properties70 = field(default_factory=Properties70)
    children: list[FBXNode] = field(default_factory=list)  # other child nodes, e.g. Vertices
    props_index: int = 0  # where Properties70 sits among `children` when written

    def child(self, name: str) -> FBXNode | None:
        return next((c for c in self.children if c.name == name), None)

    def child_value(self, name: str, default: Any = None) -> Any:
        """First value of child node `name`, e.g. the array in `Vertices`."""
        node = self.child(name)
        return default if node is None or not node.props else node.props[0].value

    @classmethod
    def from_node(cls, node: FBXNode) -> FBXObject:
        values = node.values
        id_ = values[0]
        name, sep, class_ = values[1].partition(NAME_SEP) if len(values) > 1 else ("", "", "")
        subtype = values[2] if len(values) > 2 else ""
        children, props, props_index = [], Properties70(), 0
        for child in node.children:
            if child.name == "Properties70":
                props, props_index = Properties70.from_node(child), len(children)
            else:
                children.append(child)
        return cls(node.name, id_, name, subtype, class_ if sep else None, props, children, props_index)

    def to_node(self) -> FBXNode:
        full_name = f"{self.name}{NAME_SEP}{self.cls or self.kind}"
        children = list(self.children)
        children.insert(min(self.props_index, len(children)), self.props.to_node())
        return FBXNode(self.kind, [Prop("L", self.id), Prop("S", full_name), Prop("S", self.subtype)], children)


@dataclass
class Connection:
    """`C: "OO", src, dst` (object to object) or `C: "OP", src, dst, "prop"` (object to property).

    `src` is the child and `dst` the parent: a Geometry connects to its Model,
    a root-level Model connects to ROOT_ID.
    """

    kind: str
    src: int
    dst: int
    prop: str | None = None

    @classmethod
    def from_node(cls, node: FBXNode) -> Connection:
        kind, src, dst, *rest = node.values
        return cls(kind, src, dst, rest[0] if rest else None)

    def to_node(self) -> FBXNode:
        props = [Prop("S", self.kind), Prop("L", self.src), Prop("L", self.dst)]
        if self.prop is not None:
            props.append(Prop("S", self.prop))
        return FBXNode("C", props)


@dataclass
class FBXDocument:
    version: int = 7400
    creator: str = f"fbxio {__version__}"
    global_settings: Properties70 = field(default_factory=default_global_settings)
    objects: dict[int, FBXObject] = field(default_factory=dict)
    connections: list[Connection] = field(default_factory=list)
    # Object kind -> (template name, properties) from Definitions. Missing kinds use built-in defaults.
    templates: dict[str, tuple[str, Properties70]] = field(default_factory=dict)
    # Written into CreationTimeStamp; None means the time of writing.
    timestamp: dt.datetime | None = None

    # -- building -----------------------------------------------------------

    def new_id(self) -> int:
        """Next free object id. Sequential, so writing the same scene twice gives the same file."""
        return max(_FIRST_OBJECT_ID, max(self.objects, default=0) + 1)

    def add_object(
        self,
        kind: str,
        name: str,
        subtype: str = "",
        *,
        props: Properties70 | None = None,
        children: list[FBXNode] | None = None,
        id: int | None = None,
    ) -> FBXObject:
        id_ = self.new_id() if id is None else id
        if id_ in self.objects or id_ == ROOT_ID:
            raise ValueError(f"object id {id_} already in use")
        obj = FBXObject(kind, id_, name, subtype, props=props or Properties70(), children=children or [])
        self.objects[id_] = obj
        return obj

    def connect(self, src: FBXObject | int, dst: FBXObject | int, prop: str | None = None) -> Connection:
        """Make `src` a child of `dst` (or of its property `prop`)."""
        src_id = src if isinstance(src, int) else src.id
        dst_id = dst if isinstance(dst, int) else dst.id
        conn = Connection("OO" if prop is None else "OP", src_id, dst_id, prop)
        self.connections.append(conn)
        return conn

    # -- querying -----------------------------------------------------------

    def iter_objects(self, kind: str | None = None, subtype: str | None = None) -> Iterator[FBXObject]:
        for obj in self.objects.values():
            if (kind is None or obj.kind == kind) and (subtype is None or obj.subtype == subtype):
                yield obj

    def children_of(self, obj: FBXObject | int, kind: str | None = None) -> list[FBXObject]:
        """Objects connected to `obj` as children, in connection order. Use ROOT_ID for root objects."""
        id_ = obj if isinstance(obj, int) else obj.id
        found = (self.objects.get(c.src) for c in self.connections if c.dst == id_)
        return [o for o in found if o is not None and (kind is None or o.kind == kind)]

    def parents_of(self, obj: FBXObject | int, kind: str | None = None) -> list[FBXObject]:
        """Objects `obj` is connected to. The scene root (ROOT_ID) is not an object and is left out."""
        id_ = obj if isinstance(obj, int) else obj.id
        found = (self.objects.get(c.dst) for c in self.connections if c.src == id_)
        return [o for o in found if o is not None and (kind is None or o.kind == kind)]

    def template(self, kind: str) -> Properties70:
        if kind in self.templates:
            return self.templates[kind][1]
        default = default_template(kind)
        return default[1] if default else Properties70()

    def get_property(self, obj: FBXObject, name: str, default: Any = None) -> Any:
        """Property value from the object, falling back to its kind's template, then `default`."""
        if name in obj.props:
            return obj.props.get(name)
        return self.template(obj.kind).get(name, default)

    # -- reading ------------------------------------------------------------

    @classmethod
    def from_file(cls, fbx: FBXFile) -> FBXDocument:
        doc = cls(version=fbx.version)
        if (creator := fbx.find("Creator")) is not None:
            doc.creator = creator.props[0].value
        if (settings := fbx.find("GlobalSettings")) is not None:
            doc.global_settings = Properties70.from_node(settings.find("Properties70"))
        if (definitions := fbx.find("Definitions")) is not None:
            for obj_type in definitions.iter("ObjectType"):
                if (tmpl := obj_type.find("PropertyTemplate")) is not None:
                    props = Properties70.from_node(tmpl.find("Properties70"))
                    doc.templates[obj_type.props[0].value] = (tmpl.props[0].value, props)
        if (objects := fbx.find("Objects")) is not None:
            for node in objects.children:
                obj = FBXObject.from_node(node)
                doc.objects[obj.id] = obj
        if (connections := fbx.find("Connections")) is not None:
            doc.connections = [Connection.from_node(c) for c in connections.iter("C")]
        return doc

    # -- writing ------------------------------------------------------------

    def to_file(self) -> FBXFile:
        nodes = [
            self._header_extension(),
            FBXNode("FileId", [Prop("R", FILE_ID)]),
            FBXNode("CreationTime", [Prop("S", CREATION_TIME)]),
            FBXNode("Creator", [Prop("S", self.creator)]),
            FBXNode("GlobalSettings", children=[_int("Version", 1000), self.global_settings.to_node()]),
            self._documents(),
            FBXNode("References"),
            self._definitions(),
            FBXNode("Objects", children=[o.to_node() for o in self.objects.values()]),
            FBXNode("Connections", children=[c.to_node() for c in self.connections]),
            FBXNode("Takes", children=[FBXNode("Current", [Prop("S", "")])]),
        ]
        return FBXFile(version=self.version, nodes=nodes, footer_id=DEFAULT_FOOTER_ID)

    def _header_extension(self) -> FBXNode:
        t = self.timestamp or dt.datetime.now()
        stamp = FBXNode("CreationTimeStamp", children=[
            _int("Version", 1000), _int("Year", t.year), _int("Month", t.month), _int("Day", t.day),
            _int("Hour", t.hour), _int("Minute", t.minute), _int("Second", t.second),
            _int("Millisecond", t.microsecond // 1000),
        ])  # fmt: skip
        app = "fbxio"
        info = Properties70()
        info.set("DocumentUrl", "KString", "/foobar.fbx", subtype="Url")
        info.set("SrcDocumentUrl", "KString", "/foobar.fbx", subtype="Url")
        for prefix in ("Original", "LastSaved"):
            info.set(prefix, "Compound")
            info.set(f"{prefix}|ApplicationVendor", "KString", app)
            info.set(f"{prefix}|ApplicationName", "KString", app)
            info.set(f"{prefix}|ApplicationVersion", "KString", __version__)
            info.set(f"{prefix}|DateTime_GMT", "DateTime", "01/01/1970 00:00:00.000")
        metadata = FBXNode("MetaData", children=[_int("Version", 100)] + [
            FBXNode(k, [Prop("S", "")]) for k in ("Title", "Subject", "Author", "Keywords", "Revision", "Comment")
        ])  # fmt: skip
        scene_info = FBXNode(
            "SceneInfo",
            [Prop("S", f"GlobalInfo{NAME_SEP}SceneInfo"), Prop("S", "UserData")],
            [FBXNode("Type", [Prop("S", "UserData")]), _int("Version", 100), metadata, info.to_node()],
        )
        return FBXNode("FBXHeaderExtension", children=[
            _int("FBXHeaderVersion", 1003), _int("FBXVersion", self.version), _int("EncryptionType", 0),
            stamp, FBXNode("Creator", [Prop("S", self.creator)]), scene_info,
        ])  # fmt: skip

    def _documents(self) -> FBXNode:
        props = Properties70()
        props.set("SourceObject", "object")
        props.set("ActiveAnimStackName", "KString", "")
        document = FBXNode(
            "Document",
            [Prop("L", _DOCUMENT_ID), Prop("S", "Scene"), Prop("S", "Scene")],
            [props.to_node(), FBXNode("RootNode", [Prop("L", ROOT_ID)])],
        )
        return FBXNode("Documents", children=[_int("Count", 1), document])

    def _definitions(self) -> FBXNode:
        counts: dict[str, int] = {"GlobalSettings": 1}
        for obj in self.objects.values():
            counts[obj.kind] = counts.get(obj.kind, 0) + 1
        types = []
        for kind, count in counts.items():
            node = FBXNode("ObjectType", [Prop("S", kind)], [_int("Count", count)])
            tmpl = self.templates.get(kind) or default_template(kind)
            if tmpl is not None:
                name, props = tmpl
                node.children.append(FBXNode("PropertyTemplate", [Prop("S", name)], [props.to_node()]))
            types.append(node)
        head = [_int("Version", 100), _int("Count", sum(counts.values()))]
        return FBXNode("Definitions", children=head + types)


def _int(name: str, value: int) -> FBXNode:
    return FBXNode(name, [Prop("I", value)])
