"""Command line entry point, e.g. `sceneport dump file.fbx` or `sceneport convert in.fbx out.usd`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import load, save
from .interop.fbx.binary import FBXParseError, read
from .interop.fbx.binary.dump import format_tree, to_json


def _dump(args: argparse.Namespace) -> None:
    fbx = read(args.file)
    if args.json:
        json.dump(to_json(fbx, max_depth=args.depth), sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        max_items = None if args.full else args.max_items
        print(format_tree(fbx, max_items=max_items, max_depth=args.depth))


def _convert(args: argparse.Namespace) -> None:
    scene = load(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(scene, output)
    joints = sum(node.joint for node, _ in scene.walk())
    clips = ", ".join(f"{clip.name} ({len(clip.frame_times())} frames)" for clip in scene.animations) or "none"
    print(f"{args.input} -> {output}: {len(scene.meshes)} meshes, {joints} joints, animations: {clips}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sceneport", description="3D scene file tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump", help="print the node tree of a binary FBX file")
    dump.add_argument("file")
    dump.add_argument("--json", action="store_true", help="output JSON (arrays are never truncated)")
    dump.add_argument("-d", "--depth", type=int, default=None, help="max tree depth to print")
    dump.add_argument("-n", "--max-items", type=int, default=8, help="array items to show (default 8)")
    dump.add_argument("--full", action="store_true", help="show all array items")
    dump.set_defaults(func=_dump)

    convert = sub.add_parser("convert", help="convert between formats: .fbx, .usd, .usda, .usdc, .usdz, .obj, .ply")
    convert.add_argument("input")
    convert.add_argument("output", help="the format follows the extension")
    convert.set_defaults(func=_convert)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BrokenPipeError:  # e.g. `sceneport dump x.fbx | head`
        sys.stderr.close()
    except (FBXParseError, OSError, ValueError) as e:
        sys.exit(f"sceneport: {e}")


if __name__ == "__main__":
    main()
