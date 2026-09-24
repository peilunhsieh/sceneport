"""Command line entry point, e.g. `sceneport dump file.fbx`."""

from __future__ import annotations

import argparse
import json
import sys

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

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except BrokenPipeError:  # e.g. `sceneport dump x.fbx | head`
        sys.stderr.close()
    except (FBXParseError, OSError) as e:
        sys.exit(f"sceneport: {e}")


if __name__ == "__main__":
    main()
