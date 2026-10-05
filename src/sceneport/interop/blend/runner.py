"""Find an installed Blender and run bridge actions in it, as a background process.

Nothing is installed into Blender: each run puts a temporary folder on Blender's
`sys.path` holding only a link to this sceneport package, so Blender's own Python,
numpy and pxr are used and no other host packages leak in (a venv's site-packages
would bring numpy built for another Python). Requests and results cross as pickles
in a private job folder: both sides run the same sceneport source.
"""

from __future__ import annotations

import builtins
import functools
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_VERSION = (5, 2)
ENV_VAR = "SCENEPORT_BLENDER"
_PACKAGE = Path(__file__).resolve().parents[2]  # .../sceneport
_BOOT = (
    "import sys; sys.path.insert(0, sys.argv[-2]); "
    "from sceneport.interop.blend.bridge import main; main(sys.argv[-1])"
)


class BlenderError(RuntimeError):
    """Blender couldn't be found, or a run inside Blender failed."""


@dataclass(frozen=True)
class Blender:
    path: Path  # the executable
    version: tuple[int, int, int]

    def __str__(self) -> str:
        return f"Blender {'.'.join(map(str, self.version))} ({self.path})"


def find_blender(path: str | os.PathLike | None = None) -> Blender:
    """The Blender to run: `path` (an executable or a macOS .app), else $SCENEPORT_BLENDER, else
    `blender` on PATH, else the newest Blender >= 5.2 in the usual install locations."""
    explicit = path or os.environ.get(ENV_VAR)
    if explicit:
        blender = _probe(_executable(Path(explicit).expanduser()))
        if blender is None:
            raise BlenderError(f"{explicit} isn't a runnable Blender")
        if blender.version[:2] < MIN_VERSION:
            raise BlenderError(f"{blender} is too old: sceneport needs Blender {_dotted(MIN_VERSION)} or newer")
        return blender
    candidates = [b for b in map(_probe, _candidates()) if b is not None]
    usable = [b for b in candidates if b.version[:2] >= MIN_VERSION]
    if not usable:
        found = "; found " + ", ".join(map(str, candidates)) if candidates else ""
        raise BlenderError(
            f"no Blender {_dotted(MIN_VERSION)} or newer found{found}. Pass blender=<path>, or set "
            f"${ENV_VAR} to the Blender executable (or .app on macOS)"
        )
    return max(usable, key=lambda b: b.version)


def run_in_blender(
    action: str,
    payload: Any = None,
    *,
    blender: Blender | str | os.PathLike | None = None,
    timeout: float | None = None,
) -> Any:
    """Run a bridge action (see bridge.ACTIONS) in a background Blender and return its result.
    Warnings raised inside Blender are re-raised here; errors raise BlenderError with Blender's traceback."""
    if not isinstance(blender, Blender):
        blender = find_blender(blender)
    with tempfile.TemporaryDirectory(prefix="sceneport-blender-") as tmp:
        tmp = Path(tmp)
        path_dir, job = tmp / "path", tmp / "job"
        path_dir.mkdir()
        job.mkdir()
        _link_package(path_dir / "sceneport")
        with open(job / "request.pkl", "wb") as f:
            pickle.dump({"action": action, "payload": payload}, f, protocol=pickle.HIGHEST_PROTOCOL)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        command = [str(blender.path), "-b", "--factory-startup", "--python-exit-code", "1",
                   "--python-expr", _BOOT, "--", str(path_dir), str(job)]  # fmt: skip
        try:
            run = subprocess.run(command, capture_output=True, text=True, errors="replace", env=env, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise BlenderError(f"{blender} didn't finish {action!r} within {timeout} s") from e
        response_path = job / "response.pkl"
        if not response_path.exists():
            raise BlenderError(
                f"{blender} exited with code {run.returncode} without a result for {action!r}:\n{_tail(run)}"
            )
        with open(response_path, "rb") as f:
            response = pickle.load(f)
    for category, message in response["warnings"]:
        warnings.warn(message, _warning_category(category), stacklevel=2)
    if response["error"] is not None:
        raise BlenderError(f"{action!r} failed inside {blender}:\n{response['error']}")
    return response["result"]


def _link_package(link: Path) -> None:
    """`link` -> this sceneport package: a symlink, or a copy where symlinks aren't allowed (Windows)."""
    try:
        link.symlink_to(_PACKAGE, target_is_directory=True)
    except OSError:
        shutil.copytree(_PACKAGE, link, ignore=shutil.ignore_patterns("__pycache__"))


def _executable(path: Path) -> Path:
    if path.suffix == ".app":
        return path / "Contents" / "MacOS" / "Blender"
    return path


def _candidates() -> list[Path]:
    paths = []
    if on_path := shutil.which("blender"):
        paths.append(Path(on_path))
    if sys.platform == "darwin":
        for root in (Path("/Applications"), Path.home() / "Applications"):
            paths += [_executable(app) for app in sorted(root.glob("Blender*.app"))]
    elif sys.platform == "win32":
        for root in {os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramW6432", "")} - {""}:
            paths += sorted(Path(root, "Blender Foundation").glob("Blender*/blender.exe"))
    else:
        paths += [Path("/usr/bin/blender"), Path("/usr/local/bin/blender"), Path("/snap/bin/blender")]
        for root in (Path("/opt"), Path.home()):
            paths += sorted(root.glob("blender*/blender"))
    return list(dict.fromkeys(p for p in paths if p.is_file()))


@functools.lru_cache(maxsize=None)
def _probe(path: Path) -> Blender | None:
    """The Blender at `path` with its version, or None if it doesn't run as one."""
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"^Blender (\d+)\.(\d+)(?:\.(\d+))?", out.stdout, re.MULTILINE)
    if match is None:
        return None
    return Blender(path, (int(match[1]), int(match[2]), int(match[3] or 0)))


def _warning_category(name: str) -> type[Warning]:
    category = getattr(builtins, name, None)
    return category if isinstance(category, type) and issubclass(category, Warning) else UserWarning


def _tail(run: subprocess.CompletedProcess, lines: int = 40) -> str:
    return "\n".join((run.stdout + run.stderr).splitlines()[-lines:])


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(map(str, version))
