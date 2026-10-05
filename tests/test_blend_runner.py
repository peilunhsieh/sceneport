"""Running Blender in the background: finding it, the bridge round trip, and errors / warnings reaching the host."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sceneport.interop.blend import Blender, BlenderError, find_blender, run_in_blender
from sceneport.interop.blend import runner


def test_ping(blender):
    info = run_in_blender("ping", blender=blender)
    assert info["blender"][:2] >= runner.MIN_VERSION and info["blender"] == blender.version
    assert tuple(info["python"][:2]) >= (3, 11)
    assert int(info["numpy"].split(".")[0]) >= 2  # Blender's own numpy


def test_only_sceneport_is_added_to_blenders_path(blender):
    """The venv's site-packages (numpy built for the host Python, ...) must not reach Blender."""
    info = run_in_blender("ping", blender=blender)
    host_site = {p for p in sys.path if p.endswith("site-packages")}
    assert not host_site & set(info["sys_path"])
    assert str(runner._PACKAGE.parent) not in info["sys_path"]  # the folder holding sceneport (src/ or site-packages)
    added = [p for p in info["sys_path"] if "sceneport-blender-" in p]
    assert len(added) == 1 and info["sys_path"][0] == added[0]  # the temporary folder with only the link


def test_warnings_and_errors_reach_the_host(blender):
    with pytest.warns(UserWarning, match="careful"):
        run_in_blender("ping", {"warn": "careful"}, blender=blender)
    with pytest.raises(BlenderError, match="(?s)RuntimeError: boom.*") as caught:
        run_in_blender("ping", {"fail": "boom"}, blender=blender)
    assert "bridge.py" in str(caught.value)  # Blender's traceback
    with pytest.raises(BlenderError, match="unknown bridge action 'nope'"):
        run_in_blender("nope", blender=blender)


def test_payload_and_results_cross_as_pickles(blender):
    info = run_in_blender("ping", {"array": np.arange(3)}, blender=blender)  # unused, but must load in Blender
    assert isinstance(info, dict)


def test_missing_blender(tmp_path, monkeypatch):
    with pytest.raises(BlenderError, match="isn't a runnable Blender"):
        find_blender(tmp_path / "nothing")
    monkeypatch.setenv(runner.ENV_VAR, str(tmp_path / "nothing"))
    with pytest.raises(BlenderError, match="isn't a runnable Blender"):
        find_blender()
    monkeypatch.delenv(runner.ENV_VAR)
    monkeypatch.setattr(runner, "_candidates", lambda: [])
    with pytest.raises(BlenderError, match=f"no Blender 5.2 or newer found.*{runner.ENV_VAR}"):
        find_blender()


def test_too_old_blender(monkeypatch, tmp_path):
    old = Blender(tmp_path / "blender", (4, 5, 3))
    monkeypatch.setattr(runner, "_probe", lambda path: old)
    with pytest.raises(BlenderError, match="too old"):
        find_blender(old.path)
    monkeypatch.setattr(runner, "_candidates", lambda: [old.path])
    with pytest.raises(BlenderError, match=r"found Blender 4\.5\.3"):
        find_blender()


def test_crash_without_result(blender, monkeypatch):
    monkeypatch.setattr(runner, "_BOOT", "import sys; sys.exit(3)")
    with pytest.raises(BlenderError, match="without a result"):
        run_in_blender("ping", blender=blender)
