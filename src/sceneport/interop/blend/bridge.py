"""The sceneport side inside Blender: runs one action per background Blender process.

`main(job_dir)` is called by runner.run_in_blender's boot code. It reads
`request.pkl` ({"action", "payload"}), runs ACTIONS[action](payload) and always
writes `response.pkl` ({"result", "warnings", "error"}): warnings are recorded as
(category name, message) and an exception as its traceback, so the host reports
them as if they were raised locally. This is the only sceneport module that
imports bpy.
"""

from __future__ import annotations

import pickle
import sys
import traceback
import warnings
from pathlib import Path


def ping(payload: dict | None) -> dict:
    """Blender's and its Python's versions; a diagnostic. `payload` may hold "warn" (a message to
    warn with) and "fail" (a message to raise), to exercise the error and warning paths."""
    import bpy
    import numpy

    payload = payload or {}
    if "warn" in payload:
        warnings.warn(payload["warn"])
    if "fail" in payload:
        raise RuntimeError(payload["fail"])
    return {
        "blender": tuple(bpy.app.version),
        "python": tuple(sys.version_info[:3]),
        "numpy": numpy.__version__,
        "sys_path": list(sys.path),
    }


ACTIONS = {"ping": ping}


def main(job_dir: str) -> None:
    job = Path(job_dir)
    response = {"result": None, "warnings": [], "error": None}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with open(job / "request.pkl", "rb") as f:
                request = pickle.load(f)
            action = ACTIONS.get(request["action"])
            if action is None:
                raise ValueError(f"unknown bridge action {request['action']!r}; known: {', '.join(ACTIONS)}")
            response["result"] = action(request["payload"])
        except Exception:
            response["error"] = traceback.format_exc()
    response["warnings"] = [(w.category.__name__, str(w.message)) for w in caught]
    with open(job / "response.pkl", "wb") as f:
        pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)
