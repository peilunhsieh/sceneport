import pytest

from fetch_samples import SAMPLES, fetch, sample_path


def pytest_addoption(parser):
    parser.addoption(
        "--fetch-samples",
        action="store_true",
        help="download missing sample files (see tests/fetch_samples.py) before running the tests",
    )
    parser.addoption(
        "--blender",
        metavar="PATH",
        help="the Blender (executable or .app) for the Blender tests; default: $SCENEPORT_BLENDER or the newest "
        "Blender 5.2+ installed. Blender tests are skipped when none is found",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--fetch-samples"):
        for name in SAMPLES:
            fetch(name)


@pytest.fixture
def sample():
    """`sample(name)` -> path of a downloaded sample file; skips the test when it hasn't been fetched."""

    def get(name: str):
        if name not in SAMPLES:
            raise KeyError(f"unknown sample {name!r}; add it to tests/fetch_samples.py")
        path = sample_path(name)
        if not path.exists():
            pytest.skip(f"sample {name!r} not downloaded; run pytest with --fetch-samples, "
                        f"or: uv run python tests/fetch_samples.py {name}")
        return path

    return get


@pytest.fixture(scope="session")
def blender(request):
    """The Blender to run Blender tests with; skips the test when none is found."""
    from sceneport.interop.blend import BlenderError, find_blender

    try:
        return find_blender(request.config.getoption("--blender"))
    except BlenderError as e:
        pytest.skip(f"no Blender: {e}")
