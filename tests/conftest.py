import pytest

from fetch_samples import SAMPLES, fetch, sample_path


def pytest_addoption(parser):
    parser.addoption(
        "--fetch-samples",
        action="store_true",
        help="download missing sample files (see tests/fetch_samples.py) before running the tests",
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
