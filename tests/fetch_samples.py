"""Download the sample files some tests use; they are not stored in the repository.

    uv run python tests/fetch_samples.py            # every sample
    uv run python tests/fetch_samples.py NAME ...   # some of them
    uv run python tests/fetch_samples.py --list     # what's available, with licenses

Files go to tests/samples/ (gitignored); a sample that's already there isn't
downloaded again (delete it to refresh). Zip archives are unpacked and then
deleted. Tests that need a sample skip until it has been downloaded (see
conftest.py); `pytest --fetch-samples` downloads the missing ones first.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent / "samples"


@dataclass(frozen=True)
class Sample:
    url: str
    file: str  # name under tests/samples/
    license: str
    note: str = ""
    unpacks_to: str | None = None  # for a zip: the folder it unpacks to (what tests get); the zip is deleted


SAMPLES = {
    "spring_runcycle": Sample(
        url="https://download.blender.org/institute/sybren/usd/spring.runcycle.usdc",
        file="spring.runcycle.usdc",
        license="CC BY 4.0 (Blender Studio, 'Spring' open movie); not stated at the download location",
        note="Blender 2.80 USD export (55 MB): 165 meshes, 148 with time-sampled points, 211 animated "
        "transforms, 21 UsdPreviewSurface materials, metersPerUnit stored as a float",
    ),
    "usdskel_examples": Sample(
        url="https://openusd.org/files/UsdSkelExamples.zip",
        file="UsdSkelExamples.zip",
        unpacks_to="UsdSkelExamples",
        license="Pixar USD Asset EULA (HumanFemale/LICENSE.txt): personal, non-commercial testing of USD only; "
        "no redistribution of the asset or anything derived from it",
        note="Pixar's UsdSkel example (38 MB zip): HumanFemale with skeleton, blend shapes at three "
        "levels of detail (payloads), walk and keepAlive animations, 2k textures",
    ),
    "blender_institute_hdri": Sample(
        url="https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/blender_institute_1k.exr",
        file="blender_institute_1k.exr",
        license="CC0 (Poly Haven, polyhaven.com/license): any use, no attribution required",
        note="Poly Haven's 'Blender Institute' HDRI, 1k lat-long OpenEXR: an environment (dome light) texture",
    ),
}


def sample_path(name: str) -> Path:
    """What tests use: the downloaded file, or the folder a zip unpacks to."""
    sample = SAMPLES[name]
    return SAMPLES_DIR / (sample.unpacks_to or sample.file)


def fetch(name: str) -> Path:
    """Download sample `name` unless it's already there; unpack (and delete) a zip."""
    sample, path = SAMPLES[name], sample_path(name)
    if path.exists():
        print(f"{name}: already there ({path})")
        return path
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    download = SAMPLES_DIR / sample.file
    if not download.exists():  # a zip left by an interrupted run is reused
        _download(sample.url, download)
    if sample.unpacks_to:
        _unpack(name, download)
        download.unlink()
    print(f"{name}: {path}")
    return path


def _download(url: str, path: Path) -> None:
    """Download to a ".part" file, renamed only when complete."""
    partial = path.with_name(path.name + ".part")
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "sceneport-fetch-samples"})  # some hosts refuse Python's default
    with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    partial.replace(path)


def _unpack(name: str, archive: Path) -> None:
    """Unpack into a temporary folder first, so an interrupted run leaves nothing half-done."""
    sample = SAMPLES[name]
    with tempfile.TemporaryDirectory(dir=SAMPLES_DIR) as tmp:
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(tmp)  # zipfile drops absolute and ".." parts of member names
        unpacked = Path(tmp) / sample.unpacks_to
        if not unpacked.is_dir():
            raise RuntimeError(f"{name}: {sample.file} has no {sample.unpacks_to}/ folder")
        unpacked.replace(sample_path(name))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help=f"samples to fetch (default: all): {', '.join(SAMPLES)}")
    parser.add_argument("--list", action="store_true", help="list the samples and their licenses")
    args = parser.parse_args(argv)
    if args.list:
        for name, sample in SAMPLES.items():
            state = "downloaded" if sample_path(name).exists() else "not downloaded"
            print(f"{name} ({state})\n  {sample.url}\n  license: {sample.license}\n  {sample.note}")
        return 0
    unknown = [n for n in args.names if n not in SAMPLES]
    if unknown:
        parser.error(f"unknown samples {unknown}; available: {', '.join(SAMPLES)}")
    for name in args.names or SAMPLES:
        fetch(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
