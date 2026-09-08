#!/usr/bin/env python3
"""Offline bundle generator for Kaggle notebook submission.

Kaggle's code-competition notebooks run with no internet access, so the
package can't be ``pip install``-ed at submission time. This script packages
the pure-Python ``biohub_tracking`` source tree (no compiled extensions) into
a single zip archive that can be uploaded as a private Kaggle dataset and
loaded at runtime via ``sys.path.insert(0, "/kaggle/input/<dataset>/biohub_tracking_bundle.zip")``
(Python's import system supports importing packages directly out of a zip
file placed on ``sys.path``).

Usage::

    python bundle.py [--output dist/biohub_tracking_bundle.zip]

The bundle intentionally excludes ``__pycache__``, tests, and anything
depending on the optional ``zarr``/``torch`` extras at import time (those
imports are already deferred inside the package itself).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "src" / "biohub_tracking"


def iter_package_files(package_dir: Path) -> list[Path]:
    """Return every ``.py`` file in the package, skipping caches."""
    return sorted(
        p
        for p in package_dir.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def build_bundle(output_path: Path) -> list[str]:
    """Write the zip bundle and return the list of archive member names."""
    if not PACKAGE_DIR.is_dir():
        raise FileNotFoundError(f"package directory not found: {PACKAGE_DIR}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    members: list[str] = []
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in iter_package_files(PACKAGE_DIR):
            arcname = str(Path("biohub_tracking") / file_path.relative_to(PACKAGE_DIR))
            archive.write(file_path, arcname=arcname)
            members.append(arcname)
    return members


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "biohub_tracking_bundle.zip",
        help="Output zip archive path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    members = build_bundle(args.output)
    print(f"Wrote {len(members)} files to {args.output}")
    print('Load offline with: sys.path.insert(0, "<path-to>/biohub_tracking_bundle.zip")')


if __name__ == "__main__":
    main()
