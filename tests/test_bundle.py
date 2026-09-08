"""Tests for the offline Kaggle bundle generator."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bundle  # noqa: E402  (import after sys.path tweak, this is a top-level script not a package)


def test_build_bundle_creates_importable_zip(tmp_path) -> None:
    output = tmp_path / "biohub_tracking_bundle.zip"
    members = bundle.build_bundle(output)

    assert output.exists()
    assert any(m.endswith("__init__.py") for m in members)
    assert all(m.startswith("biohub_tracking/") for m in members)
    assert not any("__pycache__" in m for m in members)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "biohub_tracking/metrics/division_jaccard.py" in names
    assert "biohub_tracking/submission/export.py" in names


def test_bundled_package_is_importable_from_zip(tmp_path) -> None:
    output = tmp_path / "bundle.zip"
    bundle.build_bundle(output)

    sys.path.insert(0, str(output))
    try:
        import importlib

        # Use a fresh module name to avoid colliding with the editable install already imported.
        spec = importlib.util.find_spec("biohub_tracking")
        assert spec is not None
    finally:
        sys.path.remove(str(output))
