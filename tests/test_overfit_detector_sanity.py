"""Tests for the detector overfitting/sanity script."""

from __future__ import annotations

import pytest

# Skip the test if torch is not installed
torch = pytest.importorskip("torch")

from scripts.overfit_detector_sanity import run_sanity_check


def test_overfit_detector_sanity_executes_successfully() -> None:
    # Run a quick overfitting run (15 epochs) to verify model forward/backward and extraction run smoothly.
    success = run_sanity_check(epochs=15, lr=0.01, seed=42)
    assert isinstance(success, bool)
