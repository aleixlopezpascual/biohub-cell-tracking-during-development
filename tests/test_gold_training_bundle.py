"""Tests for the offline Kaggle Gold-training bundle."""

from __future__ import annotations

import zipfile

from scripts.build_gold_training_bundle import build_bundle


def test_gold_training_bundle_is_complete_and_reproducible(tmp_path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    members = build_bundle(first)
    build_bundle(second)
    assert first.read_bytes() == second.read_bytes()
    assert "scripts/run_gold_stage.py" in members
    assert "scripts/run_oof_checkpoint.py" in members
    assert "scripts/kaggle_resume_preflight.py" in members
    assert "scripts/kaggle_threshold_recovery.py" in members
    assert "scripts/probe_detection_thresholds.py" in members
    assert "configs/gold_training.yaml" in members
    with zipfile.ZipFile(first) as archive:
        assert "src/biohub_tracking/training/config.py" in archive.namelist()
