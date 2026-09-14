"""CLI smoke tests for the one-command Kaggle stage runner."""

from __future__ import annotations

import subprocess
import sys


def test_gold_stage_help_is_dependency_light() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_gold_stage.py", "--help"],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--target-epoch" in completed.stdout
