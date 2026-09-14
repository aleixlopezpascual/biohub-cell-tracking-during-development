"""Smoke test for hidden-scale runtime report generation."""

from __future__ import annotations

import json
import subprocess
import sys


def test_benchmark_runtime_writes_machine_readable_report(tmp_path) -> None:
    output = tmp_path / "runtime.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_inference_runtime.py",
            "--output",
            str(output),
            "--runtime-limit-hours",
            "0.01",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["returncode"] == 0
    assert payload["within_runtime_limit"]
