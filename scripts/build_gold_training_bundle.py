#!/usr/bin/env python3
"""Build a deterministic offline bundle for the Kaggle Gold-training runner."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ROOT / "src" / "biohub_tracking"
RUNNER_SCRIPTS = (
    "compare_local_evaluations.py",
    "continue_royerlab_training.py",
    "evaluate_training_gates.py",
    "kaggle_resume_preflight.py",
    "kaggle_threshold_recovery.py",
    "overfit_detector_sanity.py",
    "prepare_gold_training.py",
    "probe_detection_thresholds.py",
    "run_gold_stage.py",
    "run_oof_checkpoint.py",
)


def bundle_files() -> list[Path]:
    """Return the minimal repository files required by the offline runner."""
    files = [
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    files.extend(ROOT / "scripts" / name for name in RUNNER_SCRIPTS)
    files.append(ROOT / "configs" / "gold_training.yaml")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Gold-training bundle inputs are missing: {missing}")
    return sorted(files)


def build_bundle(output: Path) -> list[str]:
    """Write a byte-reproducible zip and return its member names."""
    output.parent.mkdir(parents=True, exist_ok=True)
    members = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_files():
            relative = path.relative_to(ROOT)
            info = zipfile.ZipInfo(str(relative), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
            members.append(str(relative))
    return members


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "gold_training_runner.zip",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    members = build_bundle(args.output)
    print(f"wrote {len(members)} files to {args.output}")


if __name__ == "__main__":
    main()
