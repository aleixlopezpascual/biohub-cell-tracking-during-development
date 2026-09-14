#!/usr/bin/env python3
"""Calibrate and score the preserved epoch-10 checkpoint on Kaggle."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _install_dependencies() -> None:
    required = ("tracksdata", "zarr", "geff", "pyscipopt", "polars")
    if not any(importlib.util.find_spec(module) is None for module in required):
        return
    wheel_dirs = sorted(Path("/kaggle/input").glob("**/wheels"))
    if not wheel_dirs:
        raise FileNotFoundError("attach the Royerlab offline dependency wheels")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--find-links",
            str(wheel_dirs[0]),
            "tracksdata",
            "zarr",
            "pyscipopt",
            "geff",
            "ilpy",
            "polars",
            "blosc2",
            "dask",
            "imagecodecs",
            "pyarrow",
            "rustworkx",
            "sqlalchemy",
        ],
        check=True,
    )


def _run(command: list[str], repo_dir: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_dir / "src")
    print("run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_dir, env=environment, check=True)


def main() -> None:
    """Restore v9 outputs, calibrate detection, and run official OOF scoring."""
    repo_dir = Path(__file__).resolve().parent.parent
    output_dir = Path("/kaggle/working/outputs/gold_training")
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    _install_dependencies()

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("threshold recovery requires a CUDA GPU")
    print(
        f"CUDA devices={torch.cuda.device_count()} device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    checkpoints = sorted(
        Path("/kaggle/input").glob(
            "**/outputs/gold_training/weights/temporal-pu-a/split_0/epoch_010_model.pth"
        )
    )
    if len(checkpoints) != 1:
        raise FileNotFoundError("attach exactly one version-9 epoch-10 kernel output")
    source_gold = checkpoints[0].parents[3]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_gold / "run_manifest.json", output_dir / "run_manifest.json")
    shutil.copy2(
        source_gold / "prefix_holdout_splits.json",
        output_dir / "prefix_holdout_splits.json",
    )
    shutil.copytree(source_gold / "weights", output_dir / "weights")

    official_scripts = sorted(Path("/kaggle/input").glob("**/scripts/train_unet_transformer.py"))
    cv_folds = sorted(Path("/kaggle/input").glob("**/folds_prefix_holdout.csv"))
    if len(official_scripts) != 1 or len(cv_folds) != 1:
        raise FileNotFoundError("attach one support pack and one Local CV Pack")
    competition_candidates = [
        Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
    ]
    competition_data = [path for path in competition_candidates if path.is_dir()]
    if len(competition_data) != 1:
        raise FileNotFoundError("could not uniquely locate competition training data")

    runtime_config = yaml.safe_load(
        (repo_dir / "configs" / "gold_training.yaml").read_text(encoding="utf-8")
    )
    runtime_config.update(
        {
            "data_dir": str(competition_data[0]),
            "cv_pack_dir": str(cv_folds[0].parent),
            "official_source_dir": str(official_scripts[0].parent.parent),
            "output_dir": str(output_dir),
            "batch_size": 8,
        }
    )
    config_path = Path("/kaggle/working/gold_training_runtime.yaml")
    config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    split_path = output_dir / "prefix_holdout_splits.json"
    checkpoint = output_dir / "weights/temporal-pu-a/split_0/epoch_010_model.pth"
    probe_path = Path("/kaggle/working/threshold_probe.json")
    thresholds = ["0.50", "0.55", "0.60", "0.65", "0.70", "0.80", "0.90", "0.95"]
    _run(
        [
            sys.executable,
            str(repo_dir / "scripts" / "probe_detection_thresholds.py"),
            "--config",
            str(config_path),
            "--splits",
            str(split_path),
            "--fold-index",
            "0",
            "--candidate",
            "temporal-pu-a",
            "--weights",
            str(checkpoint),
            "--thresholds",
            *thresholds,
            "--max-datasets",
            "3",
            "--max-frames",
            "8",
            "--output",
            str(probe_path),
        ],
        repo_dir,
    )
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    selected = float(probe["recommended_threshold"])
    print(f"selected detection threshold={selected:.6f}", flush=True)
    _run(
        [
            sys.executable,
            str(repo_dir / "scripts" / "run_oof_checkpoint.py"),
            "--config",
            str(config_path),
            "--splits",
            str(split_path),
            "--fold-index",
            "0",
            "--candidate",
            "temporal-pu-a",
            "--epoch",
            "10",
            "--weights",
            str(checkpoint),
            "--detection-threshold",
            str(selected),
        ],
        repo_dir,
    )


if __name__ == "__main__":
    main()
