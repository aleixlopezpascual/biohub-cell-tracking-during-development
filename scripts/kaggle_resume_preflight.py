#!/usr/bin/env python3
"""Run the exact Kaggle resume path for one real batch without a full campaign."""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    """Restore the v7 campaign and execute the bounded continuation preflight."""
    repo_dir = Path(__file__).resolve().parent.parent
    output_dir = Path("/kaggle/working/outputs/gold_training")
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    required_modules = ("tracksdata", "zarr", "geff", "pyscipopt", "polars")
    if any(importlib.util.find_spec(module) is None for module in required_modules):
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

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("resume preflight requires a CUDA GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 7:
        raise RuntimeError("select a T4 or newer GPU for the Kaggle PyTorch image")
    print(
        f"CUDA devices={torch.cuda.device_count()} "
        f"device={torch.cuda.get_device_name(0)} capability={capability}",
        flush=True,
    )

    official_scripts = sorted(Path("/kaggle/input").glob("**/scripts/train_unet_transformer.py"))
    if len(official_scripts) != 1:
        raise FileNotFoundError("attach exactly one upstream Royerlab repository dataset")
    cv_folds = sorted(Path("/kaggle/input").glob("**/folds_prefix_holdout.csv"))
    if len(cv_folds) != 1:
        raise FileNotFoundError("attach exactly one Biohub Local CV Pack dataset")
    competition_candidates = [
        Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
    ]
    competition_data = [path for path in competition_candidates if path.is_dir()]
    if len(competition_data) != 1:
        raise FileNotFoundError("could not uniquely locate competition training data")
    resume_manifests = sorted(
        Path("/kaggle/input").glob("**/outputs/gold_training/run_manifest.json")
    )
    if len(resume_manifests) != 1:
        raise FileNotFoundError("attach exactly one prior gold-training output dataset")
    resume_dir = resume_manifests[0].parent
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(resume_dir, output_dir)
    print(f"restored preflight campaign from {resume_dir}", flush=True)

    source_config = repo_dir / "configs" / "gold_training.yaml"
    runtime_config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
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
    checkpoint = output_dir / "weights/temporal-pu-a/split_0/latest.pt"
    summary = Path("/kaggle/working/preflight_summary.json")
    command = [
        sys.executable,
        str(repo_dir / "scripts" / "continue_royerlab_training.py"),
        "--config",
        str(config_path),
        "--splits",
        str(output_dir / "prefix_holdout_splits.json"),
        "--fold-index",
        "0",
        "--candidate",
        "temporal-pu-a",
        "--target-epoch",
        "10",
        "--resume-checkpoint",
        str(checkpoint),
        "--preflight-output",
        str(summary),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_dir / "src")
    subprocess.run(command, cwd=repo_dir, env=environment, check=True)
    print(summary.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
