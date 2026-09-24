#!/usr/bin/env python3
"""Kaggle entry point for evaluating the clean public 50-epoch checkpoint on Holdout B."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

# 1. Dynamically locate the package code inside the mounted Kaggle datasets
print("=" * 50)
print("Searching for overfit_detector_sanity.py under /kaggle/input/...")
candidates = list(Path("/kaggle/input").glob("**/overfit_detector_sanity.py"))

if not candidates:
    raise FileNotFoundError("Could not find overfit_detector_sanity.py in mounted datasets.")

script_path = candidates[0]
scripts_dir = script_path.parent
dataset_root = scripts_dir.parent
src_dir = dataset_root / "src"

print(f"Found code dataset root at: {dataset_root}")
print(f"Adding code paths to sys.path:")
print(f"  - Src: {src_dir}")
print(f"  - Scripts: {scripts_dir}")
print("=" * 50)

# Add the mounted dataset code paths to sys.path
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(scripts_dir))

# 1.5 Install optional dependencies (zarr, geff, tracksdata, etc.) using offline wheels
required_modules = ("tracksdata", "zarr", "geff", "pyscipopt", "polars")
if any(importlib.util.find_spec(module) is None for module in required_modules):
    wheel_dirs = sorted(Path("/kaggle/input").glob("**/wheels"))
    if not wheel_dirs:
        raise FileNotFoundError("attach the Royerlab offline dependency wheels")
    print("=" * 50)
    print(f"Installing offline dependency wheels from: {wheel_dirs[0]}...")
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
    print("Dependencies installed successfully.")
    print("=" * 50)

import torch

# 2. Locate CV Pack, Competition train data, Support pack weights, and Official Source Repository
cv_packs = sorted(Path("/kaggle/input").glob("**/folds_prefix_holdout.csv"))
if len(cv_packs) != 1:
    raise FileNotFoundError("Could not uniquely locate dariushafshar/biohub-local-cv-pack")
cv_pack_dir = cv_packs[0].parent

competition_data = [
    Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
]
competition_dir = [p for p in competition_data if p.is_dir()]
if len(competition_dir) != 1:
    raise FileNotFoundError("Could not uniquely locate competition training data")
train_dir = competition_dir[0]

support_packs = sorted(Path("/kaggle/input").glob("**/weights/unet_transformer/split_0/edge_predictor_best.pth"))
if len(support_packs) != 1:
    raise FileNotFoundError("Could not uniquely locate pilkwang/biohub-tracking-support-pack-50ep-v1 weights")
public_weights = support_packs[0]

# Dynamically locate the official train_unet_transformer.py script in mounted datasets
official_scripts = sorted(Path("/kaggle/input").glob("**/scripts/train_unet_transformer.py"))
if not official_scripts:
    raise FileNotFoundError("Could not find train_unet_transformer.py in /kaggle/input")
official_source_dir = official_scripts[0].parent.parent

print("=" * 50)
print(f"CV Pack: {cv_pack_dir}")
print(f"Train Data: {train_dir}")
print(f"Public Weights: {public_weights}")
print(f"Official Source Repository: {official_source_dir}")
print("=" * 50)

# Setup environment with PYTHONPATH for subprocesses
env = os.environ.copy()
env["PYTHONPATH"] = f"{src_dir}:{scripts_dir}:{env.get('PYTHONPATH', '')}"

output_dir = Path("/kaggle/working/outputs/gold_training")
output_dir.mkdir(parents=True, exist_ok=True)

# 3. Evaluate Fold 1 (Holdout B)
for fold_index, fold_name in [(1, "B")]:
    print("\n" + "=" * 50)
    print(f"🔄 EVALUATING FOLD {fold_index} (Holdout {fold_name})")
    print("=" * 50)
    
    # 3.1 Build Gold runtime configuration with the matching fold
    source_config = dataset_root / "configs" / "gold_training.yaml"
    runtime_config = yaml.safe_load(source_config.read_text(encoding="utf-8"))

    runtime_config.update(
        {
            "data_dir": str(train_dir),
            "cv_pack_dir": str(cv_pack_dir),
            "official_source_dir": str(official_source_dir),
            "output_dir": str(output_dir),
            "fold": fold_name,  # Bind the correct fold name (A or B)
        }
    )
    # Override default threshold to optimal 0.96 for the public model with overlays
    if "oof_inference" not in runtime_config:
        runtime_config["oof_inference"] = {}
    runtime_config["oof_inference"]["detection_threshold"] = 0.96

    # Dynamically append 50 to evaluation_epochs to pass internal validation
    if "learning_curve" in runtime_config and "evaluation_epochs" in runtime_config["learning_curve"]:
        if 50 not in runtime_config["learning_curve"]["evaluation_epochs"]:
            epochs_list = list(runtime_config["learning_curve"]["evaluation_epochs"])
            epochs_list.append(50)
            runtime_config["learning_curve"]["evaluation_epochs"] = epochs_list

    runtime_config_path = Path(f"/kaggle/working/gold_training_runtime_{fold_name}.yaml")
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    print(f"Wrote runtime config to: {runtime_config_path}")

    # 3.2 Prepare gold splits matching this fold
    print(f"Preparing holdout splits for Fold {fold_name}...")
    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "prepare_gold_training.py"),
            "--config",
            str(runtime_config_path),
            "--candidate",
            "public-50ep-overlays",
        ],
        env=env,
        check=True,
    )

    splits_json = output_dir / "prefix_holdout_splits.json"
    if not splits_json.is_file():
        raise FileNotFoundError(f"splits JSON was not generated at: {splits_json}")

    # 3.3 Run OOF evaluation
    print(f"Running OOF Evaluation for Fold {fold_index} using optimal threshold 0.96 and advanced overlays...")
    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "run_oof_checkpoint.py"),
            "--config",
            str(runtime_config_path),
            "--splits",
            str(splits_json),
            "--fold-index",
            str(fold_index),
            "--candidate",
            "public-50ep-overlays",
            "--epoch",
            "50",
            "--weights",
            str(public_weights),
            "--detection-threshold",
            "0.96",
            "--use-overlay",
        ],
        env=env,
        check=True,
    )

print("\n" + "=" * 50)
print("🎉 PUBLIC MODEL OOF REFERENCE COMPLETED FOR FOLD B!")
print(f"Holdout results are saved under {output_dir / 'oof'}")
print("=" * 50)