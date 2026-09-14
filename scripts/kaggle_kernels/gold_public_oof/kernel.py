#!/usr/bin/env python3
"""Kaggle entry point for evaluating the clean public 50-epoch checkpoint on both OOF splits."""

from __future__ import annotations

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

import torch

# 2. Locate CV Pack, Competition train data, and Support pack weights
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
official_source_dir = public_weights.parent.parent.parent.parent

print("=" * 50)
print(f"CV Pack: {cv_pack_dir}")
print(f"Train Data: {train_dir}")
print(f"Public Weights: {public_weights}")
print(f"Official Source: {official_source_dir}")
print("=" * 50)

# 3. Build Gold runtime configuration
source_config = dataset_root / "configs" / "gold_training.yaml"
runtime_config = yaml.safe_load(source_config.read_text(encoding="utf-8"))

output_dir = Path("/kaggle/working/outputs/gold_training")
output_dir.mkdir(parents=True, exist_ok=True)

runtime_config.update(
    {
        "data_dir": str(train_dir),
        "cv_pack_dir": str(cv_pack_dir),
        "official_source_dir": str(official_source_dir),
        "output_dir": str(output_dir),
    }
)
# Override the default 0.99 threshold to the optimal 0.97 for the public model
if "oof_inference" not in runtime_config:
    runtime_config["oof_inference"] = {}
runtime_config["oof_inference"]["detection_threshold"] = 0.97

runtime_config_path = Path("/kaggle/working/gold_training_runtime.yaml")
runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
print(f"Wrote runtime config to: {runtime_config_path}")

# Setup environment with PYTHONPATH for subprocesses
env = os.environ.copy()
env["PYTHONPATH"] = f"{src_dir}:{scripts_dir}:{env.get('PYTHONPATH', '')}"

# 4. Prepare gold splits
print("\n" + "=" * 50)
print("Preparing holdout splits...")
subprocess.run(
    [
        sys.executable,
        str(scripts_dir / "prepare_gold_training.py"),
        "--config",
        str(runtime_config_path),
        "--candidate",
        "public-50ep",
    ],
    env=env,
    check=True,
)
print("=" * 50 + "\n")

splits_json = output_dir / "prefix_holdout_splits.json"
if not splits_json.is_file():
    raise FileNotFoundError(f"splits JSON was not generated at: {splits_json}")

# 5. Run OOF evaluations on both split_0 and split_1
for fold_index in [0, 1]:
    print("=" * 50)
    print(f"Running OOF Evaluation for Fold {fold_index} using optimal threshold 0.97...")
    print("=" * 50)
    
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
            "public-50ep",
            "--epoch",
            "50",
            "--weights",
            str(public_weights),
            "--detection-threshold",
            "0.97",
        ],
        env=env,
        check=True,
    )

print("\n" + "=" * 50)
print("🎉 PUBLIC MODEL OOF REFERENCE COMPLETED!")
print(f"Holdout results are saved under {output_dir / 'oof'}")
print("=" * 50)
