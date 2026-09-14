#!/usr/bin/env python3
"""Kaggle entry point for the detector overfitting sanity preflight."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 1. Dynamically locate the package code inside the mounted Kaggle datasets
print("=" * 50)
print("Searching for overfit_detector_sanity.py under /kaggle/input/...")
candidates = list(Path("/kaggle/input").glob("**/overfit_detector_sanity.py"))

if not candidates:
    # Print the directory structure under /kaggle/input to help debug
    print("Could not find overfit_detector_sanity.py. Listing /kaggle/input contents:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = len(Path(root).relative_to("/kaggle/input").parts)
        if depth <= 2:
            print("  " * depth + f"- {Path(root).name}/ ({len(dirs)} dirs, {len(files)} files)")
            for f in files[:5]:
                print("  " * (depth + 1) + f"- {f}")
    raise FileNotFoundError("Could not find overfit_detector_sanity.py in mounted datasets.")

script_path = candidates[0]
scripts_dir = script_path.parent
dataset_root = scripts_dir.parent
src_dir = dataset_root / "src"

print(f"Found sanity check script at: {script_path}")
print(f"Adding code paths to sys.path:")
print(f"  - Src: {src_dir}")
print(f"  - Scripts: {scripts_dir}")
print("=" * 50)

# Add the mounted dataset code paths to sys.path
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(scripts_dir))

import torch
from overfit_detector_sanity import run_sanity_check

print("=" * 50)
print(f"CUDA devices available: {torch.cuda.device_count()}")
print(f"Active Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 50)

# Run overfitting convergence check for 150 epochs on GPU
success = run_sanity_check(epochs=150, lr=0.005, seed=42)

print("\n" + "=" * 50)
if success:
    print("🎉 DETECTOR SANITY CHECK: PASSED!")
    print("Our target maps, coordinate spaces, and loss calculations are 100% bug-free.")
else:
    print("❌ DETECTOR SANITY CHECK: FAILED!")
    print("There is a pipeline alignment/coordinate space bug that needs fixing.")
print("=" * 50)

# Return exit code based on success
sys.exit(0 if success else 1)
