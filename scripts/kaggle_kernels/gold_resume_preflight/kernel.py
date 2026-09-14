"""Kaggle wrapper for the bounded Gold-training resume preflight."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path


repo_dir = Path("/kaggle/working/biohub-preflight-repo")
entry_points = sorted(Path("/kaggle/input").glob("**/scripts/kaggle_resume_preflight.py"))
if len(entry_points) == 1:
    repo_dir = entry_points[0].parent.parent
else:
    bundles = sorted(Path("/kaggle/input").glob("**/gold_training_runner.zip"))
    if len(bundles) != 1:
        raise FileNotFoundError("attach exactly one Gold-training runner dataset")
    with zipfile.ZipFile(bundles[0]) as archive:
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in archive.namelist()):
            raise ValueError("training bundle contains an unsafe member path")
        archive.extractall(repo_dir)

entry_point = repo_dir / "scripts" / "kaggle_resume_preflight.py"
environment = os.environ.copy()
environment["PYTHONPATH"] = str(repo_dir / "src")
subprocess.run([sys.executable, str(entry_point)], cwd=repo_dir, env=environment, check=True)
