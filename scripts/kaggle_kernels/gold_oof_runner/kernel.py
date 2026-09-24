"""Kaggle entry point for one exact-epoch training + OOF scoring stage."""

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

# Attach this repository, the competition data, Local CV Pack, upstream
# Royerlab repository, and its offline dependency wheels before running.
REPO_DIR = Path(os.environ.get("BIOHUB_REPO_DIR", "/kaggle/working/biohub-repo"))
CANDIDATE = os.environ.get("BIOHUB_CANDIDATE", "temporal-pu-a")
TARGET_EPOCH = int(os.environ.get("BIOHUB_TARGET_EPOCH", "10"))
# This checked-in kernel is the recovery launch for the preserved v7 state.
# Set BIOHUB_RESUME=0 only when intentionally starting a new campaign.
RESUME = os.environ.get("BIOHUB_RESUME", "1") == "1"
OUTPUT_DIR = Path("/kaggle/working/outputs/gold_training")
os.environ.setdefault("POLARS_PREFER_PKG", "32")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

if not (REPO_DIR / "scripts" / "run_gold_stage.py").is_file():
    bundles = sorted(Path("/kaggle/input").glob("**/gold_training_runner.zip"))
    extracted = sorted(Path("/kaggle/input").glob("**/scripts/run_gold_stage.py"))
    if len(bundles) == 1:
        with zipfile.ZipFile(bundles[0]) as archive:
            if any(
                Path(name).is_absolute() or ".." in Path(name).parts
                for name in archive.namelist()
            ):
                raise ValueError("training bundle contains an unsafe member path")
            archive.extractall(REPO_DIR)
    elif len(extracted) == 1:
        # Kaggle datasets may expand uploaded ZIPs during ingestion.
        REPO_DIR = extracted[0].parent.parent
    else:
        raise FileNotFoundError("attach exactly one Gold-training runner dataset")

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
    raise RuntimeError("Gold training requires a CUDA GPU")
cuda_capability = torch.cuda.get_device_capability(0)
if cuda_capability[0] < 7:
    raise RuntimeError(
        "the default Kaggle PyTorch image cannot execute on this GPU; select a T4 or newer"
    )
print(
    f"CUDA devices={torch.cuda.device_count()} "
    f"device={torch.cuda.get_device_name(0)} capability={cuda_capability}",
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
    raise FileNotFoundError("could not uniquely locate the attached competition train directory")

if RESUME and not OUTPUT_DIR.exists():
    configured_source = os.environ.get("BIOHUB_RESUME_SOURCE")
    if configured_source:
        resume_sources = [Path(configured_source)]
    else:
        resume_sources = [
            path.parent
            for path in sorted(
                Path("/kaggle/input").glob("**/outputs/gold_training/run_manifest.json")
            )
        ]
    if len(resume_sources) != 1 or not resume_sources[0].is_dir():
        raise FileNotFoundError(
            "resume requires exactly one prior gold_training output dataset; "
            "attach it or set BIOHUB_RESUME_SOURCE"
        )
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(resume_sources[0], OUTPUT_DIR)
    print(f"restored prior campaign from {resume_sources[0]}", flush=True)

source_config = REPO_DIR / "configs" / "gold_training.yaml"
runtime_config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
runtime_config.update(
    {
        "data_dir": str(competition_data[0]),
        "cv_pack_dir": str(cv_folds[0].parent),
        "official_source_dir": str(official_scripts[0].parent.parent),
        "output_dir": str(OUTPUT_DIR),
        "batch_size": int(os.environ.get("BIOHUB_BATCH_SIZE", "8")),
    }
)
CONFIG = Path("/kaggle/working/gold_training_runtime.yaml")
CONFIG.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")

DETECTION_THRESHOLD = float(os.environ.get("BIOHUB_DETECTION_THRESHOLD", "0.60"))

command = [
    sys.executable,
    str(REPO_DIR / "scripts" / "run_gold_stage.py"),
    "--config",
    str(CONFIG),
    "--candidate",
    CANDIDATE,
    "--target-epoch",
    str(TARGET_EPOCH),
    "--detection-threshold",
    str(DETECTION_THRESHOLD),
]
if RESUME:
    command.append("--resume")

environment = os.environ.copy()
environment["PYTHONPATH"] = str(REPO_DIR / "src")
subprocess.run(command, cwd=REPO_DIR, env=environment, check=True)
