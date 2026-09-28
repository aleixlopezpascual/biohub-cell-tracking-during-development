"""Tests for staging, metadata reconciliation, and AST safety of Active Mitosis Recovery candidate."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pytest

KERNEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "kaggle_kernels"
    / "biohub_active_mitosis_recovery"
)
METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"
NOTEBOOK_PATH = KERNEL_DIR / "biohub-active-mitosis-recovery.ipynb"


def test_kernel_files_exist():
    assert KERNEL_DIR.is_dir(), f"Kernel directory does not exist: {KERNEL_DIR}"
    assert METADATA_PATH.is_file(), f"Metadata file missing: {METADATA_PATH}"
    assert NOTEBOOK_PATH.is_file(), f"Notebook file missing: {NOTEBOOK_PATH}"


def test_metadata_configuration():
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["id"] == "aleixlopez/biohub-active-mitosis-recovery"
    assert meta["code_file"] == "biohub-active-mitosis-recovery.ipynb"
    assert meta["enable_gpu"] is True
    assert meta["is_private"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert "biohub-cell-tracking-during-development" in meta["competition_sources"]

    required_datasets = {
        "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
        "anvithpothula/biohub-v1284-head-s075",
        "giorgosi/biohub-divnet-v2",
    }
    assert required_datasets.issubset(set(meta["dataset_sources"]))


def test_notebook_metadata_reconciled():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    kaggle_meta = nb.get("metadata", {}).get("kaggle", {})
    assert kaggle_meta.get("isGpuEnabled") is True, "Notebook metadata isGpuEnabled must be True"
    assert kaggle_meta.get("accelerator") == "nvidiaTeslaT4"


def test_all_cells_ast_parse():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Cell {idx} failed AST parse: {e}")


def test_configuration_guard_passes():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    c0 = "".join(nb["cells"][0]["source"])
    c1 = "".join(nb["cells"][1]["source"])
    scope = {}
    exec(c0, scope)
    exec(c1, scope)


def test_strict_anti_inflation_and_mitosis_recovery():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    full_text = "".join("".join(c.get("source", [])) for c in nb["cells"])

    # Strict anti-inflation retained from 0.953 winner
    assert 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"' in full_text
    assert 'os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"' in full_text
    assert "blended_det = _torch.maximum" not in full_text

    # Active mitosis recovery parameters
    assert 'os.environ["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.80"' in full_text
    assert 'os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "18.0"' in full_text
    assert "DivNetMitosisClassifier" in full_text
    assert "divnet_score_division" in full_text
