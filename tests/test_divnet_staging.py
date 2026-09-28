"""Tests for staging, metadata reconciliation, and AST safety of the DivNet Super-Fusion candidate."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pytest

KERNEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "kaggle_kernels"
    / "biohub_super_fusion_divnet"
)
METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"
NOTEBOOK_PATH = KERNEL_DIR / "biohub-super-fusion-divnet.ipynb"


def test_kernel_files_exist():
    assert KERNEL_DIR.is_dir(), f"Kernel directory does not exist: {KERNEL_DIR}"
    assert METADATA_PATH.is_file(), f"Metadata file missing: {METADATA_PATH}"
    assert NOTEBOOK_PATH.is_file(), f"Notebook file missing: {NOTEBOOK_PATH}"


def test_metadata_configuration():
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["id"] == "aleixlopez/biohub-super-fusion-divnet"
    assert meta["code_file"] == "biohub-super-fusion-divnet.ipynb"
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


def test_divnet_modules_present():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    full_text = "".join("".join(c.get("source", [])) for c in nb["cells"])

    assert "DivNetMitosisClassifier" in full_text
    assert "load_divnet_mitosis_model" in full_text
    assert "divnet_score_division" in full_text
    assert "v1284_coordinate_refinement" in full_text
    assert "_sec_calib" in full_text
