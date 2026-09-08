"""Tests for typed YAML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from biohub_tracking.utils.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("name", ["default.yaml", "train.yaml", "inference.yaml"])
def test_shipped_configs_are_valid(name: str) -> None:
    config = load_config(CONFIGS_DIR / name)
    assert config.data.zarr_path
    assert len(config.data.voxel_size_um) == 3
    assert len(config.data.patch_size) == 3


def test_load_config_rejects_missing_required_field(tmp_path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("model:\n  architecture: unet3d\n")
    with pytest.raises(Exception):
        load_config(bad_config)


def test_load_config_coerces_yaml_lists_to_tuples(tmp_path) -> None:
    cfg_path = tmp_path / "min.yaml"
    cfg_path.write_text("data:\n  zarr_path: foo.zarr\n  voxel_size_um: [2.0, 0.5, 0.5]\n")
    config = load_config(cfg_path)
    assert config.data.voxel_size_um == (2.0, 0.5, 0.5)
