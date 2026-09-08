"""Typed configuration models and YAML loading helpers.

Configs are plain YAML files under ``configs/`` validated into pydantic
models so that typos or wrong types fail fast (before a long training run
starts) instead of surfacing deep inside the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Location and geometry of the OME-Zarr dataset."""

    zarr_path: str = Field(..., description="Path or URL to the OME-Zarr store root.")
    resolution_level: int = Field(0, ge=0, description="Multiscale pyramid level to read.")
    voxel_size_um: tuple[float, float, float] = Field(
        (1.0, 1.0, 1.0), description="Physical voxel size in microns as (z, y, x)."
    )
    channels: list[int] = Field(default_factory=lambda: [0], description="Channel indices to load.")
    patch_size: tuple[int, int, int] = Field(
        (32, 128, 128), description="3D patch size as (z, y, x) in voxels."
    )
    patch_stride: tuple[int, int, int] | None = Field(
        None, description="Stride between patches; defaults to non-overlapping (== patch_size)."
    )
    frame_pair_stride: int = Field(1, ge=1, description="Temporal gap between paired frames.")

    @field_validator("voxel_size_um", "patch_size", mode="before")
    @classmethod
    def _coerce_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value


class ModelConfig(BaseModel):
    """Baseline 3D U-Net hyperparameters."""

    architecture: Literal["unet3d"] = "unet3d"
    in_channels: int = Field(1, ge=1)
    out_channels: int = Field(1, ge=1)
    base_features: int = Field(32, ge=1)
    depth: int = Field(4, ge=1)
    norm: Literal["batch", "instance", "none"] = "instance"


class TrackingConfig(BaseModel):
    """Frame-to-frame Hungarian tracking parameters."""

    max_link_distance_um: float = Field(15.0, gt=0)
    division_search_radius_um: float = Field(20.0, gt=0)
    max_daughters: int = Field(2, ge=2)
    gating_time_window: int = Field(1, ge=1)


class MetricsConfig(BaseModel):
    """Official Biohub metric parameters (see metrics.md)."""

    node_match_max_distance_um: float = Field(7.0, gt=0)
    adjusted_jaccard_alpha: float = Field(0.1, ge=0)
    division_weight: float = Field(0.1, ge=0)
    division_window: int = Field(1, ge=0, description="Timepoints before/after a GT split allowed.")


class TrainConfig(BaseModel):
    """Training loop parameters."""

    epochs: int = Field(50, ge=1)
    batch_size: int = Field(4, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    num_workers: int = Field(0, ge=0)
    seed: int = Field(0, ge=0)
    output_dir: str = Field("outputs/train")


class PipelineConfig(BaseModel):
    """Top-level config combining all sub-configs."""

    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline config file.

    Parameters
    ----------
    path:
        Path to a YAML file matching :class:`PipelineConfig`'s schema.

    Returns
    -------
    PipelineConfig
        Fully validated, typed configuration object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return PipelineConfig.model_validate(raw)
