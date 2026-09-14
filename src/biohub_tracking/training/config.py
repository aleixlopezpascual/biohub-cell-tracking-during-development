"""Typed configuration for gated, Kaggle-hosted competitive training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LearningCurveConfig(BaseModel):
    """Official-score checkpoints and promotion thresholds."""

    model_config = ConfigDict(extra="forbid")

    evaluation_epochs: tuple[int, ...] = (50, 65, 80, 100)
    minimum_mean_gain: float = Field(0.003, ge=0)
    maximum_fold_regression: float = Field(0.001, ge=0)
    minimum_ensemble_gain: float = Field(0.002, ge=0)

    @field_validator("evaluation_epochs", mode="before")
    @classmethod
    def _coerce_epochs(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("evaluation_epochs")
    @classmethod
    def _validate_epochs(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(epoch < 1 for epoch in value):
            raise ValueError("evaluation_epochs must contain positive integers")
        if tuple(sorted(set(value))) != value:
            raise ValueError("evaluation_epochs must be strictly increasing and unique")
        return value


class OOFInferenceConfig(BaseModel):
    """Official predictor parameters frozen for checkpoint comparisons."""

    model_config = ConfigDict(extra="forbid")

    detection_threshold: float = Field(0.99, gt=0, lt=1)
    detection_tta: bool = True
    edge_activation: Literal["sigmoid", "softmax"] = "softmax"
    edge_threshold: float = Field(0.5, ge=0, le=1)
    use_ilp: bool = True
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0
    max_parents_per_node: int | None = Field(None, ge=1)
    max_children_per_node: int | None = Field(None, ge=1)

class GoldTrainingConfig(BaseModel):
    """Public workflow parameters for the first gated training campaign."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str
    cv_pack_dir: str
    official_source_dir: str
    output_dir: str = "outputs/gold_training"
    fold: Literal["A", "B"] = "A"
    seed: int = Field(42, ge=0)
    batch_size: int = Field(16, ge=1)
    learning_rate: float = Field(1e-4, gt=0)
    num_workers: int = Field(4, ge=0)
    window_size: int = Field(3, ge=2)
    downsample: tuple[int, int, int] = (1, 4, 4)
    voxel_size_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
    gaussian_sigma_um: tuple[float, float, float] = (2.0, 1.0, 1.0)
    unet_out_channels: int = Field(32, ge=1)
    unet_layers: tuple[int, ...] = (32, 64, 128)
    transformer_hidden_dim: int = Field(128, ge=1)
    transformer_heads: int = Field(4, ge=1)
    transformer_blocks: int = Field(4, ge=1)
    transformer_dropout: float = Field(0.3, ge=0, lt=1)
    pool_kernel_um: float = Field(5.0, gt=0)
    detection_loss_weight: float = Field(1.0, gt=0)
    unlabeled_negative_weight: float = Field(0.01, ge=0, le=1)
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    lr_scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    checkpoint_every_epochs: int = Field(5, ge=1)
    max_nonfinite_gradient_batches_per_epoch: int = Field(25, ge=0)
    early_stopping_patience: int = Field(2, ge=1)
    max_runtime_hours: float = Field(9.5, gt=0, le=10)
    learning_curve: LearningCurveConfig = Field(default_factory=LearningCurveConfig)
    oof_inference: OOFInferenceConfig = Field(default_factory=OOFInferenceConfig)

    @field_validator(
        "downsample",
        "voxel_size_um",
        "gaussian_sigma_um",
        "unet_layers",
        mode="before",
    )
    @classmethod
    def _coerce_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_geometry(self) -> "GoldTrainingConfig":
        if any(value < 1 for value in self.downsample):
            raise ValueError("downsample values must be positive integers")
        if any(value <= 0 for value in self.gaussian_sigma_um):
            raise ValueError("gaussian_sigma_um values must be positive")
        if any(value <= 0 for value in self.voxel_size_um):
            raise ValueError("voxel_size_um values must be positive")
        if not self.unet_layers or any(value < 1 for value in self.unet_layers):
            raise ValueError("unet_layers must contain positive integers")
        if self.transformer_hidden_dim % self.transformer_heads:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")
        return self


def load_gold_training_config(path: str | Path) -> GoldTrainingConfig:
    """Load and validate a competitive-training YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return GoldTrainingConfig.model_validate(raw)
