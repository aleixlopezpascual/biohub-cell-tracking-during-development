"""Competitive training utilities with optional deep-learning dependencies.

The module keeps experiment planning, targets, checkpoint metadata, and
promotion gates importable without PyTorch.  Actual tensor serialization only
requires PyTorch when a checkpoint is saved or loaded.
"""

from __future__ import annotations

from biohub_tracking.training.checkpoints import (
    RunManifest,
    TrainingCheckpoint,
    build_run_manifest,
    load_training_checkpoint,
    save_inference_checkpoint,
    save_training_checkpoint,
    validate_run_manifest,
)
from biohub_tracking.training.config import (
    GoldTrainingConfig,
    LearningCurveConfig,
    OOFInferenceConfig,
    load_gold_training_config,
)
from biohub_tracking.training.gates import (
    CheckpointScore,
    PromotionDecision,
    append_checkpoint_score,
    evaluate_ensemble_gate,
    evaluate_learning_curve_gate,
    load_checkpoint_scores,
    rank_complementary_models,
    select_best_checkpoint,
    write_diagnostic_curve_svgs,
    write_learning_curve_svg,
)
from biohub_tracking.training.royerlab_backend import (
    load_official_script,
    load_official_trainer,
    official_import_context,
)
from biohub_tracking.training.session import StopDecision, TrainingSession
from biohub_tracking.training.targets import (
    make_torch_gaussian_detection_loss,
    positive_unlabeled_weights,
    render_gaussian_center_target,
)

__all__ = [
    "CheckpointScore",
    "GoldTrainingConfig",
    "LearningCurveConfig",
    "OOFInferenceConfig",
    "PromotionDecision",
    "RunManifest",
    "StopDecision",
    "TrainingCheckpoint",
    "TrainingSession",
    "append_checkpoint_score",
    "build_run_manifest",
    "evaluate_ensemble_gate",
    "evaluate_learning_curve_gate",
    "load_checkpoint_scores",
    "load_gold_training_config",
    "load_official_script",
    "load_official_trainer",
    "official_import_context",
    "load_training_checkpoint",
    "make_torch_gaussian_detection_loss",
    "positive_unlabeled_weights",
    "rank_complementary_models",
    "render_gaussian_center_target",
    "save_inference_checkpoint",
    "save_training_checkpoint",
    "select_best_checkpoint",
    "validate_run_manifest",
    "write_diagnostic_curve_svgs",
    "write_learning_curve_svg",
]
