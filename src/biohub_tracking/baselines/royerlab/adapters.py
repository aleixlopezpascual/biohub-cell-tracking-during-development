"""Explicit optional-runtime boundaries for the learned Royerlab stack."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TemporalModel(Protocol):
    """Minimal model interface consumed by a Royerlab inference orchestrator."""

    def predict_heatmaps_and_features(self, frame_pair: object) -> tuple[object, object]:
        """Return detector heatmaps and node feature volumes for a frame pair."""


class EdgeModel(Protocol):
    """Minimal learned edge scorer interface."""

    def predict_edge_logits(self, source_features: object, target_features: object) -> object:
        """Return pairwise source-to-target association logits."""


def _require_module(module: str, extra: str) -> None:
    try:
        __import__(module)
    except ImportError as exc:
        raise ImportError(
            f"{module!r} is required for this Royerlab adapter. Install the {extra!r} extra "
            "or attach the compatible offline Kaggle support pack."
        ) from exc


def validate_checkpoint(path: str | Path) -> Path:
    """Validate a model artifact before an optional model adapter tries to load it."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Royerlab checkpoint not found: {checkpoint}. Provide the TemporalUNet3D/"
            "SimpleNodeTransformer weights from a compatible Kaggle support pack."
        )
    return checkpoint


class TemporalUNet3DAdapter:
    """Placeholder boundary for the support-pack TemporalUNet3D implementation."""

    def load(self, checkpoint: str | Path) -> TemporalModel:
        """Fail clearly until a compatible external TemporalUNet3D is supplied."""
        validate_checkpoint(checkpoint)
        _require_module("torch", "torch")
        raise ImportError(
            "TemporalUNet3D is supplied by the Royerlab Kaggle support pack, not this "
            "dependency-light package. Add that package to PYTHONPATH and provide a concrete adapter."
        )


class SimpleNodeTransformerAdapter:
    """Placeholder boundary for the support-pack edge-transformer implementation."""

    def load(self, checkpoint: str | Path) -> EdgeModel:
        """Fail clearly until a compatible external SimpleNodeTransformer is supplied."""
        validate_checkpoint(checkpoint)
        _require_module("torch", "torch")
        raise ImportError(
            "SimpleNodeTransformer is supplied by the Royerlab Kaggle support pack, not this "
            "dependency-light package. Add that package to PYTHONPATH and provide a concrete adapter."
        )


class TracksdataGEFFAdapter:
    """Lazy validator for the optional tracksdata/GEFF ILP graph-solving runtime."""

    def require_runtime(self) -> None:
        """Require the native graph dependencies only when solving candidate graphs."""
        _require_module("tracksdata", "royerlab")
        _require_module("geff", "royerlab")
