"""Modular components inspired by the Royerlab competition baseline.

The package is intentionally usable without PyTorch, tracksdata, GEFF, model
weights, or a Kaggle runtime.  Optional adapters validate those requirements
only when a learned-model or ILP-solver workflow is selected.
"""

from __future__ import annotations

from .candidates import CandidateEdge, EdgeCandidateConfig, generate_edge_candidates
from .features import positional_encoding, sample_node_features
from .peaks import HeatmapPeakConfig, HeatmapPeak, extract_heatmap_peaks
from .postprocess import (
    GapClosingConfig,
    NodeBudgetConfig,
    ShortComponentConfig,
    cap_node_budget,
    close_one_frame_gaps,
    filter_short_components,
)
from .tta import SpatialTransform, apply_xy_d4_tta, invert_xy_d4_tta

__all__ = [
    "CandidateEdge",
    "EdgeCandidateConfig",
    "GapClosingConfig",
    "HeatmapPeak",
    "HeatmapPeakConfig",
    "NodeBudgetConfig",
    "ShortComponentConfig",
    "SpatialTransform",
    "apply_xy_d4_tta",
    "cap_node_budget",
    "close_one_frame_gaps",
    "extract_heatmap_peaks",
    "filter_short_components",
    "generate_edge_candidates",
    "invert_xy_d4_tta",
    "positional_encoding",
    "sample_node_features",
]
