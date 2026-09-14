"""Modular components inspired by the Royerlab competition baseline.

The package is intentionally usable without PyTorch, tracksdata, GEFF, model
weights, or a Kaggle runtime.  Optional adapters validate those requirements
only when a learned-model or ILP-solver workflow is selected.
"""

from __future__ import annotations

from .candidates import CandidateEdge, EdgeCandidateConfig, generate_edge_candidates
from .features import positional_encoding, sample_node_features
from .linking import (
    apply_temperature,
    edge_motion_features,
    fit_temperature,
    fuse_bidirectional_probabilities,
    mine_hard_negative_pairs,
)
from .peaks import (
    HeatmapPeak,
    HeatmapPeakConfig,
    NodeCountCalibrator,
    extract_heatmap_peaks,
    fit_node_count_calibrator,
    refine_peak_centroid,
    select_peaks_to_budget,
)
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
    "NodeCountCalibrator",
    "NodeBudgetConfig",
    "ShortComponentConfig",
    "SpatialTransform",
    "apply_temperature",
    "apply_xy_d4_tta",
    "cap_node_budget",
    "close_one_frame_gaps",
    "edge_motion_features",
    "extract_heatmap_peaks",
    "filter_short_components",
    "fit_node_count_calibrator",
    "fit_temperature",
    "fuse_bidirectional_probabilities",
    "generate_edge_candidates",
    "invert_xy_d4_tta",
    "mine_hard_negative_pairs",
    "positional_encoding",
    "refine_peak_centroid",
    "select_peaks_to_budget",
    "sample_node_features",
]
