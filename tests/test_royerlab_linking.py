"""Tests for competitive edge features and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.baselines.royerlab import (
    HeatmapPeak,
    apply_temperature,
    edge_motion_features,
    fit_temperature,
    fuse_bidirectional_probabilities,
    mine_hard_negative_pairs,
    refine_peak_centroid,
    select_peaks_to_budget,
)
from biohub_tracking.tracking import Detection


def _node(identifier: str, frame: int, x: float) -> Detection:
    return Detection(identifier, frame, 0.0, 0.0, x)


def test_motion_features_measure_constant_velocity_residual() -> None:
    predecessor = _node("p", 0, 0)
    source = _node("s", 1, 2)
    target = _node("t", 2, 4)
    features = edge_motion_features([source], [target], predecessors={"s": predecessor})
    assert features.shape == (1, 1, 8)
    assert features[0, 0, 3] == pytest.approx(2.0)
    assert features[0, 0, 7] == pytest.approx(0.0)


def test_hard_negative_mining_prefers_high_probability_then_distance() -> None:
    sources = [_node("good", 0, 0), _node("near", 0, 1), _node("far", 0, 4)]
    targets = [_node("target", 1, 0)]
    pairs = mine_hard_negative_pairs(
        sources,
        targets,
        {("good", "target")},
        max_distance_um=5,
        max_negatives_per_target=1,
        probabilities=np.array([[0.9], [0.2], [0.8]]),
    )
    assert pairs == [(2, 0)]


def test_probability_fusion_and_temperature_calibration() -> None:
    forward = np.array([0.9, 0.8])
    reverse = np.array([0.9, 0.2])
    harmonic = fuse_bidirectional_probabilities(forward, reverse)
    arithmetic = fuse_bidirectional_probabilities(forward, reverse, mode="arithmetic")
    assert harmonic[1] < arithmetic[1]

    temperature = fit_temperature(np.array([0.6, 0.4, 0.55, 0.45]), np.array([1, 0, 1, 0]))
    calibrated = apply_temperature(np.array([0.6, 0.4]), temperature)
    assert temperature < 1
    assert calibrated[0] > 0.6
    assert calibrated[1] < 0.4


def test_budget_selection_and_intensity_refinement() -> None:
    peaks = [HeatmapPeak((0, 0, 0), 0.4), HeatmapPeak((1, 1, 1), 0.9)]
    assert select_peaks_to_budget(peaks, 1) == [peaks[1]]
    image = np.zeros((3, 3, 3), dtype=float)
    image[1, 1, 1] = 1
    image[1, 1, 2] = 3
    refined = refine_peak_centroid(
        image, (1, 1, 1), radius_um=1.0, voxel_size_um=(1.0, 1.0, 1.0)
    )
    assert refined == pytest.approx((1.0, 1.0, 1.75))
