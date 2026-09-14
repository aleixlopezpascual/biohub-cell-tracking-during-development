"""Synthetic tests for dependency-light Royerlab-style baseline components."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.baselines.royerlab.adapters import TracksdataGEFFAdapter, validate_checkpoint
from biohub_tracking.baselines.royerlab.candidates import (
    EdgeCandidateConfig,
    generate_edge_candidates,
)
from biohub_tracking.baselines.royerlab.features import positional_encoding, sample_node_features
from biohub_tracking.baselines.royerlab.peaks import (
    HeatmapPeakConfig,
    extract_heatmap_peaks,
    fit_node_count_calibrator,
    physical_pool_kernel,
)
from biohub_tracking.baselines.royerlab.postprocess import (
    GapClosingConfig,
    NodeBudgetConfig,
    ShortComponentConfig,
    cap_node_budget,
    close_one_frame_gaps,
    filter_short_components,
)
from biohub_tracking.baselines.royerlab.tta import apply_xy_d4_tta, invert_xy_d4_tta
from biohub_tracking.tracking.graph import Detection, TrackingGraph


def _detection(identifier: str, frame: int, x: float) -> Detection:
    return Detection(identifier, frame, 0.0, 0.0, x)


def test_physical_peak_extraction_respects_anisotropy_and_ties() -> None:
    heatmap = np.zeros((3, 5, 5), dtype=float)
    heatmap[1, 2, 2] = 0.9
    heatmap[2, 2, 2] = 0.9  # 2 um apart in Z, retained.
    heatmap[1, 2, 3] = 0.95  # 0.5 um from first peak, suppresses it.
    peaks = extract_heatmap_peaks(
        heatmap,
        HeatmapPeakConfig(
            threshold=0.5,
            min_distance_um=1.0,
            voxel_size_um=(2.0, 0.5, 0.5),
        ),
    )
    assert [peak.zyx for peak in peaks] == [(1, 2, 3), (2, 2, 2)]
    assert physical_pool_kernel(1.0, (2.0, 0.5, 0.5)) == (3, 5, 5)


def test_node_count_calibrator_fits_oof_image_statistics() -> None:
    features = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    counts = np.array([10, 20, 40, 80])
    calibrator = fit_node_count_calibrator(features, counts, ridge=1e-6)
    predictions = calibrator.predict(features)
    assert predictions.shape == (4,)
    assert np.all(np.diff(predictions) > 0)
    assert abs(int(predictions[-1]) - 80) <= 1


def test_xy_d4_tta_round_trips_channel_first_tensor() -> None:
    image = np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    views = apply_xy_d4_tta(image)
    restored = invert_xy_d4_tta(views)
    assert len(views) == 8
    assert np.all(restored == image)


def test_positional_encoding_and_feature_sampling_are_deterministic() -> None:
    encoding = positional_encoding(
        np.array([[0, 0, 0], [2, 4, 6]], dtype=float),
        np.array([0, 2]),
        image_shape=(2, 4, 6),
        time_length=4,
        features_per_dimension=4,
    )
    volume = np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)
    sampled = sample_node_features(volume, np.array([[-2, 1.4, 10], [1, 2, 3]], dtype=float))
    assert encoding.shape == (2, 16)
    assert not np.array_equal(encoding[0], encoding[1])
    assert np.array_equal(sampled, np.array([[7, 31], [23, 47]]))


def test_candidate_generation_keeps_strong_and_top_k_with_physical_gate() -> None:
    sources = [_detection("s0", 0, 0), _detection("s1", 0, 20), _detection("s2", 0, 2)]
    targets = [_detection("t0", 1, 1), _detection("t1", 1, 3)]
    candidates = generate_edge_candidates(
        sources,
        targets,
        np.array([[0.9, 0.1], [0.8, 0.4], [0.3, 0.5]]),
        EdgeCandidateConfig(
            strong_threshold=0.8,
            min_threshold=0.25,
            top_k_parents=1,
            max_distance_um=5,
        ),
    )
    assert [(edge.source_id, edge.target_id, edge.probability) for edge in candidates] == [
        ("s0", "t0", 0.9),
        ("s2", "t1", 0.5),
    ]


def test_gap_closing_creates_midpoint_and_honors_refinement_rejection() -> None:
    graph = TrackingGraph()
    graph.add_node(_detection("before", 0, 0))
    graph.add_node(_detection("after", 2, 4))
    closed = close_one_frame_gaps(graph, GapClosingConfig(max_distance_um=5))
    bridge = next(node for node in closed if node.frame == 1)
    assert bridge.position == (0.0, 0.0, 2.0)
    assert {("before", bridge.id), (bridge.id, "after")} <= closed.edges
    rejected = close_one_frame_gaps(graph, GapClosingConfig(max_distance_um=5), lambda *_: None)
    assert len(rejected.nodes) == 2


def test_gap_closing_can_require_learned_candidate_confidence() -> None:
    graph = TrackingGraph()
    graph.add_node(_detection("before", 0, 0))
    graph.add_node(_detection("after", 2, 4))
    config = GapClosingConfig(max_distance_um=5, min_confidence=0.8)
    with pytest.raises(ValueError, match="score_hook"):
        close_one_frame_gaps(graph, config)
    rejected = close_one_frame_gaps(graph, config, score_hook=lambda _source, _target: 0.7)
    accepted = close_one_frame_gaps(graph, config, score_hook=lambda _source, _target: 0.9)
    assert len(rejected.nodes) == 2
    assert len(accepted.nodes) == 3


def test_component_filter_and_node_budget_keep_expected_graph() -> None:
    graph = TrackingGraph()
    for node in [_detection("a", 0, 0), _detection("b", 1, 1), _detection("solo", 0, 20)]:
        graph.add_node(node)
    graph.add_edge("a", "b")
    filtered = filter_short_components(graph, ShortComponentConfig(min_nodes=2))
    capped = cap_node_budget(filtered, NodeBudgetConfig(max_nodes=1), {"a": 0.1, "b": 0.9})
    assert set(filtered.nodes) == {"a", "b"}
    assert set(capped.nodes) == {"b"}
    assert not capped.edges


def test_component_filter_preserves_confident_short_tracks() -> None:
    graph = TrackingGraph()
    graph.add_node(_detection("strong", 0, 0))
    graph.add_node(_detection("weak", 0, 10))
    config = ShortComponentConfig(min_nodes=2, keep_short_above_confidence=0.8)
    filtered = filter_short_components(
        graph,
        config,
        node_confidences={"strong": 0.9, "weak": 0.2},
    )
    assert set(filtered.nodes) == {"strong"}


def test_optional_adapters_fail_actionably_without_artifacts_or_runtime(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        validate_checkpoint(tmp_path / "missing.pt")
    with pytest.raises(ImportError, match="tracksdata"):
        TracksdataGEFFAdapter().require_runtime()
