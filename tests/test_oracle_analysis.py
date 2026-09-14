"""Tests for detection-versus-linking oracle headroom."""

from __future__ import annotations

import pytest

from biohub_tracking.evaluation.oracle import analyze_oracle_headroom, oracle_links_for_detections
from biohub_tracking.tracking import Detection, TrackingGraph


def _line(prefix: str, *, missing_middle: bool = False) -> TrackingGraph:
    graph = TrackingGraph()
    frames = (0, 2) if missing_middle else (0, 1, 2)
    for frame in frames:
        graph.add_node(Detection(f"{prefix}{frame}", frame, 0.0, 0.0, float(frame)))
    if not missing_middle:
        graph.add_edge(f"{prefix}0", f"{prefix}1")
        graph.add_edge(f"{prefix}1", f"{prefix}2")
    return graph


def test_oracle_links_recover_all_supported_gt_edges() -> None:
    gt = _line("g")
    pred = _line("p")
    pred.edges.clear()
    oracle = oracle_links_for_detections(pred, gt)
    assert oracle.edges == {("p0", "p1"), ("p1", "p2")}


def test_oracle_analysis_separates_linking_and_detection_headroom() -> None:
    gt = _line("g")
    linked_nodes = _line("p")
    linked_nodes.edges.clear()
    linking_result = analyze_oracle_headroom(
        {"sample": linked_nodes}, {"sample": gt}, node_estimates={"sample": 3}
    )
    assert linking_result.summary["priority"] == "linking"
    # A graph with no predicted forks receives the zero-event division term,
    # so the recoverable edge/linking contribution is exactly 1.0.
    assert linking_result.summary["linking_headroom"] == pytest.approx(1.0)

    missing = _line("p", missing_middle=True)
    detection_result = analyze_oracle_headroom(
        {"sample": missing}, {"sample": gt}, node_estimates={"sample": 3}
    )
    assert detection_result.summary["priority"] == "detection"
    assert detection_result.summary["detection_headroom"] > 0
