"""Tests for the sparse-aware edge Jaccard (metrics.md "Edge Jaccard")."""

from __future__ import annotations

from biohub_tracking.metrics.edge_jaccard import compute_edge_jaccard
from biohub_tracking.tracking.graph import Detection, TrackingGraph


def _det(id_, frame, z, y, x) -> Detection:
    return Detection(id=id_, frame=frame, z=z, y=y, x=x)


def test_perfect_prediction_has_jaccard_one() -> None:
    gt = TrackingGraph()
    gt.add_node(_det("g0", 0, 0, 0, 0))
    gt.add_node(_det("g1", 1, 0, 0, 1))
    gt.add_edge("g0", "g1")

    pred = TrackingGraph()
    pred.add_node(_det("p0", 0, 0, 0, 0))
    pred.add_node(_det("p1", 1, 0, 0, 1))
    pred.add_edge("p0", "p1")

    result = compute_edge_jaccard(pred, gt)
    assert (result.tp, result.fp, result.fn) == (1, 0, 0)
    assert result.jaccard == 1.0


def test_unmatched_predicted_extras_are_ignored_not_fp() -> None:
    """Sparse GT: correct predictions the GT doesn't annotate must not count as FP."""
    gt = TrackingGraph()
    gt.add_node(_det("g0", 0, 0, 0, 0))
    gt.add_node(_det("g1", 1, 0, 0, 1))
    gt.add_edge("g0", "g1")

    pred = TrackingGraph()
    pred.add_node(_det("p0", 0, 0, 0, 0))
    pred.add_node(_det("p1", 1, 0, 0, 1))
    pred.add_node(_det("extra0", 0, 50, 50, 50))
    pred.add_node(_det("extra1", 1, 50, 50, 51))
    pred.add_edge("p0", "p1")
    pred.add_edge("extra0", "extra1")  # matches nothing in GT: must be ignored

    result = compute_edge_jaccard(pred, gt)
    assert (result.tp, result.fp, result.fn) == (1, 0, 0)


def test_target_matched_to_gt_node_with_another_source_is_fp() -> None:
    gt = TrackingGraph()
    gt.add_node(_det("g1", 0, 0, 0, 0))
    gt.add_node(_det("g2", 0, 0, 0, 100))
    gt.add_node(_det("gA", 1, 0, 0, 1))
    gt.add_node(_det("gB", 1, 0, 0, 101))
    gt.add_edge("g1", "gA")
    gt.add_edge("g2", "gB")

    pred = TrackingGraph()
    pred.add_node(_det("p1", 0, 0, 0, 0))
    pred.add_node(_det("p2", 0, 0, 0, 100))
    pred.add_node(_det("pA", 1, 0, 0, 1))
    pred.add_node(_det("pB", 1, 0, 0, 101))
    pred.add_edge("p1", "pA")  # TP: matches g1->gA
    pred.add_edge("p2", "pB")  # TP: matches g2->gB
    pred.add_edge("p2", "pA")  # FP: pA matches gA, which already has source g1 != g2

    result = compute_edge_jaccard(pred, gt)
    assert result.tp == 2
    assert result.fp == 1
    assert result.fn == 0


def test_source_matched_to_gt_node_with_another_target_is_fp() -> None:
    gt = TrackingGraph()
    gt.add_node(_det("g1", 0, 0, 0, 0))
    gt.add_node(_det("g2", 0, 0, 0, 100))
    gt.add_node(_det("gA", 1, 0, 0, 1))
    gt.add_node(_det("gB", 1, 0, 0, 101))
    gt.add_edge("g1", "gA")
    gt.add_edge("g2", "gB")

    pred = TrackingGraph()
    pred.add_node(_det("p1", 0, 0, 0, 0))
    pred.add_node(_det("p2", 0, 0, 0, 100))
    pred.add_node(_det("pA", 1, 0, 0, 1))
    pred.add_node(_det("pB", 1, 0, 0, 101))
    pred.add_edge("p1", "pA")  # TP
    pred.add_edge("p2", "pB")  # TP
    pred.add_edge("p1", "pB")  # FP: p1 matches g1, which already targets gA != gB

    result = compute_edge_jaccard(pred, gt)
    assert result.tp == 2
    assert result.fp == 1
    assert result.fn == 0


def test_unrecovered_gt_edge_is_fn() -> None:
    gt = TrackingGraph()
    gt.add_node(_det("g0", 0, 0, 0, 200))
    gt.add_node(_det("g1", 1, 0, 0, 201))
    gt.add_edge("g0", "g1")

    pred = TrackingGraph()  # no predictions at all near this region

    result = compute_edge_jaccard(pred, gt)
    assert (result.tp, result.fp, result.fn) == (0, 0, 1)
    assert result.jaccard == 0.0


def test_adjusted_jaccard_penalizes_excess_predicted_nodes() -> None:
    gt = TrackingGraph()
    gt.add_node(_det("g0", 0, 0, 0, 0))
    gt.add_node(_det("g1", 1, 0, 0, 1))
    gt.add_edge("g0", "g1")

    pred = TrackingGraph()
    pred.add_node(_det("p0", 0, 0, 0, 0))
    pred.add_node(_det("p1", 1, 0, 0, 1))
    pred.add_edge("p0", "p1")

    result = compute_edge_jaccard(pred, gt)
    # T_pred == T_true: no penalty.
    assert result.adjusted_jaccard(num_pred_nodes=2, num_true_nodes=2) == 1.0
    # T_pred > T_true: penalized but not below zero.
    adjusted = result.adjusted_jaccard(num_pred_nodes=12, num_true_nodes=2, alpha=0.1)
    assert 0.0 <= adjusted < 1.0
