"""Tests for micro-averaged score aggregation (metrics.md "Final score")."""

from __future__ import annotations

import pytest

from biohub_tracking.metrics.score import aggregate_scores, evaluate_sample
from biohub_tracking.tracking.graph import Detection, TrackingGraph


def _det(id_, frame, z, y, x) -> Detection:
    return Detection(id=id_, frame=frame, z=z, y=y, x=x)


def _perfect_pair(prefix: str) -> tuple[TrackingGraph, TrackingGraph]:
    gt = TrackingGraph()
    gt.add_node(_det(f"{prefix}g0", 0, 0, 0, 0))
    gt.add_node(_det(f"{prefix}g1", 1, 0, 0, 1))
    gt.add_edge(f"{prefix}g0", f"{prefix}g1")

    pred = TrackingGraph()
    pred.add_node(_det(f"{prefix}p0", 0, 0, 0, 0))
    pred.add_node(_det(f"{prefix}p1", 1, 0, 0, 1))
    pred.add_edge(f"{prefix}p0", f"{prefix}p1")
    return pred, gt


def test_evaluate_sample_perfect_prediction() -> None:
    pred, gt = _perfect_pair("a_")
    sample = evaluate_sample("testA", pred, gt, num_true_nodes=2)
    assert sample.edge.tp == 1
    assert sample.adjusted_edge_jaccard == 1.0
    assert sample.division.jaccard == 1.0  # no divisions anywhere: 0/0 -> defined as 1.0


def test_aggregate_scores_weights_by_sample_size() -> None:
    pred_a, gt_a = _perfect_pair("a_")
    pred_b, gt_b = _perfect_pair("b_")

    sample_a = evaluate_sample("testA", pred_a, gt_a, num_true_nodes=2)
    # Make sample B imperfect: drop the predicted edge entirely -> 1 FN.
    pred_b_empty = TrackingGraph(nodes=dict(pred_b.nodes))
    sample_b = evaluate_sample("testB", pred_b_empty, gt_b, num_true_nodes=2)

    aggregate = aggregate_scores([sample_a, sample_b])
    # sample_a contributes weight 1 (tp=1), sample_b contributes weight 1 (fn=1);
    # micro-averaged adjusted edge jaccard = (1*1 + 0*1) / 2 = 0.5
    assert aggregate.adjusted_edge_jaccard == pytest.approx(0.5)
    assert aggregate.num_samples == 2


def test_aggregate_scores_requires_at_least_one_sample() -> None:
    with pytest.raises(ValueError):
        aggregate_scores([])


def test_final_score_combines_edge_and_division_with_weight() -> None:
    pred, gt = _perfect_pair("c_")
    sample = evaluate_sample("testC", pred, gt, num_true_nodes=2)
    aggregate = aggregate_scores([sample], division_weight=0.1)
    assert aggregate.score == pytest.approx(aggregate.adjusted_edge_jaccard + 0.1 * aggregate.division_jaccard)
