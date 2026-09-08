"""Tests for the local-window division Jaccard (metrics.md "Division Jaccard")."""

from __future__ import annotations

from biohub_tracking.metrics.division_jaccard import compute_division_jaccard
from biohub_tracking.tracking.graph import Detection, TrackingGraph


def _det(id_, frame, z, y, x) -> Detection:
    return Detection(id=id_, frame=frame, z=z, y=y, x=x)


def _gt_division() -> TrackingGraph:
    gt = TrackingGraph()
    gt.add_node(_det("p", 0, 0, 0, 0))
    gt.add_node(_det("c1", 1, 0, 0, 1))
    gt.add_node(_det("c2", 1, 0, 0, -1))
    gt.add_node(_det("gc1", 2, 0, 0, 2))
    gt.add_node(_det("gc2", 2, 0, 0, -2))
    gt.add_edge("p", "c1")
    gt.add_edge("p", "c2")
    gt.add_edge("c1", "gc1")
    gt.add_edge("c2", "gc2")
    return gt


def test_exact_prediction_recovers_division_as_tp() -> None:
    gt = _gt_division()
    pred = TrackingGraph()
    for node in gt.nodes.values():
        pred.add_node(Detection(id=f"pred_{node.id}", frame=node.frame, z=node.z, y=node.y, x=node.x))
    pred.add_edge("pred_p", "pred_c1")
    pred.add_edge("pred_p", "pred_c2")
    pred.add_edge("pred_c1", "pred_gc1")
    pred.add_edge("pred_c2", "pred_gc2")

    result = compute_division_jaccard(pred, gt)
    assert (result.tp, result.fp, result.fn) == (1, 0, 0)


def test_missing_second_branch_is_fn() -> None:
    gt = _gt_division()
    # Predicted graph only continues one daughter lineage: no fork at all.
    pred = TrackingGraph()
    pred.add_node(_det("p", 0, 0, 0, 0))
    pred.add_node(_det("c1", 1, 0, 0, 1))
    pred.add_node(_det("gc1", 2, 0, 0, 2))
    pred.add_edge("p", "c1")
    pred.add_edge("c1", "gc1")

    result = compute_division_jaccard(pred, gt)
    assert result.tp == 0
    assert result.fn == 1


def test_late_division_one_timepoint_after_split_still_counts_as_tp() -> None:
    gt = _gt_division()
    pred = TrackingGraph()
    pred.add_node(_det("pf", 0, 0, 0, 0))  # matches GT parent p
    pred.add_node(_det("pm", 1, 0, 0, 0))  # unmatched intermediate node (no fork yet)
    pred.add_node(_det("pb1", 2, 0, 0, 2))  # matches GT grandchild gc1
    pred.add_node(_det("pb2", 2, 0, 0, -2))  # matches GT grandchild gc2
    pred.add_edge("pf", "pm")
    pred.add_edge("pm", "pb1")
    pred.add_edge("pm", "pb2")

    result = compute_division_jaccard(pred, gt)
    assert result.tp == 1
    assert result.fn == 0


def test_spurious_fork_on_annotated_lineage_is_fp() -> None:
    gt = TrackingGraph()
    gt.add_node(_det("gp0", 0, 0, 0, 0))
    gt.add_node(_det("gp1", 1, 0, 0, 1))
    gt.add_edge("gp0", "gp1")  # a real, non-dividing GT lineage (out_degree 1)

    pred = TrackingGraph()
    pred.add_node(_det("pf0", 0, 0, 0, 0))  # matches gp0
    pred.add_node(_det("pc1", 1, 0, 0, 1))  # matches gp1
    pred.add_node(_det("pc2", 1, 0, 5, 5))  # spurious extra daughter, unmatched
    pred.add_edge("pf0", "pc1")
    pred.add_edge("pf0", "pc2")  # makes pf0 a (false) predicted fork

    result = compute_division_jaccard(pred, gt)
    assert result.fp == 1
    assert result.tp == 0


def test_completely_unmatched_fork_with_no_gt_evidence_is_ignored() -> None:
    gt = TrackingGraph()
    gt.add_node(_det("gp0", 0, 0, 0, 0))
    gt.add_node(_det("gp1", 1, 0, 0, 1))
    gt.add_edge("gp0", "gp1")

    pred = TrackingGraph()
    # Entirely disjoint region far from any GT node: no matches at all.
    pred.add_node(_det("pf0", 0, 500, 500, 500))
    pred.add_node(_det("pc1", 1, 500, 500, 501))
    pred.add_node(_det("pc2", 1, 500, 505, 500))
    pred.add_edge("pf0", "pc1")
    pred.add_edge("pf0", "pc2")

    result = compute_division_jaccard(pred, gt)
    assert result.fp == 0
    assert result.tp == 0
    assert result.fn == 0
