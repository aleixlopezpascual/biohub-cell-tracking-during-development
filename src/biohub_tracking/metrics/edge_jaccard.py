"""Sparse-aware edge accounting and the adjusted edge Jaccard (metrics.md)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from biohub_tracking.metrics.matching import match_nodes
from biohub_tracking.tracking.graph import NodeId, TrackingGraph


@dataclass(frozen=True)
class EdgeJaccardResult:
    """TP/FP/FN edge counts and the (optionally adjusted) Jaccard index."""

    tp: int
    fp: int
    fn: int

    @property
    def jaccard(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 1.0

    def adjusted_jaccard(self, num_pred_nodes: int, num_true_nodes: int, alpha: float = 0.1) -> float:
        """Penalize excess predicted nodes: ``max(0, J * (1 - a*(T_pred-T_true)/T_true))``.

        ``num_true_nodes`` is the *coarse estimate of total true nodes*
        (including ones the sparse ground truth doesn't annotate), as
        described in metrics.md - it is provided out of band, not inferred
        from the (sparse) ground-truth graph.
        """
        if num_true_nodes <= 0:
            raise ValueError("num_true_nodes must be positive")
        penalty = 1.0 - alpha * (num_pred_nodes - num_true_nodes) / num_true_nodes
        return max(0.0, self.jaccard * penalty)


def compute_edge_jaccard(
    pred_graph: TrackingGraph,
    gt_graph: TrackingGraph,
    max_distance: float = 7.0,
    match: dict[NodeId, NodeId] | None = None,
) -> EdgeJaccardResult:
    """Compute the sparse-aware edge Jaccard TP/FP/FN counts.

    Following metrics.md:

    - A predicted edge is a **TP** when both endpoints match ground-truth
      nodes that are themselves connected by a ground-truth edge.
    - A ground-truth edge with no such match is a **FN**.
    - A predicted edge that is not a TP is a **FP** when its target matches
      a GT node with a different incoming source, or its source matches a
      GT node with a different outgoing target. All other predicted edges
      (matching nothing in the sparse ground truth) are ignored.

    Parameters
    ----------
    pred_graph, gt_graph:
        Predicted and ground-truth tracking graphs.
    max_distance:
        Node-matching radius in microns (7 um officially).
    match:
        Optional precomputed ``pred_id -> gt_id`` mapping (e.g. shared with
        the division metric) to avoid recomputing the assignment problem.
    """
    if match is None:
        match = match_nodes(pred_graph.nodes.values(), gt_graph.nodes.values(), max_distance)

    gt_edges = gt_graph.edges
    gt_targets_of_source: dict[NodeId, set[NodeId]] = defaultdict(set)
    gt_sources_of_target: dict[NodeId, set[NodeId]] = defaultdict(set)
    for gs, gt in gt_edges:
        gt_targets_of_source[gs].add(gt)
        gt_sources_of_target[gt].add(gs)

    tp = 0
    fp = 0
    recovered_gt_edges: set[tuple[NodeId, NodeId]] = set()

    for ps, pt in pred_graph.edges:
        gs = match.get(ps)
        gtid = match.get(pt)
        if gs is not None and gtid is not None and (gs, gtid) in gt_edges:
            tp += 1
            recovered_gt_edges.add((gs, gtid))
            continue

        is_fp = False
        if gtid is not None:
            other_sources = gt_sources_of_target.get(gtid, set()) - ({gs} if gs is not None else set())
            if other_sources:
                is_fp = True
        if not is_fp and gs is not None:
            other_targets = gt_targets_of_source.get(gs, set()) - ({gtid} if gtid is not None else set())
            if other_targets:
                is_fp = True
        if is_fp:
            fp += 1
        # else: ignored - predicted edge unmatched to any sparse GT evidence.

    fn = len(gt_edges - recovered_gt_edges)
    return EdgeJaccardResult(tp=tp, fp=fp, fn=fn)
