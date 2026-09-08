"""Micro-averaged aggregation of per-sample metrics into the final score.

Per metrics.md, results from each video/sample are combined by summing
TP/FP/FN counts *before* computing the Jaccard, so larger samples
contribute proportionally more and zero-event samples don't skew the
average. The adjusted edge Jaccard is additionally weight-averaged per
sample by ``w_i = TP_i + FP_i + FN_i``; the division Jaccard uses summed
counts directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from biohub_tracking.metrics.division_jaccard import DivisionJaccardResult, compute_division_jaccard
from biohub_tracking.metrics.edge_jaccard import EdgeJaccardResult, compute_edge_jaccard
from biohub_tracking.metrics.matching import match_nodes
from biohub_tracking.tracking.graph import TrackingGraph


@dataclass(frozen=True)
class SampleResult:
    """Per-sample metric outputs, ready for micro-averaged aggregation."""

    sample_id: str
    edge: EdgeJaccardResult
    division: DivisionJaccardResult
    num_pred_nodes: int
    num_true_nodes: int
    adjusted_jaccard_alpha: float = 0.1

    @property
    def adjusted_edge_jaccard(self) -> float:
        return self.edge.adjusted_jaccard(self.num_pred_nodes, self.num_true_nodes, self.adjusted_jaccard_alpha)

    @property
    def edge_weight(self) -> int:
        """Sample-size weight ``w_i = TP_i + FP_i + FN_i`` used for edge averaging."""
        return self.edge.tp + self.edge.fp + self.edge.fn


def evaluate_sample(
    sample_id: str,
    pred_graph: TrackingGraph,
    gt_graph: TrackingGraph,
    num_true_nodes: int,
    max_distance: float = 7.0,
    adjusted_jaccard_alpha: float = 0.1,
) -> SampleResult:
    """Evaluate one sample (video), computing both Jaccards from a shared node match.

    Parameters
    ----------
    sample_id:
        Identifier for the sample/video (matches the submission's ``dataset`` column).
    pred_graph, gt_graph:
        Predicted and ground-truth tracking graphs for this sample.
    num_true_nodes:
        Coarse estimate of the *total* number of true nodes in this sample
        (including ones the sparse ground truth doesn't annotate); required
        for the adjusted edge Jaccard penalty.
    max_distance:
        Node-matching radius in microns (7 um officially).
    adjusted_jaccard_alpha:
        Weighting coefficient for the adjusted edge Jaccard penalty (0.1 officially).
    """
    match = match_nodes(pred_graph.nodes.values(), gt_graph.nodes.values(), max_distance)
    edge_result = compute_edge_jaccard(pred_graph, gt_graph, max_distance, match=match)
    division_result = compute_division_jaccard(pred_graph, gt_graph, max_distance, match=match)
    return SampleResult(
        sample_id=sample_id,
        edge=edge_result,
        division=division_result,
        num_pred_nodes=len(pred_graph.nodes),
        num_true_nodes=num_true_nodes,
        adjusted_jaccard_alpha=adjusted_jaccard_alpha,
    )


@dataclass(frozen=True)
class AggregateScore:
    """Final micro-averaged score across all evaluated samples."""

    adjusted_edge_jaccard: float
    division_jaccard: float
    division_weight: float
    score: float
    num_samples: int


def aggregate_scores(samples: list[SampleResult], division_weight: float = 0.1) -> AggregateScore:
    """Micro-average per-sample results into the final combined score.

    ``score = adjusted_edge_jaccard + division_weight * division_jaccard``
    """
    if not samples:
        raise ValueError("at least one sample is required to compute an aggregate score")

    total_weight = sum(s.edge_weight for s in samples)
    if total_weight > 0:
        adjusted_edge_jaccard = sum(s.adjusted_edge_jaccard * s.edge_weight for s in samples) / total_weight
    else:
        adjusted_edge_jaccard = 1.0

    total_div_tp = sum(s.division.tp for s in samples)
    total_div_fp = sum(s.division.fp for s in samples)
    total_div_fn = sum(s.division.fn for s in samples)
    div_denom = total_div_tp + total_div_fp + total_div_fn
    division_jaccard = total_div_tp / div_denom if div_denom else 1.0

    score = adjusted_edge_jaccard + division_weight * division_jaccard
    return AggregateScore(
        adjusted_edge_jaccard=adjusted_edge_jaccard,
        division_jaccard=division_jaccard,
        division_weight=division_weight,
        score=score,
        num_samples=len(samples),
    )
