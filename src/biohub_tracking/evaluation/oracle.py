"""Oracle headroom analysis separating detection and linking limitations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from biohub_tracking.evaluation.local_score import load_node_estimates, load_submission_graphs
from biohub_tracking.metrics.matching import match_nodes
from biohub_tracking.metrics.score import (
    AggregateScore,
    SampleResult,
    aggregate_scores,
    evaluate_sample,
)
from biohub_tracking.tracking.graph import TrackingGraph


@dataclass(frozen=True)
class OracleEvaluationResult:
    """Current score and two upper bounds for a set of validation datasets."""

    current: AggregateScore
    detection_constrained_oracle: AggregateScore
    perfect_oracle: AggregateScore
    per_dataset: pd.DataFrame
    summary: dict[str, object]


def oracle_links_for_detections(
    prediction: TrackingGraph,
    ground_truth: TrackingGraph,
    *,
    max_distance: float = 7.0,
) -> TrackingGraph:
    """Give current detections the best GT-consistent links they can support."""
    matched = match_nodes(
        prediction.nodes.values(), ground_truth.nodes.values(), max_distance=max_distance
    )
    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in matched.items()}
    oracle = TrackingGraph()
    for detection in prediction:
        oracle.add_node(detection)
    for gt_source, gt_target in ground_truth.edges:
        pred_source = gt_to_pred.get(gt_source)
        pred_target = gt_to_pred.get(gt_target)
        if pred_source is not None and pred_target is not None:
            oracle.add_edge(pred_source, pred_target)
    return oracle


def _score_row(
    dataset: str,
    current: SampleResult,
    detection_oracle: SampleResult,
    perfect: SampleResult,
) -> dict[str, object]:
    current_score = current.adjusted_edge_jaccard + 0.1 * current.division.jaccard
    detection_score = (
        detection_oracle.adjusted_edge_jaccard + 0.1 * detection_oracle.division.jaccard
    )
    perfect_score = perfect.adjusted_edge_jaccard + 0.1 * perfect.division.jaccard
    return {
        "dataset": dataset,
        "current_score": current_score,
        "detection_constrained_oracle_score": detection_score,
        "perfect_oracle_score": perfect_score,
        "linking_headroom": detection_score - current_score,
        "detection_headroom": perfect_score - detection_score,
        "current_edge_jaccard": current.edge.jaccard,
        "oracle_edge_jaccard": detection_oracle.edge.jaccard,
        "current_division_jaccard": current.division.jaccard,
        "oracle_division_jaccard": detection_oracle.division.jaccard,
        "num_pred_nodes": current.num_pred_nodes,
        "num_true_nodes": current.num_true_nodes,
    }


def analyze_oracle_headroom(
    submission: str | Path | pd.DataFrame | Mapping[str, TrackingGraph],
    ground_truth: str | Path | pd.DataFrame | Mapping[str, TrackingGraph],
    *,
    datasets: Sequence[str] | None = None,
    node_estimates: Mapping[str, int] | None = None,
    cv_pack_dir: str | Path | None = None,
    max_distance: float = 7.0,
    division_weight: float = 0.1,
) -> OracleEvaluationResult:
    """Measure linker headroom with fixed detections and remaining detection headroom."""
    pred_graphs = (
        dict(submission) if isinstance(submission, Mapping) else load_submission_graphs(submission)
    )
    gt_graphs = (
        dict(ground_truth)
        if isinstance(ground_truth, Mapping)
        else load_submission_graphs(ground_truth)
    )
    selected = (
        tuple(sorted(datasets))
        if datasets is not None
        else tuple(sorted(set(pred_graphs) & set(gt_graphs)))
    )
    if not selected:
        raise ValueError("no overlapping datasets to analyze")
    missing_pred = sorted(set(selected) - set(pred_graphs))
    missing_gt = sorted(set(selected) - set(gt_graphs))
    if missing_pred or missing_gt:
        raise ValueError(f"missing datasets: pred={missing_pred}, gt={missing_gt}")

    estimates = load_node_estimates(cv_pack_dir)
    if node_estimates:
        estimates.update({str(key): int(value) for key, value in node_estimates.items()})
    current_samples: list[SampleResult] = []
    detection_samples: list[SampleResult] = []
    perfect_samples: list[SampleResult] = []
    rows: list[dict[str, object]] = []
    for dataset in selected:
        pred = pred_graphs[dataset]
        gt = gt_graphs[dataset]
        estimated_nodes = estimates.get(dataset, len(gt.nodes))
        detection_oracle_graph = oracle_links_for_detections(pred, gt, max_distance=max_distance)
        current = evaluate_sample(
            dataset, pred, gt, num_true_nodes=estimated_nodes, max_distance=max_distance
        )
        detection_oracle = evaluate_sample(
            dataset,
            detection_oracle_graph,
            gt,
            num_true_nodes=estimated_nodes,
            max_distance=max_distance,
        )
        perfect = evaluate_sample(
            dataset, gt, gt, num_true_nodes=estimated_nodes, max_distance=max_distance
        )
        current_samples.append(current)
        detection_samples.append(detection_oracle)
        perfect_samples.append(perfect)
        rows.append(_score_row(dataset, current, detection_oracle, perfect))

    current_aggregate = aggregate_scores(current_samples, division_weight=division_weight)
    detection_aggregate = aggregate_scores(detection_samples, division_weight=division_weight)
    perfect_aggregate = aggregate_scores(perfect_samples, division_weight=division_weight)
    summary: dict[str, object] = {
        "current_score": current_aggregate.score,
        "detection_constrained_oracle_score": detection_aggregate.score,
        "perfect_oracle_score": perfect_aggregate.score,
        "linking_headroom": detection_aggregate.score - current_aggregate.score,
        "detection_headroom": perfect_aggregate.score - detection_aggregate.score,
        "priority": (
            "linking"
            if detection_aggregate.score - current_aggregate.score
            >= perfect_aggregate.score - detection_aggregate.score
            else "detection"
        ),
        "datasets": list(selected),
    }
    return OracleEvaluationResult(
        current=current_aggregate,
        detection_constrained_oracle=detection_aggregate,
        perfect_oracle=perfect_aggregate,
        per_dataset=pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True),
        summary=summary,
    )
