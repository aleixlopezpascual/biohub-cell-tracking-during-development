"""Submission-format local scoring and diagnostics.

This module validates strict Kaggle submission CSVs, converts each dataset's
rows into a :class:`TrackingGraph`, scores against ground-truth graphs with the
repository's Biohub metric implementation, and reports enough diagnostics to
make local CV useful before submitting to Kaggle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from biohub_tracking.metrics.matching import match_nodes
from biohub_tracking.metrics.score import AggregateScore, SampleResult, aggregate_scores, evaluate_sample
from biohub_tracking.submission.export import SUBMISSION_COLUMNS
from biohub_tracking.tracking.graph import Detection, NodeId, TrackingGraph


@dataclass(frozen=True)
class LocalEvaluationResult:
    """Local evaluation outputs for one candidate submission."""

    aggregate: AggregateScore
    per_dataset: pd.DataFrame
    summary: dict[str, object]
    scorer_provenance: str


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"submission is missing required columns: {missing}")


def _is_unused(value: object) -> bool:
    if pd.isna(value):
        return True
    try:
        return float(value) == -1.0
    except (TypeError, ValueError):
        return False


def _node_id(value: object) -> NodeId:
    if pd.isna(value):
        raise ValueError("node id cannot be missing")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if as_float.is_integer():
        return int(as_float)
    return str(value)


def graph_from_submission_rows(
    rows: pd.DataFrame,
    *,
    dataset: str,
    require_consecutive_edges: bool = True,
) -> TrackingGraph:
    """Convert one dataset's strict submission rows into a graph.

    Coordinates are interpreted as physical units because that is the internal
    convention used by tracking/metrics in this repository.
    """
    _require_columns(rows, SUBMISSION_COLUMNS)
    graph = TrackingGraph()
    node_rows = rows[rows["row_type"] == "node"].copy()
    edge_rows = rows[rows["row_type"] == "edge"].copy()
    invalid_types = sorted(set(rows["row_type"].astype(str)) - {"node", "edge"})
    if invalid_types:
        raise ValueError(f"{dataset}: invalid row_type values: {invalid_types}")

    seen: set[NodeId] = set()
    for row in node_rows.itertuples(index=False):
        node_id = _node_id(getattr(row, "node_id"))
        if node_id in seen:
            raise ValueError(f"{dataset}: duplicate node_id {node_id!r}")
        seen.add(node_id)
        if not _is_unused(getattr(row, "source_id")) or not _is_unused(getattr(row, "target_id")):
            raise ValueError(f"{dataset}: node row {node_id!r} must use -1 source_id/target_id sentinels")
        coords = [float(getattr(row, axis)) for axis in ("z", "y", "x")]
        if not np.isfinite(coords).all():
            raise ValueError(f"{dataset}: node row {node_id!r} has nonfinite coordinates")
        graph.add_node(
            Detection(
                id=node_id,
                frame=int(getattr(row, "t")),
                z=coords[0],
                y=coords[1],
                x=coords[2],
            )
        )

    for row in edge_rows.itertuples(index=False):
        if not all(_is_unused(getattr(row, column)) for column in ("node_id", "t", "z", "y", "x")):
            raise ValueError(f"{dataset}: edge rows must use -1 sentinels for node fields")
        source = _node_id(getattr(row, "source_id"))
        target = _node_id(getattr(row, "target_id"))
        if source not in graph.nodes or target not in graph.nodes:
            raise ValueError(f"{dataset}: dangling edge {source!r}->{target!r}")
        source_t = graph.nodes[source].frame
        target_t = graph.nodes[target].frame
        if require_consecutive_edges and target_t != source_t + 1:
            raise ValueError(f"{dataset}: edge {source!r}->{target!r} is not consecutive in time")
        graph.add_edge(source, target)
    return graph


def load_submission_graphs(
    submission: str | Path | pd.DataFrame,
    *,
    require_consecutive_edges: bool = True,
) -> dict[str, TrackingGraph]:
    """Load a submission CSV/DataFrame into `dataset -> TrackingGraph`."""
    frame = pd.read_csv(submission) if not isinstance(submission, pd.DataFrame) else submission.copy()
    _require_columns(frame, SUBMISSION_COLUMNS)
    if frame.empty:
        raise ValueError("submission contains no rows")
    graphs: dict[str, TrackingGraph] = {}
    for dataset, group in frame.groupby("dataset", sort=True):
        graphs[str(dataset)] = graph_from_submission_rows(
            group, dataset=str(dataset), require_consecutive_edges=require_consecutive_edges
        )
    return graphs


def _node_recall(pred: TrackingGraph, gt: TrackingGraph, max_distance: float) -> tuple[int, int, float]:
    matched = match_nodes(pred.nodes.values(), gt.nodes.values(), max_distance=max_distance)
    matched_gt = len(set(matched.values()))
    total_gt = len(gt.nodes)
    return matched_gt, total_gt, matched_gt / total_gt if total_gt else 1.0


def _diagnostics(pred: TrackingGraph, gt: TrackingGraph, max_distance: float) -> dict[str, int | float]:
    match = match_nodes(pred.nodes.values(), gt.nodes.values(), max_distance=max_distance)
    pred_to_gt = match
    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in match.items()}

    missed_gt_nodes = len(set(gt.nodes) - set(gt_to_pred))
    spurious_pred_nodes = len(set(pred.nodes) - set(pred_to_gt))
    pred_edges = set(pred.edges)
    gt_edges = set(gt.edges)
    recovered_edges = 0
    fragmented_edges = 0
    edges_lost_to_detection = 0
    wrong_association_edges = 0

    gt_successors = gt.targets_of_source()
    for gt_source, gt_target in gt_edges:
        pred_source = gt_to_pred.get(gt_source)
        pred_target = gt_to_pred.get(gt_target)
        if pred_source is None or pred_target is None:
            edges_lost_to_detection += 1
        elif (pred_source, pred_target) in pred_edges:
            recovered_edges += 1
        else:
            fragmented_edges += 1

    for pred_source, pred_target in pred_edges:
        gt_source = pred_to_gt.get(pred_source)
        gt_target = pred_to_gt.get(pred_target)
        if gt_source is not None and gt_target is not None and gt_target not in gt_successors.get(gt_source, set()):
            wrong_association_edges += 1

    return {
        "missed_gt_nodes": missed_gt_nodes,
        "spurious_pred_nodes": spurious_pred_nodes,
        "edges_recovered": recovered_edges,
        "edges_fragmented": fragmented_edges,
        "edges_lost_to_detection": edges_lost_to_detection,
        "wrong_association_edges": wrong_association_edges,
    }


def _sample_to_row(sample: SampleResult, pred: TrackingGraph, gt: TrackingGraph, max_distance: float) -> dict[str, object]:
    matched_gt, total_gt, node_recall = _node_recall(pred, gt, max_distance)
    row: dict[str, object] = {
        "dataset": sample.sample_id,
        "score": sample.adjusted_edge_jaccard + 0.1 * sample.division.jaccard,
        "edge_jaccard": sample.edge.jaccard,
        "adjusted_edge_jaccard": sample.adjusted_edge_jaccard,
        "division_jaccard": sample.division.jaccard,
        "edge_tp": sample.edge.tp,
        "edge_fp": sample.edge.fp,
        "edge_fn": sample.edge.fn,
        "division_tp": sample.division.tp,
        "division_fp": sample.division.fp,
        "division_fn": sample.division.fn,
        "node_recall": node_recall,
        "matched_gt_nodes": matched_gt,
        "num_gt_nodes": total_gt,
        "num_pred_nodes": sample.num_pred_nodes,
        "num_true_nodes": sample.num_true_nodes,
        "node_count_ratio": sample.num_pred_nodes / sample.num_true_nodes if sample.num_true_nodes else float("nan"),
        "edge_weight": sample.edge_weight,
    }
    row.update(_diagnostics(pred, gt, max_distance))
    return row


def load_node_estimates(cv_pack_dir: str | Path | None) -> dict[str, int]:
    """Load per-volume true-node estimates from the Local CV Pack when present."""
    if cv_pack_dir is None:
        return {}
    path = Path(cv_pack_dir) / "gt_per_volume_stats.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    dataset_col = "volume" if "volume" in frame.columns else "dataset" if "dataset" in frame.columns else None
    estimate_col = "est" if "est" in frame.columns else "estimated_number_of_nodes" if "estimated_number_of_nodes" in frame.columns else None
    if dataset_col is None or estimate_col is None:
        return {}
    return {str(row[dataset_col]): int(round(float(row[estimate_col]))) for _, row in frame.iterrows() if pd.notna(row[estimate_col])}


def score_submission(
    submission: str | Path | pd.DataFrame,
    ground_truth: str | Path | pd.DataFrame | Mapping[str, TrackingGraph],
    *,
    datasets: Sequence[str] | None = None,
    node_estimates: Mapping[str, int] | None = None,
    cv_pack_dir: str | Path | None = None,
    max_distance: float = 7.0,
    division_weight: float = 0.1,
) -> LocalEvaluationResult:
    """Score a candidate submission against submission-style or graph GT."""
    pred_graphs = load_submission_graphs(submission)
    if isinstance(ground_truth, Mapping):
        gt_graphs = dict(ground_truth)
    else:
        gt_graphs = load_submission_graphs(ground_truth)

    selected = tuple(sorted(datasets)) if datasets is not None else tuple(sorted(set(pred_graphs) & set(gt_graphs)))
    if not selected:
        raise ValueError("no overlapping datasets to score")
    missing_pred = sorted(set(selected) - set(pred_graphs))
    missing_gt = sorted(set(selected) - set(gt_graphs))
    if missing_pred or missing_gt:
        raise ValueError(f"missing datasets: pred={missing_pred}, gt={missing_gt}")

    estimates = dict(load_node_estimates(cv_pack_dir))
    if node_estimates:
        estimates.update({str(k): int(v) for k, v in node_estimates.items()})

    samples: list[SampleResult] = []
    rows: list[dict[str, object]] = []
    for dataset in selected:
        pred = pred_graphs[dataset]
        gt = gt_graphs[dataset]
        num_true_nodes = estimates.get(dataset, len(gt.nodes))
        sample = evaluate_sample(dataset, pred, gt, num_true_nodes=num_true_nodes, max_distance=max_distance)
        samples.append(sample)
        rows.append(_sample_to_row(sample, pred, gt, max_distance))

    aggregate = aggregate_scores(samples, division_weight=division_weight)
    per_dataset = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    summary: dict[str, object] = {
        "score": aggregate.score,
        "adjusted_edge_jaccard": aggregate.adjusted_edge_jaccard,
        "division_jaccard": aggregate.division_jaccard,
        "division_weight": aggregate.division_weight,
        "num_samples": aggregate.num_samples,
        "edge_tp": int(per_dataset["edge_tp"].sum()),
        "edge_fp": int(per_dataset["edge_fp"].sum()),
        "edge_fn": int(per_dataset["edge_fn"].sum()),
        "division_tp": int(per_dataset["division_tp"].sum()),
        "division_fp": int(per_dataset["division_fp"].sum()),
        "division_fn": int(per_dataset["division_fn"].sum()),
        "node_recall": float(
            per_dataset["matched_gt_nodes"].sum() / per_dataset["num_gt_nodes"].sum()
            if per_dataset["num_gt_nodes"].sum()
            else 1.0
        ),
        "datasets": list(selected),
    }
    return LocalEvaluationResult(
        aggregate=aggregate,
        per_dataset=per_dataset,
        summary=summary,
        scorer_provenance="biohub_tracking_internal",
    )
