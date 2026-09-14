"""Paired per-volume comparison for controlled OOF experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_PAIRED_METRICS = (
    "score",
    "adjusted_edge_jaccard",
    "division_jaccard",
    "node_recall",
    "node_count_ratio",
    "edges_fragmented",
    "edges_lost_to_detection",
    "wrong_association_edges",
)


@dataclass(frozen=True)
class PairedEvaluationResult:
    """Aligned per-volume values, deltas, and aggregate win counts."""

    per_dataset: pd.DataFrame
    summary: dict[str, object]


def compare_per_dataset(
    baseline: pd.DataFrame | str | Path,
    candidate: pd.DataFrame | str | Path,
    *,
    metrics: Sequence[str] = DEFAULT_PAIRED_METRICS,
) -> PairedEvaluationResult:
    """Compare identical OOF datasets without hiding volume-level regressions."""
    first = pd.read_csv(baseline) if not isinstance(baseline, pd.DataFrame) else baseline.copy()
    second = pd.read_csv(candidate) if not isinstance(candidate, pd.DataFrame) else candidate.copy()
    required = {"dataset", *metrics}
    for label, frame in (("baseline", first), ("candidate", second)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} per-dataset CSV is missing columns: {missing}")
        if frame["dataset"].duplicated().any():
            raise ValueError(f"{label} contains duplicate datasets")
    baseline_datasets = set(first["dataset"].astype(str))
    candidate_datasets = set(second["dataset"].astype(str))
    if not baseline_datasets:
        raise ValueError("per-dataset comparisons must not be empty")
    if baseline_datasets != candidate_datasets:
        raise ValueError("baseline and candidate must contain identical OOF datasets")

    paired = first[["dataset", *metrics]].merge(
        second[["dataset", *metrics]],
        on="dataset",
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    ).sort_values("dataset")
    for metric in metrics:
        candidate_values = pd.to_numeric(paired[f"{metric}_candidate"], errors="raise")
        baseline_values = pd.to_numeric(paired[f"{metric}_baseline"], errors="raise")
        if not np.isfinite(candidate_values).all() or not np.isfinite(baseline_values).all():
            raise ValueError(f"metric {metric!r} must contain only finite values")
        paired[f"{metric}_delta"] = candidate_values - baseline_values
    score_delta = paired["score_delta"]
    summary: dict[str, object] = {
        "num_datasets": len(paired),
        "score_wins": int((score_delta > 0).sum()),
        "score_ties": int((score_delta == 0).sum()),
        "score_losses": int((score_delta < 0).sum()),
        "mean_deltas": {
            metric: float(paired[f"{metric}_delta"].mean()) for metric in metrics
        },
    }
    return PairedEvaluationResult(paired.reset_index(drop=True), summary)
