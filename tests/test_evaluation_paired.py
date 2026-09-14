"""Tests for paired OOF candidate comparisons."""

from __future__ import annotations

import pandas as pd
import pytest

from biohub_tracking.evaluation import compare_per_dataset
from biohub_tracking.evaluation.paired import DEFAULT_PAIRED_METRICS


def _rows(scores: tuple[float, float]) -> pd.DataFrame:
    rows = []
    for dataset, score in zip(("a", "b"), scores):
        row = {metric: 0.0 for metric in DEFAULT_PAIRED_METRICS}
        row.update({"dataset": dataset, "score": score})
        rows.append(row)
    return pd.DataFrame(rows)


def test_compare_per_dataset_reports_paired_wins_and_losses() -> None:
    result = compare_per_dataset(_rows((0.8, 0.9)), _rows((0.85, 0.88)))
    assert result.summary["score_wins"] == 1
    assert result.summary["score_losses"] == 1
    assert result.per_dataset["score_delta"].tolist() == pytest.approx([0.05, -0.02])


def test_compare_per_dataset_rejects_different_oof_sets() -> None:
    candidate = _rows((0.8, 0.9)).replace({"b": "c"})
    with pytest.raises(ValueError, match="identical OOF datasets"):
        compare_per_dataset(_rows((0.8, 0.9)), candidate)
