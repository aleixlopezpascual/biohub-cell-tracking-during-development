from __future__ import annotations

import pytest

from scripts.probe_detection_thresholds import select_threshold


def test_threshold_probe_selects_node_ratio_near_one_before_recall() -> None:
    rows = [
        {"threshold": 0.5, "node_count_ratio": 1.8, "annotated_node_recall": 0.95},
        {"threshold": 0.7, "node_count_ratio": 1.05, "annotated_node_recall": 0.82},
        {"threshold": 0.9, "node_count_ratio": 0.4, "annotated_node_recall": 0.40},
    ]
    assert select_threshold(rows) == pytest.approx(0.7)


def test_threshold_probe_rejects_all_empty_candidates() -> None:
    with pytest.raises(ValueError, match="no detections"):
        select_threshold(
            [{"threshold": 0.9, "node_count_ratio": 0.0, "annotated_node_recall": 0.0}]
        )
