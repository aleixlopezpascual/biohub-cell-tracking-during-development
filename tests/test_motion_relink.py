"""Synthetic tests for the physical-unit EMA motion relinker."""

from __future__ import annotations

import pytest

from biohub_tracking.tracking.graph import Detection
from biohub_tracking.tracking.motion_relink import MotionRelinkConfig, motion_relink


def test_motion_relink_uses_velocity_prediction_over_nearest_raw_node() -> None:
    detections = {
        1: Detection(id=1, frame=0, z=0.0, y=0.0, x=0.0),
        2: Detection(id=2, frame=1, z=0.0, y=0.0, x=10.0),
        3: Detection(id=3, frame=2, z=0.0, y=0.0, x=11.0),
        4: Detection(id=4, frame=2, z=0.0, y=0.0, x=14.0),
    }

    result = motion_relink(detections)

    assert [(edge.source_id, edge.target_id) for edge in result.edges] == [
        (1, 2),
        (2, 4),
    ]


def test_motion_relink_uses_learned_score_to_resolve_geometric_choice() -> None:
    detections = {
        0: Detection(id=0, frame=0, z=0.0, y=0.0, x=0.0),
        1: Detection(id=1, frame=1, z=0.0, y=0.0, x=1.0),
        2: Detection(id=2, frame=1, z=0.0, y=0.0, x=2.0),
    }
    config = MotionRelinkConfig(velocity_weight=0.0, learned_edge_bonus=2.0)

    result = motion_relink(detections, {(0, 2): 1.0}, config)

    assert [(edge.source_id, edge.target_id) for edge in result.edges] == [(0, 2)]
    assert result.edges[0].edge_probability == 1.0


def test_motion_relink_does_not_bridge_missing_frames() -> None:
    detections = {
        0: Detection(id=0, frame=0, z=0.0, y=0.0, x=0.0),
        1: Detection(id=1, frame=2, z=0.0, y=0.0, x=1.0),
    }

    result = motion_relink(detections)

    assert result.edges == ()
    assert result.stats.frames_processed == 0


def test_motion_relink_config_rejects_nonpositive_gates() -> None:
    with pytest.raises(ValueError, match="0 < tight <= relaxed"):
        MotionRelinkConfig(tight_gate_um=0.0)
