"""Integration tests for the model-free baseline detection-to-submission path."""

from __future__ import annotations

import numpy as np

from biohub_tracking.data import OMEZarrVolume
from biohub_tracking.pipeline import BaselinePipelineConfig, infer_volume
from biohub_tracking.submission import SUBMISSION_COLUMNS, export_submission
from biohub_tracking.tracking import TrackerConfig


def test_infers_physical_centroids_links_frames_and_exports_submission(tmp_path) -> None:
    frames = np.zeros((2, 5, 7, 9), dtype=float)
    frames[0, 2, 3, 4] = 10.0
    frames[1, 2, 3, 5] = 10.0
    volume = OMEZarrVolume.from_array(frames, voxel_size_um=(2.0, 1.0, 0.5))

    graph = infer_volume(
        volume,
        BaselinePipelineConfig(
            threshold=5.0,
            min_distance=1.0,
            tracker=TrackerConfig(max_link_distance_um=2.0),
        ),
    )

    assert [(node.frame, node.position) for node in graph] == [
        (0, (4.0, 3.0, 2.0)),
        (1, (4.0, 3.0, 2.5)),
    ]
    assert graph.edges == {(0, 1)}

    output = tmp_path / "submission.csv"
    exported = export_submission({"synthetic": graph}, output)
    assert list(exported.columns) == SUBMISSION_COLUMNS
    assert list(exported["row_type"]) == ["node", "node", "edge"]
    assert output.exists()
