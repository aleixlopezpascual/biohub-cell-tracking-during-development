"""Synthetic tests for dependency-light volumetric blob detection."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.detection import LocalMaximaDetector, LocalMaximaDetectorConfig


def test_detects_separated_bright_blobs_in_descending_intensity_order() -> None:
    image = np.zeros((9, 9, 9), dtype=float)
    image[2, 3, 4] = 8.0
    image[6, 5, 1] = 10.0

    peaks = LocalMaximaDetector(LocalMaximaDetectorConfig(threshold=5.0, min_distance=2.0)).detect(image)

    assert peaks == [(6.0, 5.0, 1.0), (2.0, 3.0, 4.0)]


def test_threshold_and_minimum_distance_suppress_unwanted_peaks() -> None:
    image = np.zeros((9, 9, 9), dtype=float)
    image[4, 4, 4] = 10.0
    image[4, 4, 6] = 7.0
    image[1, 1, 1] = 2.0

    peaks = LocalMaximaDetector(LocalMaximaDetectorConfig(threshold=5.0, min_distance=3.0)).detect(image)

    assert peaks == [(4.0, 4.0, 4.0)]


def test_physical_voxel_scale_controls_anisotropic_suppression() -> None:
    image = np.zeros((8, 8, 8), dtype=float)
    image[2, 2, 2] = 10.0
    image[4, 2, 2] = 9.0

    peaks = LocalMaximaDetector(
        LocalMaximaDetectorConfig(
            threshold=5.0, min_distance=2.5, voxel_size_um=(3.0, 1.0, 1.0)
        )
    ).detect(image)

    assert peaks == [(2.0, 2.0, 2.0), (4.0, 2.0, 2.0)]


@pytest.mark.parametrize("min_distance", [0.0, -1.0])
def test_rejects_non_positive_minimum_distance(min_distance: float) -> None:
    with pytest.raises(ValueError, match="min_distance"):
        LocalMaximaDetectorConfig(min_distance=min_distance)
