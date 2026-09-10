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


def test_local_maxima_supports_subpixel_refinement() -> None:
    # A local peak at (4, 4, 4) with 10.0 intensity.
    # An asymmetric neighbor at (4, 4, 5) with 5.0 intensity.
    # Total center of mass should pull the X coordinate slightly above 4.0!
    image = np.zeros((9, 9, 9), dtype=float)
    image[4, 4, 4] = 10.0
    image[4, 4, 5] = 5.0

    # Run with refinement disabled (stays at integer 4.0, 4.0, 4.0)
    peaks_off = LocalMaximaDetector(
        LocalMaximaDetectorConfig(threshold=5.0, min_distance=2.0, use_subpixel_refinement=False)
    ).detect(image)
    assert peaks_off == [(4.0, 4.0, 4.0)]

    # Run with refinement enabled (pulled slightly towards 5)
    peaks_on = LocalMaximaDetector(
        LocalMaximaDetectorConfig(threshold=5.0, min_distance=2.0, use_subpixel_refinement=True)
    ).detect(image)
    assert len(peaks_on) == 1
    pz, py, px = peaks_on[0]
    assert pz == 4.0
    assert py == 4.0
    assert px > 4.0  # Pulled towards the right
    assert px < 5.0  # But still closer to the peak
