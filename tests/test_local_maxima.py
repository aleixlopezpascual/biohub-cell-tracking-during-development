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


def test_difference_of_gaussians_config_validation() -> None:
    # Rejects invalid dog_sigmas
    with pytest.raises(ValueError, match="dog_sigmas"):
        LocalMaximaDetectorConfig(use_dog=True, dog_sigmas=[-1.0, 2.0])
    with pytest.raises(ValueError, match="dog_sigmas"):
        LocalMaximaDetectorConfig(use_dog=True, dog_sigmas="invalid")

    # Rejects invalid dog_ratio
    with pytest.raises(ValueError, match="dog_ratio"):
        LocalMaximaDetectorConfig(use_dog=True, dog_ratio=0.9)


def test_detects_blobs_with_difference_of_gaussians() -> None:
    # Create a smooth Gaussian blob on a zero background.
    # Difference of Gaussians should easily identify the center.
    image = np.zeros((15, 15, 15), dtype=float)
    # Put a Gaussian shape at center (7, 7, 7)
    for z in range(15):
        for y in range(15):
            for x in range(15):
                dist_sq = (z - 7.0)**2 + (y - 7.0)**2 + (x - 7.0)**2
                image[z, y, x] = np.exp(-dist_sq / 8.0) * 10.0  # sigma = 2

    # DoG should successfully detect the peak at (7.0, 7.0, 7.0)
    detector = LocalMaximaDetector(
        LocalMaximaDetectorConfig(
            use_dog=True,
            dog_sigmas=[1.5, 2.0],
            threshold=0.1,
            min_distance=2.0,
        )
    )
    peaks = detector.detect(image)
    assert len(peaks) == 1
    assert peaks[0] == (7.0, 7.0, 7.0)

