"""Deterministic 3D local-maxima detection for volumetric microscopy images.

The detector relies only on NumPy and SciPy, both core package dependencies.
It finds intensity maxima above a threshold and applies greedy non-maximum
suppression, so each compact bright blob produces at most one centroid.  A
voxel scale can be supplied to make the suppression radius anisotropy-aware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass(frozen=True)
class LocalMaximaDetectorConfig:
    """Parameters controlling :class:`LocalMaximaDetector`.

    ``min_distance`` is measured in voxels when ``voxel_size_um`` is absent,
    and in microns when it is supplied. ``threshold`` is applied directly to
    image intensities.
    """

    threshold: float = 0.0
    min_distance: float = 3.0
    voxel_size_um: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.min_distance <= 0:
            raise ValueError("min_distance must be positive")
        if self.voxel_size_um is not None and (
            len(self.voxel_size_um) != 3 or any(scale <= 0 for scale in self.voxel_size_um)
        ):
            raise ValueError("voxel_size_um must contain three positive (z, y, x) values")


class LocalMaximaDetector:
    """Detect centroids of bright 3D blobs using local maxima and suppression."""

    def __init__(self, config: LocalMaximaDetectorConfig | None = None) -> None:
        self.config = config or LocalMaximaDetectorConfig()

    def detect(self, volume: np.ndarray) -> list[tuple[float, float, float]]:
        """Return deterministic ``(z, y, x)`` peak coordinates from a 3D volume.

        Plateaus and nearby peaks are reduced to one point by selecting the
        brightest voxel first; exact intensity ties are resolved lexicographically
        by coordinate. Coordinates remain in voxel space even when a physical
        voxel scale is used for distance suppression.
        """
        image = np.asarray(volume)
        if image.ndim != 3:
            raise ValueError(f"LocalMaximaDetector expects a 3D volume, got ndim={image.ndim}")

        candidates = np.argwhere(
            (image >= self.config.threshold) & (image == maximum_filter(image, size=3))
        )
        if not len(candidates):
            return []

        values = image[tuple(candidates.T)]
        order = np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0], -values))
        scale = np.asarray(self.config.voxel_size_um or (1.0, 1.0, 1.0), dtype=float)
        selected: list[np.ndarray] = []
        for candidate in candidates[order]:
            if all(np.linalg.norm((candidate - peak) * scale) >= self.config.min_distance for peak in selected):
                selected.append(candidate)
        return [tuple(float(value) for value in peak) for peak in selected]
