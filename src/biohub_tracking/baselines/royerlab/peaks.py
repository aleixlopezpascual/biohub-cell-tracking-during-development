"""Physical-unit non-maximum suppression for 3D center heatmaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass(frozen=True)
class HeatmapPeakConfig:
    """Controls peak extraction from one ``(Z, Y, X)`` heatmap.

    ``min_distance_um`` is evaluated in physical space, including during final
    greedy suppression, so anisotropic microscopy voxels are handled correctly.
    """

    threshold: float = 0.5
    min_distance_um: float = 3.0
    voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.min_distance_um <= 0:
            raise ValueError("min_distance_um must be positive")
        if len(self.voxel_size_um) != 3 or any(value <= 0 for value in self.voxel_size_um):
            raise ValueError("voxel_size_um must contain positive (z, y, x) values")


@dataclass(frozen=True)
class HeatmapPeak:
    """One heatmap maximum in voxel coordinates with its confidence score."""

    zyx: tuple[int, int, int]
    score: float

    @property
    def position_voxel(self) -> tuple[int, int, int]:
        """Return the peak's ``(z, y, x)`` voxel coordinate."""
        return self.zyx


def physical_pool_kernel(min_distance_um: float, voxel_size_um: tuple[float, float, float]) -> tuple[int, int, int]:
    """Return odd max-pooling kernel widths spanning a physical NMS radius."""
    if min_distance_um <= 0 or any(value <= 0 for value in voxel_size_um):
        raise ValueError("min_distance_um and voxel_size_um must be positive")
    radii = np.ceil(min_distance_um / np.asarray(voxel_size_um, dtype=float)).astype(int)
    return tuple(int(2 * radius + 1) for radius in radii)


def _physical_pool_footprint(
    min_distance_um: float, voxel_size_um: tuple[float, float, float]
) -> np.ndarray:
    """Build an ellipsoidal pooling footprint matching a physical NMS radius."""
    kernel = physical_pool_kernel(min_distance_um, voxel_size_um)
    offsets = np.indices(kernel, dtype=float) - (np.asarray(kernel, dtype=float)[:, None, None, None] - 1) / 2
    physical_offsets = offsets * np.asarray(voxel_size_um, dtype=float)[:, None, None, None]
    return np.sum(physical_offsets**2, axis=0) <= min_distance_um**2


def extract_heatmap_peaks(heatmap: np.ndarray, config: HeatmapPeakConfig) -> list[HeatmapPeak]:
    """Extract deterministically ordered, physically separated local maxima.

    Returned coordinates remain in voxel space because this function operates
    at the detector boundary.  Convert them to microns before graph tracking.
    """
    image = np.asarray(heatmap)
    if image.ndim != 3:
        raise ValueError(f"heatmap must have shape (Z, Y, X), got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("heatmap must contain only finite values")

    local_maximum = image == maximum_filter(
        image,
        footprint=_physical_pool_footprint(config.min_distance_um, config.voxel_size_um),
    )
    candidates = np.argwhere(local_maximum & (image >= config.threshold))
    values = image[tuple(candidates.T)]
    order = np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0], -values))
    scale = np.asarray(config.voxel_size_um, dtype=float)
    selected: list[HeatmapPeak] = []
    for candidate in candidates[order]:
        if all(
            np.linalg.norm((candidate - np.asarray(peak.zyx)) * scale) >= config.min_distance_um
            for peak in selected
        ):
            selected.append(
                HeatmapPeak(tuple(int(value) for value in candidate), float(image[tuple(candidate)]))
            )
    return selected
