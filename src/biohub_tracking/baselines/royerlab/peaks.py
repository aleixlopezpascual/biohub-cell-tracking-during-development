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


@dataclass(frozen=True)
class NodeCountCalibrator:
    """Ridge-fitted OOF mapping from image statistics to expected node count."""

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict non-negative integer node budgets for one or more volumes."""
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.coefficients):
            raise ValueError("features have the wrong shape for this calibrator")
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        standardized = (
            values - np.asarray(self.feature_mean)
        ) / np.asarray(self.feature_scale)
        log_counts = self.intercept + standardized @ np.asarray(self.coefficients)
        return np.maximum(0, np.rint(np.expm1(log_counts))).astype(int)


def fit_node_count_calibrator(
    features: np.ndarray,
    node_counts: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> NodeCountCalibrator:
    """Fit a compact log-count model using OOF volumes and image statistics."""
    values = np.asarray(features, dtype=float)
    counts = np.asarray(node_counts, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[0] != counts.size or counts.size < 2:
        raise ValueError("features and node_counts need at least two aligned samples")
    if not np.isfinite(values).all() or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("features and node_counts must be finite; counts must be non-negative")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be non-negative and finite")
    feature_mean = values.mean(axis=0)
    feature_scale = values.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    standardized = (values - feature_mean) / feature_scale
    design = np.column_stack([np.ones(counts.size), standardized])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ np.log1p(counts),
    )
    return NodeCountCalibrator(
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        coefficients=tuple(float(value) for value in parameters[1:]),
        intercept=float(parameters[0]),
    )


def select_peaks_to_budget(peaks: list[HeatmapPeak], estimated_nodes: int) -> list[HeatmapPeak]:
    """Select the highest-confidence detections for a calibrated node budget."""
    if estimated_nodes < 0:
        raise ValueError("estimated_nodes must be non-negative")
    ranked = sorted(peaks, key=lambda peak: (-peak.score, peak.zyx))
    return ranked[:estimated_nodes]


def refine_peak_centroid(
    image: np.ndarray,
    peak_zyx: tuple[int, int, int],
    *,
    radius_um: float,
    voxel_size_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Refine a peak by a non-negative local intensity center of mass."""
    volume = np.asarray(image, dtype=float)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("image must be a finite (Z, Y, X) array")
    if radius_um <= 0 or any(value <= 0 for value in voxel_size_um):
        raise ValueError("radius_um and voxel_size_um must be positive")
    peak = np.asarray(peak_zyx, dtype=int)
    if peak.shape != (3,) or np.any(peak < 0) or np.any(peak >= np.asarray(volume.shape)):
        raise IndexError("peak_zyx lies outside the image")
    radii = np.ceil(radius_um / np.asarray(voxel_size_um, dtype=float)).astype(int)
    lower = np.maximum(0, peak - radii)
    upper = np.minimum(np.asarray(volume.shape), peak + radii + 1)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    patch = volume[slices]
    weights = np.maximum(patch - float(np.min(patch)), 0.0)
    total = float(weights.sum())
    if total <= 0:
        return tuple(float(value) for value in peak)
    coordinates = np.indices(patch.shape, dtype=float)
    centroid = np.asarray(
        [(coordinates[axis] * weights).sum() / total for axis in range(3)], dtype=float
    ) + lower
    return tuple(float(value) for value in centroid)


def physical_pool_kernel(
    min_distance_um: float, voxel_size_um: tuple[float, float, float]
) -> tuple[int, int, int]:
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
    center = (np.asarray(kernel, dtype=float)[:, None, None, None] - 1) / 2
    offsets = np.indices(kernel, dtype=float) - center
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
                HeatmapPeak(
                    tuple(int(value) for value in candidate),
                    float(image[tuple(candidate)]),
                )
            )
    return selected
