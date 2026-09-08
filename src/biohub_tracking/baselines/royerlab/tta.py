"""Reversible XY dihedral test-time augmentation bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SpatialTransform:
    """A transform applied on the final two axes: rotate, then optionally flip X."""

    rotations_ccw: int
    flip_x: bool = False

    def __post_init__(self) -> None:
        if self.rotations_ccw not in range(4):
            raise ValueError("rotations_ccw must be in [0, 3]")


XY_D4_TRANSFORMS: tuple[SpatialTransform, ...] = tuple(
    SpatialTransform(rotations_ccw=rotation, flip_x=flip)
    for flip in (False, True)
    for rotation in range(4)
)


def apply_spatial_transform(array: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    """Apply one reversible XY transform to heatmaps or channel-first tensors."""
    result = np.rot90(np.asarray(array), k=transform.rotations_ccw, axes=(-2, -1))
    return np.flip(result, axis=-1) if transform.flip_x else result


def invert_spatial_transform(array: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    """Map a prediction from its augmented frame back to the original frame."""
    result = np.flip(np.asarray(array), axis=-1) if transform.flip_x else np.asarray(array)
    return np.rot90(result, k=-transform.rotations_ccw, axes=(-2, -1))


def apply_xy_d4_tta(array: np.ndarray) -> list[tuple[SpatialTransform, np.ndarray]]:
    """Return all eight XY flip/rotation views with their inverse bookkeeping."""
    return [(transform, apply_spatial_transform(array, transform)) for transform in XY_D4_TRANSFORMS]


def invert_xy_d4_tta(
    predictions: Sequence[tuple[SpatialTransform, np.ndarray]],
) -> np.ndarray:
    """Invert and stack predictions produced for :func:`apply_xy_d4_tta` views."""
    if not predictions:
        raise ValueError("predictions must contain at least one TTA view")
    restored = [invert_spatial_transform(prediction, transform) for transform, prediction in predictions]
    first_shape = restored[0].shape
    if any(value.shape != first_shape for value in restored):
        raise ValueError("all inverted TTA predictions must have the same shape")
    return np.stack(restored)
