"""Node-level feature extraction and positional encodings."""

from __future__ import annotations

import numpy as np


def positional_encoding(
    zyx: np.ndarray,
    times: int | np.ndarray,
    image_shape: tuple[int, int, int],
    time_length: int,
    features_per_dimension: int = 8,
) -> np.ndarray:
    """Create sinusoidal embeddings for physical-model ``(t, z, y, x)`` locations."""
    coordinates = np.asarray(zyx, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("zyx must have shape (N, 3)")
    if len(image_shape) != 3 or any(value <= 0 for value in image_shape):
        raise ValueError("image_shape must contain three positive values")
    if time_length <= 0 or features_per_dimension <= 0 or features_per_dimension % 2:
        raise ValueError("time_length must be positive and features_per_dimension must be positive and even")
    time_values = np.broadcast_to(np.asarray(times, dtype=float), (len(coordinates),))
    normalized = np.column_stack(
        [time_values / time_length, coordinates / np.asarray(image_shape, dtype=float)]
    )
    frequencies = 2.0 ** np.arange(features_per_dimension // 2, dtype=float)
    angles = normalized[:, :, None] * frequencies[None, None, :] * np.pi
    encoded = np.concatenate([np.sin(angles), np.cos(angles)], axis=2)
    return encoded.reshape(len(coordinates), -1)


def sample_node_features(feature_volume: np.ndarray, zyx: np.ndarray) -> np.ndarray:
    """Sample channel-first ``(C, Z, Y, X)`` features at rounded, clipped centers."""
    features = np.asarray(feature_volume)
    coordinates = np.asarray(zyx, dtype=float)
    if features.ndim != 4:
        raise ValueError("feature_volume must have shape (C, Z, Y, X)")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("zyx must have shape (N, 3)")
    indices = np.rint(coordinates).astype(int)
    indices = np.clip(indices, 0, np.asarray(features.shape[1:]) - 1)
    return features[:, indices[:, 0], indices[:, 1], indices[:, 2]].T.copy()
