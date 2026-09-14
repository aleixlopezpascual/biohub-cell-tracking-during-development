"""Motion-aware edge features, hard negatives, and probability calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from biohub_tracking.tracking.graph import Detection, NodeId


def edge_motion_features(
    sources: Sequence[Detection],
    targets: Sequence[Detection],
    *,
    predecessors: Mapping[NodeId, Detection] | None = None,
) -> np.ndarray:
    """Return physical displacement and constant-velocity residual per edge.

    The result has shape ``(N_source, N_target, 8)`` with ``dz,dy,dx``,
    distance, velocity-residual ``dz,dy,dx``, and residual distance. A source
    without predecessor uses zero velocity, making the residual equal to the
    direct displacement.
    """
    if any(target.frame <= source.frame for source in sources for target in targets):
        raise ValueError("targets must be later in time than every source")
    output = np.empty((len(sources), len(targets), 8), dtype=np.float32)
    for source_index, source in enumerate(sources):
        source_position = np.asarray(source.position, dtype=float)
        predecessor = predecessors.get(source.id) if predecessors is not None else None
        if predecessor is not None:
            if predecessor.frame >= source.frame:
                raise ValueError("predecessors must be earlier than their source")
            delta_t = source.frame - predecessor.frame
            velocity = (source_position - np.asarray(predecessor.position, dtype=float)) / delta_t
        else:
            velocity = np.zeros(3, dtype=float)
        for target_index, target in enumerate(targets):
            target_delta_t = target.frame - source.frame
            displacement = np.asarray(target.position, dtype=float) - source_position
            residual = displacement - velocity * target_delta_t
            output[source_index, target_index] = np.concatenate(
                [displacement, [np.linalg.norm(displacement)], residual, [np.linalg.norm(residual)]]
            )
    return output


def mine_hard_negative_pairs(
    sources: Sequence[Detection],
    targets: Sequence[Detection],
    positive_pairs: set[tuple[NodeId, NodeId]],
    *,
    max_distance_um: float,
    max_negatives_per_target: int = 3,
    probabilities: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Select nearby or high-scoring incorrect associations deterministically."""
    if max_distance_um <= 0 or max_negatives_per_target < 1:
        raise ValueError("distance and negative count must be positive")
    scores = None if probabilities is None else np.asarray(probabilities, dtype=float)
    expected = (len(sources), len(targets))
    if scores is not None:
        if scores.shape != expected or not np.isfinite(scores).all():
            raise ValueError(f"probabilities must be finite with shape {expected}")
    selected: list[tuple[int, int]] = []
    for target_index, target in enumerate(targets):
        candidates: list[tuple[float, float, int]] = []
        for source_index, source in enumerate(sources):
            if (source.id, target.id) in positive_pairs:
                continue
            displacement = np.asarray(source.position, dtype=float) - np.asarray(
                target.position, dtype=float
            )
            distance = float(np.linalg.norm(displacement))
            if distance <= max_distance_um:
                model_priority = (
                    -float(scores[source_index, target_index]) if scores is not None else 0.0
                )
                candidates.append((model_priority, distance, source_index))
        candidates.sort()
        selected.extend(
            (source_index, target_index)
            for _score, _distance, source_index in candidates[:max_negatives_per_target]
        )
    return selected


def fuse_bidirectional_probabilities(
    forward: np.ndarray,
    reverse_aligned: np.ndarray,
    *,
    reverse_weight: float = 0.5,
    mode: str = "harmonic",
) -> np.ndarray:
    """Fuse aligned forward/reverse edge probabilities without logit-scale drift."""
    first = np.asarray(forward, dtype=float)
    second = np.asarray(reverse_aligned, dtype=float)
    if first.shape != second.shape:
        raise ValueError("forward and reverse probabilities must have the same shape")
    if not 0 <= reverse_weight <= 1:
        raise ValueError("reverse_weight must be in [0, 1]")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("probabilities must be finite")
    if np.any((first < 0) | (first > 1) | (second < 0) | (second > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    forward_weight = 1.0 - reverse_weight
    if mode == "arithmetic":
        return forward_weight * first + reverse_weight * second
    epsilon = np.finfo(float).eps
    if mode == "geometric":
        return np.exp(
            forward_weight * np.log(np.clip(first, epsilon, 1))
            + reverse_weight * np.log(np.clip(second, epsilon, 1))
        )
    if mode == "harmonic":
        denominator = forward_weight / np.clip(first, epsilon, 1) + reverse_weight / np.clip(
            second, epsilon, 1
        )
        return 1.0 / denominator
    raise ValueError("mode must be 'harmonic', 'geometric', or 'arithmetic'")


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature calibration to binary probabilities."""
    values = np.asarray(probabilities, dtype=float)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    epsilon = np.finfo(float).eps
    clipped = np.clip(values, epsilon, 1 - epsilon)
    logits = np.log(clipped) - np.log1p(-clipped)
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fit one OOF temperature by minimizing binary cross-entropy."""
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    truth = np.asarray(labels, dtype=float).reshape(-1)
    if values.shape != truth.shape or values.size == 0:
        raise ValueError("probabilities and labels must have equal non-empty shapes")
    if np.any((truth != 0) & (truth != 1)):
        raise ValueError("labels must be binary")

    def objective(log_temperature: float) -> float:
        calibrated = apply_temperature(values, float(np.exp(log_temperature)))
        epsilon = np.finfo(float).eps
        calibrated = np.clip(calibrated, epsilon, 1 - epsilon)
        return float(-np.mean(truth * np.log(calibrated) + (1 - truth) * np.log1p(-calibrated)))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"temperature calibration failed: {result.message}")
    return float(np.exp(result.x))
