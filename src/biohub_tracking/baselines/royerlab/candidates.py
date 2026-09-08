"""Probability- and physics-gated temporal edge candidate generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub_tracking.tracking.graph import Detection, NodeId


@dataclass(frozen=True)
class EdgeCandidateConfig:
    """Filters learned source-to-target association probabilities."""

    strong_threshold: float = 0.5
    min_threshold: float = 0.2
    top_k_parents: int = 3
    max_distance_um: float = 12.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_threshold <= self.strong_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= min_threshold <= strong_threshold <= 1")
        if self.top_k_parents < 1 or self.max_distance_um <= 0:
            raise ValueError("top_k_parents and max_distance_um must be positive")


@dataclass(frozen=True)
class CandidateEdge:
    """A directed, adjacent-frame candidate with model probability and distance."""

    source_id: NodeId
    target_id: NodeId
    probability: float
    distance_um: float


def generate_edge_candidates(
    sources: list[Detection],
    targets: list[Detection],
    probabilities: np.ndarray,
    config: EdgeCandidateConfig,
) -> list[CandidateEdge]:
    """Keep strong or target-wise top-k edges that satisfy a physical gate."""
    scores = np.asarray(probabilities, dtype=float)
    expected_shape = (len(sources), len(targets))
    if scores.shape != expected_shape:
        raise ValueError(f"probabilities must have shape {expected_shape}, got {scores.shape}")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    pairs = {tuple(int(value) for value in pair) for pair in np.argwhere(scores >= config.strong_threshold)}
    for target_index in range(len(targets)):
        ranked_sources = sorted(range(len(sources)), key=lambda index: (-scores[index, target_index], index))
        for source_index in ranked_sources[: config.top_k_parents]:
            if scores[source_index, target_index] >= config.min_threshold:
                pairs.add((source_index, target_index))

    candidates: list[CandidateEdge] = []
    for source_index, target_index in pairs:
        source, target = sources[source_index], targets[target_index]
        if target.frame <= source.frame:
            raise ValueError("candidate edges must point strictly forward in time")
        distance = float(np.linalg.norm(np.asarray(source.position) - np.asarray(target.position)))
        if distance <= config.max_distance_um:
            candidates.append(CandidateEdge(source.id, target.id, float(scores[source_index, target_index]), distance))
    return sorted(candidates, key=lambda edge: (-edge.probability, repr(edge.source_id), repr(edge.target_id)))
