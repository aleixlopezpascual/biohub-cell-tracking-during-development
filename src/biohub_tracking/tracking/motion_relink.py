"""Deterministic EMA-velocity edge relinking in physical micron coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_tracking.tracking.graph import Detection


@dataclass(frozen=True)
class MotionRelinkConfig:
    """Distance gates and scoring weights for EMA motion relinking."""

    tight_gate_um: float = 6.0
    relaxed_gate_um: float = 10.0
    velocity_weight: float = 0.5
    learned_edge_bonus: float = 1.0
    max_nodes_per_frame: int = 2600

    def __post_init__(self) -> None:
        numeric = (
            self.tight_gate_um,
            self.relaxed_gate_um,
            self.velocity_weight,
            self.learned_edge_bonus,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("motion relink distances and weights must be finite")
        if self.tight_gate_um <= 0 or self.relaxed_gate_um < self.tight_gate_um:
            raise ValueError("motion relink gates must satisfy 0 < tight <= relaxed")
        if self.velocity_weight < 0 or self.learned_edge_bonus < 0:
            raise ValueError("motion relink weights must be non-negative")
        if (
            isinstance(self.max_nodes_per_frame, bool)
            or not isinstance(self.max_nodes_per_frame, int)
            or self.max_nodes_per_frame < 1
        ):
            raise ValueError("max_nodes_per_frame must be a positive integer")


@dataclass(frozen=True)
class MotionRelinkEdge:
    """One selected consecutive-frame association."""

    source_id: int
    target_id: int
    edge_probability: float
    distance_um: float
    motion_distance_um: float
    pass_name: Literal["tight", "relaxed"]


@dataclass(frozen=True)
class MotionRelinkStats:
    """Counters describing the relinking pass."""

    frames_processed: int = 0
    tight_edges: int = 0
    relaxed_edges: int = 0
    skipped_large_frame: bool = False


@dataclass(frozen=True)
class MotionRelinkResult:
    """Relinked edges and deterministic run counters."""

    edges: tuple[MotionRelinkEdge, ...]
    stats: MotionRelinkStats


def _learned_probability(value: object) -> float:
    """Normalize a probability or logit; invalid values contribute no bonus."""
    if not isinstance(value, (str, int, float, np.integer, np.floating)):
        return 0.0
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(probability):
        return 0.0
    if probability < 0.0 or probability > 1.0:
        probability = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, probability))))
    return float(np.clip(probability, 0.0, 1.0))


def motion_relink(
    detections: Mapping[int, Detection],
    learned_edge_probs: Mapping[tuple[int, int], float] | None = None,
    config: MotionRelinkConfig | None = None,
) -> MotionRelinkResult:
    """Assign nodes in adjacent frames using a smoothed velocity prediction.

    Detections must already be in physical microns. The two-pass Hungarian
    assignment first uses a tight distance gate, then a relaxed gate for the
    remaining nodes. Learned edge scores break geometric ties; the selected
    velocity is updated with an EMA (0.6 new, 0.4 previous), matching the
    staged Biohub notebook's relinking rule.
    """
    cfg = config or MotionRelinkConfig()
    if not detections:
        return MotionRelinkResult(edges=(), stats=MotionRelinkStats())

    ids_by_frame: dict[int, list[int]] = {}
    positions: dict[int, np.ndarray] = {}
    for node_id, detection in detections.items():
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            raise ValueError("motion relinking requires integer detection IDs")
        if detection.id != node_id:
            raise ValueError(f"detection key {node_id} differs from its Detection.id")
        if isinstance(detection.frame, bool) or not isinstance(detection.frame, int):
            raise ValueError(f"detection {node_id} must have an integer frame")
        if detection.frame < 0:
            raise ValueError(f"detection {node_id} has a negative frame")
        position = np.asarray(detection.position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"detection {node_id} must have finite physical coordinates")
        ids_by_frame.setdefault(detection.frame, []).append(node_id)
        positions[node_id] = position
    for ids in ids_by_frame.values():
        ids.sort()

    if max(map(len, ids_by_frame.values())) > cfg.max_nodes_per_frame:
        return MotionRelinkResult(
            edges=(),
            stats=MotionRelinkStats(skipped_large_frame=True),
        )

    probabilities = learned_edge_probs or {}
    previous_positions: dict[int, np.ndarray] = {}
    track_velocity: dict[int, np.ndarray] = {}
    selected: list[MotionRelinkEdge] = []
    frames_processed = 0
    tight_edges = 0
    relaxed_edges = 0

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        unreachable = gate_um * 1000.0 + 1.0
        costs = np.full((len(source_ids), len(target_ids)), unreachable, dtype=np.float64)
        raw_distances = np.full_like(costs, np.inf)
        motion_distances = np.full_like(costs, np.inf)
        edge_probabilities = np.zeros_like(costs)
        for row, source_id in enumerate(source_ids):
            source_position = positions[source_id]
            velocity = track_velocity.get(source_id)
            if velocity is None:
                previous = previous_positions.get(source_id)
                velocity = source_position - previous if previous is not None else np.zeros(3)
            predicted_position = source_position + cfg.velocity_weight * velocity
            for column, target_id in enumerate(target_ids):
                target_position = positions[target_id]
                raw_distance = float(np.linalg.norm(target_position - source_position))
                if raw_distance > gate_um:
                    continue
                motion_distance = float(np.linalg.norm(target_position - predicted_position))
                probability = _learned_probability(probabilities.get((source_id, target_id), 0.0))
                raw_distances[row, column] = raw_distance
                motion_distances[row, column] = motion_distance
                edge_probabilities[row, column] = probability
                costs[row, column] = (
                    motion_distance
                    + 0.05 * raw_distance
                    - cfg.learned_edge_bonus * probability
                )
        row_indices, column_indices = linear_sum_assignment(costs)
        matches: list[tuple[int, int, float, float, float]] = []
        for row, column in zip(row_indices, column_indices):
            if costs[row, column] >= unreachable:
                continue
            matches.append(
                (
                    source_ids[int(row)],
                    target_ids[int(column)],
                    float(raw_distances[row, column]),
                    float(motion_distances[row, column]),
                    float(edge_probabilities[row, column]),
                )
            )
        return matches

    for frame in sorted(ids_by_frame):
        source_ids = ids_by_frame.get(frame, [])
        target_ids = ids_by_frame.get(frame + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[
            tuple[int, int, float, float, Literal["tight", "relaxed"], float]
        ] = []
        passes: tuple[tuple[Literal["tight", "relaxed"], float], ...] = (
            ("tight", cfg.tight_gate_um),
            ("relaxed", cfg.relaxed_gate_um),
        )
        for pass_name, gate_um in passes:
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            for source_id, target_id, raw, motion, probability in assign_pass(
                pass_sources,
                pass_targets,
                gate_um,
            ):
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append(
                    (source_id, target_id, raw, motion, pass_name, probability)
                )
                if pass_name == "tight":
                    tight_edges += 1
                else:
                    relaxed_edges += 1

        for source_id, target_id, raw, motion, pass_name, probability in frame_matches:
            selected.append(
                MotionRelinkEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_probability=probability,
                    distance_um=raw,
                    motion_distance_um=motion,
                    pass_name=pass_name,
                )
            )
            instant_velocity = positions[target_id] - positions[source_id]
            previous_velocity = track_velocity.get(source_id)
            track_velocity[target_id] = (
                instant_velocity
                if previous_velocity is None
                else 0.6 * instant_velocity + 0.4 * previous_velocity
            )
            previous_positions[target_id] = positions[source_id]
        frames_processed += 1

    return MotionRelinkResult(
        edges=tuple(selected),
        stats=MotionRelinkStats(
            frames_processed=frames_processed,
            tight_edges=tight_edges,
            relaxed_edges=relaxed_edges,
        ),
    )
