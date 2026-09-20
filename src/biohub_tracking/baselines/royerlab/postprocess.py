"""Bounded metric-aware graph postprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

import numpy as np

from biohub_tracking.tracking.graph import Detection, NodeId, TrackingGraph

RefinementHook = Callable[
    [Detection, Detection, tuple[float, float, float]],
    tuple[float, float, float] | None,
]
GapScoreHook = Callable[[Detection, Detection], float]


@dataclass(frozen=True)
class GapClosingConfig:
    """One-frame-gap repair limits, all in physical microns."""

    max_distance_um: float = 12.0
    max_pairs: int = 100
    reuse_distance_um: float = 2.0
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.max_distance_um <= 0 or self.reuse_distance_um <= 0 or self.max_pairs < 0:
            raise ValueError(
                "gap-closing distances must be positive and max_pairs must be non-negative"
            )
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass(frozen=True)
class ShortComponentConfig:
    """Remove short components unless calibrated confidence preserves them."""

    min_nodes: int = 3
    keep_short_above_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.min_nodes < 1:
            raise ValueError("min_nodes must be positive")
        threshold = self.keep_short_above_confidence
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("keep_short_above_confidence must be in [0, 1]")


@dataclass(frozen=True)
class NodeBudgetConfig:
    """Limit prediction count to avoid adjusted-Jaccard node-count penalties."""

    max_nodes: int

    def __post_init__(self) -> None:
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")


def _copy_selected(graph: TrackingGraph, selected: set[NodeId]) -> TrackingGraph:
    result = TrackingGraph()
    for node_id in graph.nodes:
        if node_id in selected:
            result.add_node(graph.nodes[node_id])
    for source, target in graph.edges:
        if source in selected and target in selected:
            result.add_edge(source, target)
    return result


def close_one_frame_gaps(
    graph: TrackingGraph,
    config: GapClosingConfig,
    refinement_hook: RefinementHook | None = None,
    *,
    score_hook: GapScoreHook | None = None,
) -> TrackingGraph:
    """Bridge eligible terminal-to-root pairs two frames apart with midpoint nodes.

    A hook may refine each geometric midpoint from heatmap evidence.  Returning
    ``None`` skips that pair, making confidence-aware refinement explicit.
    """
    if config.min_confidence > 0 and score_hook is None:
        raise ValueError("score_hook is required when min_confidence is positive")
    result = _copy_selected(graph, set(graph.nodes))
    terminal = [node for node in result if result.out_degree(node.id) == 0]
    roots = [node for node in result if not result.in_edges(node.id)]
    proposals: list[tuple[float, float, Detection, Detection]] = []
    for source in terminal:
        for target in roots:
            if target.frame != source.frame + 2:
                continue
            distance = float(
                np.linalg.norm(np.asarray(source.position) - np.asarray(target.position))
            )
            if distance <= config.max_distance_um:
                confidence = 1.0 if score_hook is None else float(score_hook(source, target))
                if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("score_hook must return a finite probability in [0, 1]")
                if confidence >= config.min_confidence:
                    proposals.append((confidence, distance, source, target))
    used: set[NodeId] = set()
    next_index = 0
    for _confidence, _distance, source, target in sorted(
        proposals,
        key=lambda item: (-item[0], item[1], repr(item[2].id), repr(item[3].id)),
    ):
        if len(used) // 2 >= config.max_pairs or source.id in used or target.id in used:
            continue
        midpoint = tuple(
            float(value)
            for value in ((np.asarray(source.position) + np.asarray(target.position)) / 2)
        )
        if refinement_hook is not None:
            refined = refinement_hook(source, target, midpoint)
            if refined is None:
                continue
            if len(refined) != 3:
                raise ValueError("refinement_hook must return a (z, y, x) position or None")
            midpoint = tuple(float(value) for value in refined)
        existing = [
            node
            for node in result
            if node.frame == source.frame + 1
            and np.linalg.norm(np.asarray(node.position) - np.asarray(midpoint))
            <= config.reuse_distance_um
        ]
        if existing:
            bridge = min(
                existing,
                key=lambda node: (
                    np.linalg.norm(np.asarray(node.position) - midpoint),
                    repr(node.id),
                ),
            )
        else:
            while ("synthetic-gap", source.id, target.id, next_index) in result.nodes:
                next_index += 1
            bridge = Detection(
                id=("synthetic-gap", source.id, target.id, next_index),
                frame=source.frame + 1,
                z=midpoint[0],
                y=midpoint[1],
                x=midpoint[2],
            )
            result.add_node(bridge)
            next_index += 1
        result.add_edge(source.id, bridge.id)
        result.add_edge(bridge.id, target.id)
        used.update((source.id, target.id))
    return result


def filter_short_components(
    graph: TrackingGraph,
    config: ShortComponentConfig,
    node_confidences: dict[NodeId, float] | None = None,
) -> TrackingGraph:
    """Drop short low-confidence components, preserving deterministic order."""
    threshold = config.keep_short_above_confidence
    if threshold is not None:
        if node_confidences is None:
            raise ValueError("node_confidences are required for confidence-gated pruning")
        missing = sorted(
            (node_id for node_id in graph.nodes if node_id not in node_confidences),
            key=repr,
        )
        if missing:
            raise ValueError(f"node confidences are missing IDs: {missing}")
        if any(
            not np.isfinite(node_confidences[node_id])
            or not 0 <= node_confidences[node_id] <= 1
            for node_id in graph.nodes
        ):
            raise ValueError("node confidences must be finite probabilities")
    components = graph.weakly_connected_components()
    sizes: dict[int, int] = {}
    for component in components.values():
        sizes[component] = sizes.get(component, 0) + 1
    component_confidences: dict[int, list[float]] = {}
    if node_confidences is not None:
        for node_id, component in components.items():
            component_confidences.setdefault(component, []).append(node_confidences[node_id])
    selected = set()
    for node_id, component in components.items():
        keep_for_size = sizes[component] >= config.min_nodes
        keep_for_confidence = (
            threshold is not None
            and float(np.mean(component_confidences[component])) >= threshold
        )
        if keep_for_size or keep_for_confidence:
            selected.add(node_id)
    return _copy_selected(graph, selected)


def cap_node_budget(
    graph: TrackingGraph,
    config: NodeBudgetConfig,
    scores: dict[NodeId, float] | None = None,
) -> TrackingGraph:
    """Keep at most ``max_nodes`` by confidence, breaking all ties deterministically."""
    if scores is not None and set(scores) - set(graph.nodes):
        raise ValueError("scores contains node IDs absent from graph")
    ranked = sorted(
        graph.nodes,
        key=lambda node_id: (-(scores.get(node_id, 0.0) if scores else 0.0), repr(node_id)),
    )
    return _copy_selected(graph, set(ranked[: config.max_nodes]))


def make_deepcenter_veto_hook(
    heatmap_provider: Callable[[float, float, float], float],
    threshold: float = 0.25,
) -> RefinementHook:
    """Create a RefinementHook that vetoes (rejects) proposals using a spatial prior heatmap.

    If the probability returned by ``heatmap_provider`` at the proposed midpoint
    is below ``threshold``, the proposal is rejected (returns ``None``).
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    def veto_hook(
        _source: Detection,
        _target: Detection,
        midpoint: tuple[float, float, float],
    ) -> tuple[float, float, float] | None:
        prob = heatmap_provider(*midpoint)
        if prob < threshold:
            return None
        return midpoint

    return veto_hook

