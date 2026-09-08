"""Bounded metric-aware graph postprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

import numpy as np

from biohub_tracking.tracking.graph import Detection, NodeId, TrackingGraph

RefinementHook = Callable[[Detection, Detection, tuple[float, float, float]], tuple[float, float, float] | None]


@dataclass(frozen=True)
class GapClosingConfig:
    """One-frame-gap repair limits, all in physical microns."""

    max_distance_um: float = 12.0
    max_pairs: int = 100
    reuse_distance_um: float = 2.0

    def __post_init__(self) -> None:
        if self.max_distance_um <= 0 or self.reuse_distance_um <= 0 or self.max_pairs < 0:
            raise ValueError("gap-closing distances must be positive and max_pairs must be non-negative")


@dataclass(frozen=True)
class ShortComponentConfig:
    """Remove weak components smaller than this number of nodes."""

    min_nodes: int = 3

    def __post_init__(self) -> None:
        if self.min_nodes < 1:
            raise ValueError("min_nodes must be positive")


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
) -> TrackingGraph:
    """Bridge eligible terminal-to-root pairs two frames apart with midpoint nodes.

    A hook may refine each geometric midpoint from heatmap evidence.  Returning
    ``None`` skips that pair, making confidence-aware refinement explicit.
    """
    result = _copy_selected(graph, set(graph.nodes))
    terminal = [node for node in result if result.out_degree(node.id) == 0]
    roots = [node for node in result if not result.in_edges(node.id)]
    proposals: list[tuple[float, Detection, Detection]] = []
    for source in terminal:
        for target in roots:
            if target.frame != source.frame + 2:
                continue
            distance = float(np.linalg.norm(np.asarray(source.position) - np.asarray(target.position)))
            if distance <= config.max_distance_um:
                proposals.append((distance, source, target))
    used: set[NodeId] = set()
    next_index = 0
    for _distance, source, target in sorted(proposals, key=lambda item: (item[0], repr(item[1].id), repr(item[2].id))):
        if len(used) // 2 >= config.max_pairs or source.id in used or target.id in used:
            continue
        midpoint = tuple(float(value) for value in ((np.asarray(source.position) + np.asarray(target.position)) / 2))
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
            and np.linalg.norm(np.asarray(node.position) - np.asarray(midpoint)) <= config.reuse_distance_um
        ]
        if existing:
            bridge = min(existing, key=lambda node: (np.linalg.norm(np.asarray(node.position) - midpoint), repr(node.id)))
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


def filter_short_components(graph: TrackingGraph, config: ShortComponentConfig) -> TrackingGraph:
    """Return graph excluding weakly connected components below ``min_nodes``."""
    components = graph.weakly_connected_components()
    sizes: dict[int, int] = {}
    for component in components.values():
        sizes[component] = sizes.get(component, 0) + 1
    return _copy_selected(graph, {node_id for node_id, component in components.items() if sizes[component] >= config.min_nodes})


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
