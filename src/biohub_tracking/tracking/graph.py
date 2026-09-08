"""Core tracking graph data structures shared by tracking and metrics code.

A :class:`TrackingGraph` is intentionally simple: detections (nodes) carry a
frame index and a 3D centroid in physical units (microns), and directed
edges link a detection to itself (or its daughters, in case of division) at
a later frame. Both predicted and ground-truth tracks use this same
structure so the metrics implementation is symmetric.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Iterator

NodeId = Hashable


@dataclass(frozen=True)
class Detection:
    """A single cell detection: a node in the tracking graph.

    Coordinates are expected in physical units (microns) so that distance
    thresholds (e.g. the official 7 µm matching radius) are meaningful
    regardless of voxel anisotropy.
    """

    id: NodeId
    frame: int
    z: float
    y: float
    x: float

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.z, self.y, self.x)


@dataclass
class TrackingGraph:
    """A directed graph of detections linked across frames.

    Edges are stored as ``(source_id, target_id)`` pairs. By convention
    (matching the competition's definition) an edge points from a detection
    at frame ``t`` to a detection - or one of up to two daughters at a cell
    division - at a later frame, usually ``t + 1``.
    """

    nodes: dict[NodeId, Detection] = field(default_factory=dict)
    edges: set[tuple[NodeId, NodeId]] = field(default_factory=set)

    def add_node(self, detection: Detection) -> None:
        if detection.id in self.nodes and self.nodes[detection.id] != detection:
            raise ValueError(f"node id {detection.id!r} already exists with different attributes")
        self.nodes[detection.id] = detection

    def add_edge(self, source: NodeId, target: NodeId) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("both source and target nodes must be added before the edge")
        if self.nodes[target].frame <= self.nodes[source].frame:
            raise ValueError("edges must point strictly forward in time")
        self.edges.add((source, target))

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "TrackingGraph":
        """Build a graph from an iterable of dicts with a ``parent_id`` link.

        Each record needs keys: ``id``, ``frame``, ``z``, ``y``, ``x``, and
        an optional ``parent_id`` (``None``/absent for track starts). This
        mirrors the typical flat table representation used to exchange
        tracking results before conversion to the submission CSV schema.
        """
        graph = cls()
        for rec in records:
            graph.add_node(Detection(id=rec["id"], frame=rec["frame"], z=rec["z"], y=rec["y"], x=rec["x"]))
        for rec in records:
            parent_id = rec.get("parent_id")
            if parent_id is not None:
                graph.add_edge(parent_id, rec["id"])
        return graph

    def nodes_by_frame(self) -> dict[int, list[Detection]]:
        """Group nodes by their frame index."""
        grouped: dict[int, list[Detection]] = defaultdict(list)
        for node in self.nodes.values():
            grouped[node.frame].append(node)
        return dict(grouped)

    def out_edges(self, node_id: NodeId) -> list[tuple[NodeId, NodeId]]:
        return [e for e in self.edges if e[0] == node_id]

    def in_edges(self, node_id: NodeId) -> list[tuple[NodeId, NodeId]]:
        return [e for e in self.edges if e[1] == node_id]

    def successors(self, node_id: NodeId) -> list[NodeId]:
        return [t for (s, t) in self.edges if s == node_id]

    def predecessors(self, node_id: NodeId) -> list[NodeId]:
        return [s for (s, t) in self.edges if t == node_id]

    def out_degree(self, node_id: NodeId) -> int:
        return len(self.successors(node_id))

    def forks(self) -> list[NodeId]:
        """Nodes with two or more outgoing edges: predicted/GT division points."""
        out_count: dict[NodeId, int] = defaultdict(int)
        for s, _t in self.edges:
            out_count[s] += 1
        return [node_id for node_id, count in out_count.items() if count >= 2]

    def targets_of_source(self) -> dict[NodeId, set[NodeId]]:
        mapping: dict[NodeId, set[NodeId]] = defaultdict(set)
        for s, t in self.edges:
            mapping[s].add(t)
        return dict(mapping)

    def sources_of_target(self) -> dict[NodeId, set[NodeId]]:
        mapping: dict[NodeId, set[NodeId]] = defaultdict(set)
        for s, t in self.edges:
            mapping[t].add(s)
        return dict(mapping)

    def weakly_connected_components(self) -> dict[NodeId, int]:
        """Assign every node an integer id for its weakly-connected component.

        Two nodes are in the same component if there is a path between them
        ignoring edge direction. Used by the division metric to reason about
        which lineage a piece of local evidence belongs to.
        """
        adjacency: dict[NodeId, set[NodeId]] = defaultdict(set)
        for s, t in self.edges:
            adjacency[s].add(t)
            adjacency[t].add(s)
        component_of: dict[NodeId, int] = {}
        next_component = 0
        for node_id in self.nodes:
            if node_id in component_of:
                continue
            component_of[node_id] = next_component
            queue = deque([node_id])
            while queue:
                current = queue.popleft()
                for neighbor in adjacency.get(current, ()):
                    if neighbor not in component_of:
                        component_of[neighbor] = next_component
                        queue.append(neighbor)
            next_component += 1
        return component_of

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)
