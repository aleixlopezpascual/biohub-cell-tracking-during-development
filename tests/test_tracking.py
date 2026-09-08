"""Tests for the tracking graph and Hungarian frame-to-frame tracker."""

from __future__ import annotations

import pytest

from biohub_tracking.tracking.graph import Detection, TrackingGraph
from biohub_tracking.tracking.hungarian import HungarianTracker, TrackerConfig


def _det(id_, frame, z, y, x) -> Detection:
    return Detection(id=id_, frame=frame, z=z, y=y, x=x)


def test_graph_add_edge_rejects_backward_or_same_frame_edges() -> None:
    graph = TrackingGraph()
    graph.add_node(_det("a", 0, 0, 0, 0))
    graph.add_node(_det("b", 0, 0, 0, 1))
    with pytest.raises(ValueError):
        graph.add_edge("a", "b")


def test_graph_forks_and_degree_helpers() -> None:
    graph = TrackingGraph()
    graph.add_node(_det("p", 0, 0, 0, 0))
    graph.add_node(_det("c1", 1, 0, 0, 1))
    graph.add_node(_det("c2", 1, 0, 0, -1))
    graph.add_edge("p", "c1")
    graph.add_edge("p", "c2")
    assert graph.forks() == ["p"]
    assert graph.out_degree("p") == 2
    assert set(graph.successors("p")) == {"c1", "c2"}


def test_graph_weakly_connected_components_merges_linked_nodes() -> None:
    graph = TrackingGraph()
    graph.add_node(_det("a", 0, 0, 0, 0))
    graph.add_node(_det("b", 1, 0, 0, 0))
    graph.add_node(_det("isolated", 0, 100, 100, 100))
    graph.add_edge("a", "b")
    components = graph.weakly_connected_components()
    assert components["a"] == components["b"]
    assert components["isolated"] != components["a"]


def test_hungarian_tracker_links_nearest_pairs() -> None:
    sources = [_det("s0", 0, 0, 0, 0), _det("s1", 0, 0, 0, 20)]
    targets = [_det("t0", 1, 0, 0, 1), _det("t1", 1, 0, 0, 19)]
    tracker = HungarianTracker(TrackerConfig(max_link_distance_um=5.0))
    edges = tracker.link_frames(sources, targets)
    assert set(edges) == {("s0", "t0"), ("s1", "t1")}


def test_hungarian_tracker_respects_max_link_distance() -> None:
    sources = [_det("s0", 0, 0, 0, 0)]
    targets = [_det("t0", 1, 0, 0, 100)]
    tracker = HungarianTracker(TrackerConfig(max_link_distance_um=5.0))
    edges = tracker.link_frames(sources, targets)
    assert edges == []


def test_hungarian_tracker_detects_division_branch() -> None:
    # One parent divides into two daughters close together in the next frame.
    sources = [_det("p", 0, 0, 0, 0)]
    targets = [_det("d1", 1, 0, 0, 1), _det("d2", 1, 0, 1, 0)]
    tracker = HungarianTracker(TrackerConfig(max_link_distance_um=5.0, division_search_radius_um=5.0))
    edges = tracker.link_frames(sources, targets)
    assert set(edges) == {("p", "d1"), ("p", "d2")}


def test_hungarian_tracker_division_respects_max_daughters() -> None:
    sources = [_det("p", 0, 0, 0, 0)]
    targets = [_det("d1", 1, 0, 0, 1), _det("d2", 1, 0, 1, 0), _det("d3", 1, 0, -1, 0)]
    tracker = HungarianTracker(
        TrackerConfig(max_link_distance_um=5.0, division_search_radius_um=5.0, max_daughters=2)
    )
    edges = tracker.link_frames(sources, targets)
    parent_out = [e for e in edges if e[0] == "p"]
    assert len(parent_out) == 2


def test_hungarian_tracker_track_builds_full_graph_over_frames() -> None:
    graph = TrackingGraph()
    graph.add_node(_det("a0", 0, 0, 0, 0))
    graph.add_node(_det("a1", 1, 0, 0, 1))
    graph.add_node(_det("a2", 2, 0, 0, 2))
    tracker = HungarianTracker(TrackerConfig(max_link_distance_um=5.0))
    tracked = tracker.track(graph)
    assert ("a0", "a1") in tracked.edges
    assert ("a1", "a2") in tracked.edges
