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


def test_hungarian_tracker_respects_sister_symmetry_gate() -> None:
    # Parent is at (0, 0, 0)
    # Daughter 1 is at (0, 0, 1) -> distance = 1.0 um
    # Proposed daughter 2 is at (0, 0, 4) -> distance = 4.0 um (Asymmetrical! mean=(1+4)/2 = 2.5, ratio=3/2.5 = 1.2 > 0.6)
    sources = [_det("p", 0, 0, 0, 0)]
    targets = [_det("d1", 1, 0, 0, 1), _det("d2", 1, 0, 0, 4)]
    
    # Run with gate disabled (both matched)
    tracker_off = HungarianTracker(
        TrackerConfig(max_link_distance_um=5.0, division_search_radius_um=5.0, use_sister_symmetry_gate=False)
    )
    edges_off = tracker_off.link_frames(sources, targets)
    assert set(edges_off) == {("p", "d1"), ("p", "d2")}

    # Run with gate enabled (asymmetry prevents second daughter matching)
    tracker_on = HungarianTracker(
        TrackerConfig(max_link_distance_um=5.0, division_search_radius_um=5.0, use_sister_symmetry_gate=True, sister_symmetry_tau=0.6)
    )
    edges_on = tracker_on.link_frames(sources, targets)
    assert set(edges_on) == {("p", "d1")}


def test_tracker_config_validation_for_ema() -> None:
    with pytest.raises(ValueError, match="alpha"):
        TrackerConfig(use_ema_velocity_projection=True, ema_velocity_alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        TrackerConfig(use_ema_velocity_projection=True, ema_velocity_alpha=1.5)


def test_velocity_projection_resolves_crossover_occlusions() -> None:
    # Two trajectories crossing paths at Frame 1:
    # Trajectory 1: moving right (+2.0 um/frame) from (0,0,0) -> (0,0,2) -> expects (0,0,4)
    # Trajectory 2: moving left (-2.0 um/frame) from (0,0,5) -> (0,0,3) -> expects (0,0,1)
    graph = TrackingGraph()
    # Frame 0
    graph.add_node(_det("s1", 0, 0, 0, 0))
    graph.add_node(_det("s2", 0, 0, 0, 5))
    # Frame 1
    graph.add_node(_det("m1", 1, 0, 0, 2))
    graph.add_node(_det("m2", 1, 0, 0, 3))
    # Frame 2
    graph.add_node(_det("e1", 2, 0, 0, 4))
    graph.add_node(_det("e2", 2, 0, 0, 1))

    # 1. Run tracking with projection disabled (causes crossover error matching m1->e2 and m2->e1)
    tracker_off = HungarianTracker(TrackerConfig(max_link_distance_um=5.0, use_ema_velocity_projection=False))
    tracked_off = tracker_off.track(graph)
    assert ("m1", "e2") in tracked_off.edges
    assert ("m2", "e1") in tracked_off.edges

    # 2. Run tracking with projection enabled (correctly matches m1->e1 and m2->e2)
    tracker_on = HungarianTracker(TrackerConfig(max_link_distance_um=5.0, use_ema_velocity_projection=True, ema_velocity_alpha=1.0))
    tracked_on = tracker_on.track(graph)
    assert ("m1", "e1") in tracked_on.edges
    assert ("m2", "e2") in tracked_on.edges

