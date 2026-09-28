"""Tests for graph-level bipartite consensus ensembling."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.tracking.ensemble import ensemble_tracking_graphs
from biohub_tracking.tracking.graph import Detection, TrackingGraph


def test_ensemble_identical_graphs():
    """Ensembling identical graphs should preserve all nodes and edges."""
    g1 = TrackingGraph()
    g2 = TrackingGraph()
    for g in (g1, g2):
        g.add_node(Detection(id=1, frame=0, z=10.0, y=20.0, x=30.0))
        g.add_node(Detection(id=2, frame=1, z=10.5, y=20.2, x=30.1))
        g.add_edge(1, 2)

    ensembled = ensemble_tracking_graphs([g1, g2], min_edge_votes=2)
    assert len(ensembled.nodes) == 2
    assert len(ensembled.edges) == 1
    # Check node positions
    n1 = [n for n in ensembled.nodes.values() if n.frame == 0][0]
    n2 = [n for n in ensembled.nodes.values() if n.frame == 1][0]
    assert (n1.z, n1.y, n1.x) == (10.0, 20.0, 30.0)
    assert (n2.z, n2.y, n2.x) == (10.5, 20.2, 30.1)
    assert (n1.id, n2.id) in ensembled.edges


def test_ensemble_filters_single_vote_spurious_edge():
    """Edges present in only 1 model should be filtered when min_edge_votes=2."""
    g1 = TrackingGraph()
    g1.add_node(Detection(id=1, frame=0, z=10.0, y=10.0, x=10.0))
    g1.add_node(Detection(id=2, frame=1, z=10.0, y=10.0, x=10.0))
    g1.add_edge(1, 2)

    g2 = TrackingGraph()
    g2.add_node(Detection(id=1, frame=0, z=10.0, y=10.0, x=10.0))
    g2.add_node(Detection(id=2, frame=1, z=10.0, y=10.0, x=10.0))
    # g2 has NO edge 1->2 (e.g. tracking missed or vetoed it)

    ensembled = ensemble_tracking_graphs([g1, g2], min_edge_votes=2)
    assert len(ensembled.nodes) == 2
    assert len(ensembled.edges) == 0  # 1 vote < 2 required


def test_ensemble_resolves_contested_edge_by_proximity():
    """When models vote for different targets, pick the closer target."""
    g1 = TrackingGraph()
    g1.add_node(Detection(id="A0", frame=0, z=10.0, y=10.0, x=10.0))
    g1.add_node(Detection(id="B1_close", frame=1, z=10.5, y=10.5, x=10.5))  # dist ~0.86 um
    g1.add_node(Detection(id="B2_far", frame=1, z=14.0, y=14.0, x=14.0))    # dist ~6.9 um
    g1.add_edge("A0", "B1_close")

    g2 = TrackingGraph()
    g2.add_node(Detection(id="A0", frame=0, z=10.0, y=10.0, x=10.0))
    g2.add_node(Detection(id="B1_close", frame=1, z=10.5, y=10.5, x=10.5))
    g2.add_node(Detection(id="B2_far", frame=1, z=14.0, y=14.0, x=14.0))
    g2.add_edge("A0", "B2_far")

    # With min_edge_votes=1 and resolve_contested=True, it should break the tie towards B1_close
    ensembled = ensemble_tracking_graphs([g1, g2], min_edge_votes=1, resolve_contested=True)
    assert len(ensembled.edges) == 1
    edge = next(iter(ensembled.edges))
    target_node = ensembled.nodes[edge[1]]
    assert target_node.z < 11.0  # closer target chosen


def test_ensemble_preserves_division_when_confirmed():
    """A valid division confirmed by both models should be preserved."""
    g1 = TrackingGraph()
    g2 = TrackingGraph()
    for g in (g1, g2):
        g.add_node(Detection(id=1, frame=0, z=10.0, y=10.0, x=10.0))
        g.add_node(Detection(id=2, frame=1, z=8.0, y=10.0, x=10.0))   # daughter 1
        g.add_node(Detection(id=3, frame=1, z=12.0, y=10.0, x=10.0))  # daughter 2
        g.add_edge(1, 2)
        g.add_edge(1, 3)

    ensembled = ensemble_tracking_graphs([g1, g2], min_edge_votes=2)
    assert len(ensembled.nodes) == 3
    assert len(ensembled.edges) == 2
    assert len(ensembled.forks()) == 1


def test_ensemble_submission_files(tmp_path):
    """Test full file-based submission ensembling with strict schema export."""
    from biohub_tracking.submission.export import export_submission
    from biohub_tracking.tracking.ensemble import ensemble_submission_files
    from biohub_tracking.evaluation.local_score import load_submission_graphs

    g1 = TrackingGraph()
    g1.add_node(Detection(id=1, frame=0, z=10.0, y=10.0, x=10.0))
    g1.add_node(Detection(id=2, frame=1, z=10.0, y=10.0, x=10.0))
    g1.add_edge(1, 2)

    g2 = TrackingGraph()
    g2.add_node(Detection(id="A", frame=0, z=10.2, y=10.1, x=10.0))  # within 3.5 um of 1
    g2.add_node(Detection(id="B", frame=1, z=10.1, y=10.0, x=10.2))  # within 3.5 um of 2
    g2.add_edge("A", "B")

    sub1_path = tmp_path / "sub1.csv"
    sub2_path = tmp_path / "sub2.csv"
    out_path = tmp_path / "ensemble_out.csv"

    export_submission({"movie_01": g1}, sub1_path)
    export_submission({"movie_01": g2}, sub2_path)

    df_ens = ensemble_submission_files([sub1_path, sub2_path], out_path, min_edge_votes=2)
    assert out_path.is_file()
    assert len(df_ens) > 0

    # Verify that the exported ensemble loads cleanly into TrackingGraph
    loaded = load_submission_graphs(out_path)
    assert "movie_01" in loaded
    assert len(loaded["movie_01"].nodes) == 2
    assert len(loaded["movie_01"].edges) == 1

