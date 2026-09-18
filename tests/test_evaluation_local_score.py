from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from biohub_tracking.evaluation.local_score import graph_from_submission_rows, load_submission_graphs, score_submission
from biohub_tracking.submission import export_submission, graphs_to_dataframe
from biohub_tracking.tracking import Detection, TrackingGraph


def _graph(prefix: str = "") -> TrackingGraph:
    graph = TrackingGraph()
    graph.add_node(Detection(id=f"{prefix}0", frame=0, z=0.0, y=0.0, x=0.0))
    graph.add_node(Detection(id=f"{prefix}1", frame=1, z=0.0, y=0.0, x=1.0))
    graph.add_edge(f"{prefix}0", f"{prefix}1")
    return graph


def test_load_submission_graphs_round_trips_exported_schema(tmp_path) -> None:
    path = tmp_path / "submission.csv"
    export_submission({"44b6_a": _graph()}, path)
    graphs = load_submission_graphs(path)
    assert sorted(graphs) == ["44b6_a"]
    assert len(graphs["44b6_a"].nodes) == 2
    assert len(graphs["44b6_a"].edges) == 1


def test_graph_from_submission_rows_rejects_bad_sentinels() -> None:
    frame = graphs_to_dataframe({"d": _graph()})
    frame.loc[frame["row_type"] == "edge", "node_id"] = 3
    with pytest.raises(ValueError, match="edge rows must use -1"):
        graph_from_submission_rows(frame, dataset="d")


def test_graph_from_submission_rows_rejects_dangling_edges() -> None:
    frame = graphs_to_dataframe({"d": _graph()})
    frame.loc[frame["row_type"] == "edge", "target_id"] = "missing"
    with pytest.raises(ValueError, match="dangling edge"):
        graph_from_submission_rows(frame, dataset="d")


def test_score_submission_reports_aggregate_and_diagnostics(tmp_path) -> None:
    pred = tmp_path / "pred.csv"
    gt = tmp_path / "gt.csv"
    export_submission({"44b6_a": _graph("p")}, pred)
    export_submission({"44b6_a": _graph("g")}, gt)
    result = score_submission(pred, gt, node_estimates={"44b6_a": 2})
    assert result.summary["score"] == pytest.approx(1.1)
    row = result.per_dataset.iloc[0]
    assert row["edge_tp"] == 1
    assert row["edges_recovered"] == 1
    assert row["node_recall"] == pytest.approx(1.0)


def test_score_submission_requires_requested_dataset_coverage(tmp_path) -> None:
    pred = tmp_path / "pred.csv"
    gt = tmp_path / "gt.csv"
    export_submission({"44b6_a": _graph("p")}, pred)
    export_submission({"44b6_a": _graph("g")}, gt)
    with pytest.raises(ValueError, match="missing datasets"):
        score_submission(pred, gt, datasets=["6bba_b"])


def test_local_eval_cli_writes_artifacts_and_log(tmp_path) -> None:
    pred = tmp_path / "pred.csv"
    gt = tmp_path / "gt.csv"
    out = tmp_path / "eval"
    export_submission({"44b6_a": _graph("p")}, pred)
    export_submission({"44b6_a": _graph("g")}, gt)
    cmd = [
        sys.executable,
        "scripts/local_eval.py",
        "--submission",
        str(pred),
        "--gt-submission",
        str(gt),
        "--candidate",
        "unit-test",
        "--output-dir",
        str(out),
        "--log-experiment",
        "--oracle-analysis",
    ]
    completed = subprocess.run(cmd, cwd=".", env={"PYTHONPATH": "src"}, text=True, capture_output=True, check=True)
    assert "unit-test" in completed.stdout
    summary = json.loads((out / "summary.json").read_text())
    assert summary["score"] == pytest.approx(1.1)
    assert (out / "per_dataset.csv").exists()
    assert pd.read_csv(out / "local_eval_log.csv").iloc[-1]["candidate"] == "unit-test"
    assert json.loads((out / "oracle_summary.json").read_text())["linking_headroom"] == pytest.approx(0.0)
    assert (out / "oracle_per_dataset.csv").exists()


def test_local_eval_cli_postprocessing(tmp_path) -> None:
    pred = tmp_path / "pred.csv"
    gt = tmp_path / "gt.csv"
    out = tmp_path / "eval"
    postprocessed_out = tmp_path / "postprocessed.csv"
    
    # Construct prediction graph with size 2 component and size 1 component
    pred_graph = TrackingGraph()
    pred_graph.add_node(Detection(id="p0", frame=0, z=0.0, y=0.0, x=0.0))
    pred_graph.add_node(Detection(id="p1", frame=1, z=0.0, y=0.0, x=1.0))
    pred_graph.add_edge("p0", "p1")
    pred_graph.add_node(Detection(id="solo", frame=0, z=0.0, y=0.0, x=10.0))
    export_submission({"44b6_a": pred_graph}, pred)
    
    # Construct a simple ground truth graph matching only the size 2 component
    gt_graph = TrackingGraph()
    gt_graph.add_node(Detection(id="g0", frame=0, z=0.0, y=0.0, x=0.0))
    gt_graph.add_node(Detection(id="g1", frame=1, z=0.0, y=0.0, x=1.0))
    gt_graph.add_edge("g0", "g1")
    export_submission({"44b6_a": gt_graph}, gt)
    
    # Run with --min-component-nodes 2 (should drop "solo" node)
    cmd = [
        sys.executable,
        "scripts/local_eval.py",
        "--submission",
        str(pred),
        "--gt-submission",
        str(gt),
        "--candidate",
        "postprocess-test",
        "--output-dir",
        str(out),
        "--min-component-nodes",
        "2",
        "--postprocessed-output",
        str(postprocessed_out),
    ]
    subprocess.run(cmd, cwd=".", env={"PYTHONPATH": "src"}, text=True, capture_output=True, check=True)
    
    # Verify post-processed output exists and "solo" node was pruned
    assert postprocessed_out.exists()
    postprocessed_df = pd.read_csv(postprocessed_out)
    nodes = postprocessed_df[postprocessed_df["row_type"] == "node"]
    node_ids = set(nodes["node_id"].astype(str))
    assert node_ids == {"p0", "p1"}  # "solo" is gone!
    
    # Run with --max-nodes 1 (should keep only 1 node)
    postprocessed_out2 = tmp_path / "postprocessed2.csv"
    cmd = [
        sys.executable,
        "scripts/local_eval.py",
        "--submission",
        str(pred),
        "--gt-submission",
        str(gt),
        "--candidate",
        "postprocess-test2",
        "--output-dir",
        str(out),
        "--max-nodes",
        "1",
        "--postprocessed-output",
        str(postprocessed_out2),
    ]
    subprocess.run(cmd, cwd=".", env={"PYTHONPATH": "src"}, text=True, capture_output=True, check=True)
    
    # Verify post-processed output has exactly 1 node
    assert postprocessed_out2.exists()
    postprocessed_df2 = pd.read_csv(postprocessed_out2)
    nodes2 = postprocessed_df2[postprocessed_df2["row_type"] == "node"]
    assert len(nodes2) == 1


def test_local_eval_cli_gap_closing(tmp_path) -> None:
    pred = tmp_path / "pred.csv"
    gt = tmp_path / "gt.csv"
    out = tmp_path / "eval"
    postprocessed_out = tmp_path / "postprocessed.csv"
    
    # Prediction graph has two nodes with a 1-frame gap: p0 at t=0, p1 at t=2
    pred_graph = TrackingGraph()
    pred_graph.add_node(Detection(id="p0", frame=0, z=0.0, y=0.0, x=0.0))
    pred_graph.add_node(Detection(id="p1", frame=2, z=0.0, y=0.0, x=2.0))
    export_submission({"44b6_a": pred_graph}, pred)
    
    # Ground truth matches the full bridge (p0, midpoint, p1)
    gt_graph = TrackingGraph()
    gt_graph.add_node(Detection(id="g0", frame=0, z=0.0, y=0.0, x=0.0))
    gt_graph.add_node(Detection(id="g1", frame=1, z=0.0, y=0.0, x=1.0))
    gt_graph.add_node(Detection(id="g2", frame=2, z=0.0, y=0.0, x=2.0))
    gt_graph.add_edge("g0", "g1")
    gt_graph.add_edge("g1", "g2")
    export_submission({"44b6_a": gt_graph}, gt)
    
    cmd = [
        sys.executable,
        "scripts/local_eval.py",
        "--submission",
        str(pred),
        "--gt-submission",
        str(gt),
        "--candidate",
        "gap-test",
        "--output-dir",
        str(out),
        "--close-gaps",
        "--gap-max-distance-um",
        "5.0",
        "--postprocessed-output",
        str(postprocessed_out),
    ]
    subprocess.run(cmd, cwd=".", env={"PYTHONPATH": "src"}, text=True, capture_output=True, check=True)
    
    # Verify post-processed output contains midpoint node at t=1, plus the two bridged edges
    assert postprocessed_out.exists()
    postprocessed_df = pd.read_csv(postprocessed_out)
    
    nodes = postprocessed_df[postprocessed_df["row_type"] == "node"]
    edges = postprocessed_df[postprocessed_df["row_type"] == "edge"]
    
    # Check node times: should be t=0, t=1 (inserted midpoint), t=2
    assert sorted(nodes["t"].astype(int)) == [0, 1, 2]
    # Check edges count: should have 2 edges bridging them
    assert len(edges) == 2
