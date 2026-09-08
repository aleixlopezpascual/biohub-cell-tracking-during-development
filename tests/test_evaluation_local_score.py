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
    ]
    completed = subprocess.run(cmd, cwd=".", env={"PYTHONPATH": "src"}, text=True, capture_output=True, check=True)
    assert "unit-test" in completed.stdout
    summary = json.loads((out / "summary.json").read_text())
    assert summary["score"] == pytest.approx(1.1)
    assert (out / "per_dataset.csv").exists()
    assert pd.read_csv(out / "local_eval_log.csv").iloc[-1]["candidate"] == "unit-test"
