"""Tests for the strict, deterministic submission CSV exporter."""

from __future__ import annotations

from biohub_tracking.submission.export import SUBMISSION_COLUMNS, export_submission, graphs_to_dataframe
from biohub_tracking.tracking.graph import Detection, TrackingGraph


def _sample_graph() -> TrackingGraph:
    graph = TrackingGraph()
    graph.add_node(Detection(id=2, frame=0, z=0.0, y=0.0, x=0.0))
    graph.add_node(Detection(id=1, frame=0, z=1.0, y=1.0, x=1.0))
    graph.add_node(Detection(id=5, frame=1, z=0.0, y=0.0, x=1.0))
    graph.add_edge(2, 5)
    graph.add_edge(1, 5)
    return graph


def test_dataframe_has_exact_schema_and_sentinel_values() -> None:
    df = graphs_to_dataframe({"testA": _sample_graph()})
    assert list(df.columns) == SUBMISSION_COLUMNS

    node_rows = df[df["row_type"] == "node"]
    assert (node_rows["source_id"] == -1).all()
    assert (node_rows["target_id"] == -1).all()

    edge_rows = df[df["row_type"] == "edge"]
    assert (edge_rows["node_id"] == -1).all()
    assert (edge_rows["t"] == -1).all()
    assert (edge_rows["z"] == -1).all()


def test_export_is_deterministic_across_repeated_calls() -> None:
    graph = _sample_graph()
    df1 = graphs_to_dataframe({"testA": graph})
    df2 = graphs_to_dataframe({"testA": graph})
    assert df1.equals(df2)


def test_nodes_are_sorted_before_edges_within_dataset() -> None:
    df = graphs_to_dataframe({"testA": _sample_graph()})
    row_types = list(df["row_type"])
    # All node rows must precede all edge rows for a given dataset.
    first_edge_index = row_types.index("edge")
    assert all(rt == "node" for rt in row_types[:first_edge_index])
    assert all(rt == "edge" for rt in row_types[first_edge_index:])


def test_multiple_datasets_are_sorted_and_grouped() -> None:
    df = graphs_to_dataframe({"testB": _sample_graph(), "testA": _sample_graph()})
    datasets_in_order = list(dict.fromkeys(df["dataset"]))
    assert datasets_in_order == ["testA", "testB"]


def test_export_submission_writes_csv(tmp_path) -> None:
    path = tmp_path / "submission.csv"
    df = export_submission({"testA": _sample_graph()}, path)
    assert path.exists()
    written = path.read_text().strip().splitlines()
    assert written[0] == ",".join(SUBMISSION_COLUMNS)
    assert len(written) == 1 + len(df)
