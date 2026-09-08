"""Deterministic Kaggle submission CSV export.

The official schema (verified against the competition's own CSV conversion
script) is a single flat table per dataset with node rows and edge rows
interleaved by ``row_type``, unused fields sentinel-filled with ``-1``:

``dataset, row_type, node_id, t, z, y, x, source_id, target_id``

Row ordering is made fully deterministic (sorted datasets, then all node
rows sorted by ``(t, node_id)`` followed by all edge rows sorted by
``(source_id, target_id)``) so repeated exports of the same graph always
produce byte-identical output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biohub_tracking.tracking.graph import TrackingGraph

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

_UNUSED = -1


def graphs_to_dataframe(graphs: dict[str, TrackingGraph]) -> pd.DataFrame:
    """Convert ``{dataset_name: TrackingGraph}`` into the strict submission schema.

    Parameters
    ----------
    graphs:
        Mapping from dataset name (as used in the ``dataset`` column) to its
        predicted :class:`TrackingGraph`.

    Returns
    -------
    pandas.DataFrame
        Columns exactly match :data:`SUBMISSION_COLUMNS`, in deterministic
        row order.
    """
    rows: list[dict[str, object]] = []
    for dataset in sorted(graphs):
        graph = graphs[dataset]
        seen_ids: set[object] = set()
        for node in sorted(graph.nodes.values(), key=lambda n: (n.frame, str(n.id))):
            if node.id in seen_ids:
                raise ValueError(f"duplicate node id {node.id!r} in dataset {dataset!r}")
            seen_ids.add(node.id)
            rows.append(
                {
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": node.id,
                    "t": node.frame,
                    "z": node.z,
                    "y": node.y,
                    "x": node.x,
                    "source_id": _UNUSED,
                    "target_id": _UNUSED,
                }
            )
        for source_id, target_id in sorted(graph.edges, key=lambda e: (str(e[0]), str(e[1]))):
            rows.append(
                {
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": _UNUSED,
                    "t": _UNUSED,
                    "z": _UNUSED,
                    "y": _UNUSED,
                    "x": _UNUSED,
                    "source_id": source_id,
                    "target_id": target_id,
                }
            )
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def export_submission(graphs: dict[str, TrackingGraph], path: str | Path) -> pd.DataFrame:
    """Write the strict, deterministic submission CSV to ``path``.

    Returns the exported :class:`pandas.DataFrame` for convenience (e.g. for
    a caller that also wants to log summary statistics).
    """
    df = graphs_to_dataframe(graphs)
    df.to_csv(path, index=False)
    return df
