"""Evaluation entry point: compute the official Biohub tracking score.

Usage::

    python scripts/evaluate.py --pred pred_submission.csv --gt gt_submission.csv \\
        --true-node-counts '{"testA": 5000}'

Reads predicted and ground-truth tracks from submission-schema CSVs (one or
more ``dataset`` values each), computes the sparse-aware edge Jaccard and
local-window division Jaccard per dataset, and micro-averages them into the
final combined score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from biohub_tracking.metrics import aggregate_scores, evaluate_sample
from biohub_tracking.tracking.graph import Detection, TrackingGraph
from biohub_tracking.utils import get_logger

logger = get_logger(__name__)


def _dataframe_to_graphs(df: pd.DataFrame) -> dict[str, TrackingGraph]:
    """Parse a submission-schema DataFrame into ``{dataset: TrackingGraph}``."""
    graphs: dict[str, TrackingGraph] = {}
    for dataset, group in df.groupby("dataset"):
        graph = TrackingGraph()
        for row in group[group["row_type"] == "node"].itertuples():
            graph.add_node(Detection(id=row.node_id, frame=int(row.t), z=float(row.z), y=float(row.y), x=float(row.x)))
        for row in group[group["row_type"] == "edge"].itertuples():
            graph.add_edge(row.source_id, row.target_id)
        graphs[str(dataset)] = graph
    return graphs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="Predicted submission CSV.")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth submission CSV.")
    parser.add_argument(
        "--true-node-counts",
        type=str,
        default=None,
        help='JSON mapping dataset -> coarse total true node count, e.g. \'{"testA": 5000}\'. '
        "Defaults to each dataset's GT node count (no adjustment penalty).",
    )
    parser.add_argument("--max-distance-um", type=float, default=7.0)
    parser.add_argument("--division-weight", type=float, default=0.1)
    parser.add_argument("--adjusted-jaccard-alpha", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pred_df = pd.read_csv(args.pred)
    gt_df = pd.read_csv(args.gt)
    pred_graphs = _dataframe_to_graphs(pred_df)
    gt_graphs = _dataframe_to_graphs(gt_df)

    true_node_counts: dict[str, int] = json.loads(args.true_node_counts) if args.true_node_counts else {}

    samples = []
    for dataset, gt_graph in gt_graphs.items():
        pred_graph = pred_graphs.get(dataset, TrackingGraph())
        num_true_nodes = true_node_counts.get(dataset, len(gt_graph.nodes))
        sample = evaluate_sample(
            dataset,
            pred_graph,
            gt_graph,
            num_true_nodes=num_true_nodes,
            max_distance=args.max_distance_um,
            adjusted_jaccard_alpha=args.adjusted_jaccard_alpha,
        )
        samples.append(sample)
        logger.info(
            "dataset=%s edge(tp=%d fp=%d fn=%d) division(tp=%d fp=%d fn=%d) adj_jaccard=%.4f",
            dataset,
            sample.edge.tp,
            sample.edge.fp,
            sample.edge.fn,
            sample.division.tp,
            sample.division.fp,
            sample.division.fn,
            sample.adjusted_edge_jaccard,
        )

    aggregate = aggregate_scores(samples, division_weight=args.division_weight)
    print(
        json.dumps(
            {
                "adjusted_edge_jaccard": aggregate.adjusted_edge_jaccard,
                "division_jaccard": aggregate.division_jaccard,
                "score": aggregate.score,
                "num_samples": aggregate.num_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
