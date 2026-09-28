#!/usr/bin/env python3
"""CLI utility to ensemble multiple Kaggle cell tracking submission CSVs into a consensus CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from biohub_tracking.tracking.ensemble import ensemble_submission_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble multiple cell tracking submission CSVs into a consensus submission CSV."
    )
    parser.add_argument(
        "--submissions",
        nargs="+",
        required=True,
        help="Paths to submission CSV files to ensemble.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the ensembled submission CSV.",
    )
    parser.add_argument(
        "--matching-radius-um",
        type=float,
        default=3.5,
        help="Maximum distance in microns for bipartite node clustering (default: 3.5).",
    )
    parser.add_argument(
        "--min-node-votes",
        type=int,
        default=1,
        help="Minimum model votes required to keep a detection node (default: 1).",
    )
    parser.add_argument(
        "--min-edge-votes",
        type=int,
        default=2,
        help="Minimum model votes required to retain a temporal edge (default: 2).",
    )
    parser.add_argument(
        "--no-resolve-contested",
        action="store_true",
        help="Disable degree constraints (out_degree <= 2, in_degree <= 1).",
    )

    args = parser.parse_args()

    sub_paths = [Path(p) for p in args.submissions]
    for p in sub_paths:
        if not p.is_file():
            print(f"Error: Submission file does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ensembling {len(sub_paths)} submissions:")
    for p in sub_paths:
        print(f"  - {p}")
    print(f"  Parameters: matching_radius={args.matching_radius_um} µm, min_node_votes={args.min_node_votes}, min_edge_votes={args.min_edge_votes}")

    df = ensemble_submission_files(
        sub_paths,
        out_path,
        matching_radius_um=args.matching_radius_um,
        min_node_votes=args.min_node_votes,
        min_edge_votes=args.min_edge_votes,
        resolve_contested=not args.no_resolve_contested,
    )

    print(f"Ensemble complete! Exported {len(df)} total rows to {out_path}")


if __name__ == "__main__":
    main()
