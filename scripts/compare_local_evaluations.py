#!/usr/bin/env python3
"""Write paired per-volume deltas for two identical OOF evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_tracking.evaluation import compare_per_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = compare_per_dataset(args.baseline, args.candidate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "paired_per_dataset.csv"
    summary_path = args.output_dir / "paired_summary.json"
    result.per_dataset.to_csv(paired_path, index=False)
    summary_path.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"wrote {paired_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
