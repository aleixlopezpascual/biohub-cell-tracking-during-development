#!/usr/bin/env python3
"""Validate paired fold-disjoint OOF evidence and write a structural report."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from biohub_tracking.evaluation.promotion_gate import (
    PromotionEvidenceError,
    evaluate_promotion_evidence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evidence and receipt paths from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
        help="Paired OOF evidence manifest JSON.",
    )
    parser.add_argument(
        "--output-receipt",
        "--output-report",
        dest="output_receipt",
        required=True,
        type=Path,
        help="Structural report path; this validator does not authorize promotion.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate evidence, write a report, and return pass/no-go/error status."""
    args = parse_args(argv)
    evidence_path = args.evidence.resolve()
    try:
        receipt = evaluate_promotion_evidence(evidence_path)
    except (PromotionEvidenceError, OSError) as exc:
        print(f"paired OOF evidence rejected: {exc}", file=sys.stderr)
        return 2

    destination = args.output_receipt.resolve()
    input_paths = {evidence_path}
    input_paths.update(
        (evidence_path.parent / Path(item["path"])).resolve()
        for item in receipt["verified_artifacts"]
    )
    if destination in input_paths:
        print(
            "receipt path must not overwrite an evidence input or verified artifact",
            file=sys.stderr,
        )
        return 2
    try:
        if destination.exists() and any(
            path.exists() and os.path.samefile(destination, path) for path in input_paths
        ):
            print(
                "receipt path must not overwrite an evidence input or verified artifact",
                file=sys.stderr,
            )
            return 2
    except OSError as exc:
        print(f"cannot verify receipt path: {exc}", file=sys.stderr)
        return 2
    temp_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temp_path, destination)
    except OSError as exc:
        print(f"cannot write structural report: {exc}", file=sys.stderr)
        return 2
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
    print(
        "structural assessment: "
        f"metrics_gate_passed={receipt['metrics_gate_passed']} "
        f"provenance_gate_passed={receipt['provenance_gate_passed']} "
        f"promote={receipt['promote']} "
        f"pooled_score_delta={receipt['pooled_score_delta']:.6f}"
    )
    print(f"wrote structural report: {destination}")
    return 0 if receipt["promote"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
