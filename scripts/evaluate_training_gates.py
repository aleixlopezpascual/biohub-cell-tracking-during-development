#!/usr/bin/env python3
"""Select checkpoints and enforce learning-curve/ensemble promotion gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from biohub_tracking.training import (
    evaluate_ensemble_gate,
    evaluate_learning_curve_gate,
    load_checkpoint_scores,
    load_gold_training_config,
    rank_complementary_models,
    select_best_checkpoint,
    write_diagnostic_curve_svgs,
    write_learning_curve_svg,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--baseline-epoch", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ensemble-score", type=float)
    parser.add_argument("--runtime-hours", type=float)
    parser.add_argument("--runtime-limit-hours", type=float, default=12.0)
    parser.add_argument(
        "--oof-errors",
        type=Path,
        help="Optional long CSV with checkpoint,sample,error for diversity ranking.",
    )
    parser.add_argument("--max-models", type=int, default=3)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Optional fold-level epoch diagnostics CSV for acceptance plots.",
    )
    return parser.parse_args(argv)


def _load_oof_errors(path: Path) -> dict[str, list[float]]:
    frame = pd.read_csv(path)
    required = {"checkpoint", "sample", "error"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF error CSV is missing columns: {missing}")
    if frame.duplicated(["checkpoint", "sample"]).any():
        raise ValueError("OOF error CSV contains duplicate checkpoint/sample rows")
    pivot = frame.pivot(index="sample", columns="checkpoint", values="error").sort_index()
    if pivot.isna().any().any():
        raise ValueError("each checkpoint must contain the same OOF samples")
    return {
        str(checkpoint): [float(value) for value in pivot[checkpoint].to_numpy()]
        for checkpoint in pivot.columns
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_gold_training_config(args.config)
    scores = load_checkpoint_scores(args.scores)
    baseline_candidates = [score for score in scores if score.epoch == args.baseline_epoch]
    if len(baseline_candidates) != 1:
        raise ValueError(
            f"expected exactly one baseline at epoch {args.baseline_epoch}, "
            f"found {len(baseline_candidates)}"
        )
    best = select_best_checkpoint(scores)
    curve = evaluate_learning_curve_gate(
        baseline_candidates[0],
        scores,
        minimum_mean_gain=config.learning_curve.minimum_mean_gain,
        maximum_fold_regression=config.learning_curve.maximum_fold_regression,
    )
    payload: dict[str, object] = {
        "best_checkpoint": best.checkpoint,
        "best_epoch": best.epoch,
        "best_mean_score": best.mean_score,
        "learning_curve_gate": asdict(curve),
    }
    if args.oof_errors is not None:
        shortlist = rank_complementary_models(
            scores,
            _load_oof_errors(args.oof_errors),
            max_models=args.max_models,
        )
        payload["diverse_model_shortlist"] = [
            {
                "checkpoint": score.checkpoint,
                "epoch": score.epoch,
                "mean_score": score.mean_score,
            }
            for score in shortlist
        ]
    if (args.ensemble_score is None) != (args.runtime_hours is None):
        raise ValueError("--ensemble-score and --runtime-hours must be supplied together")
    if args.ensemble_score is not None and args.runtime_hours is not None:
        payload["ensemble_gate"] = asdict(
            evaluate_ensemble_gate(
                best_single_score=best.mean_score,
                ensemble_score=args.ensemble_score,
                measured_runtime_hours=args.runtime_hours,
                runtime_limit_hours=args.runtime_limit_hours,
                minimum_gain=config.learning_curve.minimum_ensemble_gain,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curve_path = args.output.with_suffix(".svg")
    write_learning_curve_svg(scores, curve_path)
    if args.diagnostics is not None:
        diagnostic_paths = write_diagnostic_curve_svgs(
            args.diagnostics,
            args.output.with_name(f"{args.output.stem}_diagnostics"),
        )
        payload["diagnostic_plots"] = [str(path) for path in diagnostic_paths]
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {curve_path}")


if __name__ == "__main__":
    main()
