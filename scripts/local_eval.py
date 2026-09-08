#!/usr/bin/env python3
"""Run local Biohub validation before submitting a candidate to Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_tracking.evaluation import ExperimentLogEntry, append_experiment_log, load_cv_split_plan, score_submission
from biohub_tracking.evaluation.experiment_log import current_commit_sha, file_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path, help="Candidate submission.csv to score.")
    parser.add_argument("--gt-submission", required=True, type=Path, help="Ground-truth submission-style CSV.")
    parser.add_argument("--cv-pack-dir", type=Path, help="Directory containing Biohub Local CV Pack CSVs.")
    parser.add_argument("--fold", choices=["A", "B", "all"], default="all", help="Fold to evaluate.")
    parser.add_argument("--candidate", default="candidate", help="Candidate/run name for logs.")
    parser.add_argument("--config", type=Path, help="Config file used to produce this candidate.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local_eval"), help="Evaluation artifacts directory.")
    parser.add_argument("--max-distance", type=float, default=7.0, help="Node matching radius in microns.")
    parser.add_argument("--log-experiment", action="store_true", help="Append summary to local_eval_log.csv.")
    parser.add_argument("--public-lb", type=float, help="Optional public leaderboard score to record.")
    parser.add_argument("--private-lb", type=float, help="Optional private leaderboard score to record.")
    parser.add_argument("--notes", default="", help="Optional experiment notes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = None
    split_provenance = "all_overlapping_datasets"
    if args.cv_pack_dir is not None:
        plan = load_cv_split_plan(cv_pack_dir=args.cv_pack_dir)
        if args.fold == "all":
            datasets = plan.evaluation_stems()
            split_provenance = plan.provenance
        else:
            fold = plan.fold(args.fold)
            datasets = fold.evaluate
            split_provenance = fold.provenance

    result = score_submission(
        args.submission,
        args.gt_submission,
        datasets=datasets,
        cv_pack_dir=args.cv_pack_dir,
        max_distance=args.max_distance,
    )

    summary = dict(result.summary)
    summary.update(
        {
            "candidate": args.candidate,
            "fold": args.fold,
            "split_provenance": split_provenance,
            "scorer_provenance": result.scorer_provenance,
        }
    )
    summary_path = args.output_dir / "summary.json"
    per_dataset_path = args.output_dir / "per_dataset.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    result.per_dataset.to_csv(per_dataset_path, index=False)

    if args.log_experiment:
        log_path = args.output_dir / "local_eval_log.csv"
        append_experiment_log(
            ExperimentLogEntry(
                candidate=args.candidate,
                score=float(result.aggregate.score),
                adjusted_edge_jaccard=float(result.aggregate.adjusted_edge_jaccard),
                division_jaccard=float(result.aggregate.division_jaccard),
                split=args.fold,
                split_provenance=split_provenance,
                scorer_provenance=result.scorer_provenance,
                commit_sha=current_commit_sha(Path.cwd()),
                config_path=str(args.config or ""),
                config_sha256=file_sha256(args.config),
                public_lb=args.public_lb,
                private_lb=args.private_lb,
                notes=args.notes,
            ),
            log_path,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")
    print(f"wrote {per_dataset_path}")


if __name__ == "__main__":
    main()
