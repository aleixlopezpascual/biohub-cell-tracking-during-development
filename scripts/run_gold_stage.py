#!/usr/bin/env python3
"""Run one gated training epoch target followed by official OOF scoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from biohub_tracking.training import (
    GoldTrainingConfig,
    evaluate_learning_curve_gate,
    load_checkpoint_scores,
    load_gold_training_config,
    load_training_checkpoint,
    save_inference_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--target-epoch", required=True, type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest training state and reuse completed OOF predictions.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip training and only finish/score OOF inference for the target.",
    )
    return parser.parse_args(argv)


def _run(command: list[str], repo_root: Path) -> None:
    print("run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def _fold_index(split_path: Path, fold_name: str) -> int:
    folds = json.loads(split_path.read_text(encoding="utf-8"))
    matches = [index for index, fold in enumerate(folds) if fold.get("name") == fold_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one split named {fold_name!r}")
    return matches[0]


def _previous_score(scores_path: Path, epoch: int, fold: str) -> float:
    scores = load_checkpoint_scores(scores_path)
    matches = [score for score in scores if score.epoch == epoch and fold in score.fold_scores]
    if len(matches) != 1:
        raise ValueError(f"expected one official score for fold {fold} at epoch {epoch}")
    return float(matches[0].fold_scores[fold])


def _enforce_later_stage_gate(
    scores_path: Path,
    epochs: tuple[int, ...],
    target_epoch: int,
    config: GoldTrainingConfig,
) -> None:
    target_index = epochs.index(target_epoch)
    if target_index < 2:
        return
    scores = load_checkpoint_scores(scores_path)
    baseline = [score for score in scores if score.epoch == epochs[0]]
    candidates = [score for score in scores if score.epoch < target_epoch]
    if len(baseline) != 1:
        raise ValueError("later stages require exactly one first-gate baseline score")
    learning_curve = config.learning_curve
    decision = evaluate_learning_curve_gate(
        baseline[0],
        candidates,
        minimum_mean_gain=learning_curve.minimum_mean_gain,
        maximum_fold_regression=learning_curve.maximum_fold_regression,
    )
    if not decision.promote:
        raise RuntimeError(f"quota gate rejected epoch {target_epoch}: {decision.reason}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_gold_training_config(args.config)
    epochs = config.learning_curve.evaluation_epochs
    if args.target_epoch not in epochs:
        raise ValueError("target epoch must be listed in learning_curve.evaluation_epochs")
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(config.output_dir)
    split_path = output_dir / "prefix_holdout_splits.json"
    manifest_path = output_dir / "run_manifest.json"
    if split_path.exists() != manifest_path.exists():
        raise RuntimeError(
            "campaign has only one of split/manifest; recover both from the same saved run"
        )
    if not split_path.exists():
        if args.target_epoch != epochs[0]:
            raise FileNotFoundError("prepare the first stage before requesting a later epoch")
        stale_weights = output_dir / "weights" / args.candidate
        if stale_weights.exists():
            raise RuntimeError(
                f"refusing to prepare new provenance over existing weights: {stale_weights}"
            )
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "prepare_gold_training.py"),
                "--config",
                str(args.config),
                "--candidate",
                args.candidate,
            ],
            repo_root,
        )
    fold_index = _fold_index(split_path, config.fold)
    weights_dir = output_dir / "weights" / args.candidate / f"split_{fold_index}"
    full_checkpoint = weights_dir / f"epoch_{args.target_epoch:03d}.pt"
    inference_weights = weights_dir / f"epoch_{args.target_epoch:03d}_model.pth"
    scores_path = output_dir / "checkpoint_scores.csv"
    _enforce_later_stage_gate(scores_path, epochs, args.target_epoch, config)

    if not args.score_only and not full_checkpoint.exists():
        command = [
            sys.executable,
            str(repo_root / "scripts" / "continue_royerlab_training.py"),
            "--config",
            str(args.config),
            "--splits",
            str(split_path),
            "--fold-index",
            str(fold_index),
            "--candidate",
            args.candidate,
            "--target-epoch",
            str(args.target_epoch),
        ]
        target_index = epochs.index(args.target_epoch)
        if target_index > 0:
            previous_epoch = epochs[target_index - 1]
            resume_checkpoint = weights_dir / f"epoch_{previous_epoch:03d}.pt"
            if args.resume and (weights_dir / "latest.pt").exists():
                resume_checkpoint = weights_dir / "latest.pt"
            command.extend(
                [
                    "--resume-checkpoint",
                    str(resume_checkpoint),
                    "--previous-official-score",
                    str(_previous_score(scores_path, previous_epoch, config.fold)),
                    "--previous-official-epoch",
                    str(previous_epoch),
                ]
            )
        elif args.resume and (weights_dir / "latest.pt").exists():
            command.extend(["--resume-checkpoint", str(weights_dir / "latest.pt")])
        _run(command, repo_root)
    if full_checkpoint.is_file() and not inference_weights.is_file():
        if not args.resume:
            raise FileNotFoundError(
                "full checkpoint exists but inference state is missing; rerun with --resume"
            )
        restored = load_training_checkpoint(full_checkpoint)
        save_inference_checkpoint(inference_weights, restored.model_state)
        print(f"recovered {inference_weights} from {full_checkpoint}", flush=True)
    if not inference_weights.is_file():
        raise FileNotFoundError(f"stage did not produce inference weights: {inference_weights}")

    oof_command = [
        sys.executable,
        str(repo_root / "scripts" / "run_oof_checkpoint.py"),
        "--config",
        str(args.config),
        "--splits",
        str(split_path),
        "--fold-index",
        str(fold_index),
        "--candidate",
        args.candidate,
        "--epoch",
        str(args.target_epoch),
        "--weights",
        str(inference_weights),
    ]
    if args.device:
        oof_command.extend(["--device", args.device])
    if args.resume:
        oof_command.append("--resume")
    if args.score_only:
        oof_command.append("--score-only")
    _run(oof_command, repo_root)
    summary_path = (
        output_dir
        / "oof"
        / args.candidate
        / f"epoch_{args.target_epoch:03d}"
        / f"split_{fold_index}"
        / "official_summary.json"
    )
    print(summary_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
