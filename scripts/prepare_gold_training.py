#!/usr/bin/env python3
"""Prepare a reproducible prefix-holdout bootstrap run for Kaggle GPUs.

This command never invents five folds. It converts the Local CV Pack's two
prefix holdouts to the JSON format expected by the official Royerlab trainer,
writes immutable provenance, and optionally launches only the first gated
learning-curve stage. Later checkpoints must be promoted with official graph
scores before additional folds consume quota.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from biohub_tracking.evaluation import load_cv_split_plan
from biohub_tracking.training import build_run_manifest, load_gold_training_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--execute-bootstrap",
        action="store_true",
        help="Run the official trainer through the first evaluation epoch after preparation.",
    )
    return parser.parse_args(argv)


def _write_split_manifest(config_path: Path, output_dir: Path) -> tuple[Path, dict[str, int]]:
    config = load_gold_training_config(config_path)
    plan = load_cv_split_plan(cv_pack_dir=config.cv_pack_dir)
    folds = []
    fold_indices: dict[str, int] = {}
    for index, fold in enumerate(plan.folds):
        fold_indices[fold.name] = index
        folds.append(
            {
                "split": index,
                "name": fold.name,
                "train": list(fold.train),
                "test": list(fold.evaluate),
                "excluded": list(fold.excluded),
                "provenance": fold.provenance,
            }
        )
    path = output_dir / "prefix_holdout_splits.json"
    path.write_text(json.dumps(folds, indent=2, sort_keys=True), encoding="utf-8")
    return path, fold_indices


def _trainer_command(
    *,
    config_path: Path,
    split_path: Path,
    fold_index: int,
    candidate: str,
) -> list[str]:
    config = load_gold_training_config(config_path)
    trainer = Path(config.official_source_dir) / "scripts" / "train_unet_transformer.py"
    if not trainer.is_file():
        raise FileNotFoundError(
            f"official Royerlab trainer not found: {trainer}. Attach or checkout "
            "royerlab/kaggle-cell-tracking-competition before enabling the GPU."
        )
    wrapper = Path(__file__).resolve().parent / "continue_royerlab_training.py"
    return [
        sys.executable,
        str(wrapper),
        "--config",
        str(config_path.resolve()),
        "--splits",
        str(split_path.resolve()),
        "--fold-index",
        str(fold_index),
        "--candidate",
        candidate,
        "--target-epoch",
        str(config.learning_curve.evaluation_epochs[0]),
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_gold_training_config(args.config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path, fold_indices = _write_split_manifest(args.config, output_dir)
    if config.fold not in fold_indices:
        raise ValueError(f"configured fold {config.fold!r} is absent from the split manifest")
    manifest = build_run_manifest(
        candidate=args.candidate,
        config_path=args.config,
        split_path=split_path,
        fold=config.fold,
        seed=config.seed,
        max_runtime_hours=config.max_runtime_hours,
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest.write_json(manifest_path)
    command = _trainer_command(
        config_path=args.config,
        split_path=split_path,
        fold_index=fold_indices[config.fold],
        candidate=args.candidate,
    )
    command_path = output_dir / "bootstrap_command.json"
    command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")
    print(f"wrote {split_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {command_path}")
    if args.execute_bootstrap:
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            timeout=config.max_runtime_hours * 3600,
        )


if __name__ == "__main__":
    main()
