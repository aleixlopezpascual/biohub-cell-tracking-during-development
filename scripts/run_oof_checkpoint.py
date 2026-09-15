#!/usr/bin/env python3
"""Run exact-split OOF inference and persist official Royerlab graph scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from biohub_tracking.training import (
    append_checkpoint_score,
    load_gold_training_config,
    load_official_script,
    official_import_context,
    validate_run_manifest,
)
from biohub_tracking.training.checkpoints import RunManifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--fold-index", required=True, type=int)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", help="Torch device, defaulting to CUDA when available.")
    parser.add_argument(
        "--detection-threshold",
        type=float,
        help="Override the configured sigmoid detection threshold for a calibrated OOF run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep already completed GEFF predictions and finish missing datasets.",
    )
    parser.add_argument(
        "--use-overlay",
        action="store_true",
        help="Enable advanced inference overlays (Edge-Feature TTA and Bidirectional Tracking).",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip GPU inference and score an already complete output directory.",
    )
    return parser.parse_args(argv)


def _manifest(path: Path) -> RunManifest:
    return RunManifest(**json.loads(path.read_text(encoding="utf-8")))


def _dataset_artifacts(data_dir: Path, name: str) -> tuple[Path, Path]:
    stem = name.removesuffix(".zarr").removesuffix(".geff")
    return data_dir / f"{stem}.zarr", data_dir / f"{stem}.geff"


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class OverlayPredictorWrapper:
    """Wraps any loaded model instance to run Feature-Level TTA and Bidirectional Tracking."""

    def __init__(self, model: Any, use_tta: bool = True, use_bidirectional: bool = True) -> None:
        self.model = model
        self.use_tta = use_tta
        self.use_bidirectional = use_bidirectional

    def __getattr__(self, name: str) -> Any:
        # Fallback to the underlying model for all unhandled attributes/methods (like training, state_dict, etc.)
        return getattr(self.model, name)

    def encode(self, frames: Any) -> tuple[Any, Any]:
        """Intercept the U-Net forward encoder pass to apply Feature-Level TTA in canonical space."""
        if not self.use_tta:
            return self.model.encode(frames)
            
        from biohub_tracking.baselines.royerlab.tta import XY_D4_TRANSFORMS, apply_spatial_transform, invert_spatial_transform
        import numpy as np
        
        heatmaps = []
        features = []
        for transform in XY_D4_TRANSFORMS:
            aug_frames = apply_spatial_transform(frames, transform)
            # Call encode on the underlying model
            f_aug, h_aug = self.model.encode(aug_frames)
            heatmaps.append(invert_spatial_transform(h_aug, transform))
            features.append(invert_spatial_transform(f_aug, transform))
            
        return np.mean(features, axis=0), np.mean(heatmaps, axis=0)

    def predict_heatmaps_and_features(self, frames: Any) -> tuple[Any, Any]:
        """Direct alias in case the codebase protocol name is called."""
        f, h = self.encode(frames)
        return h, f

    def predict_edge_logits(self, src_features: Any, tgt_features: Any) -> Any:
        """Intercept the edge model pass to apply Bidirectional Tracking probability fusion."""
        if not self.use_bidirectional:
            return self.model.predict_edge_logits(src_features, tgt_features)
            
        from biohub_tracking.baselines.royerlab.linking import fuse_bidirectional_probabilities
        import numpy as np
        
        # Predict forward probabilities
        p_forward = self.model.predict_edge_logits(src_features, tgt_features)
        
        # Predict reverse probabilities
        p_reverse = self.model.predict_edge_logits(tgt_features, src_features)
        p_reverse_aligned = p_reverse.T
        
        # Fuse using existing harmonic block
        return fuse_bidirectional_probabilities(p_forward, p_reverse_aligned, mode="harmonic")


def _load_model(predictor: Any, config: Any, weights: Path, device: Any) -> Any:
    import torch

    unet = predictor.TemporalUNet3D(
        in_channels=1,
        out_channels=config.unet_out_channels,
        layers=list(config.unet_layers),
    )
    model = predictor.UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config.unet_out_channels,
        pos_feat_dim=4 * predictor._POS_EMBED_DIM,
        hidden_dim=config.transformer_hidden_dim,
        n_heads=config.transformer_heads,
        n_blocks=config.transformer_blocks,
        dropout=config.transformer_dropout,
    )
    state = torch.load(weights, map_location=device, weights_only=True)
    if not isinstance(state, dict) or "model_state" in state:
        raise ValueError("--weights must be the adjacent plain *_model.pth inference state")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _predict_config(
    predictor: Any,
    config: Any,
    detection_threshold: float | None = None,
) -> Any:
    inference = config.oof_inference
    threshold = inference.detection_threshold if detection_threshold is None else detection_threshold
    if not 0 < threshold < 1:
        raise ValueError("detection threshold must be between zero and one")
    return predictor.PredictConfig(
        det_threshold=threshold,
        det_tta=inference.detection_tta,
        pool_kernel_um=config.pool_kernel_um,
        edge_activation=inference.edge_activation,
        threshold=inference.edge_threshold,
        use_ilp=inference.use_ilp,
        ilp_edge_weight=inference.ilp_edge_weight,
        ilp_appearance_weight=inference.ilp_appearance_weight,
        ilp_disappearance_weight=inference.ilp_disappearance_weight,
        ilp_division_weight=inference.ilp_division_weight,
        max_parents_per_node=inference.max_parents_per_node,
        max_children_per_node=inference.max_children_per_node,
    )


def _write_submission(
    source_dir: Path,
    predictions: Path,
    output: Path,
    evaluator: Any,
) -> None:
    converter = source_dir / "scripts" / "geffs_to_csv.py"
    if converter.is_file():
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--in-dir",
                str(predictions),
                "--csv",
                str(output),
            ],
            cwd=source_dir,
            check=True,
        )
        return

    rows: list[dict[str, object]] = []
    for geff_path in sorted(predictions.glob("*.geff")):
        result = evaluator.td.graph.IndexedRXGraph.from_geff(geff_path)
        graph = result[0] if isinstance(result, tuple) else result
        nodes = sorted(
            graph.node_attrs().to_dicts(),
            key=lambda row: (int(row["t"]), int(row["node_id"])),
        )
        edges = sorted(
            graph.edge_attrs().to_dicts(),
            key=lambda row: (int(row["source_id"]), int(row["target_id"])),
        )
        for node in nodes:
            rows.append(
                {
                    "dataset": geff_path.stem,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": round(float(node["z"])),
                    "y": round(float(node["y"])),
                    "x": round(float(node["x"])),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
        for edge in edges:
            rows.append(
                {
                    "dataset": geff_path.stem,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": int(edge["source_id"]),
                    "target_id": int(edge["target_id"]),
                }
            )
    submission = pd.DataFrame(
        rows,
        columns=(
            "dataset",
            "row_type",
            "node_id",
            "t",
            "z",
            "y",
            "x",
            "source_id",
            "target_id",
        ),
    )
    submission.insert(0, "id", range(len(submission)))
    submission.to_csv(output, index=False)


def _evaluate_pairs(
    evaluator: Any,
    predictions_dir: Path,
    data_dir: Path,
    names: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Call either the current or pinned-support-pack official scorer API."""
    if hasattr(evaluator, "evaluate_pairs"):
        return evaluator.evaluate_pairs(predictions_dir, data_dir, max_distance=7.0)
    if not hasattr(evaluator, "evaluate_run"):
        raise ImportError("official evaluator exposes neither evaluate_pairs nor evaluate_run")
    evaluator.DATA_DIR = data_dir
    run = {
        "username": "oof",
        "method": "exact_checkpoint",
        "split": "split_0",
        "dir": predictions_dir,
        "geffs": [predictions_dir / f"{name}.geff" for name in sorted(names)],
    }
    rows = evaluator.evaluate_run(run, max_distance=7.0)
    skipped = []
    for row in rows:
        try:
            valid = math.isfinite(float(row["edge_tp"]))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            skipped.append(str(row.get("dataset", "unknown")))
    return rows, skipped


def _append_oof_errors(path: Path, checkpoint: str, frame: pd.DataFrame) -> None:
    records = pd.DataFrame(
        {
            "checkpoint": checkpoint,
            "sample": frame["dataset"].astype(str),
            "error": 1.0 - frame["score"].astype(float),
        }
    )
    if path.exists():
        existing = pd.read_csv(path)
        required = {"checkpoint", "sample", "error"}
        missing = sorted(required - set(existing.columns))
        if missing:
            raise ValueError(f"OOF error log is missing columns: {missing}")
        combined = pd.concat([existing, records], ignore_index=True)
        duplicate = combined.duplicated(["checkpoint", "sample"], keep=False)
        if duplicate.any():
            duplicated = combined[duplicate]
            for _keys, group in duplicated.groupby(["checkpoint", "sample"]):
                values = group["error"].astype(float).tolist()
                if any(
                    not math.isclose(value, values[0], rel_tol=0.0, abs_tol=1e-12)
                    for value in values[1:]
                ):
                    raise ValueError("OOF error log contains a conflicting checkpoint/sample")
            existing_keys = set(
                zip(existing["checkpoint"].astype(str), existing["sample"].astype(str))
            )
            keep = [
                (str(row.checkpoint), str(row.sample)) not in existing_keys
                for row in records.itertuples(index=False)
            ]
            records = records.loc[keep]
    if not records.empty:
        records.to_csv(path, mode="a", header=not path.exists(), index=False)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_gold_training_config(args.config)
    if args.epoch not in config.learning_curve.evaluation_epochs:
        raise ValueError("epoch must be one of learning_curve.evaluation_epochs")
    if args.fold_index < 0:
        raise ValueError("fold index must be non-negative")
    folds = json.loads(args.splits.read_text(encoding="utf-8"))
    try:
        fold = folds[args.fold_index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"fold index {args.fold_index} is absent") from exc
    if not isinstance(fold, dict) or not {"name", "test"} <= set(fold):
        raise ValueError("selected fold must contain name and test fields")
    run_manifest = _manifest(Path(config.output_dir) / "run_manifest.json")
    validate_run_manifest(
        run_manifest,
        config_path=args.config,
        split_path=args.splits,
        candidate=args.candidate,
    )
    if run_manifest.fold != fold["name"]:
        raise ValueError("run manifest fold differs from the selected OOF fold")
    if not args.weights.is_file():
        raise FileNotFoundError(f"inference weights not found: {args.weights}")

    names = [str(name).removesuffix(".zarr").removesuffix(".geff") for name in fold["test"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("OOF test list must be non-empty and unique")
    data_dir = Path(config.data_dir)
    missing = [
        str(path)
        for name in names
        for path in _dataset_artifacts(data_dir, name)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"OOF datasets are incomplete; missing: {missing}")

    output_dir = (
        Path(config.output_dir)
        / "oof"
        / args.candidate
        / f"epoch_{args.epoch:03d}"
        / f"split_{args.fold_index}"
    )
    effective_detection_threshold = (
        config.oof_inference.detection_threshold
        if args.detection_threshold is None
        else args.detection_threshold
    )
    if not 0 < effective_detection_threshold < 1:
        raise ValueError("detection threshold must be between zero and one")
    if args.detection_threshold is not None:
        threshold_name = f"det_{effective_detection_threshold:.6f}".rstrip("0").rstrip(".")
        output_dir = output_dir / threshold_name.replace(".", "p")
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in names if (predictions_dir / f"{name}.geff").exists()]
    if existing and not (args.resume or args.score_only):
        raise FileExistsError(
            f"OOF predictions already exist for {existing}; pass --resume or --score-only"
        )

    source_dir = Path(config.official_source_dir).resolve()
    if not args.score_only:
        import torch

        predictor = load_official_script(
            source_dir,
            "predict_unet_transformer.py",
            module_name="biohub_official_prediction",
            required={
                "PredictConfig",
                "TemporalUNet3D",
                "UNetNodeTransformer",
                "_POS_EMBED_DIM",
                "build_graph",
                "predict_video",
                "save_graph",
                "td",
            },
        )
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = _load_model(predictor, config, args.weights, device)
        if args.use_overlay:
            print("Applying Advanced Inference Overlays wrapper (Edge-Feature TTA + Bidirectional Tracking)...", flush=True)
            model = OverlayPredictorWrapper(model, use_tta=True, use_bidirectional=True)
        predict_config = _predict_config(
            predictor,
            config,
            detection_threshold=effective_detection_threshold,
        )
        prediction_node_counts: list[int] = []
        for name in names:
            prediction_path = predictions_dir / f"{name}.geff"
            if prediction_path.exists() and args.resume:
                print(f"reuse {prediction_path}", flush=True)
                continue
            coords, edges = predictor.predict_video(
                model,
                data_dir / name,
                device,
                cfg=predict_config,
                window_size=config.window_size,
                downsample=config.downsample,
            )
            graph = predictor.build_graph(coords, edges)
            if predict_config.use_ilp and graph.num_edges() > 0:
                solver = predictor.td.solvers.ILPSolver(
                    edge_weight=predict_config.ilp_edge_weight
                    * predictor.td.EdgeAttr("edge_prob"),
                    appearance_weight=predict_config.ilp_appearance_weight,
                    disappearance_weight=predict_config.ilp_disappearance_weight,
                    division_weight=predict_config.ilp_division_weight,
                )
                graph = solver.solve(graph)
            prediction_node_counts.append(int(graph.num_nodes()))
            predictor.save_graph(graph, prediction_path)
            print(f"wrote {prediction_path}", flush=True)
            if len(prediction_node_counts) == min(3, len(names)) and not any(
                prediction_node_counts
            ):
                raise RuntimeError(
                    "detection threshold produced zero nodes for the first "
                    f"{len(prediction_node_counts)} OOF datasets; aborting before full inference"
                )

    missing_predictions = [
        name for name in names if not (predictions_dir / f"{name}.geff").exists()
    ]
    if missing_predictions:
        raise FileNotFoundError(f"OOF predictions are incomplete: {missing_predictions}")
    actual_predictions = {path.stem for path in predictions_dir.glob("*.geff")}
    if actual_predictions != set(names):
        raise ValueError("prediction directory contains datasets outside the selected OOF fold")
    evaluator = load_official_script(
        source_dir,
        "evaluate.py",
        module_name="biohub_official_evaluation",
        required={"summarise", "td"},
    )
    with official_import_context(source_dir):
        rows, skipped = _evaluate_pairs(evaluator, predictions_dir, data_dir, names)
    if skipped or len(rows) != len(names):
        raise RuntimeError(f"official evaluation was incomplete; skipped={skipped}")
    per_dataset = pd.DataFrame(rows)
    if "dataset" in per_dataset:
        if set(per_dataset["dataset"].astype(str)) != set(names):
            raise ValueError("official evaluator returned unexpected dataset names")
        per_dataset = per_dataset.sort_values("dataset", kind="stable").reset_index(drop=True)
    else:
        per_dataset.insert(0, "dataset", sorted(names))
    division_total = (
        per_dataset["division_tp"]
        + per_dataset["division_fp"]
        + per_dataset["division_fn"]
    )
    per_dataset["division_jaccard"] = per_dataset["division_tp"] / division_total
    per_dataset.loc[division_total == 0, "division_jaccard"] = float("nan")
    per_dataset["score"] = per_dataset["adj_edge_jaccard"]
    has_divisions = division_total > 0
    per_dataset.loc[has_divisions, "score"] = (
        per_dataset.loc[has_divisions, "adj_edge_jaccard"]
        + 0.1 * per_dataset.loc[has_divisions, "division_jaccard"]
    )
    per_dataset["node_count_ratio"] = 1.0 + per_dataset["total_node_ratio"]
    with official_import_context(source_dir):
        summary = _json_value(evaluator.summarise(rows))
    if int(summary.get("n", -1)) != len(names):
        raise RuntimeError("official summary does not cover every selected OOF dataset")
    if "score" not in summary:
        raise ValueError("official summary is missing its score")
    if not math.isfinite(float(summary["score"])):
        raise ValueError("official score is non-finite; check estimated node-count metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_dataset_path = output_dir / "official_per_dataset.csv"
    summary_path = output_dir / "official_summary.json"
    submission_path = output_dir / "submission.csv"
    per_dataset.to_csv(per_dataset_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with official_import_context(source_dir):
        _write_submission(source_dir, predictions_dir, submission_path, evaluator)

    checkpoint = (
        f"{args.weights.resolve()}#det_threshold={effective_detection_threshold:.6f}"
    )
    append_checkpoint_score(
        Path(config.output_dir) / "checkpoint_scores.csv",
        epoch=args.epoch,
        checkpoint=checkpoint,
        fold=str(fold["name"]),
        score=float(summary["score"]),
    )
    _append_oof_errors(Path(config.output_dir) / "oof_errors.csv", checkpoint, per_dataset)
    artifact_manifest = {
        "candidate": args.candidate,
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "config_sha256": run_manifest.config_sha256,
        "datasets": sorted(names),
        "epoch": args.epoch,
        "fold": fold["name"],
        "official_score": float(summary["score"]),
        "oof_inference": {
            **config.oof_inference.model_dump(mode="json"),
            "detection_threshold": effective_detection_threshold,
        },
        "split_sha256": run_manifest.split_sha256,
    }
    (output_dir / "oof_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
