#!/usr/bin/env python3
"""Calibrate an OOF detection threshold on a small prefix-held frame probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from biohub_tracking.metrics.matching import match_nodes
from biohub_tracking.tracking.graph import Detection
from biohub_tracking.training import (
    load_gold_training_config,
    load_official_script,
    validate_run_manifest,
)

try:
    from scripts.run_oof_checkpoint import _load_model, _manifest
except ModuleNotFoundError:
    from run_oof_checkpoint import _load_model, _manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded calibration probe arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--fold-index", required=True, type=int)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, nargs="+", type=float)
    parser.add_argument("--max-datasets", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def select_threshold(rows: list[dict[str, Any]]) -> float:
    """Prefer estimated node ratio near one, then higher annotated recall."""
    usable = [row for row in rows if float(row["node_count_ratio"]) > 0]
    if not usable:
        raise ValueError("threshold probe produced no detections")
    best = min(
        usable,
        key=lambda row: (
            abs(math.log(float(row["node_count_ratio"]))),
            -float(row["annotated_node_recall"]),
            -float(row["threshold"]),
        ),
    )
    return float(best["threshold"])


def _gt_probe_nodes(dataset: Any, max_frames: int) -> list[Detection]:
    scale = tuple(float(value) for value in dataset.scale)
    nodes = []
    for row in dataset.tracks.node_attrs(
        attr_keys=["node_id", "t", "z", "y", "x"]
    ).to_dicts():
        if int(row["t"]) >= max_frames:
            continue
        nodes.append(
            Detection(
                id=int(row["node_id"]),
                frame=int(row["t"]),
                z=float(row["z"]) * scale[0],
                y=float(row["y"]) * scale[1],
                x=float(row["x"]) * scale[2],
            )
        )
    return nodes


def _detector_probe(
    predictor: Any,
    model: Any,
    dataset: Any,
    device: Any,
    *,
    max_frames: int,
    window_size: int,
    downsample: tuple[int, int, int],
    pool_kernel_um: float,
    use_tta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the detector once and return every local maximum and probability."""
    import torch
    import torch.nn.functional as functional

    zarr_array = predictor.zarr.open_group(str(dataset.zarr_path), mode="r")["0"]
    q_low = float(dataset.quantiles["0.001"])
    q_high = float(dataset.quantiles["0.999"])
    frame_count = min(max_frames, int(dataset.image_shape[0]))
    target_shape = list(dataset.image_shape[1:])
    voxel_size = tuple(
        float(scale) * int(stride)
        for scale, stride in zip(dataset.scale, downsample)
    )
    pool_kernel = predictor.pool_kernel_from_um(pool_kernel_um, voxel_size)
    padding = tuple(size // 2 for size in pool_kernel)
    stride = max(window_size - 1, 1)
    starts = list(range(0, frame_count - window_size + 1, stride))
    if not starts or starts[-1] + window_size < frame_count:
        final_start = max(frame_count - window_size, 0)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)

    seen_frames: set[int] = set()
    coordinates = []
    probabilities = []
    with torch.no_grad():
        for start in starts:
            frame_indices = list(range(start, start + window_size))
            images = torch.stack(
                [
                    predictor._load_frame(zarr_array, frame, target_shape, downsample)
                    for frame in frame_indices
                ]
            )
            images = ((images - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
            images = images.unsqueeze(0).to(device)
            _features, logits = model.encode(images)
            if use_tta:
                for dims in [(-1,), (-2,), (-2, -1)]:
                    flipped_images = images.flip(dims)
                    _flipped_features, flipped_logits = model.encode(flipped_images)
                    for index in range(window_size):
                        logits[index] = logits[index] + flipped_logits[index].flip(dims)
                logits = [value / 4 for value in logits]
            for index, frame in enumerate(frame_indices):
                if frame in seen_frames:
                    continue
                frame_logits = logits[index][0].unsqueeze(0)
                pooled = functional.max_pool3d(
                    frame_logits,
                    pool_kernel,
                    stride=1,
                    padding=padding,
                )
                maxima = frame_logits == pooled
                frame_coordinates = torch.nonzero(maxima[0, 0]).cpu().numpy()
                frame_probabilities = torch.sigmoid(frame_logits[0, 0][maxima[0, 0]])
                if len(frame_coordinates):
                    original_coordinates = frame_coordinates.astype(np.float32)
                    original_coordinates *= np.asarray(downsample, dtype=np.float32)
                    time_column = np.full((len(original_coordinates), 1), frame)
                    coordinates.append(
                        np.concatenate([time_column, original_coordinates], axis=1)
                    )
                    probabilities.append(frame_probabilities.cpu().numpy())
                seen_frames.add(frame)
    if not coordinates:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(coordinates), np.concatenate(probabilities)


def main(argv: list[str] | None = None) -> None:
    """Run detector probes and persist the recommended threshold."""
    args = parse_args(argv)
    if args.max_datasets < 1 or args.max_frames < 2:
        raise ValueError("max-datasets must be positive and max-frames must be at least two")
    thresholds = sorted(set(args.thresholds))
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("thresholds must be unique probabilities between zero and one")
    config = load_gold_training_config(args.config)
    folds = json.loads(args.splits.read_text(encoding="utf-8"))
    try:
        fold = folds[args.fold_index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"fold index {args.fold_index} is absent") from exc
    manifest = _manifest(Path(config.output_dir) / "run_manifest.json")
    validate_run_manifest(
        manifest,
        config_path=args.config,
        split_path=args.splits,
        candidate=args.candidate,
    )
    names = [
        str(name).removesuffix(".zarr").removesuffix(".geff")
        for name in fold["test"][: args.max_datasets]
    ]
    if not names:
        raise ValueError("selected fold has no probe datasets")

    import torch
    from geff import GeffMetadata

    source_dir = Path(config.official_source_dir).resolve()
    predictor = load_official_script(
        source_dir,
        "predict_unet_transformer.py",
        module_name="biohub_threshold_probe_prediction",
        required={
            "PredictConfig",
            "TemporalUNet3D",
            "UNetNodeTransformer",
            "_POS_EMBED_DIM",
            "_load_frame",
            "open_dataset",
            "pool_kernel_from_um",
            "zarr",
        },
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(predictor, config, args.weights, device)
    data_dir = Path(config.data_dir)
    probes = []
    for name in names:
        dataset = predictor.open_dataset(
            data_dir / name,
            normalize=False,
            load_image=False,
            require_tracks=True,
            downsample=config.downsample,
        )
        frame_count = min(args.max_frames, int(dataset.image_shape[0]))
        coordinates, probabilities = _detector_probe(
            predictor,
            model,
            dataset,
            device,
            max_frames=frame_count,
            window_size=config.window_size,
            downsample=config.downsample,
            pool_kernel_um=config.pool_kernel_um,
            use_tta=config.oof_inference.detection_tta,
        )
        ground_truth = _gt_probe_nodes(dataset, frame_count)
        metadata = GeffMetadata.read(data_dir / f"{name}.geff")
        estimated_full = float((metadata.extra or {})["estimated_number_of_nodes"])
        estimated_probe = estimated_full * frame_count / int(dataset.image_shape[0])
        probes.append(
            {
                "name": name,
                "dataset": dataset,
                "frames": frame_count,
                "coordinates": coordinates,
                "probabilities": probabilities,
                "ground_truth": ground_truth,
                "estimated_nodes": estimated_probe,
            }
        )

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted_total = 0
        estimated_total = 0.0
        annotated_total = 0
        matched_total = 0
        details = []
        for probe in probes:
            dataset = probe["dataset"]
            selected = probe["probabilities"] > threshold
            coords = probe["coordinates"][selected]
            scale = tuple(float(value) for value in dataset.scale)
            predictions = [
                Detection(
                    id=index,
                    frame=int(row[0]),
                    z=float(row[1]) * scale[0],
                    y=float(row[2]) * scale[1],
                    x=float(row[3]) * scale[2],
                )
                for index, row in enumerate(coords)
            ]
            ground_truth = probe["ground_truth"]
            matched = len(match_nodes(predictions, ground_truth, max_distance=7.0))
            estimated_probe = float(probe["estimated_nodes"])
            predicted_total += len(predictions)
            estimated_total += estimated_probe
            annotated_total += len(ground_truth)
            matched_total += matched
            details.append(
                {
                    "dataset": probe["name"],
                    "frames": probe["frames"],
                    "predicted_nodes": len(predictions),
                    "estimated_nodes": estimated_probe,
                    "annotated_nodes": len(ground_truth),
                    "matched_nodes": matched,
                }
            )
        rows.append(
            {
                "threshold": threshold,
                "node_count_ratio": predicted_total / estimated_total,
                "annotated_node_recall": (
                    matched_total / annotated_total if annotated_total else 0.0
                ),
                "predicted_nodes": predicted_total,
                "estimated_nodes": estimated_total,
                "annotated_nodes": annotated_total,
                "matched_nodes": matched_total,
                "datasets": details,
            }
        )
    selected = select_threshold(rows)
    payload = {
        "status": "passed",
        "candidate": args.candidate,
        "checkpoint": str(args.weights),
        "fold": fold["name"],
        "max_datasets": len(names),
        "max_frames": args.max_frames,
        "recommended_threshold": selected,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
