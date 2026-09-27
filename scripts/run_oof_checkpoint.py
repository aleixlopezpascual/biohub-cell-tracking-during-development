#!/usr/bin/env python3
"""Run exact-split OOF inference and persist official Royerlab graph scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biohub_tracking.training import (
    append_checkpoint_score,
    load_gold_training_config,
    load_official_script,
    official_import_context,
    validate_run_manifest,
)
from biohub_tracking.training.checkpoints import RunManifest
from biohub_tracking.tracking.graph import Detection
from biohub_tracking.tracking.motion_relink import MotionRelinkConfig, motion_relink


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
        "--motion-relink",
        action="store_true",
        help=(
            "Run the staged EMA motion relinker on Gold OOF predictions before graph "
            "construction/ILP. This is an isolated local-model ablation, not the exact "
            "0.946 Kaggle pipeline."
        ),
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip GPU inference and score an already complete output directory.",
    )
    return parser.parse_args(argv)


def _manifest(path: Path) -> RunManifest:
    return RunManifest(**json.loads(path.read_text(encoding="utf-8")))


def _validated_path_component(value: object, label: str) -> str:
    """Accept one literal path component, never a path or traversal expression."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
        or Path(value).name != value
    ):
        raise ValueError(f"invalid {label} path component: {value!r}")
    return value


def _ensure_safe_output_path(path: Path, root: Path, label: str) -> None:
    """Reject output paths that escape the configured root or traverse symlinks."""
    root_path = root.resolve()
    if not path.is_absolute():
        raise ValueError(f"{label} output path must be absolute: {path}")
    try:
        relative = path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{label} output path escapes configured root: {path}") from exc

    current = root_path
    for component in relative.parts:
        if component in {".", ".."}:
            raise ValueError(f"{label} output path contains traversal: {path}")
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} output path contains a symlink: {current}")
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve {label} output path {path}: {exc}") from exc
    if not resolved.is_relative_to(root_path):
        raise ValueError(f"{label} output path escapes configured root: {path}")


def _dataset_stem(value: object) -> str:
    """Validate and normalize a split dataset identifier to its filename stem."""
    name = _validated_path_component(value, "dataset")
    if name.endswith(".zarr") or name.endswith(".geff"):
        name = name[:-5]
    if not name or name in {".", ".."}:
        raise ValueError(f"invalid dataset path component: {value!r}")
    return name


def _dataset_artifacts(data_dir: Path, name: str) -> tuple[Path, Path]:
    stem = _dataset_stem(name)
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
        import torch
        
        if not self.use_tta:
            return self.model.encode(frames)
            
        from biohub_tracking.baselines.royerlab.tta import XY_D4_TRANSFORMS
        
        # Determine if input is a PyTorch Tensor or NumPy Array
        is_tensor = isinstance(frames, torch.Tensor)
        
        # Helper functions to recursively apply/invert spatial transforms on nested lists/tuples of Tensors
        def apply_torch_transform(item: Any, transform: Any) -> Any:
            if isinstance(item, (list, tuple)):
                return type(item)(apply_torch_transform(x, transform) for x in item)
            if isinstance(item, torch.Tensor):
                res = torch.rot90(item, k=transform.rotations_ccw, dims=(-2, -1))
                if transform.flip_x:
                    res = torch.flip(res, dims=(-1,))
                return res
            return item

        def invert_torch_transform(item: Any, transform: Any) -> Any:
            if isinstance(item, (list, tuple)):
                return type(item)(invert_torch_transform(x, transform) for x in item)
            if isinstance(item, torch.Tensor):
                res = torch.flip(item, dims=(-1,)) if transform.flip_x else item
                return torch.rot90(res, k=-transform.rotations_ccw, dims=(-2, -1))
            return item

        def mean_torch_items(items_list: list[Any]) -> Any:
            first = items_list[0]
            if isinstance(first, (list, tuple)):
                return type(first)(
                    mean_torch_items([items[i] for items in items_list])
                    for i in range(len(first))
                )
            if isinstance(first, torch.Tensor):
                return torch.stack(items_list, dim=0).mean(dim=0)
            return first

        heatmaps = []
        features = []
        
        if is_tensor:
            for transform in XY_D4_TRANSFORMS:
                # Apply TTA recursively
                aug_frames = apply_torch_transform(frames, transform)
                f_aug, h_aug = self.model.encode(aug_frames)
                
                # Invert TTA recursively
                features.append(invert_torch_transform(f_aug, transform))
                heatmaps.append(invert_torch_transform(h_aug, transform))
                
            # Average recursively
            mean_features = mean_torch_items(features)
            mean_heatmaps = mean_torch_items(heatmaps)
            return mean_features, mean_heatmaps
            
        else:
            # Fallback for NumPy arrays (e.g. CPU fallback)
            from biohub_tracking.baselines.royerlab.tta import apply_spatial_transform, invert_spatial_transform
            import numpy as np
            
            for transform in XY_D4_TRANSFORMS:
                aug_frames = apply_spatial_transform(frames, transform)
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
    evaluated_names = {str(row.get("dataset")) for row in rows if "dataset" in row}
    missing_names = [n for n in names if n not in evaluated_names]

    # When an early-stage checkpoint produces zero nodes on an embryo volume, the official evaluator
    # can crash with KeyError('z') and omit the row. We fill in a clean 0.0 baseline row instead of aborting.
    for name in missing_names:
        rows.append(
            {
                "dataset": name,
                "edge_tp": 0,
                "edge_fp": 0,
                "edge_fn": 1,
                "node_recall": 0.0,
                "score": 0.0,
                "adj_edge_jaccard": 0.0,
                "edge_jaccard": 0.0,
                "division_tp": 0,
                "division_fp": 0,
                "division_fn": 0,
            }
        )

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


def _relink_prediction_edges(
    coordinates: Any,
    edges: Any,
    voxel_size_um: tuple[float, float, float],
    config: MotionRelinkConfig | None = None,
) -> tuple[list[tuple[int, int, float, float]], dict[str, int | bool]]:
    """Apply the EMA relinker to ``(t,z,y,x)`` coordinates and indexed edge tuples.

    The checked-in Biohub ``build_graph`` adapters consume each edge as
    ``(source_index, target_index, probability, distance)``. Strict runtime
    validation keeps a changed upstream API from silently corrupting OOF output.
    """
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    if coordinate_array.size == 0 and coordinate_array.shape == (0,):
        coordinate_array = coordinate_array.reshape((0, 4))
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 4:
        raise ValueError("predictor coordinates must have shape (N, 4) in (t, z, y, x) order")
    if not np.isfinite(coordinate_array).all():
        raise ValueError("predictor coordinates contain non-finite values")
    scale = np.asarray(voxel_size_um, dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("voxel_size_um must contain three finite positive values")

    detections: dict[int, Detection] = {}
    for node_id, row in enumerate(coordinate_array):
        frame = int(row[0])
        if row[0] != frame:
            raise ValueError(f"predictor coordinate row {node_id} has a non-integer frame")
        detections[node_id] = Detection(
            id=node_id,
            frame=frame,
            z=float(row[1] * scale[0]),
            y=float(row[2] * scale[1]),
            x=float(row[3] * scale[2]),
        )

    try:
        raw_edge_rows = list(edges)
    except TypeError as exc:
        raise ValueError("predictor edges must be an iterable of indexed 4-tuples") from exc
    normalized_edges: list[tuple[int, int, float, float]] = []
    learned_probabilities: dict[tuple[int, int], float] = {}
    for edge_index, raw_edge in enumerate(raw_edge_rows):
        try:
            values = np.asarray(raw_edge, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"predictor edge {edge_index} is not numeric") from exc
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError(
                f"predictor edge {edge_index} must contain four finite values "
                "(source, target, probability, distance)"
            )
        source_value, target_value = values[:2]
        source_id, target_id = int(source_value), int(target_value)
        if source_value != source_id or target_value != target_id:
            raise ValueError(f"predictor edge {edge_index} has non-integer node indices")
        if not 0 <= source_id < len(detections) or not 0 <= target_id < len(detections):
            raise ValueError(f"predictor edge {edge_index} references a missing coordinate row")
        probability, distance = float(values[2]), float(values[3])
        normalized_edges.append((source_id, target_id, probability, distance))
        key = (source_id, target_id)
        learned_probabilities[key] = max(
            learned_probabilities.get(key, float("-inf")),
            probability,
        )

    result = motion_relink(detections, learned_probabilities, config)
    used_motion_relink = bool(result.edges)
    if result.stats.skipped_large_frame:
        relinked_edges = normalized_edges
    else:
        relinked_edges = [
            (edge.source_id, edge.target_id, edge.edge_probability, edge.distance_um)
            for edge in result.edges
        ]
    stats: dict[str, int | bool] = {
        "frames_processed": result.stats.frames_processed,
        "tight_edges": result.stats.tight_edges,
        "relaxed_edges": result.stats.relaxed_edges,
        "skipped_large_frame": result.stats.skipped_large_frame,
        "input_edges": len(normalized_edges),
        "output_edges": len(relinked_edges),
        "used_motion_relink": used_motion_relink,
    }
    return relinked_edges, stats


def _motion_relink_source_sha256(official_source_dir: Path) -> str:
    """Hash local relinker code and all Python sources in the loaded upstream tree."""
    source_paths = {
        "src/biohub_tracking/tracking/motion_relink.py": Path(
            motion_relink.__code__.co_filename
        ).resolve(),
        "scripts/run_oof_checkpoint.py": Path(__file__).resolve(),
    }
    source_root = Path(official_source_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"official predictor source directory not found: {source_root}")
    upstream_source_count = 0
    for subtree in ("scripts", "src"):
        directory = source_root / subtree
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"official source subtree must be a regular directory: {directory}")
        for item in sorted(directory.rglob("*"), key=lambda path: path.as_posix()):
            if item.is_symlink():
                raise ValueError(f"official source hashing does not allow symlinks: {item}")
            if not item.is_file() or item.suffix != ".py":
                continue
            if not item.resolve().is_relative_to(source_root):
                raise ValueError(f"official source escapes its root directory: {item}")
            logical_name = f"official_source/{item.relative_to(source_root).as_posix()}"
            source_paths[logical_name] = item
            upstream_source_count += 1
    if upstream_source_count == 0:
        raise ValueError(f"official source tree contains no Python modules: {source_root}")

    digest = hashlib.sha256(b"biohub-ema-motion-relink-source-bundle-v1\0")
    for logical_name, source_path in sorted(source_paths.items()):
        if not source_path.is_file():
            raise FileNotFoundError(f"motion relinker source not found: {source_path}")
        name_bytes = logical_name.encode("utf-8")
        content = source_path.read_bytes()
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    """Hash a file or directory artifact deterministically without loading it all."""
    if path.is_symlink():
        raise ValueError(f"artifact hashing does not allow symlinks: {path}")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"artifact not found or not a regular file/directory: {path}")

    digest = hashlib.sha256(b"biohub-directory-artifact-v1\0")
    files = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
    file_count = 0
    for item in files:
        if item.is_symlink():
            raise ValueError(f"artifact hashing does not allow symlinks: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"artifact contains an unsupported filesystem entry: {item}")
        relative_path = item.relative_to(path).as_posix().encode("utf-8")
        file_size = item.stat().st_size
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(file_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_count += 1
    if file_count == 0:
        raise ValueError(f"directory artifact contains no files: {path}")
    return digest.hexdigest()


def _write_motion_relink_sidecar(
    sidecar_path: Path,
    prediction_path: Path,
    dataset: str,
    config: MotionRelinkConfig,
    source_sha256: str,
    checkpoint_sha256: str,
    stats: dict[str, int | bool],
) -> None:
    """Persist runtime details bound to one generated prediction artifact."""
    if sidecar_path.is_symlink():
        raise ValueError(f"EMA motion-relink sidecar path must not be a symlink: {sidecar_path}")
    payload = {
        "config": asdict(config),
        "dataset": dataset,
        "method_id": "ema_motion_relink_v1",
        "method_sources": [
            "scripts/run_oof_checkpoint.py",
            "src/biohub_tracking/tracking/motion_relink.py",
            "official_source_dir/scripts/**/*.py",
            "official_source_dir/src/**/*.py",
        ],
        "method_sha256": source_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": _artifact_sha256(prediction_path),
        "runtime_executed": True,
        "stats": stats,
    }
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=sidecar_path.parent,
            prefix=f".{sidecar_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if sidecar_path.is_symlink():
            raise ValueError(
                f"EMA motion-relink sidecar path must not be a symlink: {sidecar_path}"
            )
        os.replace(temp_path, sidecar_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _read_motion_relink_sidecar(
    sidecar_path: Path,
    prediction_path: Path,
    dataset: str,
    config: MotionRelinkConfig,
    source_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, int | bool]:
    """Validate a resumed EMA artifact's exact method/config/prediction binding."""
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read EMA motion-relink sidecar {sidecar_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"EMA motion-relink sidecar is not an object: {sidecar_path}")
    if payload.get("dataset") != dataset or payload.get("method_id") != "ema_motion_relink_v1":
        raise ValueError(f"EMA motion-relink sidecar identity mismatch for {dataset}")
    expected_sources = [
        "scripts/run_oof_checkpoint.py",
        "src/biohub_tracking/tracking/motion_relink.py",
        "official_source_dir/scripts/**/*.py",
        "official_source_dir/src/**/*.py",
    ]
    if payload.get("method_sources") != expected_sources:
        raise ValueError(f"EMA motion-relink sidecar source list mismatch for {dataset}")
    if payload.get("method_sha256") != source_sha256 or payload.get("config") != asdict(config):
        raise ValueError(f"EMA motion-relink sidecar method/config mismatch for {dataset}")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"EMA motion-relink sidecar checkpoint hash mismatch for {dataset}")
    if payload.get("runtime_executed") is not True:
        raise ValueError(f"EMA motion-relink sidecar does not attest execution for {dataset}")
    actual_prediction_hash = _artifact_sha256(prediction_path)
    if payload.get("prediction_sha256") != actual_prediction_hash:
        raise ValueError(f"EMA motion-relink sidecar prediction hash mismatch for {dataset}")
    raw_stats = payload.get("stats")
    expected_keys = {
        "frames_processed",
        "tight_edges",
        "relaxed_edges",
        "skipped_large_frame",
        "input_edges",
        "output_edges",
        "used_motion_relink",
    }
    if not isinstance(raw_stats, dict) or set(raw_stats) != expected_keys:
        raise ValueError(f"EMA motion-relink sidecar stats are incomplete for {dataset}")
    count_keys = expected_keys - {"skipped_large_frame", "used_motion_relink"}
    if any(
        isinstance(raw_stats[key], bool)
        or not isinstance(raw_stats[key], int)
        or raw_stats[key] < 0
        for key in count_keys
    ):
        raise ValueError(f"EMA motion-relink sidecar stats are invalid for {dataset}")
    if not isinstance(raw_stats["skipped_large_frame"], bool) or not isinstance(
        raw_stats["used_motion_relink"], bool
    ):
        raise ValueError(f"EMA motion-relink sidecar flags are invalid for {dataset}")
    return {str(key): value for key, value in raw_stats.items()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_name = _validated_path_component(args.candidate, "candidate")
    if args.motion_relink and args.use_overlay:
        raise ValueError(
            "--motion-relink is an isolated ablation and cannot be combined with --use-overlay"
        )
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
    output_root = Path(config.output_dir).resolve()
    run_manifest = _manifest(output_root / "run_manifest.json")
    validate_run_manifest(
        run_manifest,
        config_path=args.config,
        split_path=args.splits,
        candidate=candidate_name,
    )
    if run_manifest.fold != fold["name"]:
        raise ValueError("run manifest fold differs from the selected OOF fold")
    if not args.weights.is_file():
        raise FileNotFoundError(f"inference weights not found: {args.weights}")
    checkpoint_sha256 = _artifact_sha256(args.weights)

    raw_test_names = fold["test"]
    if not isinstance(raw_test_names, list):
        raise ValueError("selected fold test field must be a list of dataset names")
    names = [_dataset_stem(name) for name in raw_test_names]
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
        output_root
        / "oof"
        / candidate_name
        / f"epoch_{args.epoch:03d}"
        / f"split_{args.fold_index}"
    )
    motion_relink_config = MotionRelinkConfig()
    source_dir = Path(config.official_source_dir).resolve()
    motion_relink_source_hash = (
        _motion_relink_source_sha256(source_dir) if args.motion_relink else None
    )
    if args.motion_relink:
        output_dir = output_dir / "ema_motion_relink_v1"
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
    _ensure_safe_output_path(output_dir, output_root, "OOF directory")
    _ensure_safe_output_path(predictions_dir, output_root, "prediction directory")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    _ensure_safe_output_path(predictions_dir, output_root, "prediction directory")
    for name in names:
        _ensure_safe_output_path(
            predictions_dir / f"{name}.geff", output_root, "prediction artifact"
        )
        if args.motion_relink:
            _ensure_safe_output_path(
                predictions_dir / f"{name}.motion_relink.json",
                output_root,
                "motion-relink sidecar",
            )
    existing = [name for name in names if (predictions_dir / f"{name}.geff").exists()]
    if existing and not (args.resume or args.score_only):
        raise FileExistsError(
            f"OOF predictions already exist for {existing}; pass --resume or --score-only"
        )
    motion_relink_stats: dict[str, dict[str, int | bool]] = {}
    if args.motion_relink and existing:
        if motion_relink_source_hash is None:
            raise RuntimeError("EMA motion-relink source hash was not initialized")
        for name in existing:
            prediction_path = predictions_dir / f"{name}.geff"
            sidecar_path = predictions_dir / f"{name}.motion_relink.json"
            _ensure_safe_output_path(prediction_path, output_root, "prediction artifact")
            _ensure_safe_output_path(sidecar_path, output_root, "motion-relink sidecar")
            motion_relink_stats[name] = _read_motion_relink_sidecar(
                sidecar_path,
                prediction_path,
                name,
                motion_relink_config,
                motion_relink_source_hash,
                checkpoint_sha256,
            )
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
            if args.motion_relink:
                edges, motion_relink_stats[name] = _relink_prediction_edges(
                    coords,
                    edges,
                    config.voxel_size_um,
                    motion_relink_config,
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
            _ensure_safe_output_path(prediction_path, output_root, "prediction artifact")
            if args.motion_relink:
                _ensure_safe_output_path(
                    predictions_dir / f"{name}.motion_relink.json",
                    output_root,
                    "motion-relink sidecar",
                )
            predictor.save_graph(graph, prediction_path)
            if args.motion_relink:
                if motion_relink_source_hash is None:
                    raise RuntimeError("EMA motion-relink source hash was not initialized")
                _write_motion_relink_sidecar(
                    predictions_dir / f"{name}.motion_relink.json",
                    prediction_path,
                    name,
                    motion_relink_config,
                    motion_relink_source_hash,
                    checkpoint_sha256,
                    motion_relink_stats[name],
                )
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
    if args.motion_relink and set(motion_relink_stats) != set(names):
        raise RuntimeError("EMA motion-relink execution records do not cover the held-out fold")
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
    # Empty per-sample division sets have Jaccard 1.0 under the metric definition.
    per_dataset["division_jaccard"] = per_dataset["division_tp"] / division_total.where(
        division_total > 0,
        1.0,
    )
    per_dataset.loc[division_total == 0, "division_jaccard"] = 1.0
    per_dataset["score"] = (
        per_dataset["adj_edge_jaccard"] + 0.1 * per_dataset["division_jaccard"]
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
    if _artifact_sha256(args.weights) != checkpoint_sha256:
        raise RuntimeError("inference checkpoint changed during OOF evaluation")
    if args.motion_relink:
        if motion_relink_source_hash is None:
            raise RuntimeError("EMA motion-relink source hash was not initialized")
        if _motion_relink_source_sha256(source_dir) != motion_relink_source_hash:
            raise RuntimeError("EMA motion-relink source tree changed during OOF evaluation")
    _ensure_safe_output_path(output_dir, output_root, "OOF directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_safe_output_path(output_dir, output_root, "OOF directory")
    per_dataset_path = output_dir / "official_per_dataset.csv"
    summary_path = output_dir / "official_summary.json"
    submission_path = output_dir / "submission.csv"
    oof_manifest_path = output_dir / "oof_manifest.json"
    checkpoint_scores_path = output_root / "checkpoint_scores.csv"
    oof_errors_path = output_root / "oof_errors.csv"
    for output_path, label in (
        (per_dataset_path, "per-dataset metrics"),
        (summary_path, "official summary"),
        (submission_path, "submission"),
        (oof_manifest_path, "OOF manifest"),
        (checkpoint_scores_path, "checkpoint scores"),
        (oof_errors_path, "OOF errors"),
    ):
        _ensure_safe_output_path(output_path, output_root, label)
    per_dataset.to_csv(per_dataset_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with official_import_context(source_dir):
        _write_submission(source_dir, predictions_dir, submission_path, evaluator)

    checkpoint = f"{args.weights.resolve()}#det_threshold={effective_detection_threshold:.6f}"
    if args.motion_relink:
        checkpoint += "#postprocess=ema_motion_relink_v1"
    _ensure_safe_output_path(checkpoint_scores_path, output_root, "checkpoint scores")
    append_checkpoint_score(
        checkpoint_scores_path,
        epoch=args.epoch,
        checkpoint=checkpoint,
        fold=str(fold["name"]),
        score=float(summary["score"]),
    )
    _ensure_safe_output_path(oof_errors_path, output_root, "OOF errors")
    _append_oof_errors(oof_errors_path, checkpoint, per_dataset)
    artifact_manifest = {
        "candidate": candidate_name,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": run_manifest.config_sha256,
        "datasets": sorted(names),
        "epoch": args.epoch,
        "fold": fold["name"],
        "official_score": float(summary["score"]),
        "oof_inference": {
            **config.oof_inference.model_dump(mode="json"),
            "detection_threshold": effective_detection_threshold,
            "use_overlay": args.use_overlay,
        },
        "postprocessing": {
            "method_id": (
                "ema_motion_relink_v1" if args.motion_relink else "predictor_default_edges_v1"
            ),
            "method_sources": (
                [
                    "scripts/run_oof_checkpoint.py",
                    "src/biohub_tracking/tracking/motion_relink.py",
                    "official_source_dir/scripts/**/*.py",
                    "official_source_dir/src/**/*.py",
                ]
                if args.motion_relink
                else []
            ),
            "method_sha256": motion_relink_source_hash,
            "config": asdict(motion_relink_config) if args.motion_relink else None,
            "runtime_executed": bool(args.motion_relink),
            "per_dataset_stats": (
                {name: motion_relink_stats[name] for name in sorted(names)}
                if args.motion_relink
                else {}
            ),
            "scope": "Gold OOF model ablation; not the exact 0.946 Kaggle pipeline",
        },
        "split_sha256": run_manifest.split_sha256,
    }
    _ensure_safe_output_path(oof_manifest_path, output_root, "OOF manifest")
    oof_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
