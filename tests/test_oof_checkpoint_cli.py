"""CLI smoke test for the Kaggle-only OOF runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.run_oof_checkpoint import (
    _append_oof_errors,
    _artifact_sha256,
    _evaluate_pairs,
    main,
    _motion_relink_source_sha256,
    _predict_config,
    _relink_prediction_edges,
    _read_motion_relink_sidecar,
    _write_motion_relink_sidecar,
)
from biohub_tracking.tracking.motion_relink import MotionRelinkConfig
import scripts.run_oof_checkpoint as oof_runner


def test_oof_checkpoint_help_does_not_import_kaggle_dependencies() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_oof_checkpoint.py", "--help"],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--score-only" in completed.stdout
    assert "--motion-relink" in completed.stdout


def _patch_minimal_oof_runner(
    monkeypatch, tmp_path, dataset_name: str, *, create_data: bool = False
):
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    if create_data:
        (data_dir / f"{dataset_name}.zarr").mkdir()
        (data_dir / f"{dataset_name}.geff").mkdir()
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"synthetic weights")
    config = SimpleNamespace(
        output_dir=output_dir,
        data_dir=data_dir,
        official_source_dir=tmp_path / "official",
        learning_curve=SimpleNamespace(evaluation_epochs=[1]),
        oof_inference=SimpleNamespace(detection_threshold=0.99),
    )
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps([{"name": "fold0", "test": [dataset_name]}]), encoding="utf-8"
    )
    monkeypatch.setattr(oof_runner, "load_gold_training_config", lambda _path: config)
    monkeypatch.setattr(oof_runner, "_manifest", lambda _path: SimpleNamespace(fold="fold0"))
    monkeypatch.setattr(oof_runner, "validate_run_manifest", lambda *_args, **_kwargs: None)
    return config, splits_path, weights_path


def test_score_only_runner_scores_zero_division_sample_with_unit_jaccard(
    monkeypatch, tmp_path
) -> None:
    dataset_name = "zero-divisions"
    config, splits_path, weights_path = _patch_minimal_oof_runner(
        monkeypatch, tmp_path, dataset_name, create_data=True
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.oof_inference.model_dump = lambda **_kwargs: {}
    run_manifest = SimpleNamespace(
        fold="fold0",
        config_sha256="c" * 64,
        split_sha256="d" * 64,
    )
    monkeypatch.setattr(oof_runner, "_manifest", lambda _path: run_manifest)
    monkeypatch.setattr(oof_runner, "official_import_context", lambda _path: nullcontext())
    monkeypatch.setattr(oof_runner, "load_official_script", lambda *_args, **_kwargs: Evaluator)
    monkeypatch.setattr(
        oof_runner,
        "_evaluate_pairs",
        lambda *_args: (
            [
                {
                    "dataset": dataset_name,
                    "adj_edge_jaccard": 0.8,
                    "edge_tp": 80.0,
                    "edge_fp": 10.0,
                    "edge_fn": 10.0,
                    "division_tp": 0.0,
                    "division_fp": 0.0,
                    "division_fn": 0.0,
                    "total_node_ratio": 0.0,
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(oof_runner, "append_checkpoint_score", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(oof_runner, "_append_oof_errors", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(oof_runner, "_write_submission", lambda *_args, **_kwargs: None)

    class Evaluator:
        @staticmethod
        def summarise(rows):
            return {"score": 0.9, "n": len(rows)}

    predictions_dir = (
        config.output_dir
        / "oof"
        / "safe-candidate"
        / "epoch_001"
        / "split_0"
        / "predictions"
    )
    predictions_dir.mkdir(parents=True)
    (predictions_dir / f"{dataset_name}.geff").mkdir()

    oof_runner.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--splits",
            str(splits_path),
            "--fold-index",
            "0",
            "--candidate",
            "safe-candidate",
            "--epoch",
            "1",
            "--weights",
            str(weights_path),
            "--score-only",
        ]
    )

    per_dataset = pd.read_csv(predictions_dir.parent / "official_per_dataset.csv")
    assert per_dataset.loc[0, "division_jaccard"] == pytest.approx(1.0)
    assert per_dataset.loc[0, "score"] == pytest.approx(0.9)


@pytest.mark.parametrize("dataset_name", ["../escape", "/absolute", "..\\escape"])
def test_oof_runner_rejects_escaping_dataset_name_before_data_access(
    dataset_name: str, monkeypatch, tmp_path
) -> None:
    _config, splits_path, weights_path = _patch_minimal_oof_runner(
        monkeypatch, tmp_path, dataset_name
    )

    with pytest.raises(ValueError, match="invalid.*dataset.*component"):
        oof_runner.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--splits",
                str(splits_path),
                "--fold-index",
                "0",
                "--candidate",
                "safe-candidate",
                "--epoch",
                "1",
                "--weights",
                str(weights_path),
                "--score-only",
            ]
        )


def test_oof_runner_rejects_escaping_candidate_before_output_creation(
    monkeypatch, tmp_path
) -> None:
    config, splits_path, weights_path = _patch_minimal_oof_runner(
        monkeypatch, tmp_path, "safe-dataset", create_data=True
    )

    with pytest.raises(ValueError, match="invalid.*candidate.*component"):
        oof_runner.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--splits",
                str(splits_path),
                "--fold-index",
                "0",
                "--candidate",
                "../escape_candidate",
                "--epoch",
                "1",
                "--weights",
                str(weights_path),
                "--score-only",
            ]
        )
    assert not (config.output_dir / "escape_candidate").exists()


def test_oof_runner_rejects_symlinked_output_directory_ancestor(monkeypatch, tmp_path) -> None:
    config, splits_path, weights_path = _patch_minimal_oof_runner(
        monkeypatch, tmp_path, "safe-dataset", create_data=True
    )
    config.output_dir.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (config.output_dir / "oof").symlink_to(external_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        oof_runner.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--splits",
                str(splits_path),
                "--fold-index",
                "0",
                "--candidate",
                "safe-candidate",
                "--epoch",
                "1",
                "--weights",
                str(weights_path),
                "--score-only",
            ]
        )

    assert list(external_dir.iterdir()) == []


def test_oof_runner_rejects_dangling_prediction_symlink(monkeypatch, tmp_path) -> None:
    config, splits_path, weights_path = _patch_minimal_oof_runner(
        monkeypatch, tmp_path, "safe-dataset", create_data=True
    )
    prediction_dir = (
        config.output_dir
        / "oof"
        / "safe-candidate"
        / "epoch_001"
        / "split_0"
        / "predictions"
    )
    prediction_dir.mkdir(parents=True)
    prediction_path = prediction_dir / "safe-dataset.geff"
    prediction_path.symlink_to(tmp_path / "not-created.geff", target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        oof_runner.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--splits",
                str(splits_path),
                "--fold-index",
                "0",
                "--candidate",
                "safe-candidate",
                "--epoch",
                "1",
                "--weights",
                str(weights_path),
                "--score-only",
            ]
        )

    assert prediction_path.is_symlink()
    assert not (tmp_path / "not-created.geff").exists()


def test_oof_error_log_resumes_without_duplicate_rows(tmp_path) -> None:
    path = tmp_path / "errors.csv"
    first = pd.DataFrame({"dataset": ["a"], "score": [0.9]})
    second = pd.DataFrame({"dataset": ["a", "b"], "score": [0.9, 0.8]})
    _append_oof_errors(path, "model.pth", first)
    _append_oof_errors(path, "model.pth", second)
    result = pd.read_csv(path)
    assert result["sample"].tolist() == ["a", "b"]
    assert result["error"].tolist() == pytest.approx([0.1, 0.2])


def test_oof_evaluator_supports_pinned_evaluate_run_api(tmp_path) -> None:
    class PinnedEvaluator:
        DATA_DIR = None

        @staticmethod
        def evaluate_run(run, max_distance):
            assert max_distance == 7.0
            return [
                {"dataset": path.stem, "edge_tp": 1.0}
                for path in run["geffs"]
            ]

    rows, skipped = _evaluate_pairs(
        PinnedEvaluator,
        tmp_path / "predictions",
        tmp_path / "ground_truth",
        ["b", "a"],
    )
    assert [row["dataset"] for row in rows] == ["a", "b"]
    assert skipped == []
    assert PinnedEvaluator.DATA_DIR == tmp_path / "ground_truth"


def test_predict_config_accepts_calibrated_detection_threshold() -> None:
    class PredictConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Predictor:
        pass

    Predictor.PredictConfig = PredictConfig

    class Inference:
        detection_threshold = 0.99
        detection_tta = True
        edge_activation = "softmax"
        edge_threshold = 0.5
        use_ilp = True
        ilp_edge_weight = -1.0
        ilp_appearance_weight = 0.1
        ilp_disappearance_weight = 0.1
        ilp_division_weight = 1.0
        max_parents_per_node = None
        max_children_per_node = None

    class Config:
        oof_inference = Inference()
        pool_kernel_um = 5.0

    result = _predict_config(Predictor, Config(), detection_threshold=0.6)
    assert result.det_threshold == pytest.approx(0.6)


def test_relink_prediction_edges_converts_voxels_to_physical_microns() -> None:
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 10],
            [2, 0, 0, 11],
            [2, 0, 0, 14],
        ],
        dtype=np.float64,
    )
    raw_edges = [(0, 1, 0.8, 4.0), (1, 2, 0.9, 1.0), (1, 3, 0.1, 4.0)]

    relinked, stats = _relink_prediction_edges(
        coordinates,
        raw_edges,
        (1.625, 0.40625, 0.40625),
    )

    assert [(edge[0], edge[1]) for edge in relinked] == [(0, 1), (1, 3)]
    assert [edge[2] for edge in relinked] == pytest.approx([0.8, 0.1])
    assert [edge[3] for edge in relinked] == pytest.approx([4.0625, 1.625])
    assert stats["frames_processed"] == 2
    assert stats["tight_edges"] == 2
    assert stats["relaxed_edges"] == 0
    assert stats["skipped_large_frame"] is False
    assert stats["used_motion_relink"] is True


def test_relink_prediction_edges_falls_back_when_frame_node_cap_is_exceeded() -> None:
    coordinates = np.array([[0, 0, 0, 0], [0, 0, 0, 2], [1, 0, 0, 1]], dtype=np.float64)
    raw_edges = [(0, 2, 0.7, 1.0)]

    relinked, stats = _relink_prediction_edges(
        coordinates,
        raw_edges,
        (1.625, 0.40625, 0.40625),
        MotionRelinkConfig(max_nodes_per_frame=1),
    )

    assert relinked == raw_edges
    assert stats["skipped_large_frame"] is True
    assert stats["used_motion_relink"] is False


def test_relink_prediction_edges_does_not_restore_rejected_gap_edges() -> None:
    coordinates = np.array([[0, 0, 0, 0], [2, 0, 0, 1]], dtype=np.float64)
    raw_edges = [(0, 1, 0.9, 1.0)]

    relinked, stats = _relink_prediction_edges(
        coordinates,
        raw_edges,
        (1.625, 0.40625, 0.40625),
    )

    assert relinked == []
    assert stats["skipped_large_frame"] is False
    assert stats["output_edges"] == 0
    assert stats["used_motion_relink"] is False


def test_relink_prediction_edges_rejects_unknown_node_indices() -> None:
    coordinates = np.array([[0, 0, 0, 0]], dtype=np.float64)

    with pytest.raises(ValueError, match="missing coordinate row"):
        _relink_prediction_edges(
            coordinates,
            [(0, 1, 0.7, 1.0)],
            (1.625, 0.40625, 0.40625),
        )


def test_motion_relink_sidecar_binds_stats_to_prediction_hash(tmp_path) -> None:
    prediction_path = tmp_path / "video.geff"
    prediction_path.mkdir()
    (prediction_path / "metadata.json").write_text("synthetic GEFF artifact", encoding="utf-8")
    sidecar_path = tmp_path / "video.motion_relink.json"
    stats = {
        "frames_processed": 1,
        "tight_edges": 1,
        "relaxed_edges": 0,
        "skipped_large_frame": False,
        "input_edges": 1,
        "output_edges": 1,
        "used_motion_relink": True,
    }
    config = MotionRelinkConfig()
    source_hash = "a" * 64
    checkpoint_hash = "b" * 64

    _write_motion_relink_sidecar(
        sidecar_path,
        prediction_path,
        "video",
        config,
        source_hash,
        checkpoint_hash,
        stats,
    )

    assert _read_motion_relink_sidecar(
        sidecar_path,
        prediction_path,
        "video",
        config,
        source_hash,
        checkpoint_hash,
    ) == stats

    # Wrong checkpoint hash fails closed
    wrong_checkpoint_hash = "c" * 64
    with pytest.raises(ValueError, match="checkpoint hash"):
        _read_motion_relink_sidecar(
            sidecar_path,
            prediction_path,
            "video",
            config,
            source_hash,
            wrong_checkpoint_hash,
        )

    (prediction_path / "metadata.json").write_text("changed GEFF artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction hash"):
        _read_motion_relink_sidecar(
            sidecar_path,
            prediction_path,
            "video",
            config,
            source_hash,
            checkpoint_hash,
        )


def test_motion_relink_sidecar_refuses_symlink_destination(tmp_path) -> None:
    prediction_path = tmp_path / "video.geff"
    prediction_path.mkdir()
    (prediction_path / "metadata.json").write_text("synthetic GEFF artifact", encoding="utf-8")
    outside_path = tmp_path / "outside.json"
    outside_path.write_text("keep this content", encoding="utf-8")
    sidecar_path = tmp_path / "video.motion_relink.json"
    sidecar_path.symlink_to(outside_path)

    with pytest.raises(ValueError, match="sidecar.*symlink"):
        _write_motion_relink_sidecar(
            sidecar_path,
            prediction_path,
            "video",
            MotionRelinkConfig(),
            "a" * 64,
            "b" * 64,
            {
                "frames_processed": 1,
                "tight_edges": 1,
                "relaxed_edges": 0,
                "skipped_large_frame": False,
                "input_edges": 1,
                "output_edges": 1,
                "used_motion_relink": True,
            },
        )

    assert outside_path.read_text(encoding="utf-8") == "keep this content"
    assert sidecar_path.is_symlink()


def test_motion_relink_ablation_rejects_overlay_confounds() -> None:
    with pytest.raises(ValueError, match="cannot be combined with --use-overlay"):
        main(
            [
                "--config",
                "unused.yaml",
                "--splits",
                "unused.json",
                "--fold-index",
                "0",
                "--candidate",
                "ema-test",
                "--epoch",
                "1",
                "--weights",
                "unused.pth",
                "--motion-relink",
                "--use-overlay",
            ]
        )


def test_motion_relink_source_hash_is_sha256(tmp_path) -> None:
    upstream_dir = tmp_path / "upstream"
    scripts_dir = upstream_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "predict_unet_transformer.py").write_text(
        "PREDICTOR_VERSION = 1\n", encoding="utf-8"
    )
    assert len(_motion_relink_source_sha256(upstream_dir)) == 64


def test_motion_relink_source_hash_binds_upstream_source(tmp_path) -> None:
    upstream_dir = tmp_path / "upstream"
    scripts_dir = upstream_dir / "scripts"
    src_dir = upstream_dir / "src" / "biohub_tracking"
    scripts_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)

    script_file = scripts_dir / "predict_unet_transformer.py"
    script_file.write_text("print('hello')", encoding="utf-8")

    src_file = src_dir / "some_module.py"
    src_file.write_text("def foo(): pass", encoding="utf-8")

    hash1 = _motion_relink_source_sha256(upstream_dir)

    script_file.write_text("print('world')", encoding="utf-8")
    hash2 = _motion_relink_source_sha256(upstream_dir)

    assert hash1 != hash2


def test_artifact_sha256_hashes_regular_files(tmp_path) -> None:
    artifact = tmp_path / "prediction.geff"
    content = b"synthetic file artifact"
    artifact.write_bytes(content)

    assert _artifact_sha256(artifact) == hashlib.sha256(content).hexdigest()
