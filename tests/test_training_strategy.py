"""Synthetic tests for the gated Gold-training utilities."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from biohub_tracking.training import (
    CheckpointScore,
    TrainingSession,
    append_checkpoint_score,
    build_run_manifest,
    evaluate_ensemble_gate,
    evaluate_learning_curve_gate,
    load_checkpoint_scores,
    load_gold_training_config,
    load_official_script,
    official_import_context,
    positive_unlabeled_weights,
    rank_complementary_models,
    render_gaussian_center_target,
    select_best_checkpoint,
    validate_run_manifest,
    write_diagnostic_curve_svgs,
    write_learning_curve_svg,
)


def test_gaussian_target_uses_physical_anisotropy_and_max_composition() -> None:
    target = render_gaussian_center_target(
        (7, 7, 7),
        np.array([[3.0, 3.0, 3.0], [3.0, 3.0, 4.0]]),
        voxel_size_um=(2.0, 1.0, 1.0),
        sigma_um=(1.0, 1.0, 1.0),
    )
    assert target[3, 3, 3] == pytest.approx(1.0)
    assert target[3, 3, 4] == pytest.approx(1.0)
    assert target[4, 3, 3] < target[3, 4, 3]
    assert target.dtype == np.float32


def test_positive_unlabeled_weights_downweight_only_unlabeled_voxels() -> None:
    target = np.array([0.0, 0.2, 1.0], dtype=np.float32)
    weights = positive_unlabeled_weights(target, negative_weight=0.02)
    assert np.array_equal(weights, np.array([0.02, 1.0, 1.0], dtype=np.float32))


def test_learning_curve_gate_requires_mean_gain_and_fold_stability() -> None:
    baseline = CheckpointScore(50, "epoch50.pt", {"A": 0.940, "B": 0.942})
    passing = CheckpointScore(65, "epoch65.pt", {"A": 0.944, "B": 0.946})
    unstable = CheckpointScore(80, "epoch80.pt", {"A": 0.939, "B": 0.953})
    decision = evaluate_learning_curve_gate(baseline, [passing, unstable])
    assert decision.promote
    assert select_best_checkpoint([baseline, passing]).epoch == 65

    rejected = evaluate_learning_curve_gate(baseline, [unstable])
    assert not rejected.promote
    assert "regressed" in rejected.reason


def test_ensemble_gate_requires_score_and_runtime() -> None:
    assert evaluate_ensemble_gate(
        best_single_score=0.95,
        ensemble_score=0.953,
        measured_runtime_hours=10,
        runtime_limit_hours=12,
    ).promote


def test_complementary_model_ranking_prefers_low_oof_error_correlation() -> None:
    scores = [
        CheckpointScore(50, "best.pt", {"A": 0.95, "B": 0.95}),
        CheckpointScore(65, "clone.pt", {"A": 0.949, "B": 0.949}),
        CheckpointScore(80, "diverse.pt", {"A": 0.948, "B": 0.948}),
    ]
    ranked = rank_complementary_models(
        scores,
        {
            "best.pt": [1.0, 2.0, 3.0, 4.0],
            "clone.pt": [2.0, 4.0, 6.0, 8.0],
            "diverse.pt": [4.0, 3.0, 2.0, 1.0],
        },
        max_models=2,
    )
    assert [score.checkpoint for score in ranked] == ["best.pt", "diverse.pt"]
    assert not evaluate_ensemble_gate(
        best_single_score=0.95,
        ensemble_score=0.953,
        measured_runtime_hours=13,
        runtime_limit_hours=12,
    ).promote


def test_checkpoint_scores_load_long_form_and_reject_duplicates(tmp_path) -> None:
    path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {"epoch": 50, "checkpoint": "a.pt", "fold": "A", "score": 0.9},
            {"epoch": 50, "checkpoint": "a.pt", "fold": "B", "score": 0.8},
        ]
    ).to_csv(path, index=False)
    loaded = load_checkpoint_scores(path)
    assert loaded[0].mean_score == pytest.approx(0.85)
    svg = tmp_path / "curve.svg"
    write_learning_curve_svg(loaded, svg)
    assert "Official prefix-held learning curve" in svg.read_text(encoding="utf-8")

    duplicate = pd.read_csv(path)
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate fold"):
        load_checkpoint_scores(path)


def test_checkpoint_score_log_is_append_only_and_idempotent(tmp_path) -> None:
    path = tmp_path / "scores.csv"
    assert append_checkpoint_score(
        path,
        epoch=50,
        checkpoint="model.pth",
        fold="A",
        score=0.94,
    )
    assert not append_checkpoint_score(
        path,
        epoch=50,
        checkpoint="model.pth",
        fold="A",
        score=0.94,
    )
    with pytest.raises(ValueError, match="different official"):
        append_checkpoint_score(
            path,
            epoch=50,
            checkpoint="model.pth",
            fold="A",
            score=0.95,
        )


def test_diagnostic_curve_writer_covers_acceptance_metrics(tmp_path) -> None:
    rows = []
    for epoch in (50, 65):
        for fold in ("A", "B"):
            rows.append(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "edge_jaccard": 0.9 + epoch / 10_000,
                    "node_recall": 0.95,
                    "node_count_ratio": 1.0,
                    "edges_fragmented": 10 - epoch / 10,
                    "edges_lost_to_detection": 4,
                    "wrong_association_edges": 2,
                }
            )
    paths = write_diagnostic_curve_svgs(pd.DataFrame(rows), tmp_path / "plots")
    assert len(paths) == 6
    assert all(path.exists() for path in paths)


def test_gold_config_and_run_manifest_hash_inputs(tmp_path) -> None:
    config_path = tmp_path / "gold.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_dir: train",
                "cv_pack_dir: cv",
                "official_source_dir: official",
                "learning_curve:",
                "  evaluation_epochs: [50, 65]",
            ]
        ),
        encoding="utf-8",
    )
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps([{"split": 0}]), encoding="utf-8")
    config = load_gold_training_config(config_path)
    assert config.oof_inference.use_ilp
    manifest = build_run_manifest(
        candidate="test",
        config_path=config_path,
        split_path=split_path,
        fold=config.fold,
        seed=config.seed,
        max_runtime_hours=config.max_runtime_hours,
        repo_root=tmp_path,
    )
    assert manifest.config_sha256
    assert manifest.split_sha256
    output = tmp_path / "manifest.json"
    manifest.write_json(output)
    assert json.loads(output.read_text(encoding="utf-8"))["candidate"] == "test"
    validate_run_manifest(
        manifest,
        config_path=config_path,
        split_path=split_path,
        candidate="test",
    )
    split_path.write_text(json.dumps([{"split": 1}]), encoding="utf-8")
    with pytest.raises(ValueError, match="split hash"):
        validate_run_manifest(
            manifest,
            config_path=config_path,
            split_path=split_path,
        )


def test_official_script_loader_fails_before_optional_imports(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="official Royerlab script"):
        load_official_script(
            tmp_path,
            "predict_unet_transformer.py",
            module_name="missing_official_predictor",
        )


def test_official_script_loader_isolates_same_named_upstream_package(tmp_path) -> None:
    source = tmp_path / "official"
    scripts = source / "scripts"
    package = source / "src" / "biohub_tracking"
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "models.py").write_text("MARKER = 'upstream'\n", encoding="utf-8")
    (scripts / "probe.py").write_text(
        "from biohub_tracking.models import MARKER\n", encoding="utf-8"
    )

    loaded = load_official_script(source, "probe.py", module_name="test_upstream_probe")

    assert loaded.MARKER == "upstream"
    from biohub_tracking.models import UNet3D

    assert UNet3D.__name__ == "UNet3D"


def test_official_import_context_supports_lazy_sibling_imports(tmp_path) -> None:
    import biohub_tracking as local_package

    source = tmp_path / "official"
    package = source / "src" / "biohub_tracking"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "lazy_metric.py").write_text("MARKER = 'upstream-lazy'\n", encoding="utf-8")

    with official_import_context(source):
        from biohub_tracking.lazy_metric import MARKER

        assert MARKER == "upstream-lazy"

    import biohub_tracking as restored_package

    assert restored_package is local_package


def test_prepare_gold_training_writes_real_two_fold_manifest(tmp_path) -> None:
    cv_pack = tmp_path / "cv"
    cv_pack.mkdir()
    pd.DataFrame(
        [
            {
                "volume": "44b6_train",
                "fold_A_role": "evaluate",
                "fold_B_role": "tune",
                "is_public_test_twin": False,
            },
            {
                "volume": "6bba_train",
                "fold_A_role": "tune",
                "fold_B_role": "evaluate",
                "is_public_test_twin": False,
            },
        ]
    ).to_csv(cv_pack / "folds_prefix_holdout.csv", index=False)
    official = tmp_path / "official" / "scripts"
    official.mkdir(parents=True)
    (official / "train_unet_transformer.py").write_text("def train(): pass\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    config_path = tmp_path / "gold.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'train'}",
                f"cv_pack_dir: {cv_pack}",
                f"official_source_dir: {official.parent}",
                f"output_dir: {output_dir}",
                "fold: A",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_gold_training.py",
            "--config",
            str(config_path),
            "--candidate",
            "unit-test",
        ],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )
    folds = json.loads((output_dir / "prefix_holdout_splits.json").read_text(encoding="utf-8"))
    command = json.loads((output_dir / "bootstrap_command.json").read_text(encoding="utf-8"))
    assert [fold["name"] for fold in folds] == ["A", "B"]
    assert folds[0]["test"] == ["44b6_train"]
    assert "continue_royerlab_training.py" in command[1]
    assert command[-2:] == ["--target-epoch", "50"]
    assert "run_manifest.json" in completed.stdout


def test_training_session_enforces_checkpoint_early_stop_and_runtime() -> None:
    session = TrainingSession(
        max_runtime_hours=1,
        early_stopping_patience=2,
        checkpoint_every_epochs=5,
        started_at=100.0,
    )
    assert session.record_evaluation(0, 0.9)
    assert not session.record_evaluation(0, 0.9)
    assert not session.record_evaluation(1, 0.8)
    assert not session.record_evaluation(2, 0.85)
    assert session.stop_decision(now=200.0).reason == "official-score early stopping"
    assert session.checkpoint_due(4)

    fresh = TrainingSession(1, 3, 5, started_at=100.0)
    assert fresh.stop_decision(now=3701.0).reason == "runtime budget exhausted"
