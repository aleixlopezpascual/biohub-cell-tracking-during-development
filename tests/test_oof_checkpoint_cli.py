"""CLI smoke test for the Kaggle-only OOF runner."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from scripts.run_oof_checkpoint import _append_oof_errors, _evaluate_pairs, _predict_config


def test_oof_checkpoint_help_does_not_import_kaggle_dependencies() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_oof_checkpoint.py", "--help"],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--score-only" in completed.stdout


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
