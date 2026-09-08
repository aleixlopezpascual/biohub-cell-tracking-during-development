from __future__ import annotations

import pandas as pd

from biohub_tracking.evaluation.experiment_log import ExperimentLogEntry, append_experiment_log, file_sha256


def test_append_experiment_log_preserves_existing_rows(tmp_path) -> None:
    path = tmp_path / "logs" / "local_eval_log.csv"
    entry = ExperimentLogEntry(
        candidate="a",
        score=1.0,
        adjusted_edge_jaccard=0.9,
        division_jaccard=1.0,
        split="A",
        split_provenance="unit",
        scorer_provenance="internal",
        commit_sha="abc",
    )
    append_experiment_log(entry, path)
    append_experiment_log(entry, path)
    frame = pd.read_csv(path)
    assert len(frame) == 2
    assert list(frame["candidate"]) == ["a", "a"]


def test_file_sha256_returns_empty_for_missing_path(tmp_path) -> None:
    assert file_sha256(tmp_path / "missing.yaml") == ""
