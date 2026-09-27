"""CPU-only synthetic tests for the paired OOF promotion evidence gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from biohub_tracking.evaluation.promotion_gate import (
    PromotionEvidenceError,
    evaluate_promotion_evidence,
)
from scripts.evaluate_promotion_gate import main


FOLDS = {
    "A": {
        "train": ["6bba_train", "6bba_train_2"],
        "test": ["44b6_holdout", "44b6_holdout_2"],
    },
    "B": {
        "train": ["44b6_holdout", "44b6_holdout_2"],
        "test": ["6bba_holdout", "6bba_holdout_2"],
    },
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, relative_path: str, content: str | bytes) -> dict[str, str]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode() if isinstance(content, str) else content
    path.write_bytes(payload)
    return {"path": relative_path, "sha256": _hash(path)}


def _write_json(root: Path, relative_path: str, payload: object) -> dict[str, str]:
    return _artifact(root, relative_path, json.dumps(payload, sort_keys=True))


def _make_evidence(root: Path, *, candidate_gain: float = 0.002) -> Path:
    split_rows = [
        {
            "name": name,
            "train": data["train"],
            "test": data["test"],
            "excluded": [],
        }
        for name, data in FOLDS.items()
    ]
    split_ref = _write_json(root, "splits.json", split_rows)
    metric_ref = _artifact(root, "sources/official_metric.py", "metric source v1\n")
    parent_method_ref = _artifact(root, "sources/parent_method.py", "parent method source\n")
    candidate_method_ref = _artifact(root, "sources/ema_method.py", "ema method source\n")
    parent_name = "parent-0.946"
    candidate_name = "ema-candidate"
    metric_id = "biohub-patched-official"
    metric_version = "local-2026-09"
    runs: list[dict[str, object]] = []

    for fold_name, split in FOLDS.items():
        for variant, name, method_id, method_ref, score in (
            ("parent", parent_name, "baseline-relinker", parent_method_ref, 0.900),
            (
                "candidate",
                candidate_name,
                "ema-relinker",
                candidate_method_ref,
                0.900 + candidate_gain,
            ),
        ):
            run_id = f"{variant}-{fold_name}-run"
            config_ref = _artifact(
                root,
                f"artifacts/{variant}/{fold_name}/inference.yaml",
                "fixed inference config\n",
            )
            checkpoint_ref = _artifact(
                root,
                f"artifacts/{variant}/{fold_name}/weights.pt",
                f"fold weights {fold_name}\n",
            )
            prediction_refs = [
                {
                    "dataset": dataset,
                    **_artifact(
                        root,
                        f"artifacts/{variant}/{fold_name}/{dataset}.geff",
                        f"{variant}:{fold_name}:{dataset}\n",
                    ),
                }
                for dataset in split["test"]
            ]
            edge_tp = (
                80.0 + candidate_gain * 100.0
                if variant == "candidate"
                else 80.0
            )
            edge_fp = 10.0
            edge_fn = 10.0 - (candidate_gain * 100.0 if variant == "candidate" else 0.0)
            node_count_ratio = 1.0
            J = edge_tp / (edge_tp + edge_fp + edge_fn)
            adj = max(0.0, J * (1.0 - 0.1 * (node_count_ratio - 1.0)))
            per_dataset_rows = [
                {
                    "dataset": dataset,
                    "score": score,
                    "edge_tp": edge_tp,
                    "edge_fp": edge_fp,
                    "edge_fn": edge_fn,
                    "adj_edge_jaccard": adj,
                    "division_tp": 10.0,
                    "division_fp": 0.0,
                    "division_fn": 0.0,
                    "node_recall": 0.95,
                    "node_count_ratio": node_count_ratio,
                    "edges_fragmented": 1.0,
                    "edges_lost_to_detection": 0.0,
                    "wrong_association_edges": 1.0,
                }
                for dataset in split["test"]
            ]
            per_dataset_ref = _artifact(
                root,
                f"artifacts/{variant}/{fold_name}/official_per_dataset.csv",
                pd.DataFrame(per_dataset_rows).to_csv(index=False),
            )
            summary_ref = _write_json(
                root,
                f"artifacts/{variant}/{fold_name}/official_summary.json",
                {"score": score, "n": len(split["test"])},
            )
            run_manifest = {
                "run_id": run_id,
                "candidate": name,
                "fold": fold_name,
                "commit_sha": "a" * 40 if variant == "parent" else "b" * 40,
                "split_sha256": split_ref["sha256"],
                "config_sha256": config_ref["sha256"],
                "training_datasets": split["train"],
                "evaluation_datasets": split["test"],
                "method_id": method_id,
                "method_sha256": method_ref["sha256"],
                "metric_id": metric_id,
                "metric_version": metric_version,
                "metric_source_sha256": metric_ref["sha256"],
            }
            run_ref = _write_json(
                root,
                f"artifacts/{variant}/{fold_name}/run_manifest.json",
                run_manifest,
            )
            oof_manifest = {
                "run_id": run_id,
                "candidate": name,
                "fold": fold_name,
                "datasets": split["test"],
                "split_sha256": split_ref["sha256"],
                "config_sha256": config_ref["sha256"],
                "checkpoint_sha256": checkpoint_ref["sha256"],
                "run_manifest_sha256": run_ref["sha256"],
                "method_id": method_id,
                "method_sha256": method_ref["sha256"],
                "metric_id": metric_id,
                "metric_version": metric_version,
                "metric_source_sha256": metric_ref["sha256"],
                "predictions_sha256": {
                    item["dataset"]: item["sha256"] for item in prediction_refs
                },
                "oof_inference": {"threshold": 0.96, "tta": True},
            }
            oof_ref = _write_json(
                root,
                f"artifacts/{variant}/{fold_name}/oof_manifest.json",
                oof_manifest,
            )
            runs.append(
                {
                    "variant": variant,
                    "fold": fold_name,
                    "run_manifest": run_ref,
                    "oof_manifest": oof_ref,
                    "checkpoint": checkpoint_ref,
                    "config": config_ref,
                    "predictions": prediction_refs,
                    "official_summary": summary_ref,
                    "per_dataset": per_dataset_ref,
                }
            )

    evidence = {
        "schema_version": 1,
        "experiment": {
            "parent": {
                "name": parent_name,
                "method_id": "baseline-relinker",
                "method_source": parent_method_ref,
            },
            "candidate": {
                "name": candidate_name,
                "method_id": "ema-relinker",
                "method_source": candidate_method_ref,
            },
            "changed_factor": "EMA motion relinking",
        },
        "metric": {"id": metric_id, "version": metric_version, "source": metric_ref},
        "split": split_ref,
        "runs": runs,
    }
    _write_json(root, "evidence.json", evidence)
    return root / "evidence.json"


@pytest.fixture
def valid_evidence(tmp_path: Path) -> Path:
    """Synthetic, temporary evidence only; never represents a real score result."""
    return _make_evidence(tmp_path)


def test_valid_synthetic_evidence_computes_pooled_score_and_receipt(valid_evidence: Path) -> None:
    receipt = evaluate_promotion_evidence(valid_evidence)
    assert receipt["promote"] is False
    assert receipt.get("trusted_runtime_provenance") is False
    assert "blockers" in receipt
    assert any("trusted runtime provenance" in str(b).lower() for b in receipt["blockers"])
    assert receipt["pooled_score_delta"] == pytest.approx(0.002)
    assert receipt["folds_evaluated"] == ["A", "B"]
    assert receipt["verified_artifacts"]


def test_single_heldout_dataset_drop_blocks_promotion(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    run = next(
        item
        for item in evidence["runs"]
        if item["variant"] == "candidate" and item["fold"] == "A"
    )
    metric_path = valid_evidence.parent / run["per_dataset"]["path"]
    frame = pd.read_csv(metric_path)
    frame.loc[0, ["score", "adj_edge_jaccard", "edge_tp", "edge_fp", "edge_fn"]] = [
        0.896,
        0.796,
        79.6,
        10,
        10.4,
    ]
    frame.loc[1, ["score", "adj_edge_jaccard", "edge_tp", "edge_fp", "edge_fn"]] = [
        0.908,
        0.808,
        80.8,
        10,
        9.2,
    ]
    metric_path.write_text(frame.to_csv(index=False))
    run["per_dataset"]["sha256"] = _hash(metric_path)
    valid_evidence.write_text(json.dumps(evidence))

    receipt = evaluate_promotion_evidence(valid_evidence)
    assert receipt["pooled_score_delta"] == pytest.approx(0.002)
    assert receipt["score_gate_passed"] is False
    assert min(row["score_delta"] for row in receipt["paired_per_dataset"]) == pytest.approx(-0.004)
    assert receipt["metrics_regression_review_required"] is True
    assert any(
        "dataset 44b6_holdout adj_edge_jaccard regressed" in reason
        for reason in receipt["metrics_regression_reasons"]
    )


def test_hash_tampering_fails_closed(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    evidence["runs"][0]["checkpoint"]["sha256"] = "0" * 64
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="hash mismatch"):
        evaluate_promotion_evidence(valid_evidence)


def test_training_overlap_or_wrong_fold_training_set_fails(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    run_ref = evidence["runs"][0]["run_manifest"]
    run_path = valid_evidence.parent / run_ref["path"]
    run = json.loads(run_path.read_text())
    run["training_datasets"] = ["44b6_holdout"]
    run_path.write_text(json.dumps(run, sort_keys=True))
    evidence["runs"][0]["run_manifest"]["sha256"] = _hash(run_path)
    oof_ref = evidence["runs"][0]["oof_manifest"]
    oof_path = valid_evidence.parent / oof_ref["path"]
    oof = json.loads(oof_path.read_text())
    oof["run_manifest_sha256"] = _hash(run_path)
    oof_path.write_text(json.dumps(oof, sort_keys=True))
    oof_ref["sha256"] = _hash(oof_path)
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="training_datasets"):
        evaluate_promotion_evidence(valid_evidence)


def test_split_file_must_match_all_run_and_oof_manifests(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    split_path = valid_evidence.parent / evidence["split"]["path"]
    split_path.write_text(split_path.read_text() + " ")
    with pytest.raises(PromotionEvidenceError, match="split.*hash"):
        evaluate_promotion_evidence(valid_evidence)


def test_exactly_both_folds_and_no_duplicates_are_required(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    evidence["runs"] = [
        run
        for run in evidence["runs"]
        if not (run["variant"] == "candidate" and run["fold"] == "B")
    ]
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="exactly one run for each variant and fold"):
        evaluate_promotion_evidence(valid_evidence)


def test_prediction_and_metric_rows_must_cover_exact_fold_datasets(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    run = next(
        item
        for item in evidence["runs"]
        if item["variant"] == "candidate" and item["fold"] == "A"
    )
    run["predictions"] = []
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="prediction datasets"):
        evaluate_promotion_evidence(valid_evidence)


def test_method_source_and_metric_provenance_must_match_artifacts(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    source_ref = evidence["experiment"]["candidate"]["method_source"]
    source_path = valid_evidence.parent / source_ref["path"]
    source_path.write_text("tampered source")
    with pytest.raises(PromotionEvidenceError, match="hash mismatch"):
        evaluate_promotion_evidence(valid_evidence)


def test_paired_checkpoint_must_be_identical(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    candidate = next(
        item
        for item in evidence["runs"]
        if item["variant"] == "candidate" and item["fold"] == "A"
    )
    candidate["checkpoint"]["sha256"] = "f" * 64
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="hash mismatch|checkpoint"):
        evaluate_promotion_evidence(valid_evidence)


def test_failed_score_gate_emits_nonpromotable_result(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    # Keep the candidate reports valid but lower-scoring, and update their hashes.
    for run in evidence["runs"]:
        if run["variant"] != "candidate":
            continue
        summary_path = valid_evidence.parent / run["official_summary"]["path"]
        summary = json.loads(summary_path.read_text())
        summary["score"] = 0.900
        summary_path.write_text(json.dumps(summary, sort_keys=True))
        run["official_summary"]["sha256"] = _hash(summary_path)
        per_path = valid_evidence.parent / run["per_dataset"]["path"]
        frame = pd.read_csv(per_path)
        frame["score"] = 0.900
        frame["adj_edge_jaccard"] = 0.800
        frame["edge_tp"] = 80
        frame["edge_fp"] = 10
        frame["edge_fn"] = 10
        per_path.write_text(frame.to_csv(index=False))
        run["per_dataset"]["sha256"] = _hash(per_path)
    valid_evidence.write_text(json.dumps(evidence))
    receipt = evaluate_promotion_evidence(valid_evidence)
    assert receipt["score_gate_passed"] is False
    assert receipt["promote"] is False


def test_inconsistent_adjusted_edge_jaccard_fails_validation(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    run = evidence["runs"][0]
    metric_path = valid_evidence.parent / run["per_dataset"]["path"]
    frame = pd.read_csv(metric_path)
    # Change the adjusted edge score while leaving TP/FP/FN/node_count_ratio unchanged.
    frame["adj_edge_jaccard"] = 0.999
    frame["score"] = 1.099
    metric_path.write_text(frame.to_csv(index=False))

    summary_path = valid_evidence.parent / run["official_summary"]["path"]
    summary_path.write_text(json.dumps({"score": 1.099, "n": len(frame)}, sort_keys=True))
    run["official_summary"]["sha256"] = _hash(summary_path)
    # Update hashes so it passes standard tampering checks
    run["per_dataset"]["sha256"] = _hash(metric_path)
    valid_evidence.write_text(json.dumps(evidence))

    with pytest.raises(PromotionEvidenceError, match="inconsistent adjusted edge Jaccard"):
        evaluate_promotion_evidence(valid_evidence)


def test_underprediction_adjustment_is_not_upper_clipped(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    for run in evidence["runs"]:
        metric_path = valid_evidence.parent / run["per_dataset"]["path"]
        frame = pd.read_csv(metric_path)
        frame["edge_tp"] = 100.0
        frame["edge_fp"] = 0.0
        frame["edge_fn"] = 0.0
        frame["node_count_ratio"] = 0.5
        frame["adj_edge_jaccard"] = 1.05
        frame["score"] = 1.15
        metric_path.write_text(frame.to_csv(index=False))
        run["per_dataset"]["sha256"] = _hash(metric_path)

        summary_path = valid_evidence.parent / run["official_summary"]["path"]
        summary = json.loads(summary_path.read_text())
        summary["score"] = 1.15
        summary_path.write_text(json.dumps(summary, sort_keys=True))
        run["official_summary"]["sha256"] = _hash(summary_path)
    valid_evidence.write_text(json.dumps(evidence))

    receipt = evaluate_promotion_evidence(valid_evidence)

    assert receipt["parent_submetrics"]["adjusted_edge_jaccard"] == pytest.approx(1.05)
    assert receipt["parent_submetrics"]["score"] == pytest.approx(1.15)
    assert receipt["promote"] is False


def test_zero_division_counts_use_per_dataset_and_pooled_conventions(
    valid_evidence: Path,
) -> None:
    evidence = json.loads(valid_evidence.read_text())
    for run in evidence["runs"]:
        metric_path = valid_evidence.parent / run["per_dataset"]["path"]
        frame = pd.read_csv(metric_path)
        frame[["division_tp", "division_fp", "division_fn"]] = 0.0
        frame["score"] = frame["adj_edge_jaccard"] + 0.1
        metric_path.write_text(frame.to_csv(index=False))
        run["per_dataset"]["sha256"] = _hash(metric_path)
    valid_evidence.write_text(json.dumps(evidence))

    receipt = evaluate_promotion_evidence(valid_evidence)

    assert receipt["parent_submetrics"]["division_jaccard"] == pytest.approx(1.0)
    assert receipt["candidate_submetrics"]["division_jaccard"] == pytest.approx(1.0)
    assert receipt["pooled_score_parent"] == pytest.approx(0.9)
    assert receipt["pooled_score_candidate"] == pytest.approx(0.902)


def test_component_regression_requires_review_even_if_score_passes(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    run = next(
        item
        for item in evidence["runs"]
        if item["variant"] == "candidate" and item["fold"] == "A"
    )
    path = valid_evidence.parent / run["per_dataset"]["path"]
    frame = pd.read_csv(path)
    frame["adj_edge_jaccard"] = 0.799
    frame["edge_tp"] = 79.9
    frame["edge_fp"] = 10
    frame["edge_fn"] = 10.1
    frame["score"] = 0.899
    path.write_text(frame.to_csv(index=False))
    run["per_dataset"]["sha256"] = _hash(path)
    summary_path = valid_evidence.parent / run["official_summary"]["path"]
    summary_path.write_text(json.dumps({"score": 0.899, "n": len(frame)}, sort_keys=True))
    run["official_summary"]["sha256"] = _hash(summary_path)
    # Fold B has a larger valid gain, so the pooled score gate still passes.
    fold_b = next(
        item
        for item in evidence["runs"]
        if item["variant"] == "candidate" and item["fold"] == "B"
    )
    fold_b_path = valid_evidence.parent / fold_b["per_dataset"]["path"]
    fold_b_frame = pd.read_csv(fold_b_path)
    fold_b_frame["adj_edge_jaccard"] = 0.804
    fold_b_frame["edge_tp"] = 80.4
    fold_b_frame["edge_fp"] = 10
    fold_b_frame["edge_fn"] = 9.6
    fold_b_frame["score"] = 0.904
    fold_b_path.write_text(fold_b_frame.to_csv(index=False))
    fold_b["per_dataset"]["sha256"] = _hash(fold_b_path)
    fold_b_summary = valid_evidence.parent / fold_b["official_summary"]["path"]
    fold_b_summary.write_text(json.dumps({"score": 0.904, "n": len(fold_b_frame)}, sort_keys=True))
    fold_b["official_summary"]["sha256"] = _hash(fold_b_summary)
    valid_evidence.write_text(json.dumps(evidence))
    receipt = evaluate_promotion_evidence(valid_evidence)
    assert receipt["score_gate_passed"] is True
    assert receipt["metrics_regression_review_required"] is True
    assert receipt["promote"] is False


def test_invalid_evidence_does_not_write_a_receipt(valid_evidence: Path, tmp_path: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    evidence["runs"] = []
    valid_evidence.write_text(json.dumps(evidence))
    receipt_path = tmp_path / "receipt.json"
    status = main(["--evidence", str(valid_evidence), "--output-receipt", str(receipt_path)])
    assert status == 2
    assert not receipt_path.exists()


def test_cli_refuses_receipt_aliasing_a_verified_artifact(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    checkpoint = evidence["runs"][0]["checkpoint"]
    artifact_path = valid_evidence.parent / checkpoint["path"]
    original = artifact_path.read_bytes()

    status = main(
        ["--evidence", str(valid_evidence), "--output-receipt", str(artifact_path)]
    )

    assert status == 2
    assert artifact_path.read_bytes() == original


def test_cli_returns_error_for_non_utf8_evidence(valid_evidence: Path, tmp_path: Path) -> None:
    valid_evidence.write_bytes(b"\xff")
    receipt_path = tmp_path / "receipt.json"

    status = main(["--evidence", str(valid_evidence), "--output-receipt", str(receipt_path)])

    assert status == 2
    assert not receipt_path.exists()


def test_cli_returns_error_for_empty_metrics_csv(valid_evidence: Path, tmp_path: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    record = evidence["runs"][0]
    metric_path = valid_evidence.parent / record["per_dataset"]["path"]
    metric_path.write_bytes(b"")
    record["per_dataset"]["sha256"] = _hash(metric_path)
    valid_evidence.write_text(json.dumps(evidence))
    receipt_path = tmp_path / "receipt.json"

    status = main(["--evidence", str(valid_evidence), "--output-receipt", str(receipt_path)])

    assert status == 2
    assert not receipt_path.exists()


def test_cli_returns_error_when_report_directory_cannot_be_created(
    valid_evidence: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_mkdir(*args: object, **kwargs: object) -> None:
        raise OSError("simulated report-directory write failure")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    receipt_path = tmp_path / "receipts" / "receipt.json"

    status = main(["--evidence", str(valid_evidence), "--output-receipt", str(receipt_path)])

    assert status == 2


def test_run_method_hash_must_bind_to_declared_source(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    record = next(
        item
        for item in evidence["runs"]
        if item["variant"] == "candidate" and item["fold"] == "A"
    )
    for ref_key in ("run_manifest", "oof_manifest"):
        ref = record[ref_key]
        manifest_path = valid_evidence.parent / ref["path"]
        manifest = json.loads(manifest_path.read_text())
        manifest["method_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        ref["sha256"] = _hash(manifest_path)
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="method provenance mismatch"):
        evaluate_promotion_evidence(valid_evidence)


def test_duplicate_metric_rows_fail_closed(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    record = next(
        item
        for item in evidence["runs"]
        if item["variant"] == "parent" and item["fold"] == "A"
    )
    metric_path = valid_evidence.parent / record["per_dataset"]["path"]
    frame = pd.read_csv(metric_path)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    metric_path.write_text(frame.to_csv(index=False))
    record["per_dataset"]["sha256"] = _hash(metric_path)
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="duplicate dataset rows"):
        evaluate_promotion_evidence(valid_evidence)


def test_nonfinite_metrics_fail_closed(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    record = next(
        item
        for item in evidence["runs"]
        if item["variant"] == "parent" and item["fold"] == "A"
    )
    metric_path = valid_evidence.parent / record["per_dataset"]["path"]
    frame = pd.read_csv(metric_path)
    frame.loc[0, "node_recall"] = float("nan")
    metric_path.write_text(frame.to_csv(index=False))
    record["per_dataset"]["sha256"] = _hash(metric_path)
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="node_recall contains non-finite"):
        evaluate_promotion_evidence(valid_evidence)


def test_cli_writes_receipt_only_for_valid_synthetic_evidence(
    valid_evidence: Path, tmp_path: Path
) -> None:
    """A valid synthetic manifest exercises the CLI write path only."""
    receipt_path = tmp_path / "receipts" / "synthetic-only.json"
    status = main(["--evidence", str(valid_evidence), "--output-receipt", str(receipt_path)])

    assert status == 1

    receipt = json.loads(receipt_path.read_text())
    assert receipt["promote"] is False
    assert receipt.get("trusted_runtime_provenance") is False
    assert receipt.get("provenance_limitation", "").startswith("Hashes verify artifact")


def test_missing_prediction_artifact_fails_closed(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    prediction = evidence["runs"][0]["predictions"][0]
    (valid_evidence.parent / prediction["path"]).unlink()
    with pytest.raises(PromotionEvidenceError, match="artifact is missing"):
        evaluate_promotion_evidence(valid_evidence)


def test_excluded_dataset_may_not_be_used_for_training_or_evaluation(valid_evidence: Path) -> None:
    evidence = json.loads(valid_evidence.read_text())
    split_path = valid_evidence.parent / evidence["split"]["path"]
    split = json.loads(split_path.read_text())
    split[0]["excluded"] = [split[0]["train"][0]]
    split_path.write_text(json.dumps(split, sort_keys=True))
    evidence["split"]["sha256"] = _hash(split_path)
    valid_evidence.write_text(json.dumps(evidence))
    with pytest.raises(PromotionEvidenceError, match="excluded datasets overlap"):
        evaluate_promotion_evidence(valid_evidence)
