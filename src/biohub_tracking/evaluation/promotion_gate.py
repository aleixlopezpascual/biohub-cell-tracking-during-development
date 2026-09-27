"""Fail-closed provenance and promotion checks for paired two-fold OOF results."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

FOLDS = frozenset({"A", "B"})
MIN_POOLED_GAIN = 0.001
MAX_FOLD_DROP = 0.003
EPSILON = 1e-9
ADJUSTED_EDGE_ALPHA = 0.1
COUNTS = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
DIAGNOSTICS = (
    "node_recall", "node_count_ratio", "edges_fragmented",
    "edges_lost_to_detection", "wrong_association_edges",
)


class PromotionEvidenceError(ValueError):
    """Raised when paired OOF evidence is missing, inconsistent, or tampered."""


def _obj(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PromotionEvidenceError(f"{label} must be an object")
    return value


def _text(obj: Mapping[str, Any], key: str, label: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PromotionEvidenceError(f"{label}.{key} must be a non-empty string")
    return value


def _names(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise PromotionEvidenceError(f"{label} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise PromotionEvidenceError(f"{label} contains duplicate identifiers")
    return tuple(sorted(value))


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise PromotionEvidenceError(f"cannot read {path}: {exc}") from exc
    return hasher.hexdigest()


def _verify(
    root: Path, ref: object, label: str, verified: list[dict[str, str]]
) -> tuple[Path, str]:
    item = _obj(ref, label)
    relative = Path(_text(item, "path", label))
    expected = _text(item, "sha256", label).lower()
    if relative.is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PromotionEvidenceError(f"{label} requires a relative path and 64-character SHA256")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise PromotionEvidenceError(f"{label}.path escapes the evidence directory")
    if not path.is_file():
        raise PromotionEvidenceError(f"{label} artifact is missing: {relative}")
    actual = _digest(path)
    if actual != expected:
        raise PromotionEvidenceError(f"{label} hash mismatch: expected {expected}, got {actual}")
    verified.append({"label": label, "path": relative.as_posix(), "sha256": actual})
    return path, actual


def _json_artifact(
    root: Path, ref: object, label: str, verified: list[dict[str, str]]
) -> tuple[Mapping[str, Any], str]:
    path, digest = _verify(root, ref, label, verified)
    try:
        return _obj(json.loads(path.read_text(encoding="utf-8")), label), digest
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionEvidenceError(f"{label} is invalid JSON: {exc}") from exc


def _read_splits(
    root: Path, ref: object, verified: list[dict[str, str]]
) -> tuple[dict[str, dict[str, tuple[str, ...]]], str]:
    path, digest = _verify(root, ref, "split", verified)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionEvidenceError(f"split file is invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise PromotionEvidenceError("split file must be a list")
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for index, raw in enumerate(data):
        item = _obj(raw, f"split[{index}]")
        name = _text(item, "name", f"split[{index}]")
        if name in result:
            raise PromotionEvidenceError(f"duplicate split fold {name}")
        train = _names(item.get("train"), f"split.{name}.train")
        test = _names(item.get("test"), f"split.{name}.test")
        excluded = _names(item.get("excluded", []), f"split.{name}.excluded")
        if not train or not test or set(train) & set(test):
            raise PromotionEvidenceError(
                f"fold {name} must have non-empty, disjoint training_datasets and test datasets"
            )
        if set(excluded) & (set(train) | set(test)):
            raise PromotionEvidenceError(f"fold {name} excluded datasets overlap train or test")
        result[name] = {"train": train, "test": test}
    if set(result) != FOLDS:
        raise PromotionEvidenceError(
            f"split file must define exactly A and B; got {sorted(result)}"
        )
    if set(result["A"]["test"]) & set(result["B"]["test"]):
        raise PromotionEvidenceError("fold A and B held-out datasets overlap")
    return result, digest


def _aggregate(frame: pd.DataFrame) -> dict[str, float]:
    weight = frame["edge_tp"] + frame["edge_fp"] + frame["edge_fn"]
    weight_sum = float(weight.sum())
    edge = (
        float((frame["adj_edge_jaccard"] * weight).sum() / weight_sum)
        if weight_sum
        else 1.0
    )
    div_sum = float(
        frame["division_tp"].sum() + frame["division_fp"].sum() + frame["division_fn"].sum()
    )
    div = float(frame["division_tp"].sum() / div_sum) if div_sum else 1.0
    return {
        "score": edge + ADJUSTED_EDGE_ALPHA * div,
        "adjusted_edge_jaccard": edge,
        "division_jaccard": div,
        "node_recall_macro": float(frame["node_recall"].mean()),
        "node_count_ratio_macro": float(frame["node_count_ratio"].mean()),
    }


def _load_metrics(path: Path, label: str, expected: tuple[str, ...]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise PromotionEvidenceError(f"cannot read {label}: {exc}") from exc
    if "adj_edge_jaccard" not in frame and "adjusted_edge_jaccard" in frame:
        frame = frame.rename(columns={"adjusted_edge_jaccard": "adj_edge_jaccard"})
    required = {"dataset", "score", "adj_edge_jaccard", *COUNTS, *DIAGNOSTICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PromotionEvidenceError(f"{label} missing columns: {missing}")
    if frame["dataset"].isna().any() or frame["dataset"].duplicated().any():
        raise PromotionEvidenceError(f"{label} has missing or duplicate dataset rows")
    if tuple(sorted(frame["dataset"].astype(str))) != expected:
        raise PromotionEvidenceError(f"{label} dataset coverage differs from held-out split")
    numeric = ["score", "adj_edge_jaccard", *COUNTS, *DIAGNOSTICS]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not frame[column].map(math.isfinite).all():
            raise PromotionEvidenceError(f"{label}.{column} contains non-finite values")
    count_columns = (
        *COUNTS,
        "edges_fragmented",
        "edges_lost_to_detection",
        "wrong_association_edges",
    )
    for column in count_columns:
        if (frame[column] < 0).any():
            raise PromotionEvidenceError(f"{label}.{column} contains negative counts")
    if (frame["node_count_ratio"] <= 0).any():
        raise PromotionEvidenceError(f"{label}.node_count_ratio must be positive")
    edge_total = frame["edge_tp"] + frame["edge_fp"] + frame["edge_fn"]
    edge_jaccard = (frame["edge_tp"] / edge_total.where(edge_total > 0, 1.0)).where(
        edge_total > 0,
        1.0,
    )
    # Match metrics.edge_jaccard.adjusted_jaccard: only lower-clamp, so node under-counting
    # can make this adjusted metric exceed 1.0.
    edge_penalty = 1.0 - ADJUSTED_EDGE_ALPHA * (frame["node_count_ratio"] - 1.0)
    expected_adjusted_edge = (edge_jaccard * edge_penalty).clip(lower=0.0)
    if not (frame["adj_edge_jaccard"] - expected_adjusted_edge).abs().le(EPSILON).all():
        raise PromotionEvidenceError(
            f"{label} has inconsistent adjusted edge Jaccard with TP/FP/FN and node_count_ratio"
        )
    div_total = frame["division_tp"] + frame["division_fp"] + frame["division_fn"]
    div_score = frame["division_tp"] / div_total.where(div_total > 0, 1.0)
    # Per-dataset division Jaccard defines the empty/empty case as 1.0, so the score
    # includes the weighted division term even when that dataset has no division events.
    expected_score = frame["adj_edge_jaccard"] + ADJUSTED_EDGE_ALPHA * div_score.where(
        div_total > 0,
        1.0,
    )
    if not (frame["score"] - expected_score).abs().le(EPSILON).all():
        raise PromotionEvidenceError(f"{label} per-dataset scores disagree with metric components")
    # Metric subreports use the no-event Jaccard convention (1.0); pooled scoring
    # applies the same convention to the aggregate counts in _aggregate.
    frame["division_jaccard"] = div_score.where(div_total > 0, 1.0)
    return frame.sort_values("dataset", kind="stable").reset_index(drop=True)


def _validate_run(
    root: Path,
    raw: object,
    variant: str,
    fold: str,
    split: dict[str, tuple[str, ...]],
    variant_spec: Mapping[str, Any],
    metric: Mapping[str, Any],
    split_hash: str,
    verified: list[dict[str, str]],
) -> dict[str, Any]:
    record = _obj(raw, f"run.{variant}.{fold}")
    if record.get("variant") != variant or record.get("fold") != fold:
        raise PromotionEvidenceError(f"run identity mismatch for {variant} fold {fold}")
    run, run_hash = _json_artifact(
        root, record.get("run_manifest"), f"{variant}.{fold}.run_manifest", verified
    )
    oof, _ = _json_artifact(
        root, record.get("oof_manifest"), f"{variant}.{fold}.oof_manifest", verified
    )
    _, checkpoint_hash = _verify(
        root, record.get("checkpoint"), f"{variant}.{fold}.checkpoint", verified
    )
    _, config_hash = _verify(root, record.get("config"), f"{variant}.{fold}.config", verified)
    expected = split["test"]
    for manifest, label in ((run, "run_manifest"), (oof, "oof_manifest")):
        if manifest.get("candidate") != variant_spec["name"] or manifest.get("fold") != fold:
            raise PromotionEvidenceError(f"{variant}.{fold}.{label} candidate/fold mismatch")
        if manifest.get("split_sha256") != split_hash:
            raise PromotionEvidenceError(f"{variant}.{fold}.{label} split hash mismatch")
        if manifest.get("config_sha256") != config_hash:
            raise PromotionEvidenceError(f"{variant}.{fold}.{label} config hash mismatch")
        if (
            manifest.get("method_id") != variant_spec["method_id"]
            or manifest.get("method_sha256") != variant_spec["method_sha256"]
        ):
            raise PromotionEvidenceError(f"{variant}.{fold}.{label} method provenance mismatch")
        for field in ("metric_id", "metric_version", "metric_source_sha256"):
            expected_value = metric[field]
            if manifest.get(field) != expected_value:
                raise PromotionEvidenceError(f"{variant}.{fold}.{label} {field} mismatch")
    run_id = _text(run, "run_id", f"{variant}.{fold}.run_manifest")
    commit = _text(run, "commit_sha", f"{variant}.{fold}.run_manifest")
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit):
        raise PromotionEvidenceError(f"{variant}.{fold} commit_sha is invalid")
    if oof.get("run_id") != run_id or oof.get("run_manifest_sha256") != run_hash:
        raise PromotionEvidenceError(f"{variant}.{fold} OOF manifest is not bound to run manifest")
    if (
        _names(run.get("training_datasets"), f"{variant}.{fold}.training_datasets")
        != split["train"]
    ):
        raise PromotionEvidenceError(f"{variant}.{fold} training_datasets do not match split")
    if (
        _names(run.get("evaluation_datasets"), f"{variant}.{fold}.evaluation_datasets")
        != expected
    ):
        raise PromotionEvidenceError(f"{variant}.{fold} evaluation_datasets do not match split")
    if _names(oof.get("datasets"), f"{variant}.{fold}.datasets") != expected:
        raise PromotionEvidenceError(
            f"{variant}.{fold} OOF datasets do not match held-out split"
        )
    if oof.get("checkpoint_sha256") != checkpoint_hash:
        raise PromotionEvidenceError(f"{variant}.{fold} checkpoint hash mismatch")

    predictions = record.get("predictions")
    if not isinstance(predictions, list):
        raise PromotionEvidenceError(f"{variant}.{fold} predictions must be a list")
    prediction_hashes: dict[str, str] = {}
    for i, raw_prediction in enumerate(predictions):
        pred = _obj(raw_prediction, f"{variant}.{fold}.prediction[{i}]")
        dataset = _text(pred, "dataset", f"{variant}.{fold}.prediction[{i}]")
        if dataset in prediction_hashes:
            raise PromotionEvidenceError(f"{variant}.{fold} duplicate prediction dataset {dataset}")
        _, prediction_hashes[dataset] = _verify(
            root, pred, f"{variant}.{fold}.prediction[{dataset}]", verified
        )
    if set(prediction_hashes) != set(expected):
        raise PromotionEvidenceError(
            f"{variant}.{fold} prediction datasets do not match held-out split"
        )
    if oof.get("predictions_sha256") != prediction_hashes:
        raise PromotionEvidenceError(f"{variant}.{fold} prediction hashes mismatch")
    inference = oof.get("oof_inference")
    if not isinstance(inference, dict) or not inference:
        raise PromotionEvidenceError(f"{variant}.{fold} oof_inference is missing")

    summary, _ = _json_artifact(
        root, record.get("official_summary"), f"{variant}.{fold}.official_summary", verified
    )
    per_path, _ = _verify(
        root, record.get("per_dataset"), f"{variant}.{fold}.per_dataset", verified
    )
    frame = _load_metrics(per_path, f"{variant}.{fold}.official_per_dataset", expected)
    if summary.get("n") != len(expected):
        raise PromotionEvidenceError(f"{variant}.{fold} summary does not cover held-out datasets")
    result = _aggregate(frame)
    score = summary.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise PromotionEvidenceError(f"{variant}.{fold} summary score is invalid")
    if not math.isclose(result["score"], float(score), rel_tol=0.0, abs_tol=EPSILON):
        raise PromotionEvidenceError(
            f"{variant}.{fold} recomputed score disagrees with official summary"
        )
    frame["fold"] = fold
    return {
        "name": variant_spec["name"], "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash, "inference": inference,
        "frame": frame, "summary": result,
    }


def _same_json(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _review_reasons(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    by_fold: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    p_all, c_all = _aggregate(parent), _aggregate(candidate)
    core = ("adjusted_edge_jaccard", "division_jaccard", "node_recall_macro")
    for metric in core:
        if c_all[metric] < p_all[metric] - EPSILON:
            reasons.append(f"pooled {metric} regressed")
        for fold, pair in by_fold.items():
            if pair["candidate"][metric] < pair["parent"][metric] - EPSILON:
                reasons.append(f"fold {fold} {metric} regressed")
    if (
        abs(c_all["node_count_ratio_macro"] - 1)
        > abs(p_all["node_count_ratio_macro"] - 1) + EPSILON
    ):
        reasons.append("pooled node_count_ratio moved farther from 1.0")
    for fold, pair in by_fold.items():
        if (
            abs(pair["candidate"]["node_count_ratio_macro"] - 1)
            > abs(pair["parent"]["node_count_ratio_macro"] - 1) + EPSILON
        ):
            reasons.append(f"fold {fold} node_count_ratio moved farther from 1.0")
    paired_rows = parent.merge(
        candidate,
        on=["fold", "dataset"],
        suffixes=("_parent", "_candidate"),
        validate="one_to_one",
    )
    for row in paired_rows.itertuples(index=False):
        for column in ("adj_edge_jaccard", "division_jaccard", "node_recall"):
            parent_value = getattr(row, f"{column}_parent")
            candidate_value = getattr(row, f"{column}_candidate")
            if candidate_value < parent_value - EPSILON:
                reasons.append(f"dataset {row.dataset} {column} regressed")
        if (
            abs(row.node_count_ratio_candidate - 1)
            > abs(row.node_count_ratio_parent - 1) + EPSILON
        ):
            reasons.append(f"dataset {row.dataset} node_count_ratio moved farther from 1.0")
        diagnostic_counts = (
            "edges_fragmented",
            "edges_lost_to_detection",
            "wrong_association_edges",
        )
        for column in diagnostic_counts:
            candidate_value = getattr(row, f"{column}_candidate")
            parent_value = getattr(row, f"{column}_parent")
            if candidate_value > parent_value + EPSILON:
                reasons.append(f"dataset {row.dataset} {column} increased")
    diagnostic_counts = (
        "edges_fragmented",
        "edges_lost_to_detection",
        "wrong_association_edges",
    )
    for column in diagnostic_counts:
        if candidate[column].sum() > parent[column].sum() + EPSILON:
            reasons.append(f"pooled {column} increased")
        for fold in sorted(FOLDS):
            candidate_total = candidate.loc[candidate.fold == fold, column].sum()
            parent_total = parent.loc[parent.fold == fold, column].sum()
            if candidate_total > parent_total + EPSILON:
                reasons.append(f"fold {fold} {column} increased")
    return reasons


def evaluate_promotion_evidence(evidence_path: str | Path) -> dict[str, Any]:
    """Validate and compare a complete parent/candidate A/B evidence manifest.

    Hashes verify artifact identity/integrity, not that a declared program
    generated a file. Schema v1 has no trusted runtime attestation, so this
    report never authorizes promotion even when its metric checks pass.
    """
    path = Path(evidence_path).resolve()
    if not path.is_file():
        raise PromotionEvidenceError(f"evidence manifest does not exist: {path}")
    try:
        evidence = _obj(json.loads(path.read_text(encoding="utf-8")), "evidence")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionEvidenceError(f"evidence manifest is invalid JSON: {exc}") from exc
    if evidence.get("schema_version") != 1:
        raise PromotionEvidenceError("evidence schema_version must be 1")
    root = path.parent
    verified: list[dict[str, str]] = []
    splits, split_hash = _read_splits(root, evidence.get("split"), verified)
    experiment = _obj(evidence.get("experiment"), "experiment")
    _text(experiment, "changed_factor", "experiment")
    specs: dict[str, dict[str, Any]] = {}
    for variant in ("parent", "candidate"):
        item = _obj(experiment.get(variant), f"experiment.{variant}")
        _, method_hash = _verify(
            root, item.get("method_source"), f"experiment.{variant}.method_source", verified
        )
        specs[variant] = {
            "name": _text(item, "name", f"experiment.{variant}"),
            "method_id": _text(item, "method_id", f"experiment.{variant}"),
            "method_sha256": method_hash,
        }
    if (
        specs["parent"]["name"] == specs["candidate"]["name"]
        or specs["parent"]["method_id"] == specs["candidate"]["method_id"]
    ):
        raise PromotionEvidenceError("parent and candidate must have distinct names and method IDs")
    if specs["parent"]["method_sha256"] == specs["candidate"]["method_sha256"]:
        raise PromotionEvidenceError("parent and candidate method source hashes must differ")
    metric_record = _obj(evidence.get("metric"), "metric")
    _, metric_hash = _verify(root, metric_record.get("source"), "metric.source", verified)
    metric = {
        "metric_id": _text(metric_record, "id", "metric"),
        "metric_version": _text(metric_record, "version", "metric"),
        "metric_source_sha256": metric_hash,
    }
    raw_runs = evidence.get("runs")
    if not isinstance(raw_runs, list):
        raise PromotionEvidenceError("evidence.runs must be a list")
    expected_pairs = {(v, f) for v in ("parent", "candidate") for f in FOLDS}
    records: dict[tuple[str, str], object] = {}
    for index, raw in enumerate(raw_runs):
        item = _obj(raw, f"runs[{index}]")
        key = (
            _text(item, "variant", f"runs[{index}]"),
            _text(item, "fold", f"runs[{index}]"),
        )
        if key not in expected_pairs or key in records:
            raise PromotionEvidenceError(
                "evidence must contain exactly one run for each variant and fold A/B; "
                f"invalid {key}"
            )
        records[key] = item
    if set(records) != expected_pairs:
        raise PromotionEvidenceError(
            "evidence must contain exactly one run for each variant and fold A/B"
        )

    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    for fold in sorted(FOLDS):
        for variant in ("parent", "candidate"):
            outputs[variant, fold] = _validate_run(
                root,
                records[variant, fold],
                variant,
                fold,
                splits[fold],
                specs[variant],
                metric,
                split_hash,
                verified,
            )
        parent_run, candidate_run = outputs["parent", fold], outputs["candidate", fold]
        if parent_run["checkpoint_sha256"] != candidate_run["checkpoint_sha256"]:
            raise PromotionEvidenceError(f"fold {fold} parent/candidate checkpoint hashes differ")
        if parent_run["config_sha256"] != candidate_run["config_sha256"]:
            raise PromotionEvidenceError(f"fold {fold} parent/candidate configs differ")
        if not _same_json(parent_run["inference"], candidate_run["inference"]):
            raise PromotionEvidenceError(f"fold {fold} parent/candidate inference settings differ")

    parent = pd.concat(
        [outputs["parent", f]["frame"] for f in sorted(FOLDS)], ignore_index=True
    )
    candidate = pd.concat(
        [outputs["candidate", f]["frame"] for f in sorted(FOLDS)], ignore_index=True
    )
    if parent.dataset.duplicated().any() or candidate.dataset.duplicated().any():
        raise PromotionEvidenceError("held-out datasets must be unique across folds")
    parent_agg, candidate_agg = _aggregate(parent), _aggregate(candidate)
    fold_info: dict[str, dict[str, Any]] = {}
    fold_deltas: dict[str, float] = {}
    paired_rows: list[dict[str, Any]] = []
    for fold in sorted(FOLDS):
        p_run, c_run = outputs["parent", fold], outputs["candidate", fold]
        fold_deltas[fold] = (
            c_run["summary"]["score"] - p_run["summary"]["score"]
        )
        fold_info[fold] = {
            "parent": p_run["summary"], "candidate": c_run["summary"],
            "score_delta": fold_deltas[fold],
            "submetric_deltas": {
                key: c_run["summary"][key] - p_run["summary"][key]
                for key in p_run["summary"]
            },
        }
        p_rows, c_rows = p_run["frame"].set_index("dataset"), c_run["frame"].set_index("dataset")
        for dataset in sorted(p_rows.index):
            p, c = p_rows.loc[dataset], c_rows.loc[dataset]
            paired_rows.append({
                "fold": fold, "dataset": dataset,
                "score_delta": float(c.score - p.score),
                "adj_edge_jaccard_delta": float(c.adj_edge_jaccard - p.adj_edge_jaccard),
                "division_jaccard_delta": float(c.division_jaccard - p.division_jaccard),
                "node_recall_delta": float(c.node_recall - p.node_recall),
                "node_count_ratio_delta": float(c.node_count_ratio - p.node_count_ratio),
                "edges_fragmented_delta": float(c.edges_fragmented - p.edges_fragmented),
                "edges_lost_to_detection_delta": float(
                    c.edges_lost_to_detection - p.edges_lost_to_detection
                ),
                "wrong_association_edges_delta": float(
                    c.wrong_association_edges - p.wrong_association_edges
                ),
            })
    pooled_delta = candidate_agg["score"] - parent_agg["score"]
    score_gate_failures: list[str] = []
    if pooled_delta < MIN_POOLED_GAIN:
        score_gate_failures.append("pooled gain is below the minimum")
    for fold, delta in fold_deltas.items():
        if delta < -MAX_FOLD_DROP:
            score_gate_failures.append(
                f"fold {fold} score dropped by more than the allowed maximum"
            )
    for row in paired_rows:
        if row["score_delta"] < -MAX_FOLD_DROP:
            score_gate_failures.append(
                f"held-out dataset {row['dataset']} score dropped by more than the allowed maximum"
            )
    score_pass = not score_gate_failures
    review_reasons = _review_reasons(parent, candidate, fold_info)
    provenance_limitation = (
        "Hashes verify artifact identity/integrity, not that the declared method or scorer "
        "executed; schema v1 has no trusted signed runtime attestation."
    )
    blockers = [*score_gate_failures, *review_reasons, "Trusted runtime provenance is absent."]
    verified.sort(key=lambda item: (item["label"], item["path"], item["sha256"]))
    return {
        "schema_version": 1,
        "parent_candidate_name": specs["parent"]["name"],
        "candidate_name": specs["candidate"]["name"],
        "changed_factor": experiment["changed_factor"],
        "metric": {
            "id": metric["metric_id"],
            "version": metric["metric_version"],
            "source_sha256": metric_hash,
        },
        "split_sha256": split_hash,
        "folds_evaluated": sorted(FOLDS),
        "thresholds": {
            "minimum_pooled_gain": MIN_POOLED_GAIN,
            "maximum_fold_drop": MAX_FOLD_DROP,
            "maximum_held_out_dataset_drop": MAX_FOLD_DROP,
            "diagnostic_policy": (
                "Any core-metric decline or diagnostic-count increase blocks promotion; "
                "no unstated numeric tolerance is applied."
            ),
        },
        "pooled_score_parent": parent_agg["score"],
        "pooled_score_candidate": candidate_agg["score"],
        "pooled_score_delta": pooled_delta,
        "parent_submetrics": parent_agg,
        "candidate_submetrics": candidate_agg,
        "fold_scores": fold_info,
        "paired_per_dataset": paired_rows,
        "score_gate_passed": score_pass,
        "score_gate_failures": score_gate_failures,
        "metrics_regression_review_required": bool(review_reasons),
        "metrics_regression_reasons": review_reasons,
        "metrics_gate_passed": bool(score_pass and not review_reasons),
        "trusted_runtime_provenance": False,
        "provenance_gate_passed": False,
        "blockers": blockers,
        "promote": False,
        "verified_artifacts": verified,
        "evidence_manifest_sha256": _digest(path),
        "provenance_limitation": provenance_limitation,
    }
