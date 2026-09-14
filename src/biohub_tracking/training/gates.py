"""Official-score checkpoint selection and compute-promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

DIAGNOSTIC_METRICS = (
    "edge_jaccard",
    "node_recall",
    "node_count_ratio",
    "edges_fragmented",
    "edges_lost_to_detection",
    "wrong_association_edges",
)


@dataclass(frozen=True)
class CheckpointScore:
    """Official graph scores for one checkpoint across validation folds."""

    epoch: int
    checkpoint: str
    fold_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.epoch < 1 or not self.checkpoint:
            raise ValueError("epoch must be positive and checkpoint must not be empty")
        if not self.fold_scores:
            raise ValueError("fold_scores must not be empty")
        if any(not 0 <= float(score) <= 1.1 for score in self.fold_scores.values()):
            raise ValueError("official fold scores must be in [0, 1.1]")

    @property
    def mean_score(self) -> float:
        """Equal-weight mean across the configured prefix holdouts."""
        return sum(float(value) for value in self.fold_scores.values()) / len(self.fold_scores)


@dataclass(frozen=True)
class PromotionDecision:
    """A deterministic pass/fail decision with measurable gains."""

    promote: bool
    reason: str
    mean_gain: float
    worst_fold_delta: float | None = None


def select_best_checkpoint(scores: Sequence[CheckpointScore]) -> CheckpointScore:
    """Select by official mean score, then earlier epoch and path for stable ties."""
    if not scores:
        raise ValueError("at least one checkpoint score is required")
    return sorted(scores, key=lambda item: (-item.mean_score, item.epoch, item.checkpoint))[0]


def rank_complementary_models(
    scores: Sequence[CheckpointScore],
    error_vectors: Mapping[str, Sequence[float]],
    *,
    max_models: int = 3,
    maximum_score_drop: float = 0.01,
) -> list[CheckpointScore]:
    """Greedily shortlist strong models with low OOF error correlation."""
    if max_models < 1 or maximum_score_drop < 0:
        raise ValueError("max_models must be positive and maximum_score_drop non-negative")
    best = select_best_checkpoint(scores)
    eligible = [
        score
        for score in scores
        if best.mean_score - score.mean_score <= maximum_score_drop
    ]
    vectors: dict[str, np.ndarray] = {}
    expected_size: int | None = None
    for score in eligible:
        if score.checkpoint not in error_vectors:
            raise ValueError(f"missing OOF error vector for {score.checkpoint}")
        vector = np.asarray(error_vectors[score.checkpoint], dtype=float).reshape(-1)
        if vector.size < 2 or not np.isfinite(vector).all():
            raise ValueError("OOF error vectors must contain at least two finite values")
        if expected_size is None:
            expected_size = vector.size
        elif vector.size != expected_size:
            raise ValueError("OOF error vectors must have equal lengths")
        vectors[score.checkpoint] = vector

    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        first_centered = first - first.mean()
        second_centered = second - second.mean()
        denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
        if denominator == 0:
            return 1.0 if np.array_equal(first, second) else 0.0
        return float(np.dot(first_centered, second_centered) / denominator)

    selected = [best]
    remaining = [score for score in eligible if score.checkpoint != best.checkpoint]
    while remaining and len(selected) < max_models:
        remaining.sort(
            key=lambda score: (
                sum(
                    correlation(vectors[score.checkpoint], vectors[item.checkpoint])
                    for item in selected
                )
                / len(selected),
                -score.mean_score,
                score.epoch,
                score.checkpoint,
            )
        )
        selected.append(remaining.pop(0))
    return selected


def evaluate_learning_curve_gate(
    baseline: CheckpointScore,
    candidates: Sequence[CheckpointScore],
    *,
    minimum_mean_gain: float = 0.003,
    maximum_fold_regression: float = 0.001,
) -> PromotionDecision:
    """Decide whether later training merits spending quota on more folds."""
    if minimum_mean_gain < 0 or maximum_fold_regression < 0:
        raise ValueError("gate thresholds must be non-negative")
    later = [candidate for candidate in candidates if candidate.epoch > baseline.epoch]
    if not later:
        return PromotionDecision(False, "no later checkpoint was evaluated", 0.0, None)
    ranked = sorted(later, key=lambda item: (-item.mean_score, item.epoch, item.checkpoint))
    evaluated: list[tuple[CheckpointScore, float, float]] = []
    for candidate in ranked:
        if set(candidate.fold_scores) != set(baseline.fold_scores):
            raise ValueError("baseline and candidate must contain the same folds")
        deltas = {
            fold: float(candidate.fold_scores[fold]) - float(baseline.fold_scores[fold])
            for fold in baseline.fold_scores
        }
        evaluated.append(
            (candidate, candidate.mean_score - baseline.mean_score, min(deltas.values()))
        )
    eligible = [
        item
        for item in evaluated
        if item[1] >= minimum_mean_gain and item[2] >= -maximum_fold_regression
    ]
    if eligible:
        best, mean_gain, worst_delta = eligible[0]
        return PromotionDecision(
            True,
            f"checkpoint at epoch {best.epoch} passed the learning-curve gate",
            mean_gain,
            worst_delta,
        )

    best, mean_gain, worst_delta = evaluated[0]
    if mean_gain < minimum_mean_gain:
        return PromotionDecision(
            False,
            f"best later checkpoint gain {mean_gain:.6f} is below {minimum_mean_gain:.6f}",
            mean_gain,
            worst_delta,
        )
    if worst_delta < -maximum_fold_regression:
        return PromotionDecision(
            False,
            f"worst fold regressed {worst_delta:.6f}, beyond {-maximum_fold_regression:.6f}",
            mean_gain,
            worst_delta,
        )
    raise AssertionError("unreachable learning-curve gate state")


def evaluate_ensemble_gate(
    *,
    best_single_score: float,
    ensemble_score: float,
    measured_runtime_hours: float,
    runtime_limit_hours: float,
    minimum_gain: float = 0.002,
) -> PromotionDecision:
    """Require both an official-score gain and feasible end-to-end runtime."""
    if minimum_gain < 0 or measured_runtime_hours < 0 or runtime_limit_hours <= 0:
        raise ValueError("invalid ensemble gate inputs")
    gain = float(ensemble_score) - float(best_single_score)
    if measured_runtime_hours > runtime_limit_hours:
        return PromotionDecision(
            False,
            f"runtime {measured_runtime_hours:.3f}h exceeds {runtime_limit_hours:.3f}h",
            gain,
        )
    if gain < minimum_gain:
        return PromotionDecision(
            False,
            f"ensemble gain {gain:.6f} is below {minimum_gain:.6f}",
            gain,
        )
    return PromotionDecision(True, "ensemble passed score and runtime gates", gain)


def load_checkpoint_scores(path: str | Path) -> list[CheckpointScore]:
    """Load long-form ``epoch,checkpoint,fold,score`` official-score CSV data."""
    frame = pd.read_csv(path)
    required = {"epoch", "checkpoint", "fold", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"checkpoint score CSV is missing columns: {missing}")
    records: list[CheckpointScore] = []
    for (epoch, checkpoint), group in frame.groupby(["epoch", "checkpoint"], sort=True):
        if group["fold"].duplicated().any():
            raise ValueError(f"duplicate fold score for epoch={epoch}, checkpoint={checkpoint}")
        records.append(
            CheckpointScore(
                epoch=int(epoch),
                checkpoint=str(checkpoint),
                fold_scores={
                    str(row.fold): float(row.score) for row in group.itertuples(index=False)
                },
            )
        )
    return records


def append_checkpoint_score(
    path: str | Path,
    *,
    epoch: int,
    checkpoint: str,
    fold: str,
    score: float,
) -> bool:
    """Append one official OOF score, idempotently rejecting conflicting reruns."""
    record = CheckpointScore(epoch, checkpoint, {fold: score})
    destination = Path(path)
    columns = ["epoch", "checkpoint", "fold", "score"]
    if destination.exists():
        existing = pd.read_csv(destination)
        missing = sorted(set(columns) - set(existing.columns))
        if missing:
            raise ValueError(f"checkpoint score CSV is missing columns: {missing}")
        matches = existing[
            (existing["epoch"] == epoch)
            & (existing["checkpoint"].astype(str) == checkpoint)
            & (existing["fold"].astype(str) == fold)
        ]
        if not matches.empty:
            values = matches["score"].astype(float).tolist()
            if len(values) == 1 and values[0] == float(score):
                return False
            raise ValueError("checkpoint already has a different official fold score")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "epoch": record.epoch,
                "checkpoint": record.checkpoint,
                "fold": fold,
                "score": float(score),
            }
        ],
        columns=columns,
    ).to_csv(destination, mode="a", header=not destination.exists(), index=False)
    return True


def write_learning_curve_svg(scores: Sequence[CheckpointScore], path: str | Path) -> None:
    """Write a dependency-free SVG of per-fold and mean official scores."""
    if not scores:
        raise ValueError("at least one checkpoint score is required")
    folds = sorted({fold for score in scores for fold in score.fold_scores})
    if any(set(score.fold_scores) != set(folds) for score in scores):
        raise ValueError("every checkpoint must contain the same folds")
    ordered = sorted(scores, key=lambda score: (score.epoch, score.checkpoint))
    minimum_epoch = min(score.epoch for score in ordered)
    maximum_epoch = max(score.epoch for score in ordered)
    all_values = [float(value) for score in ordered for value in score.fold_scores.values()]
    all_values.extend(score.mean_score for score in ordered)
    minimum_score = min(all_values)
    maximum_score = max(all_values)
    if maximum_epoch == minimum_epoch:
        maximum_epoch += 1
    if maximum_score == minimum_score:
        maximum_score += 1e-6
    width, height = 720, 420
    left, right, top, bottom = 70, 25, 35, 55

    def x_position(epoch: int) -> float:
        return left + (epoch - minimum_epoch) / (maximum_epoch - minimum_epoch) * (
            width - left - right
        )

    def y_position(value: float) -> float:
        return top + (maximum_score - value) / (maximum_score - minimum_score) * (
            height - top - bottom
        )

    series: dict[str, list[tuple[int, float]]] = {
        fold: [(score.epoch, float(score.fold_scores[fold])) for score in ordered]
        for fold in folds
    }
    series["mean"] = [(score.epoch, score.mean_score) for score in ordered]
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14">'
        "Official prefix-held learning curve</text>",
    ]
    for index, (name, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coordinates = " ".join(
            f"{x_position(epoch):.1f},{y_position(value):.1f}" for epoch, value in points
        )
        dash = ' stroke-dasharray="5,4"' if name != "mean" else ""
        elements.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            f'stroke-width="2"{dash}/>'
        )
        elements.append(
            f'<text x="{width-right-5}" y="{top + 16*index}" text-anchor="end" '
            f'font-size="12" fill="{color}">{escape(name)}</text>'
        )
    elements.extend(
        [
            f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="12">epoch</text>',
            f'<text x="14" y="{height/2}" transform="rotate(-90 14 {height/2})" '
            'text-anchor="middle" font-size="12">official score</text>',
            "</svg>",
        ]
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(elements), encoding="utf-8")


def write_diagnostic_curve_svgs(
    diagnostics: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = DIAGNOSTIC_METRICS,
) -> list[Path]:
    """Plot fold-level acceptance diagnostics against checkpoint epoch."""
    frame = pd.read_csv(diagnostics) if not isinstance(diagnostics, pd.DataFrame) else diagnostics
    required = {"epoch", "fold", *metrics}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"diagnostic CSV is missing columns: {missing}")
    if frame.duplicated(["epoch", "fold"]).any():
        raise ValueError("diagnostics contain duplicate epoch/fold rows")
    epochs = sorted(int(value) for value in frame["epoch"].unique())
    folds = sorted(str(value) for value in frame["fold"].unique())
    if not epochs or not folds:
        raise ValueError("diagnostics must not be empty")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    width, height = 720, 420
    left, right, top, bottom = 70, 25, 35, 55
    minimum_epoch, maximum_epoch = min(epochs), max(epochs)
    if maximum_epoch == minimum_epoch:
        maximum_epoch += 1

    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"diagnostic metric {metric!r} must be finite")
        minimum_value, maximum_value = float(values.min()), float(values.max())
        if maximum_value == minimum_value:
            maximum_value += 1e-6

        def x_position(epoch: int) -> float:
            return left + (epoch - minimum_epoch) / (maximum_epoch - minimum_epoch) * (
                width - left - right
            )

        def y_position(value: float) -> float:
            return top + (maximum_value - value) / (maximum_value - minimum_value) * (
                height - top - bottom
            )

        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="#333"/>',
            f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14">'
            f'{escape(metric)}</text>',
        ]
        for index, fold in enumerate(folds):
            rows = frame[frame["fold"].astype(str) == fold].sort_values("epoch")
            points = " ".join(
                f"{x_position(int(row.epoch)):.1f},{y_position(float(getattr(row, metric))):.1f}"
                for row in rows.itertuples(index=False)
            )
            color = colors[index % len(colors)]
            elements.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            elements.append(
                f'<text x="{width-right-5}" y="{top + 16*index}" text-anchor="end" '
                f'font-size="12" fill="{color}">{escape(fold)}</text>'
            )
        elements.extend(
            [
                f'<text x="{width/2}" y="{height-12}" text-anchor="middle" '
                'font-size="12">epoch</text>',
                "</svg>",
            ]
        )
        output_path = destination / f"{metric}.svg"
        output_path.write_text("\n".join(elements), encoding="utf-8")
        output_paths.append(output_path)
    return output_paths
