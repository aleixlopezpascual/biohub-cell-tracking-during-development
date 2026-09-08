"""Local cross-validation split helpers for the Biohub competition.

The preferred protocol is the community Biohub Local CV Pack prefix-holdout
split: evaluate on the held-out embryo prefix while excluding public-test twin
volumes. When that CSV is not available, this module can build the same style
of deterministic prefix holdout from available train stems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

PUBLIC_TEST_TWINS = frozenset(
    {"44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"}
)
FoldName = Literal["A", "B"]


@dataclass(frozen=True)
class CVFold:
    """One deterministic local-validation fold."""

    name: str
    train: tuple[str, ...]
    evaluate: tuple[str, ...]
    excluded: tuple[str, ...]
    provenance: str


@dataclass(frozen=True)
class CVSplitPlan:
    """Collection of local-validation folds plus source metadata."""

    folds: tuple[CVFold, ...]
    provenance: str

    def fold(self, name: str) -> CVFold:
        """Return a fold by name, raising clearly when unavailable."""
        for fold in self.folds:
            if fold.name == name:
                return fold
        raise KeyError(f"unknown fold {name!r}; available folds: {[f.name for f in self.folds]}")

    def evaluation_stems(self, fold: str | None = None) -> tuple[str, ...]:
        """Return deterministic evaluation stems for one fold or all folds."""
        if fold is not None:
            return self.fold(fold).evaluate
        stems: set[str] = set()
        for cv_fold in self.folds:
            stems.update(cv_fold.evaluate)
        return tuple(sorted(stems))


def _prefix(stem: str) -> str:
    return stem.split("_", 1)[0]


def _clean_stems(stems: Iterable[str], exclude_public_twins: bool) -> tuple[str, ...]:
    values = {str(stem).removesuffix(".zarr").removesuffix(".geff") for stem in stems}
    if exclude_public_twins:
        values -= set(PUBLIC_TEST_TWINS)
    return tuple(sorted(values))


def _role_column(frame: pd.DataFrame, fold_name: FoldName) -> str:
    candidates = [f"fold_{fold_name}_role", f"fold_{fold_name.lower()}_role"]
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"fold CSV is missing a role column for fold {fold_name!r}: tried {candidates}")


def load_local_cv_pack_folds(
    folds_csv: str | Path,
    *,
    exclude_public_twins: bool = True,
) -> CVSplitPlan:
    """Load `folds_prefix_holdout.csv` from the Biohub Local CV Pack."""
    path = Path(folds_csv)
    frame = pd.read_csv(path)
    if "volume" not in frame.columns:
        raise ValueError(f"{path} must contain a 'volume' column")
    if "is_public_test_twin" not in frame.columns:
        frame["is_public_test_twin"] = frame["volume"].isin(PUBLIC_TEST_TWINS)

    folds: list[CVFold] = []
    for fold_name in ("A", "B"):
        role_col = _role_column(frame, fold_name)  # type: ignore[arg-type]
        eligible = frame.copy()
        if exclude_public_twins:
            eligible = eligible[~eligible["is_public_test_twin"].astype(bool)]
        train = tuple(sorted(eligible.loc[eligible[role_col] == "tune", "volume"].astype(str)))
        evaluate = tuple(sorted(eligible.loc[eligible[role_col] == "evaluate", "volume"].astype(str)))
        excluded = tuple(sorted(frame.loc[frame["is_public_test_twin"].astype(bool), "volume"].astype(str)))
        folds.append(
            CVFold(
                name=fold_name,
                train=train,
                evaluate=evaluate,
                excluded=excluded,
                provenance=f"local_cv_pack:{path}",
            )
        )
    return CVSplitPlan(tuple(folds), provenance=f"local_cv_pack:{path}")


def build_prefix_holdout_plan(
    stems: Iterable[str],
    *,
    exclude_public_twins: bool = True,
) -> CVSplitPlan:
    """Build deterministic prefix-holdout folds from available dataset stems.

    For the competition's known two embryo prefixes, fold A evaluates on
    `44b6` and fold B evaluates on `6bba`, matching the Local CV Pack. For
    other data, the first two sorted prefixes are used in the same pattern.
    """
    clean = _clean_stems(stems, exclude_public_twins)
    by_prefix: dict[str, list[str]] = {}
    for stem in clean:
        by_prefix.setdefault(_prefix(stem), []).append(stem)
    if len(by_prefix) < 2:
        raise ValueError("prefix-holdout CV requires at least two dataset prefixes")

    ordered_prefixes = [p for p in ("44b6", "6bba") if p in by_prefix]
    ordered_prefixes.extend(p for p in sorted(by_prefix) if p not in ordered_prefixes)
    eval_a, eval_b = ordered_prefixes[:2]
    excluded = tuple(sorted(set(stems) & set(PUBLIC_TEST_TWINS))) if exclude_public_twins else tuple()
    folds = (
        CVFold(
            name="A",
            train=tuple(sorted(s for p, xs in by_prefix.items() if p != eval_a for s in xs)),
            evaluate=tuple(sorted(by_prefix[eval_a])),
            excluded=excluded,
            provenance="prefix_holdout_fallback",
        ),
        CVFold(
            name="B",
            train=tuple(sorted(s for p, xs in by_prefix.items() if p != eval_b for s in xs)),
            evaluate=tuple(sorted(by_prefix[eval_b])),
            excluded=excluded,
            provenance="prefix_holdout_fallback",
        ),
    )
    return CVSplitPlan(folds=folds, provenance="prefix_holdout_fallback")


def load_cv_split_plan(
    *,
    cv_pack_dir: str | Path | None = None,
    stems: Iterable[str] | None = None,
    exclude_public_twins: bool = True,
) -> CVSplitPlan:
    """Load Local CV Pack folds when available, otherwise build fallback folds."""
    if cv_pack_dir is not None:
        folds_csv = Path(cv_pack_dir) / "folds_prefix_holdout.csv"
        if folds_csv.exists():
            return load_local_cv_pack_folds(folds_csv, exclude_public_twins=exclude_public_twins)
        raise FileNotFoundError(f"Local CV Pack folds not found: {folds_csv}")
    if stems is None:
        raise ValueError("either cv_pack_dir or stems must be provided")
    return build_prefix_holdout_plan(stems, exclude_public_twins=exclude_public_twins)
