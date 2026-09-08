from __future__ import annotations

import pandas as pd
import pytest

from biohub_tracking.evaluation.cv_splits import PUBLIC_TEST_TWINS, build_prefix_holdout_plan, load_cv_split_plan


def test_build_prefix_holdout_excludes_public_twins() -> None:
    stems = ["44b6_a", "44b6_0113de3b", "6bba_b", "6bba_05b6850b"]
    plan = build_prefix_holdout_plan(stems)
    assert plan.fold("A").evaluate == ("44b6_a",)
    assert plan.fold("B").evaluate == ("6bba_b",)
    assert set(plan.fold("A").excluded) == {"44b6_0113de3b", "6bba_05b6850b"}


def test_load_local_cv_pack_folds(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"volume": "44b6_a", "fold_A_role": "evaluate", "fold_B_role": "tune", "is_public_test_twin": False},
            {"volume": "6bba_b", "fold_A_role": "tune", "fold_B_role": "evaluate", "is_public_test_twin": False},
            {"volume": next(iter(PUBLIC_TEST_TWINS)), "fold_A_role": "evaluate", "fold_B_role": "tune", "is_public_test_twin": True},
        ]
    )
    frame.to_csv(tmp_path / "folds_prefix_holdout.csv", index=False)
    plan = load_cv_split_plan(cv_pack_dir=tmp_path)
    assert plan.fold("A").evaluate == ("44b6_a",)
    assert plan.fold("B").evaluate == ("6bba_b",)
    assert "local_cv_pack" in plan.provenance


def test_build_prefix_holdout_requires_two_prefixes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_prefix_holdout_plan(["44b6_a"])
