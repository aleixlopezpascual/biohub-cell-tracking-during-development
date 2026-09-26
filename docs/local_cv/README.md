# Local CV results

> The recorded `public-rule-control-est-budget` result is a different, rule-based control—not OOF evidence for the 0.946-vs-EMA candidate. The current search found no matching fold-excluded candidate predictions/weights or receipt in the project and known prior CV workspace. See [`../endgame_next_steps_and_blockers.md`](../endgame_next_steps_and_blockers.md) for the missing artifacts and next validation steps.

## public-rule-control-est-budget

| Metric | Fold A (44b6) | Fold B (6bba) | Equal-weight CV |
|---|---:|---:|---:|
| Score | 0.5313966754463341 | 0.6220321552223572 | 0.5767144153343456 |
| Edge Jaccard | 0.5179759704251387 | 0.6148127402805713 |  |
| Division Jaccard | 0.0 | 0.0 |  |
| Node recall | 0.7253436273123322 | 0.8130155286874892 |  |

This run used extracted Kaggle training data, the official Royerlab
`tracking_cellmot` scorer, and the Biohub Local CV Pack prefix-holdout split.
It used the CV-pack `est` node budgets because the local quantile-budget path
failed with the extracted Zarr layout. The four public-test twins were
excluded from evaluation.

The metrics are also recorded in
[`results/local_cv/public-rule-control-est-budget.csv`](../../results/local_cv/public-rule-control-est-budget.csv)
and indexed in [`results/local_cv/experiments.csv`](../../results/local_cv/experiments.csv)
for programmatic comparisons. Competition data, extracted Zarr stores, and
temporary scorer artifacts are intentionally not tracked.
