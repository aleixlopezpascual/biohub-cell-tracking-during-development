# Experiment tracking

This repository keeps explanatory documentation under `docs/` and
machine-readable experiment outputs under `results/`.

## Local CV results

Use `results/local_cv/experiments.csv` as the append-only index for local
validation runs. Every candidate should have a local CV row before it is used
for a Kaggle submission.

Required columns:

| Column | Meaning |
|---|---|
| `candidate` | Stable candidate name, matching configs/submission names where possible. |
| `run_at` | ISO-8601 timestamp for the local evaluation run. |
| `commit_sha` | Code commit used to generate the candidate. |
| `config` | Config path or short config identifier. |
| `fold_a_score` | Local CV score on the `44b6` evaluate fold. |
| `fold_b_score` | Local CV score on the `6bba` evaluate fold. |
| `cv_score` | Equal-weight mean of Fold A and Fold B. |
| `edge_jaccard_a` | Edge Jaccard on Fold A. |
| `edge_jaccard_b` | Edge Jaccard on Fold B. |
| `division_jaccard_a` | Division Jaccard on Fold A. |
| `division_jaccard_b` | Division Jaccard on Fold B. |
| `node_recall_a` | Node recall on Fold A. |
| `node_recall_b` | Node recall on Fold B. |
| `scorer` | Scorer implementation and provenance. |
| `split` | Split protocol, normally Biohub Local CV Pack prefix holdout. |
| `artifact` | Path to the detailed report/log for the run. |
| `notes` | Short caveats or follow-up context. |

Detailed markdown summaries for notable runs belong under `docs/local_cv/`.
Per-candidate CSV/JSON artifacts may live under `results/local_cv/`.

## Kaggle live scores

Use `results/kaggle_lb/submissions.csv` as the append-only index for Kaggle
submissions and leaderboard feedback.

Required columns:

| Column | Meaning |
|---|---|
| `submission_id` | Kaggle submission ID when available. |
| `candidate` | Candidate name, matching the local CV row. |
| `submitted_at` | ISO-8601 timestamp for the Kaggle submission. |
| `commit_sha` | Code commit used to generate the submitted file. |
| `submission_csv_path` | Local or artifact path for the submitted CSV. |
| `local_cv_score` | Local equal-weight CV score available before submission. |
| `local_fold_a` | Local Fold A score. |
| `local_fold_b` | Local Fold B score. |
| `public_score` | Kaggle public leaderboard score. |
| `private_score` | Kaggle private leaderboard score, filled after final scoring. |
| `kaggle_message` | Kaggle status/message for the submission. |
| `notes` | Short caveats, ensemble details, or interpretation notes. |

Do not submit candidates without first registering their local CV result unless
the row explicitly documents why local CV was impossible.
