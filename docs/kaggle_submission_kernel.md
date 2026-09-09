# Kaggle submission kernel

This competition is a **code competition**: submissions must come from a
Kaggle kernel (`kaggle competitions submit -k <kernel-ref> -v <version> -f
<file>`), not a raw file upload (`-f` alone returns a `400 Bad Request` from
`CreateSubmission`).

The submission kernel source lives at
`scripts/kaggle_kernels/public_rule_control_est_budget_submit/` and is pushed
with `kaggle kernels push -p <dir>`.

## Incident: v1 static-copy format error (submission 56101369)

The first kernel version only copied a precomputed `submission.csv` (staged
via a private Kaggle dataset) into `/kaggle/working/submission.csv`. Kaggle
accepted and ran it (`SubmissionStatus.COMPLETE`), but both `publicScore` and
`privateScore` came back blank. The Kaggle CLI table does not surface *why*;
the reason is only visible through the raw API object:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("biohub-cell-tracking-during-development")
print(subs[0]._error_description)
# "Your notebook generated a submission file with incorrect format. Some
#  examples causing this are: wrong number of rows or columns, empty values,
#  an incorrect data type for a value, or invalid submission values from
#  what is expected."
```

Locally, the CSV's schema, dtypes, node/edge referential integrity, and
forest structure (no node with two parents) all validated cleanly against
`sample_submission.csv`. The real problem is structural to code
competitions: **a kernel that emits a hardcoded/precomputed file instead of
computing it from the mounted test data at run time is exactly what this
class of error message describes**, since the scorer's rerun can present
different test data/row counts than what was baked into the static file.

## Fix (kernel v2, submission 56107186)

`kernel.py` was rewritten to run inference live, every time the kernel runs:

1. Locate the mounted competition root (`/kaggle/input/biohub-cell-tracking-during-development`
   or `/kaggle/input/competitions/biohub-cell-tracking-during-development`).
2. For each `test/*.zarr` dataset, read only metadata embedded in that
   dataset's own OME-Zarr group (`image_statistics` quantiles for the
   detection threshold, physical `scale` for the frame-to-frame linking
   distance gate). No lookups into `train/*.geff` or any other dataset's
   ground truth are performed, so the kernel generalizes to test data it has
   never seen.
3. Bounded local-maxima detection (`uniform_filter` + `maximum_filter` +
   weighted sub-voxel refinement) per frame, followed by Hungarian
   (`linear_sum_assignment`) frame-to-frame linking gated by physical
   distance, matching the approach validated in `results/local_cv/`.
4. Nodes with no incident edge are pruned (they only inflate `T_pred` in the
   adjusted edge Jaccard without contributing recall).
5. Write `/kaggle/working/submission.csv` with columns
   `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.

Verified before submitting:
- Local dry-run against the same 4 public test volumes (pointed at a local
  mirror of `test/`) produced 4422 / 671 / 3111 / 4506 nodes across the four
  datasets, matching the Kaggle-run kernel output exactly.
- Re-validated schema: no NaNs, sequential `id`, unique `node_id` per
  dataset, no dangling edge references, no duplicated `target_id` (i.e. no
  node with two parents — the forest constraint the metric's local-window
  division logic depends on).
- Kernel run time: ~55s end-to-end for all four test volumes, well within
  Kaggle's execution limits.

`dataset_sources` was removed from `kernel-metadata.json`; the kernel now
only depends on `competition_sources`.

## Result

Submission `56107186` scored successfully: **public score 0.251**
(`SubmissionStatus.COMPLETE`). This is well below the local CV estimate of
0.5767 for the same candidate. The gap is expected, not a bug:

- Local CV scored a bounded rule-based detector using GT-derived node
  budgets and per-prefix distance gates tuned against the CV-pack's known
  train folds (`results/local_cv/README.md`).
- The live Kaggle kernel must generalize to test data with no ground truth
  available, so it derives its detection threshold and linking distance gate
  only from generic per-dataset OME-Zarr metadata (`image_statistics`
  quantiles, physical `scale`) rather than any oracle node-count/gate value.
  This is a materially weaker heuristic than the CV-pack's GT-informed
  settings, so a substantial LB/CV gap for this exact rule-based baseline is
  expected.

This LB score should be treated as the calibration point for "rule-based
detector with no oracle budget," not as a discrepancy to chase down further.

