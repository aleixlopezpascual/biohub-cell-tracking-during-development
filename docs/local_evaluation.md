# Local evaluation before Kaggle submission

Every candidate must be evaluated locally before submission. The public leaderboard is useful feedback, but participants found it can be optimistic or misleading because public test volumes appear to include train-like twins and because sparse ground truth makes single aggregate scores hard to interpret.

## Recommended protocol

1. Generate a candidate `submission.csv`.
2. Score it locally using `scripts/local_eval.py`.
3. Inspect per-dataset diagnostics, not only the final score.
4. Append an experiment-log row with the commit/config provenance.
5. Submit to Kaggle only when local CV improves without obvious metric regressions.

## Preferred split source

Use the Biohub Local CV Pack when available:

- Dataset: https://www.kaggle.com/datasets/dariushafshar/biohub-local-cv-pack
- Key file: `folds_prefix_holdout.csv`
- Fold A evaluates on `44b6` and tunes on `6bba`.
- Fold B evaluates on `6bba` and tunes on `44b6`.
- Public-test twins are excluded by default:
  - `44b6_0113de3b`
  - `44b6_0b24845f`
  - `6bba_05b6850b`
  - `6bba_05db0fb1`

The pack also provides `gt_per_volume_stats.csv`, which supplies `estimated_number_of_nodes` for the adjusted edge Jaccard node-count term.

## Local scoring command

For synthetic or converted ground truth in the same CSV schema as submissions:

```bash
PYTHONPATH=src python3 scripts/local_eval.py \
  --submission outputs/candidate/submission.csv \
  --gt-submission data/validation_gt_submission.csv \
  --cv-pack-dir /kaggle/input/biohub-local-cv-pack \
  --fold all \
  --candidate local-maxima-v1 \
  --config configs/inference.yaml \
  --output-dir outputs/local_eval/local-maxima-v1 \
  --log-experiment
```

Outputs:

- `summary.json`: aggregate score and provenance.
- `per_dataset.csv`: per-volume metrics and diagnostics.
- `local_eval_log.csv`: append-only experiment tracker when `--log-experiment` is used.
- `oracle_summary.json` and `oracle_per_dataset.csv`: optional detection-versus-linking
  headroom outputs when `--oracle-analysis` is used.

## Metrics to inspect

The CLI reports:

- final score
- adjusted edge Jaccard
- raw edge Jaccard
- division Jaccard
- edge TP/FP/FN
- division TP/FP/FN
- node recall
- predicted node count
- estimated true node count
- node-count ratio
- missed GT nodes
- spurious predicted nodes
- recovered edges
- fragmented edges
- detection-lost edges
- wrong-association edges

Do not promote a candidate based on final score alone. A healthy change should improve the component it targets without worsening node-count ratio, fragmentation, or wrong associations beyond what the score justifies.

For model-training decisions, add `--oracle-analysis`. The detection-constrained
oracle keeps the candidate's nodes and assigns only GT-consistent edges. The
gain from the current score to that oracle is linking/postprocessing headroom;
the remaining gap to perfect GT nodes and links is detection headroom.

Compare a candidate to the frozen baseline with aligned per-volume deltas:

```bash
PYTHONPATH=src python3 scripts/compare_local_evaluations.py \
  --baseline outputs/local_eval/baseline/per_dataset.csv \
  --candidate outputs/local_eval/candidate/per_dataset.csv \
  --output-dir outputs/local_eval/candidate/paired
```

The comparison fails if the OOF dataset sets differ and writes win/tie/loss
counts plus deltas for the official components and failure categories.

## Community references

- Official scorer and metric docs: https://github.com/royerlab/kaggle-cell-tracking-competition
- Top host discussion pointing to the scorer: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716062
- Local scorer notebook: https://www.kaggle.com/code/busyaprime/score-your-cell-tracker-locally
- Local CV Pack: https://www.kaggle.com/datasets/dariushafshar/biohub-local-cv-pack
- Rule-based CV/LB calibration: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716952
- CV/LB gap discussion: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160
- Metric patch discussion: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154
- Division-metric diagnostic notebook: https://www.kaggle.com/code/nekkon/your-linker-cannot-score-a-single-division

## Current limitation

`scripts/local_eval.py` scores submission-style CSVs with this repository's internal metric implementation. Real train GEFF scoring should be added as an optional adapter around `tracksdata`/`geff`/royerlab `tracking_cellmot`, keeping those dependencies outside the core install.
