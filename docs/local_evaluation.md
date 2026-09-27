# Local evaluation before Kaggle submission

> **Current candidate status:** the 0.948 EMA notebook has no verified paired fold-disjoint evaluation or receipt producer yet, and no matching candidate OOF predictions were found in the searched project/CV paths. This generic CSV scorer can evaluate predictions once produced; it does not generate model predictions or validate the notebook's declared receipt by itself. Follow [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md) for the exact current gate and CPU/GPU split.

Every candidate must be evaluated locally before submission. The public leaderboard is useful feedback, but participants found it can be optimistic or misleading because public test volumes appear to include train-like twins and because sparse ground truth makes single aggregate scores hard to interpret.

## ⚠️ Crucial Validation Pitfalls & Kaggle Secrets

Based on community research and competition-wide analysis:

1. **The "Dummy Test Set" Trap:** The 4 `.zarr` volumes provided in the public test directory of the competition are **dummies copied directly from the train data**. Tuning hyperparameters (like detection threshold or division gates) directly on these 4 dummy volumes causes severe overfitting. Kaggle swaps this directory with the real, hidden test set during submission. Therefore, **our embryo-disjoint local CV (using real embryo-prefix holdouts) is the only source of truth**.
2. **Linear Track Dominance ($\ge$ 96.6%):** There are 4,435 disjoint tracking trees in the ground truth, but **at most 151 actually branch**. Because division events are so extremely rare, naive validation splits can suffer from high noise: matching or missing a single division can swing the division Jaccard score by over 10% ($>0.10$).
3. **"The Metric Pays You to Delete Nodes" (FP Node Penalization):** Because of the 7 µm matching threshold, false positive node detections are heavily penalized by the adjusted edge Jaccard. Ranking predicted nodes by confidence and cutting them off via a strict budget (`cap_node_budget`) or pruning isolated short components (`filter_short_components`) yields significant score improvements.
4. **No Long Steps for Divisions:** The displacement IQR of division events is completely within the normal cell displacement range. The apparent "long steps" are actually **detector localization errors** (up to 7 µm) compounding across the edge.
5. **The Mid-Volume "New Track" Gotcha:** Tracks starting mid-volume are almost always **broken continuations** (where the detector endpoint was off and the linker refused them) rather than missed daughters or new tracks. Linking them as forks to "recover divisions" creates massive false divisions and destroys scores.

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

## Paired OOF promotion evidence gate

`scripts/evaluate_promotion_gate.py` is a CPU-only **structural validator and scorer**. It does not run inference or train fold-excluded weights. It checks the declared metrics, including the arithmetic relationship between edge TP/FP/FN, `node_count_ratio`, and adjusted edge Jaccard, but schema v1 has no trusted signed runtime/scorer attestation. Therefore it always reports `promote=false`; it cannot authorize promotion, even when the metric checks pass. `scripts/run_oof_checkpoint.py --motion-relink` offers a local Gold-model ablation that applies the EMA relinker before graph construction/ILP. This one-to-one relinking path does not reconstruct the staged Kaggle candidate's later division-repair step, and it does **not** reproduce or validate the exact `0.946` Kaggle pipeline.

### Run or reuse the isolated Gold OOF ablation

Run the baseline with the flag off and the candidate with `--motion-relink`, using the **same fold-excluded weights, fold, split, config, and detection threshold**. Supply the actual prepared split JSON, fold checkpoint, and data paths for this workspace; these artifacts are not currently checked in. Example candidate invocation:

```bash
PYTHONPATH=src python3 scripts/run_oof_checkpoint.py \
  --config configs/gold_training.yaml \
  --splits /path/to/prepared_folds.json \
  --fold-index 0 \
  --candidate gold-candidate-name \
  --epoch 20 \
  --weights /path/to/fold-excluded/checkpoint.pth \
  --motion-relink
```

The flag is off by default. It uses tight/relaxed physical gates of 6/10 µm, velocity weight `0.5`, learned-edge bonus `1.0`, and a 2,600-node-per-frame cap. It requires indexed edge tuples `(source_index, target_index, probability, distance)` and fails closed if the upstream contract changes. If a frame exceeds the cap, relinking is skipped and the predictor's original edges are retained; otherwise only edges selected by the relinker survive, including an empty result when all proposed edges are rejected. `--motion-relink` cannot be combined with `--use-overlay`, so the ablation does not conflate those changes. Candidate outputs go under an `ema_motion_relink_v1/` subdirectory; sidecars bind config/counters, the checkpoint SHA256, the local runner/helper, and all Python files under the configured upstream `scripts/` and `src/` trees to each GEFF file/directory hash. Matching `--resume` or `--score-only` runs verify these bindings before reuse. These hashes are integrity metadata, not trusted proof that the declared code executed.

Output paths beneath the resolved configured output root reject symlinked descendants (including intermediate directories and final GEFF/sidecar paths) before writes.

Fresh runs execute neural inference (CUDA is the practical route, though `--device cpu` is supported); `--score-only` evaluates an already complete, matching output directory and is CPU-capable. Even with `--score-only`, the checkpoint path and matching upstream source tree must remain present because the runner hashes them before validating reuse. The run manifest and sidecars are local execution records, not a complete `evidence.json` and not independent proof that the stated source was executed. No real prediction or score has been generated for this ablation.

Once complete parent/candidate results from both folds have been converted into the documented `evidence.json` contract, run:

```bash
PYTHONPATH=src python3 scripts/evaluate_promotion_gate.py \
  --evidence outputs/ema_oof/evidence.json \
  --output-receipt outputs/ema_oof/promotion_receipt.json
```

The manifest uses schema version 1. Its paths are relative to and must remain inside the manifest's directory, and every artifact reference contains `path` and a 64-character `sha256`. It records:

- the exact parent/candidate names, distinct method IDs, source files/hashes, and one declared changed factor;
- the A/B split file and official metric ID/version/source hash;
- exactly four run records: parent and candidate on each of folds A and B;
- each run's run manifest, OOF manifest, checkpoint, fixed inference config, one prediction artifact per held-out dataset, official summary, and per-dataset metrics CSV.

The run manifest must bind the run/candidate/commit to its fold, split and config hashes, exact training and evaluation dataset lists, method source, and scorer. The OOF manifest must bind those values to the held-out dataset list, checkpoint and prediction hashes, and inference settings. The paired parent/candidate runs must use identical weights, config, scorer, split, and inference settings; the declared method IDs/source hashes must identify the one intended delta. The validator re-hashes the referenced files, checks train/test/excluded separation and exact A/B coverage, rejects duplicate or non-finite metrics, verifies adjusted edge Jaccard from edge counts and node-count ratio, and cross-checks each fold's recomputed score against its recorded official summary. It follows `J * (1 - 0.1 * (node_count_ratio - 1))`, lower-clamped at zero but not upper-clamped; under-counting can therefore yield an adjusted value above 1.0. For each per-dataset row, a zero division denominator maps to Jaccard 1.0 and contributes the weighted `+0.1` term to that row's diagnostic score; the pooled score instead computes division Jaccard from summed counts, using 1.0 only when the pooled denominator is zero. Review the actual code diff independently: the changed-factor label and source hash do not prove that only that code path changed or that the source was executed. Schema v1 cannot establish trusted runtime/scorer provenance.

Exact v1 producer keys: `evidence.json` contains `schema_version`, `experiment` (`parent`, `candidate`, `changed_factor`), `metric` (`id`, `version`, `source`), `split`, and `runs`. Each variant has `name`, `method_id`, and `method_source`; each run has `variant`, `fold`, `run_manifest`, `oof_manifest`, `checkpoint`, `config`, `predictions` (a list of `{dataset, path, sha256}`), `official_summary`, and `per_dataset`. Run-manifest JSON requires `run_id`, `candidate`, `fold`, `commit_sha`, `split_sha256`, `config_sha256`, `training_datasets`, `evaluation_datasets`, `method_id`, `method_sha256`, `metric_id`, `metric_version`, and `metric_source_sha256`. OOF-manifest JSON requires `run_id`, `candidate`, `fold`, `datasets`, `split_sha256`, `config_sha256`, `checkpoint_sha256`, `run_manifest_sha256`, the same method/metric identifiers and hashes, `predictions_sha256` (dataset-to-hash mapping), and a non-empty `oof_inference` object. Each official summary requires `score` and `n`. Each per-dataset CSV requires `dataset`, `score`, `adj_edge_jaccard` (or `adjusted_edge_jaccard`), `edge_tp/fp/fn`, `division_tp/fp/fn`, `node_recall`, `node_count_ratio`, `edges_fragmented`, `edges_lost_to_detection`, and `wrong_association_edges`. The split JSON is the A/B list with `name`, `train`, `test`, and `excluded` fields used by `scripts/prepare_gold_training.py`.

The score gate requires a pooled micro-averaged gain of at least `+0.001`, no fold-level score drop below `-0.003`, and no individual held-out dataset score drop below `-0.003`. For the currently undefined “material regression” clause, the implementation is deliberately conservative: at pooled, fold, **and individual-dataset** level, any decline beyond a `1e-9` floating-point comparison epsilon in adjusted-edge Jaccard, division Jaccard, or node recall, any move in node-count ratio farther from 1.0, or any increase in fragmentation/detection-loss/wrong-association counts blocks promotion and requires review. No material-regression tolerance is invented. The structural report includes per-video deltas and the hashes of all verified artifacts.

Exit status: `1` means valid structural evidence was assessed but promotion remains blocked; `2` means evidence was invalid or incomplete. A valid v1 report is written with `promote=false` because no trusted runtime/scorer attestation is supported. Status `0` is unreachable until the evidence contract and validator gain an independently trusted attestation path. SHA256 checks prove that files match their declared hashes, not that a source file was executed. The local Gold OOF mode above does not yet produce the four-run v1 package, and no real EMA predictions or paired score evidence exist. Never hand-author `evidence.json` or set `promote` yourself. Passing synthetic tests validate only the structural checks; they are not score evidence.

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
