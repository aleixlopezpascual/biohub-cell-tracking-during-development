# Gold-training session handoff

## Decision at handoff

Do **not** start another long training run, add folds, build an ensemble, or
submit the current `temporal-pu-a` checkpoint. The broad competitive strategy
is still sound, but the active scratch-training campaign has not passed its
first viability gate.

The next session should treat detector diagnosis and a trustworthy OOF
reference as the highest-priority work. It should not interpret training loss,
the public leaderboard, or the existence of a completed Kaggle GPU run as
evidence of competitive performance.

## What is sound in the strategy

The following parts of
[the gated cloud design](cloud_kfold_training_design.md) remain the plan:

- use the TemporalUNet3D + node-transformer + graph-solver family, rather
  than expanding the dependency-light Hungarian baseline;
- evaluate on the two real embryo-prefix holdouts with the official graph
  metric;
- use detector/linker oracle analysis to choose the next subsystem to improve;
- select checkpoints by complete official OOF score, not trainer proxy metrics;
- promote longer runs, additional splits, and ensembles only after their
  measured OOF gates and final runtime gate pass;
- retain legitimate prior art such as learned edge scoring, physical candidate
  gates, TTA, ILP graph solving, conservative gap repair, and component
  filtering.

The public-notebook exploit must remain excluded. See
[the metric-hack finding](public_notebook_metric_hack_finding.md).

## Evidence available at handoff

| Candidate / evidence | Result | Interpretation |
| --- | ---: | --- |
| `public-rule-control-est-budget` local prefix CV | `0.5767` | Only registered, genuine local-CV reference. It is a rule-based baseline, not Gold-capable. |
| Clean public `biohub-v6-ultra-best` reproduction | `0.883` public Kaggle score | Legitimate and reproducible, but the shared support pack supplies one 50-epoch checkpoint, not the author's private multi-checkpoint ensemble. |
| `temporal-pu-a`, epoch 10, fold A | `0.007871` official OOF score | Not a viable candidate. The score came from 69 held-out datasets with TTA, a `0.50` detection threshold, and ILP enabled. |
| Epoch-10 detector node recall | `0.0174` | The main observed problem is detection/localization, before graph-linking refinements can matter. |
| Epoch-10 detector node-count ratio | `0.0747` | The model emits far too few nodes at the tested best threshold. |

The epoch-10 score is recorded in the ignored recovery artifact:
`outputs/kaggle_threshold_recovery/v2/det_0p5/oof_manifest.json`.
Its per-dataset error decomposition is in the adjacent
`official_per_dataset.csv`. It is intentionally **not** appended to
`results/local_cv/experiments.csv`: that registry is for a real local
evaluation, and this score was obtained on the Kaggle-attached CV data.

The detector-only threshold probe tested `0.50` through `0.95` on a bounded
sample. At `0.50`, it matched 19 of 96 annotated nodes (`0.1979`) in that
small probe; at `0.55` and above it was effectively empty. This is not a
complete threshold calibration: thresholds below `0.50` were not tested.
Nevertheless, the full OOF result at `0.50` is poor enough that threshold
tuning alone must not be assumed to close the gap.

## What happened during the GPU campaign

The infrastructure is now substantially more robust, but its successful
execution does not validate the model quality.

1. Early Kaggle versions failed from packaging, a P100/PyTorch incompatibility,
   then transformer-backpropagation OOM with three-frame windows.
2. Two-frame versions then encountered NaNs because the pinned upstream
   transformer gave `MultiheadAttention` an all-padding key mask for detector
   samples with no peaks.
3. The continuation runner now inserts one zero-valued dummy key for exactly
   those all-empty rows while preserving the original loss mask. It also checks
   finite gradients.
4. Resume initially failed because `map_location="cuda"` moved the saved CPU
   RNG byte tensor. The runner now restores RNG state on CPU.
5. A Kaggle resume preflight successfully exercised checkpoint restore,
   all-empty attention forward/backward, one real training batch, and
   checkpoint/RNG round-trip.
6. Epochs 5--9 then completed cleanly, producing `epoch_010.pt` and
   `epoch_010_model.pth`; zero batches were skipped.
7. OOF scoring initially failed on a lazy sibling import in the official
   scorer. The runner now evaluates inside the upstream import context.
8. The repaired OOF evaluation completed and exposed the real issue: the
   detector is severely underperforming.

These fixes are described in more detail in
[the implementation notes](gold_training_implementation.md). They should be
kept, but they are reliability fixes, not evidence that the changed Gaussian
positive-unlabeled target or scratch-training recipe is correct.

## Critical interpretation

The original premise, “train longer and add folds to improve a 0.946 model,”
is not supported by repository evidence:

- no recorded robust prefix-CV result establishes a `0.946` starting point;
- public code plus the only shared 50-epoch checkpoint reaches about `0.883`
  on the public leaderboard, not the claimed `0.946`;
- an epoch-10 scratch run at `0.007871` cannot justify an extrapolation that
  five to ten times more epochs will reach the competitive range;
- the present failure occurs before the proposed high-ROI linker, graph repair,
  fold-diversity, and ensembling work can add value.

Longer training can still be valuable **after** the detector passes basic
sanity checks and produces a credible early learning curve. It is not the next
experiment by default.

## Required next actions, in order

1. **Create a trustworthy learned-model reference.** Run the clean public
   50-epoch checkpoint on both prefix holdouts using the same official scorer
   and register the result. Do not substitute a public-LB score for this OOF
   result.
2. **Run detector sanity checks before consuming another long GPU block.**
   Overfit one or two fixed batches; inspect target maps and logits; verify
   coordinate transforms, anisotropic physical scales, temporal augmentation,
   target amplitude, and peak extraction. The test should show near-complete
   recovery of the annotated centers on those fixed batches.
3. **Produce a short, diagnostic learning curve.** For one frozen split,
   checkpoint at closely spaced early epochs (for example 1, 5, 10, and 20)
   and score with the official metric. Sweep a justified wider detection
   threshold range only after the raw detector-output distribution is checked.
4. **Run detector and linker oracle analyses on the validated reference.**
   If detections have high recall but edges are wrong, invest in hard negatives,
   motion features, edge calibration, bidirectional consistency, and graph
   repair. If detection is limiting, fix the detector first.
5. **Only then run the documented epoch gates.** Continue toward epochs 50,
   65, 80, and 100 only if the prior checkpoint improves the equal-weight
   two-prefix OOF score by at least `0.003` without a material fold regression.
6. **Add a second model or fold only after validation.** Require at least
   `+0.002` OOF ensemble gain and a hidden-test-scale runtime rehearsal. Do
   not use fold count as a proxy for diversity.

## Operational guardrails

- Preserve all existing uncommitted files and ignored `outputs/` artifacts;
  do not reset or clean the worktree.
- Do not launch a new Kaggle kernel or upload/checkpoint dataset merely to
  continue the current epoch-10 model. A new run needs the next action above
  to define a falsifiable objective.
- Continue to checkpoint each epoch, keep jobs below about ten hours, and use
  the preflight runner before an altered Kaggle environment or resume path.
- Before promoting or submitting any candidate, follow the mandatory local
  evaluation command in `AGENTS.md`, inspect `summary.json` and
  `per_dataset.csv`, and append the resulting real evaluation to
  `results/local_cv/experiments.csv`.
- Keep the final submission exploit-free and validate schema, node IDs,
  coordinates, and dangling edges.

## Useful artifacts and code paths

- Strategy: [cloud_kfold_training_design.md](cloud_kfold_training_design.md)
- Reliability notes: [gold_training_implementation.md](gold_training_implementation.md)
- Public baseline review: [kaggle_baseline_review.md](../research/kaggle_baseline_review.md)
- Clean-public-model result and exploit decision:
  [public_notebook_metric_hack_finding.md](public_notebook_metric_hack_finding.md)
- Resumable trainer: `scripts/continue_royerlab_training.py`
- Official OOF runner: `scripts/run_oof_checkpoint.py`
- Threshold probe: `scripts/probe_detection_thresholds.py`
- Kaggle preflight: `scripts/kaggle_resume_preflight.py`
- Epoch-10 recovery artifacts: `outputs/kaggle_recovery/v9/` and
  `outputs/kaggle_threshold_recovery/v2/`

## One-sentence instruction for the next session

**Do not scale the current scratch model; first establish a real learned-model
OOF baseline and prove that the detector can recover annotated centers on a
small fixed training set.**
