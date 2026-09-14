# Gated cloud training design

This replaces the rejected five-fold-first design. The objective is to spend
Kaggle GPU quota only after the preceding experiment demonstrates official
prefix-held graph-score headroom.

## Decision gates

1. Establish the current candidate on both Local CV Pack prefix holdouts using
   the official Royerlab scorer. Public leaderboard values are never copied
   into fold score fields.
2. Run detection/linking oracle analysis with `scripts/local_eval.py
   --oracle-analysis`. Prioritize the subsystem with the larger measured
   headroom.
3. Continue one fold and evaluate checkpoints at epochs 50, 65, 80, and 100.
   Additional folds are allowed only when the best later checkpoint improves
   equal-weight prefix CV by at least 0.003 and no fold regresses by more than
   0.001.
4. Add a second complementary checkpoint only when its OOF ensemble improves
   at least 0.002 and a hidden-test-scale rehearsal remains within the runtime
   limit.

The metric used for checkpoint selection is the complete official graph score,
not training loss, pairwise accuracy, public LB, or `accuracy * node_recall`.

## Training target and model priorities

- The first campaign uses the official TemporalUNet3D + node-transformer path
  and a two-frame window. A measured T4x2 benchmark showed that three-frame
  windows exhaust GPU 0 during transformer backpropagation even at batch 8,
  so three-frame context is a later, separately budgeted ablation. The first
  high-impact linker variant adds the provided physical motion features and
  distance-gated hard negatives.
- Detection supervision uses anisotropic physical Gaussian targets and a low
  weight for unlabeled voxels. Sparse annotations must not be interpreted as
  dense negative labels.
- Offset regression remains out of scope until oracle diagnostics show that
  centroid localization near the 7 um boundary is a material source of loss.
- Probability calibration is fitted on OOF edges before forward/reverse or
  multi-checkpoint blending. A single ILP solve consumes the blended candidate
  probabilities.

## Reproducible Kaggle workflow

Prepare the two real prefix folds and immutable provenance:

```bash
PYTHONPATH=src python3 scripts/prepare_gold_training.py \
  --config configs/gold_training.yaml \
  --candidate temporal-link-v1
```

This produces `prefix_holdout_splits.json`, `run_manifest.json`, and the exact
bootstrap command as JSON. Add `--execute-bootstrap` only in a Kaggle GPU
session with the official repository attached. The command uses the upstream
TemporalUNet3D, transformer, lazy dataset, and training epoch functions, while
bypassing its `accuracy * recall` checkpoint selector. It fixes the random
seed, replaces point-voxel BCE with the configured physical Gaussian PU loss,
and saves the exact first-gate state. Later training is conditional on local
evaluation.

On Kaggle, the preferred entry point runs one exact target and its official
OOF evaluation end to end:

```bash
PYTHONPATH=src python3 scripts/run_gold_stage.py \
  --config configs/gold_training.yaml \
  --candidate temporal-pu-a \
  --target-epoch 50 \
  --resume
```

The first invocation prepares the real prefix split and manifest. Interrupted
training resumes from `latest.pt`; completed GEFF predictions are reused. The
stage writes `official_summary.json`, `official_per_dataset.csv`,
`submission.csv`, `oof_manifest.json`, the append-only `checkpoint_scores.csv`,
and per-volume `oof_errors.csv`. Epoch 80 and later are refused unless an
earlier longer-training checkpoint passed the configured official-score gate.

After producing OOF predictions and an official score for the first-gate model,
continue only to the next configured epoch:

```bash
PYTHONPATH=src:scripts python3 scripts/continue_royerlab_training.py \
  --config configs/gold_training.yaml \
  --splits outputs/gold_training/prefix_holdout_splits.json \
  --fold-index 0 \
  --candidate temporal-link-v1 \
  --target-epoch 65 \
  --resume-checkpoint outputs/gold_training/weights/temporal-link-v1/split_0/epoch_050.pt \
  --previous-official-score 0.942 \
  --previous-official-epoch 50
```

Each following stage likewise resumes from the preceding full-state `.pt`
file. Every native checkpoint includes model, optimizer, scheduler, scaler
placeholder, RNG and session state, epoch, best official score, and the
immutable run manifest. Its adjacent `_model.pth` is the plain inference state
expected by the official pipeline. `--bootstrap-checkpoint` exists only to
migrate a legacy plain upstream model; it cannot recover missing optimizer
history and is not the default campaign path.

Record official scores in a long-form CSV:

```text
epoch,checkpoint,fold,score
50,weights/epoch_050.pt,A,0.941
50,weights/epoch_050.pt,B,0.943
65,weights/epoch_065.pt,A,0.945
65,weights/epoch_065.pt,B,0.946
```

Then enforce the promotion gate:

```bash
PYTHONPATH=src python3 scripts/evaluate_training_gates.py \
  --config configs/gold_training.yaml \
  --scores outputs/gold_training/checkpoint_scores.csv \
  --baseline-epoch 50 \
  --output outputs/gold_training/promotion.json
```

To shortlist at most three strong, complementary models, also provide
`--oof-errors` as a long `checkpoint,sample,error` CSV. The shortlist uses OOF
error correlation; actual inclusion still requires the measured ensemble gain
and end-to-end runtime gate. Provide `--diagnostics` with fold-level rows for
each epoch to emit the required edge Jaccard, node recall, node-count ratio,
fragmentation, detection-lost-edge, and wrong-association SVG curves.

## Quota and runtime policy

For each approximately 30-hour allocation, reserve 2 hours for throughput
profiling, 10 for the one-fold learning curve, 10 for one high-impact variant,
and 8 for OOF inference, recovery, and runtime rehearsal. Data preparation and
metric aggregation run without a GPU accelerator.

Jobs must checkpoint at least every five epochs, remain below 10 hours, and
resume rather than restart. Benchmark P100 versus T4x2, AMP, batch size,
gradient checkpointing, and data loading using measured examples per quota
hour. Five-fold training is not an accepted default.

Run the final candidate through the same entry point and test-volume scale as
the submission notebook:

```bash
python3 scripts/benchmark_inference_runtime.py \
  --output outputs/gold_training/runtime.json \
  --runtime-limit-hours 12 \
  -- python3 path/to/end_to_end_inference.py
```

The report records wall time, exit status, timeout status, and sampled peak GPU
memory. Feed `runtime_hours` into the ensemble promotion gate.

## Offline Kaggle runner

Build and upload the runner bundle as a private Kaggle dataset:

```bash
PYTHONPATH=src python3 scripts/build_gold_training_bundle.py
```

Use `scripts/kaggle_kernels/gold_oof_runner/` as the Kaggle script source and
attach `gold_training_runner.zip`, the competition, Local CV Pack, Royerlab
offline wheels, and a dataset containing the upstream Royerlab repository.
The kernel discovers these inputs, writes a runtime config under
`/kaggle/working`, and invokes `run_gold_stage.py`. Set `BIOHUB_TARGET_EPOCH`,
`BIOHUB_CANDIDATE`, and `BIOHUB_RESUME=1` when advancing or recovering a stage.
Because Kaggle working storage does not carry into a new notebook version,
save each successful version's output and attach that output as an input dataset
to the next version. The runner restores the single attached
`outputs/gold_training` campaign before resuming. If more than one prior output
is attached, set `BIOHUB_RESUME_SOURCE` to the exact campaign directory.

## Acceptance outputs

Each candidate must retain:

- strict OOF submission CSVs for folds A and B;
- `summary.json`, `per_dataset.csv`, `oracle_summary.json`, and
  `oracle_per_dataset.csv`;
- checkpoint-score CSV and `promotion.json`;
- `paired_per_dataset.csv` and `paired_summary.json` against the frozen baseline;
- run manifest with commit, config, and split hashes;
- hidden-test-scale inference timing and peak GPU memory;
- an append-only row in `results/local_cv/experiments.csv` only after a real
  local evaluation.
