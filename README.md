# Biohub Cell Tracking During Development

Production-oriented, modular pipeline for the Kaggle **Biohub - Cell Tracking
During Development** competition: 3D+time OME-Zarr data loading, a baseline
3D U-Net, Hungarian frame-to-frame tracking with division branching, the
official competition metrics, and a deterministic submission exporter.

Metric behavior follows the official reference:
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

## Install

```bash
pip install -e '.[dev]'
```

Core dependencies are `numpy`, `scipy`, `pandas`, `pydantic`, and `pyyaml` only.
Two integrations are optional and imported lazily so the core package never
requires them:

```bash
pip install -e '.[zarr]'    # real OME-Zarr store reading (zarr, ome-zarr)
pip install -e '.[torch]'   # building/training/running the 3D U-Net
```

## Layout

```
configs/                  YAML configs validated by biohub_tracking.utils.config.PipelineConfig
src/biohub_tracking/
  data/                    Lazy OME-Zarr reader (OMEZarrVolume) + 3D patch/frame-pair iteration
  detection/               Dependency-light 3D local-maxima/blob detector
  baselines/royerlab/      Optional high-ceiling Royerlab-style reusable stages
  models/                  Baseline 3D U-Net (torch-optional interface)
  pipeline/                Model-free detect-and-track baseline composition
  tracking/                TrackingGraph + Hungarian frame-to-frame linker with division branching
  metrics/                 Official edge Jaccard, division Jaccard, and micro-averaged scoring
  submission/              Strict, deterministic Kaggle submission CSV export
  utils/                   Config loading, logging, seeding
scripts/
  train.py                 Training entry point (requires the torch extra)
  evaluate.py               Compute the official score from predicted/GT submission CSVs
  infer.py                  Detect + track a volume into a submission CSV (requires the torch extra)
  baseline_infer.py         Detect blobs + track a volume without a model (requires the zarr extra)
  royerlab_infer.py         Preflight the optional TemporalUNet3D + transformer + ILP workflow
  prepare_gold_training.py  Build real prefix folds and a hashed Kaggle training manifest
  continue_royerlab_training.py Train one Royerlab model to the next gated epoch
  run_oof_checkpoint.py     Predict and officially score one exact OOF checkpoint
  run_gold_stage.py         Run one training gate and its OOF evaluation end to end
  build_gold_training_bundle.py Build the offline Kaggle runner bundle
  evaluate_training_gates.py Enforce official-score learning-curve and runtime gates
  compare_local_evaluations.py Write paired per-volume OOF experiment deltas
  benchmark_inference_runtime.py Rehearse hidden-scale runtime and peak GPU memory
bundle.py                  Zips the pure-Python package for offline Kaggle notebook submission
tests/                     Focused synthetic tests (no real data required)
```

## Metrics

`biohub_tracking.metrics` implements, faithfully to metrics.md:

- **Node matching**: optimal bipartite assignment per frame, gated at 7 um.
- **Edge Jaccard**: sparse-aware TP/FP/FN edge accounting - predictions the
  (deliberately sparse) ground truth doesn't cover are ignored rather than
  penalized, while edges that contradict already-matched GT structure are FP.
- **Adjusted edge Jaccard**: penalizes excess predicted nodes relative to a
  coarse true-node-count estimate.
- **Division Jaccard**: a local window (grandparent -> parent -> children ->
  grandchildren) around each GT split, tolerant of a predicted fork one
  timepoint early or late, using bipartite daughter-branch matching,
  connected-component evidence, and rejection of merged branches.
- **Micro-averaging**: per-sample TP/FP/FN are summed before computing each
  Jaccard, so `score = adjusted_edge_jaccard + 0.1 * division_jaccard`
  weights samples by size rather than averaging per-sample scores directly.

The division-Jaccard implementation documents one deliberate simplification:
node matching is shared globally (once per frame) with the edge metric
rather than re-solved independently inside every local window; see the
module docstring in `src/biohub_tracking/metrics/division_jaccard.py` for
details and rationale.

## Submission schema

`biohub_tracking.submission.export_submission` writes the exact competition
schema, sentinel-filling unused fields with `-1`, with fully deterministic
row ordering (sorted datasets, then all node rows, then all edge rows):

```
dataset, row_type, node_id, t, z, y, x, source_id, target_id
```

## Quickstart

```bash
# Train (requires the torch extra and a real OME-Zarr dataset)
python scripts/train.py --config configs/train.yaml

# Run inference -> submission.csv
python scripts/infer.py --config configs/inference.yaml \
  --checkpoint outputs/train/unet3d.pt --dataset testA --output submission.csv

# Run the dependency-light blob-detection baseline -> submission.csv
python scripts/baseline_infer.py --input data/test/testA.zarr --dataset testA \
  --output submission.csv --threshold 100 --min-distance 5

# Score a prediction against ground truth
python scripts/evaluate.py --pred submission.csv --gt gt_submission.csv

# Build an offline bundle for a no-internet Kaggle notebook
python bundle.py --output dist/biohub_tracking_bundle.zip
```

## Testing

```bash
pip install -e '.[dev]'
pytest -q
```

Tests are synthetic (hand-constructed graphs/arrays) and require no real
dataset; `torch`-dependent tests are skipped automatically when the `torch`
extra isn't installed.

## Local validation before submission

## Recommended competitive baseline

Use `configs/royerlab_inference.yaml` as the starting point for a competitive
Kaggle candidate. Its portable stages are independently usable with only NumPy
and SciPy: physical 3D heatmap peaks, reversible XY D4 TTA, `(t,z,y,x)`
sinusoidal encodings, center feature sampling, probability/distance-gated
edge candidates, and bounded gap/component/node-count postprocessing.

The learned `TemporalUNet3D`/`SimpleNodeTransformer` and `tracksdata`/GEFF ILP
solver are deliberately **not** bundled. Attach compatible offline Kaggle
weights and wheels, then preflight them:

```bash
PYTHONPATH=src python3 scripts/royerlab_infer.py \
  --config configs/royerlab_inference.yaml \
  --temporal-checkpoint /kaggle/input/support/weights/temporal.pt \
  --transformer-checkpoint /kaggle/input/support/weights/transformer.pt \
  --require-ilp
```

The entry point stops with an actionable dependency/integration error until a
concrete support-pack model adapter is provided; it never silently substitutes
an incompatible model or solver. Run the local-evaluation gate below before
promoting any resulting submission.

Before submitting any candidate to Kaggle, run local evaluation and inspect the component metrics. The recommended protocol is documented in `docs/local_evaluation.md` and follows the community Local CV Pack/prefix-holdout strategy.

```bash
PYTHONPATH=src python3 scripts/local_eval.py \
  --submission outputs/candidate/submission.csv \
  --gt-submission data/validation_gt_submission.csv \
  --cv-pack-dir /path/to/biohub-local-cv-pack \
  --fold all \
  --candidate candidate-name \
  --config configs/inference.yaml \
  --output-dir outputs/local_eval/candidate-name \
  --log-experiment
```

The command writes `summary.json`, `per_dataset.csv`, and optionally an append-only `local_eval_log.csv` containing commit/config/scorer/split provenance.

Add `--oracle-analysis` to write detection-versus-linking headroom reports.
The gated Kaggle training workflow is documented in
`docs/cloud_kfold_training_design.md` and configured by
`configs/gold_training.yaml`.
