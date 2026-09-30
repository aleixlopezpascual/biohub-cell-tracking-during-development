# Biohub Cell Tracking During Development

Production-oriented, modular pipeline for the Kaggle **Biohub - Cell Tracking
During Development** competition: 3D+time OME-Zarr data loading, a baseline
3D U-Net, Hungarian frame-to-frame tracking with division branching, the
official competition metrics, and a deterministic submission exporter.

Metric behavior follows the official reference:
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

## Competition endgame status & official outcome

- **Official Private Leaderboard:** **Rank 653 of 4,017 teams** (Top 16.2%, Selected Private Score `0.917`).
- **Peak Leaderboard Marks:** **`0.956` Public LB** (Rank 376) and **`0.919` Private LB** (achieved by Candidate `56639822`, Active Mitosis Recovery).
- **Final Retrospective & Analysis:** See [`docs/final_competition_retrospective_and_results.md`](docs/final_competition_retrospective_and_results.md) and [`docs/overfitting_risk_and_final_selection.md`](docs/overfitting_risk_and_final_selection.md).
- **Top Solutions & Meta Learnings Post-Mortem:** See [`docs/top_solutions_and_competitive_postmortem.md`](docs/top_solutions_and_competitive_postmortem.md) for an in-depth review of the winning approaches (3rd, 12th, 14th, 361st) and comparative analysis against this codebase.
- **Complete Submission Ledger:** See [`results/kaggle_lb/submissions.csv`](results/kaggle_lb/submissions.csv).

## Key Algorithmic Contributions

1. **Continuous Sub-Voxel Coordinate Regression (`v1284_head.pt`):**
   Regresses continuous $(z, y, x)$ offsets from 3D U-Net feature gradients around peak detections and applies continuous trilinear feature interpolation in `UNetNodeTransformer`, eliminating grid-quantization errors.
2. **Collective Neighborhood Tissue Flow Prior:**
   Estimates developmental drift velocity from the collective motion of the 12–16 nearest neighbors within $40\text{--}48\,\mu\text{m}$, stabilizing track association through cellular crossings.
3. **DivNet 3D-CNN Active Mitosis Recovery:**
   Loosens rigid geometric sister symmetry gates ($\tau \le 0.80$) and uses a 3D-CNN patch classifier (`giorgosi/biohub-divnet-v2`) to visually confirm cytokinesis, delivering our top generalization mark of **`0.919` Private LB**.
4. **In-Line Bipartite Graph Consensus Ensembling:**
   Fuses distinct cell tracking graphs in memory frame-by-frame (`src/biohub_tracking/tracking/ensemble.py`), resolving contested links with 3D spatial proximity while preserving graph degree invariants.
5. **Strict Denominator Penalty Protection ($N_{\text{pred}}$ Shield):**
   Identified and eliminated element-wise `torch.maximum` detection ensembling (which dragged scores to $0.901$ by inflating false positives), keeping detection thresholds at $0.965$ and pruning isolated singletons.

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

Use [`docs/local_evaluation.md`](docs/local_evaluation.md) for the generic evaluation workflow. For the staged 0.948 candidate specifically, no valid paired fold-disjoint evaluation/receipt currently exists; see the current endgame status document before attempting promotion or a Kaggle run.

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

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

