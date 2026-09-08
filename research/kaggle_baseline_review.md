# Kaggle Biohub baseline notebook review

Reviewed notebooks downloaded with `kaggle kernels pull` on 2026-09-08:

| Notebook | Local file | Family | Main approach |
|---|---|---|---|
| `amanatar/improved-metric-hack-last-call` | `research/kaggle_notebooks/amanatar_improved-metric-hack-last-call/` | Metric-hack/royerlab | TemporalUNet3D + node transformer + tracksdata ILP, followed by gap-closing and short-component filtering. |
| `kaiwalyaatulraut/biohub-solution` | `research/kaggle_notebooks/kaiwalyaatulraut_biohub-solution/` | Compact custom baseline | Small 3D U-Net ensemble on pooled XY volumes, local-maxima detection, Hungarian linking with velocity, optional gap repair/line fitting/short-track filtering. |
| `kaiwalyaatulraut/biohub-competition-solution` | `research/kaggle_notebooks/kaiwalyaatulraut_biohub-competition-solution/` | Metric-hack/royerlab | Same royerlab TemporalUNet3D+transformer+ILP family, shorter/earlier variant with post-processing. |
| `amanatar/biohub-v6-ultra-best` | `research/kaggle_notebooks/amanatar_biohub-v6-ultra-best/` | Best metric-hack/royerlab variant | Multi-checkpoint ensemble over TemporalUNet3D+transformer predictions, TTA, tracksdata ILP, aggressive metric-aware cleanup. |
| `ravi123a321at/metric-hack-last-call` | `research/kaggle_notebooks/ravi123a321at_metric-hack-last-call/` | Duplicate metric-hack/royerlab | Exact code hash match with `amanatar/biohub-metric-hack-last-call`. |
| `amanatar/biohub-metric-hack-last-call` | `research/kaggle_notebooks/amanatar_biohub-metric-hack-last-call/` | Metric-hack/royerlab | TemporalUNet3D + transformer + ILP plus gap-closing/cleanup; copied by Ravi notebook. |

## Common competition insights

All serious notebooks exploit the same scoring properties:

1. Detection quality matters, but over-detection is punished through adjusted edge Jaccard, so high-recall union-style candidate generation is dangerous unless filtered.
2. Physical distance in microns is central. The common scale is `(z, y, x) = (1.625, 0.40625, 0.40625)` and linking gates are typically 10-12 µm before final metric matching at 7 µm.
3. Edge structure matters more than visual segmentation masks. The stronger notebooks predict graph links explicitly instead of relying only on nearest-neighbor tracking.
4. Post-processing is a large part of leaderboard performance: one-frame gap closing, synthetic midpoint insertion/refinement, removal of short isolated components, and limiting node additions to avoid the `T_pred` penalty.
5. Division events are relatively rare and low-weighted in the final score, so notebooks optimize edge Jaccard first and treat division support as a secondary graph-topology improvement.

## Notebook families

### 1. Amanatar / Ravi / Kaiwalya competition-solution: TemporalUNet3D + transformer + tracksdata ILP

Representative: `amanatar/biohub-v6-ultra-best`.

Pipeline:

1. Load royerlab support package objects: `TemporalUNet3D`, `SimpleNodeTransformer`, `tracksdata`, and `geff`.
2. Run a temporal 3D U-Net on two adjacent frames at a time. The model predicts:
   - point heatmaps for cell-center detection;
   - per-node features for edge scoring.
3. Convert heatmap probabilities to centroids with 3D max-pooling NMS (`prob_to_zyx`) using a physical pooling kernel.
4. Embed `(t, z, y, x)` positions with sinusoidal position encodings and sample U-Net features at detected centers.
5. Score all candidate links between frame `t` and `t+1` with a node transformer.
6. Keep candidate edges using probability/top-k parent heuristics plus a physical distance gate.
7. Build a `tracksdata` `InMemoryGraph` and solve the graph with an ILP-like tracker using weights for edge, appearance, disappearance, and division.
8. Convert predicted GEFF graphs to submission CSV.
9. Apply metric-aware cleanup:
   - one-frame gap closing;
   - reuse isolated middle-frame detections if close enough;
   - otherwise insert synthetic midpoint nodes refined toward local image brightness;
   - remove short connected components while optionally preserving boundary/division components;
   - cap added nodes/components to avoid the adjusted-Jaccard node-count penalty.

Important parameter patterns:

- `POINT_THRESHOLD`: around `0.95-0.97`.
- `EDGE_MAX_DISTANCE_UM`: `10-12` µm.
- `EDGE_STRONG_THRESHOLD`: about `0.50`.
- `EDGE_MIN_THRESHOLD`: `0.20-0.25`.
- `EDGE_TOPK_PARENTS`: `2-3`.
- ILP weights: edge negative, disappearance positive, division positive, appearance near zero.
- `biohub-v6-ultra-best` adds multi-checkpoint ensembling and heavier TTA compared with the earlier metric-hack variants.

Strengths:

- Closest to the official royerlab baseline and competition graph format.
- Explicit learned link scoring, not just geometric nearest-neighbor tracking.
- Strong post-processing aligned with sparse edge/adjusted-node metric behavior.
- `biohub-v6-ultra-best` is the most advanced of the listed notebooks.

Weaknesses:

- Relies on external Kaggle input datasets/weights/support package.
- Notebook code is large and monolithic, with several hard-coded paths and leaderboard-tuned constants.
- Some post-processing is metric-hack oriented and should be modularized before serious experimentation.
- Requires GPU for practical inference.

### 2. Kaiwalya `biohub-solution`: compact U-Net + local peaks + Hungarian + repair

Pipeline:

1. Load two small `unet3d_*.pt` models (`bright` and `traintophat`) if attached as Kaggle inputs.
2. Read Zarr frames lazily; has a fallback direct Blosc2 chunk reader.
3. Pool XY by factor 4 to reduce memory/compute.
4. Normalize intensity by percentile range; optionally use a top-hat preprocessor.
5. Average model heatmaps, detect peaks with `skimage.feature.peak_local_max`, refine peaks by local center-of-mass on raw volume, and apply physical-radius NMS with `cKDTree`.
6. Link consecutive frames using Hungarian assignment with a tight pass and a looser unmatched pass; includes a velocity predictor.
7. Repair tracks with optional gap closing, short-track filtering, line-fit smoothing, and simple CSV emission.

Strengths:

- Much simpler to understand and port into our repository.
- Good engineering signal for fast baseline inference: pooled XY frames, percentile normalization, physical NMS, velocity-aware Hungarian linking.
- Does not require `tracksdata`/GEFF ILP machinery for inference.
- Easy to adapt into our existing local-maxima and Hungarian baseline modules.

Weaknesses:

- Link scoring is geometric rather than learned.
- Less faithful to the official graph baseline and likely lower ceiling than the TemporalUNet3D+transformer+ILP family.
- Still depends on external model weights for real performance.

## Duplicate / near-duplicate findings

- `ravi123a321at/metric-hack-last-call` and `amanatar/biohub-metric-hack-last-call` are exact code duplicates by SHA-256.
- `amanatar/improved-metric-hack-last-call` and `amanatar/biohub-metric-hack-last-call` are near-duplicates with threshold/cap changes (`POINT_THRESHOLD`, `MAX_ADDED_FRAC`, `MAX_COMPONENTS`).
- `kaiwalyaatulraut/biohub-competition-solution` is a shorter member of the same royerlab/metric-hack lineage.
- `amanatar/biohub-v6-ultra-best` is the most complete evolution of that lineage: more TTA, checkpoint discovery, and ensembling.

## Recommendation

Use **`amanatar/biohub-v6-ultra-best` as the primary competitive baseline**, but do not copy it as a notebook blob. Instead, port it into our modular codebase in stages:

1. Add adapters for the official royerlab/tracksdata/GEFF graph baseline so we can run compatible predictions and local evaluation.
2. Add a model-wrapper interface for TemporalUNet3D + SimpleNodeTransformer checkpoints.
3. Port candidate edge generation: heatmap NMS, feature sampling, positional encoding, top-k parent filtering, distance gating.
4. Port graph solving/post-processing as separate modules:
   - ILP/tracksdata solver adapter;
   - one-frame gap closing;
   - synthetic node refinement;
   - short-component filtering;
   - metric-aware caps on added nodes.
5. Keep our current dependency-light local-maxima baseline as the fallback/debug baseline and as a fast smoke-test path.

Why this choice:

- Among the listed notebooks, `biohub-v6-ultra-best` contains the strongest technical ingredients: learned detection, learned edge scoring, TTA, ensembling, graph optimization, and metric-aware post-processing.
- The compact Kaiwalya notebook is useful as implementation inspiration, especially for preprocessing/NMS/velocity Hungarian/gap repair, but it is less competitive as the main baseline.
- The metric-hack notebooks are already tuned for the competition's sparse graph metric; ignoring them would leave leaderboard performance on the table.

## Immediate next engineering task

Create a `biohub_tracking.baselines.royerlab` package with a clean adapter around the `biohub-v6-ultra-best` approach:

- `models/temporal_unet_transformer.py`
- `detection/heatmap_peaks.py`
- `tracking/candidate_edges.py`
- `tracking/tracksdata_solver.py`
- `postprocessing/gap_closing.py`
- `postprocessing/component_filter.py`
- tests built from synthetic graphs/heatmaps without requiring Kaggle weights

This lets us start from the strongest public baseline while preserving the production-ready architecture we already created.
