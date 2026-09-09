# AGENTS.md

Guidance for GitHub Copilot, Claude, Gemini, Cursor, and any other AI development assistant working in this repository.

This file is the single source of truth for assistant behavior. Do not create parallel instruction files with divergent rules; point them here with symlinks.

All repository documentation must live under `docs/`. Use root files only for established project conventions such as `README.md`, `AGENTS.md`, and tool-specific symlinks that point back to this guidance. Machine-readable outputs may live in purpose-specific directories such as `results/`, but any explanatory markdown for those outputs belongs in `docs/`.

Register experiment results consistently: local CV scores go in `results/local_cv/experiments.csv`, Kaggle live/private leaderboard submissions go in `results/kaggle_lb/submissions.csv`, and explanatory writeups for either belong under `docs/`. Keep these files append-only unless fixing a factual error.

## Project mission

Build a production-ready, research-friendly codebase for the Kaggle **Biohub - Cell Tracking During Development** competition. The task is to detect 3D cell centroids in OME-Zarr microscopy volumes, link them across time, reconstruct cell lineages including mitosis, and export strict Kaggle submission CSVs.

The repository should remain both:

- **Competitive**: able to absorb strong public baselines such as `amanatar/biohub-v6-ultra-best`.
- **Maintainable**: modular, typed, tested, dependency-light at the core, and safe to import inside offline Kaggle notebooks.

## Architecture and stack

### Core package

Source code lives under `src/biohub_tracking/` using a `src/` layout.

| Area | Path | Responsibility |
|---|---|---|
| Data access | `data/ome_zarr.py`, `data/patches.py` | Lazy OME-Zarr-compatible volume reads, 5D `(T, C, Z, Y, X)` normalization, deterministic 3D patch grids, frame-pair datasets. |
| Detection | `detection/local_maxima.py` | Dependency-light 3D local-maxima/blob detector for smoke tests and fallback inference. |
| Models | `models/unet3d.py` | Torch-optional 3D U-Net interface. Imports must work without torch installed. |
| Pipeline | `pipeline/baseline.py` | Model-free detect -> track composition for reproducible baseline inference. |
| Tracking | `tracking/graph.py`, `tracking/hungarian.py` | `Detection`, `TrackingGraph`, weak components, forks, and distance-gated Hungarian linking with division branching. |
| Metrics | `metrics/matching.py`, `metrics/edge_jaccard.py`, `metrics/division_jaccard.py`, `metrics/score.py` | Official-style 7 um matching, sparse-aware edge Jaccard, local-window division Jaccard, adjusted/micro-averaged scoring. |
| Submission | `submission/export.py` | Strict deterministic Kaggle CSV export. |
| Utilities | `utils/config.py`, `utils/logging.py`, `utils/seed.py` | Pydantic/YAML validation, logging, reproducibility helpers. |

### Entry points

- `scripts/baseline_infer.py`: dependency-light blob detector + Hungarian tracker + CSV export.
- `scripts/infer.py`: model-based inference entry point.
- `scripts/train.py`: training entry point for torch-backed models.
- `scripts/evaluate.py`: local metric evaluation from submission-style CSVs.
- `bundle.py`: builds an offline package bundle for Kaggle notebooks.

### Dependencies

Core dependencies are intentionally small: `numpy`, `scipy`, `pandas`, `pydantic`, and `pyyaml`.

Optional integrations must stay lazy:

- `zarr` / `ome-zarr`: only required when opening real OME-Zarr stores.
- `torch`: only required when building, training, loading, or running neural models.
- Future `tracksdata` / `geff` / royerlab baseline support must be optional adapters, not core import requirements.

Never add heavyweight libraries to core imports unless they are truly required by every user path.

## Competitive landscape and repository edge

The reviewed Kaggle notebooks are documented in `research/kaggle_baseline_review.md`.

### Strongest public baseline family

The most competitive listed notebooks are the royerlab-style **TemporalUNet3D + SimpleNodeTransformer + tracksdata/ILP + metric-aware postprocessing** variants, especially `amanatar/biohub-v6-ultra-best`.

They use:

1. Temporal 3D U-Net heatmaps for cell-center detection.
2. Node feature pooling at detected centroids.
3. Transformer-based learned edge probabilities.
4. Candidate edge filtering by probability, top-k parent heuristics, and physical distance gates.
5. tracksdata/GEFF graph solving with ILP-like weights.
6. Test-time augmentation, checkpoint ensembling, one-frame gap closing, synthetic midpoint refinement, short-component filtering, and node-count-aware caps.

### Compact baseline family

`kaiwalyaatulraut/biohub-solution` is simpler: pooled XY U-Net heatmaps, local peak detection, physical NMS, velocity-aware Hungarian linking, gap repair, line fitting, and short-track cleanup. It is easier to port but likely lower ceiling because edge linking is mostly geometric.

### Our required edge

This repository should combine the best of both worlds:

- Keep the **clean modular architecture** and typed APIs already in this repo.
- Port strong public-baseline ideas as **small, testable modules**, never as a monolithic copied notebook.
- Preserve a **fast dependency-light fallback path** for debugging, CI, and smoke tests.
- Treat physical units and memory behavior as first-class constraints.
- Keep metric-aware postprocessing explicit and configurable instead of hiding leaderboard constants in notebook cells.

### Required patterns

- Use `TrackingGraph`/`Detection` as internal graph exchange types.
- Keep coordinates in physical microns once detections enter tracking/metrics/submission logic; raw voxel coordinates are allowed only inside data/detection modules and must be converted at boundaries.
- Use deterministic sorting for outputs and tie-breaking.
- Use pydantic config models for public workflow parameters and YAML files for runnable configurations.
- Add synthetic tests for algorithms so behavior is validated without Kaggle data or model weights.

### Anti-patterns to avoid

- Do not paste an entire Kaggle notebook into `src/`.
- Do not make torch, zarr, tracksdata, geff, skimage, or notebook-only packages mandatory core imports.
- Do not silently swallow invalid graph/submission data; raise clear exceptions.
- Do not optimize only leaderboard hacks without documenting the metric assumption.
- Do not materialize full 4D/5D volumes unless the method name/documentation makes that explicit and tests cover small-only behavior.
- Do not emit nondeterministic CSVs, unsorted graph rows, or ambiguous node IDs.
- Do not introduce global mutable state for model/config/runtime behavior.

## Code style and guardrails

### Typing and public APIs

- Use `from __future__ import annotations` in Python modules.
- Public functions/classes need type hints and docstrings.
- Prefer dataclasses for lightweight immutable configs/results and pydantic models for YAML/user-facing configuration.
- Avoid `Any` except at dependency boundaries such as array protocols or optional third-party adapters.
- Avoid broad casts and `# type: ignore`; if unavoidable, localize and explain them.

### Error handling

- Validate inputs at module boundaries: dimensions, channel indices, voxel scales, positive radii, missing columns, dangling edges, duplicate IDs.
- Raise `ValueError`, `IndexError`, `FileNotFoundError`, or `ImportError` with actionable messages.
- Optional dependencies must fail with clear install guidance only when the optional feature is used.
- Do not catch broad `Exception` unless wrapping a known optional backend fallback, and do not hide failures that would corrupt outputs.

### Performance and memory

- Read OME-Zarr data lazily: frame-by-frame or patch-by-patch.
- Keep memory proportional to the requested frame/patch, not total dataset size.
- Use vectorized NumPy/SciPy operations for distances, matching, and NMS where practical.
- Use physical-distance gates before expensive graph/linking operations.
- Cap postprocessing insertions and component expansion to avoid adjusted-Jaccard penalties and memory blowups.
- Document any method that intentionally materializes a whole volume.

### Testing expectations

- Add or update tests for every behavior change.
- Tests must be synthetic and small by default; do not require Kaggle data, GPU, torch, zarr, tracksdata, or external weights unless marked/skipped appropriately.
- Preserve deterministic tests: fixed arrays, fixed graph IDs, fixed sorting.
- Run the smallest relevant test first, then full suite before declaring completion.

Current validation command:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Expected current baseline: `54 passed, 1 skipped` in this environment. The skip is torch-dependent when torch is unavailable.

### Linting and packaging

Configured in `pyproject.toml`:

- Python `>=3.10`
- pytest test discovery in `tests/`
- mypy with Python 3.10 target and missing imports ignored for optional deps
- ruff line length 100, target `py310`

Use when available:

```bash
python3 -m pytest -q
python3 -m ruff check .
python3 -m mypy src
```

If `pip install -e '.[dev]'` fails because of a corporate package index/auth issue, validate via `PYTHONPATH=src` and note the environment limitation instead of treating it as a code failure.

## Common workflows

### Add a new module

1. Place code in the appropriate `src/biohub_tracking/<area>/` package.
2. Keep imports dependency-light and localize optional dependency imports inside methods that require them.
3. Add dataclass or pydantic config objects for parameters that users will tune.
4. Export stable symbols from the package `__init__.py` only when they are intended public API.
5. Add focused tests in `tests/test_<area>.py`.
6. Run targeted tests, then the full suite.

### Extend detection

1. Keep detector inputs as small 3D NumPy volumes or frame patches.
2. Return deterministic `(z, y, x)` centroid coordinates.
3. State whether coordinates are voxel-space or micron-space.
4. Use physical voxel scale for anisotropic suppression when possible.
5. Add tests for thresholding, tie-breaking, suppression, anisotropic scaling, and empty frames.

### Extend tracking

1. Accept/return `TrackingGraph` or `Detection` objects.
2. Keep matching/linking in physical microns.
3. Gate impossible candidates before Hungarian/ILP solving.
4. Enforce graph invariants: no dangling edges, valid node IDs, directed temporal links unless explicitly documented.
5. Add tests for one-to-one links, unmatched nodes, divisions, merge rejection, and deterministic output order.

### Extend metrics

1. Read `metrics.md` behavior first; do not simplify without documenting it in the module docstring.
2. Keep node matching time-aware and distance-gated at 7 um by default.
3. Preserve sparse-ground-truth semantics: unannotated predictions are not automatically false positives.
4. Keep adjusted edge Jaccard, division Jaccard, and final score aggregation separate and testable.
5. Add synthetic graph tests for TP/FP/FN edge cases, divisions, forks, merges, and zero-event behavior.

### Extend submission/export

1. Use the strict columns exactly:

   ```text
   dataset,row_type,node_id,t,z,y,x,source_id,target_id
   ```

2. Use `-1` sentinels for unused fields.
3. Sort datasets, node rows, and edge rows deterministically.
4. Validate duplicate node IDs and dangling edges before writing.
5. Add round-trip or schema tests whenever changing export logic.

### Port a public Kaggle baseline

1. Summarize the notebook in `research/` before changing production code.
2. Extract one reusable concept at a time: heatmap peaks, TTA, edge candidate scoring, ILP adapter, gap closing, component filtering.
3. Put optional dependency integrations behind adapters, e.g. future `biohub_tracking.baselines.royerlab`.
4. Write synthetic tests that do not need the public notebook's weights.
5. Keep notebook-derived constants in config files, not hidden globals.
6. Document leaderboard/metric assumptions and risks.

### Run baseline inference

```bash
PYTHONPATH=src python3 scripts/baseline_infer.py \
  --input data/test/testA.zarr \
  --dataset testA \
  --output submission.csv \
  --threshold 100 \
  --min-distance 5 \
  --voxel-size-um 1.625 0.40625 0.40625
```

### Evaluate predictions

```bash
PYTHONPATH=src python3 scripts/evaluate.py \
  --pred submission.csv \
  --gt gt_submission.csv
```

### Build offline Kaggle bundle

```bash
PYTHONPATH=src python3 bundle.py --output dist/biohub_tracking_bundle.zip
```

## Symlink setup for AI assistants

All assistant-specific instruction files should point to this file:

```bash
mkdir -p .github
ln -sf ../AGENTS.md .github/copilot-instructions.md
ln -sf AGENTS.md CLAUDE.md
ln -sf AGENTS.md GEMINI.md
```

Or run:

```bash
bash scripts/setup_agent_instructions.sh
```

After setup, verify:

```bash
ls -l AGENTS.md .github/copilot-instructions.md CLAUDE.md GEMINI.md
```

## Commit and collaboration rules

- Keep changes small and coherent.
- Do not commit downloaded credentials, Kaggle tokens, datasets, model checkpoints, generated submissions, or local virtual environments.
- Research artifacts under `research/` are allowed when they are intentionally downloaded public notebooks or written analysis.
- Include the Copilot co-author trailer when creating commits unless explicitly told otherwise:

```text
Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>
```

## Mandatory local evaluation gate

Before recommending or submitting any Kaggle candidate, run local validation or explicitly document why it cannot run. Use:

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

Promotion decisions must inspect `summary.json` and `per_dataset.csv`, especially adjusted edge Jaccard, division Jaccard, node recall, node-count ratio, fragmented edges, detection-lost edges, and wrong-association edges. Public LB alone is not an acceptance criterion.
