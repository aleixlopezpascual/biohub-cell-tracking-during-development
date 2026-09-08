# Biohub Cell Tracking During Development

Production-oriented baseline for 3D+time cell detection, lineage tracking, evaluation, and Kaggle submission export.

## Install

```bash
pip install -e '.[dev]'
```

Core APIs use typed NumPy/SciPy/Pandas components. Install `.[zarr]` for OME-Zarr loading and `.[torch]` for the optional 3D model.

## Layout

`src/biohub_tracking/data` contains lazy Zarr and patch loaders; `tracking` links centroids with Hungarian assignment and divisions; `metrics` implements sparse-aware Biohub Jaccards; `utils` writes the strict CSV schema.

The official metric reference is the [royerlab competition baseline](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md).
