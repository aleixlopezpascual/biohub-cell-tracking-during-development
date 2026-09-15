# Advanced Inference Overlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a fully native, clean prediction orchestrator module with Feature-Level TTA, temporal Bidirectional Tracking, and Multi-Model ensembling.

**Architecture:** Model-free/Model-based composition. Loads $N$ TemporalUNet3D and SimpleNodeTransformer checkpoints, computes averaged logits across 8 reversible spatial transforms (D4) and $N$ checkpoints in canonical space, and applies bidirectional temporal harmonic mean probability fusion before Graph Solving.

**Tech Stack:** PyTorch, NumPy, SciPy, PyYAML

**Spec:** `docs/advanced_inference_overlays_spec.md`

## Global Constraints
*   **Dependency Limits:** No third-party heavyweight imports like tracksdata or GEFF in core files; optional adapters are used at execution runtime only.
*   **Physical Scale:** All coordinates entering tracking/metrics/submission must be in physical microns.
*   **Reproducibility:** Seed all PyTorch/NumPy RNGs.
*   **Strict Typing:** Type-hints and docstrings required on all public methods.

---

## 🛠️ File Structure

*   Create: **`src/biohub_tracking/baselines/royerlab/prediction.py`**
    *   *Responsibility:* Orchestrates model loading, TTA, ensembling, and bidirectional prediction.
*   Create: **`tests/test_royerlab_prediction.py`**
    *   *Responsibility:* Synthetic TDD testing of the prediction, TTA, and Bidirectional fusion interfaces.

---

## 📋 Detailed Tasks

### Task 1: Scaffolding and Feature-Level TTA

**Files:**
- Create: `src/biohub_tracking/baselines/royerlab/prediction.py`
- Test: `tests/test_royerlab_prediction.py`

**Interfaces:**
- Consumes: `SpatialTransform` and `apply_xy_d4_tta` from `biohub_tracking.baselines.royerlab.tta`
- Produces: `NativePredictor` and `NativePredictor.predict_heatmaps_and_features`

- [ ] **Step 1: Write the failing TTA and scaffolding test**

```python
import numpy as np
import pytest
from biohub_tracking.baselines.royerlab.prediction import NativePredictor

def test_tta_canonical_averaging_is_invariant() -> None:
    # A fake mock model that returns the input as features
    class MockModel:
        def predict_heatmaps_and_features(self, frames):
            # returns (heatmap, features)
            return frames, frames

    predictor = NativePredictor(detector_weights=[], edge_weights=[])
    predictor.detectors = [MockModel()]
    
    # Create a simple asymmetric 3D volume (B=1, C=1, Z=4, Y=8, X=8)
    image = np.zeros((1, 1, 4, 8, 8), dtype=float)
    image[0, 0, 2, 4, 5] = 10.0  # single bright peak
    
    mean_heatmap, mean_features = predictor.predict_heatmaps_and_features(image, use_tta=True)
    
    # Verify peak coordinates are correctly localized at (2, 4, 5) after TTA inversion
    peak_idx = np.unravel_index(np.argmax(mean_heatmap), mean_heatmap.shape)
    assert peak_idx == (0, 0, 2, 4, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_royerlab_prediction.py -k test_tta_canonical_averaging_is_invariant`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/biohub_tracking/baselines/royerlab/prediction.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from biohub_tracking.baselines.royerlab.tta import XY_D4_TRANSFORMS, apply_spatial_transform, invert_spatial_transform

class NativePredictor:
    def __init__(self, detector_weights: list[Path], edge_weights: list[Path]) -> None:
        self.detector_weights = detector_weights
        self.edge_weights = edge_weights
        self.detectors = []
        self.edge_scorers = []

    def predict_heatmaps_and_features(self, frames: np.ndarray, use_tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if not self.detectors:
            raise ValueError("No detectors loaded")
        model = self.detectors[0]
        if not use_tta:
            return model.predict_heatmaps_and_features(frames)
            
        heatmaps = []
        features = []
        for transform in XY_D4_TRANSFORMS:
            # Apply transform to spatial axes (last two axes Y, X)
            aug_frames = apply_spatial_transform(frames, transform)
            h, f = model.predict_heatmaps_and_features(aug_frames)
            # Invert transform
            heatmaps.append(invert_spatial_transform(h, transform))
            features.append(invert_spatial_transform(f, transform))
            
        return np.mean(heatmaps, axis=0), np.mean(features, axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_royerlab_prediction.py -k test_tta_canonical_averaging_is_invariant`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/biohub_tracking/baselines/royerlab/prediction.py tests/test_royerlab_prediction.py
git commit -m "feat: scaffold NativePredictor and implement Feature-Level TTA"
```

---

### Task 2: Bidirectional Track Fusion

**Files:**
- Modify: `src/biohub_tracking/baselines/royerlab/prediction.py`
- Modify: `tests/test_royerlab_prediction.py`

**Interfaces:**
- Consumes: `fuse_bidirectional_probabilities` from `biohub_tracking.baselines.royerlab.linking`
- Produces: `NativePredictor.predict_bidirectional_edges`

- [ ] **Step 1: Write the failing Bidirectional Veto test**

```python
# Add to tests/test_royerlab_prediction.py
def test_bidirectional_harmonic_veto() -> None:
    class MockEdgeModel:
        def __init__(self, forward_prob, reverse_prob):
            self.forward_prob = forward_prob
            self.reverse_prob = reverse_prob
            
        def predict_edge_logits(self, src_feats, tgt_feats):
            # Dummy logic returning pre-defined prob as logits for testing
            # (Assuming model returns probabilities as values)
            if len(src_feats) == 2: # Forward
                return np.array([[self.forward_prob]])
            else: # Reverse
                return np.array([[self.reverse_prob]])

    # Case A: Forward high, backward low (should be vetoed)
    predictor = NativePredictor(detector_weights=[], edge_weights=[])
    predictor.edge_scorers = [MockEdgeModel(0.99, 0.01)]
    
    # Dummy mock feature vectors
    src_feats, tgt_feats = np.zeros((2, 10)), np.zeros((1, 10))
    probs = predictor.predict_bidirectional_edges(src_feats, tgt_feats)
    
    # Harmonic mean of 0.99 and 0.01 is highly penalized (~0.0198)
    assert probs[0, 0] < 0.05
    
    # Case B: Both high (should pass)
    predictor.edge_scorers = [MockEdgeModel(0.95, 0.95)]
    probs_pass = predictor.predict_bidirectional_edges(src_feats, tgt_feats)
    assert probs_pass[0, 0] > 0.90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_royerlab_prediction.py -k test_bidirectional_harmonic_veto`
Expected: FAIL with `AttributeError: 'NativePredictor' object has no attribute 'predict_bidirectional_edges'`.

- [ ] **Step 3: Write minimal implementation**

```python
# Add import to src/biohub_tracking/baselines/royerlab/prediction.py
from biohub_tracking.baselines.royerlab.linking import fuse_bidirectional_probabilities

# Inside NativePredictor class:
    def predict_bidirectional_edges(self, src_features: np.ndarray, tgt_features: np.ndarray) -> np.ndarray:
        if not self.edge_scorers:
            raise ValueError("No edge scorers loaded")
        model = self.edge_scorers[0]
        
        # Predict forward probabilities (u -> v)
        p_forward = model.predict_edge_logits(src_features, tgt_features)
        
        # Predict reverse probabilities (v -> u)
        p_reverse = model.predict_edge_logits(tgt_features, src_features)
        p_reverse_aligned = p_reverse.T
        
        # Fuse using existing harmonic block
        return fuse_bidirectional_probabilities(p_forward, p_reverse_aligned, mode="harmonic")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_royerlab_prediction.py -k test_bidirectional_harmonic_veto`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/biohub_tracking/baselines/royerlab/prediction.py tests/test_royerlab_prediction.py
git commit -m "feat: implement Bidirectional Edge Probability Fusion with harmonic veto"
```

---

### Task 3: Multi-Model Ensembling

**Files:**
- Modify: `src/biohub_tracking/baselines/royerlab/prediction.py`
- Modify: `tests/test_royerlab_prediction.py`

**Interfaces:**
- Consumes: `TemporalUNet3DAdapter` and `SimpleNodeTransformerAdapter`
- Produces: `NativePredictor._load_models` and ensembled average paths

- [ ] **Step 1: Write the failing Ensembling average test**

```python
# Add to tests/test_royerlab_prediction.py
def test_ensemble_logits_averaging() -> None:
    class MockModelA:
        def predict_heatmaps_and_features(self, frames):
            return frames * 2.0, frames
            
    class MockModelB:
        def predict_heatmaps_and_features(self, frames):
            return frames * 4.0, frames

    predictor = NativePredictor(detector_weights=[], edge_weights=[])
    predictor.detectors = [MockModelA(), MockModelB()]
    
    image = np.ones((1, 1, 4, 8, 8), dtype=float)
    # Average across models A (weight 2.0) and B (weight 4.0) should be 3.0
    mean_heatmap, _ = predictor.predict_heatmaps_and_features(image, use_tta=False)
    assert np.allclose(mean_heatmap, 3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_royerlab_prediction.py -k test_ensemble_logits_averaging`
Expected: FAIL with `AssertionError`.

- [ ] **Step 3: Write minimal implementation**

```python
# Modify predict_heatmaps_and_features inside src/biohub_tracking/baselines/royerlab/prediction.py:
    def predict_heatmaps_and_features(self, frames: np.ndarray, use_tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if not self.detectors:
            raise ValueError("No detectors loaded")
            
        all_heatmaps = []
        all_features = []
        
        for model in self.detectors:
            if not use_tta:
                h, f = model.predict_heatmaps_and_features(frames)
                all_heatmaps.append(h)
                all_features.append(f)
                continue
                
            for transform in XY_D4_TRANSFORMS:
                aug_frames = apply_spatial_transform(frames, transform)
                h, f = model.predict_heatmaps_and_features(aug_frames)
                all_heatmaps.append(invert_spatial_transform(h, transform))
                all_features.append(invert_spatial_transform(f, transform))
                
        return np.mean(all_heatmaps, axis=0), np.mean(all_features, axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_royerlab_prediction.py -k test_ensemble_logits_averaging`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/biohub_tracking/baselines/royerlab/prediction.py tests/test_royerlab_prediction.py
git commit -m "feat: complete multi-model ensembling averages across checkpoints"
```
