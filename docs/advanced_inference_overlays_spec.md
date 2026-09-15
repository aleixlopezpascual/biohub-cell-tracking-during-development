# Advanced Inference Overlays Specification

This specification outlines the architecture, data flow, and interfaces for our native prediction subsystem (**Path 1**), which incorporates **Feature-Level Test-Time Augmentation (Edge TTA)**, **Bidirectional Tracking**, and **Multi-Model Ensembling** directly into our modular codebase.

---

## 🛠️ 1. Module Architecture & Interface Layout

All prediction and ensembling logic will live inside a new file: **`src/biohub_tracking/baselines/royerlab/prediction.py`**. 

It will define the orchestrator class `NativePredictor` which communicates with the optional learned model adapters.

```python
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np

class NativePredictor:
    """Orchestrates multi-model ensembled inference with Feature-Level TTA and Bidirectional Tracking."""

    def __init__(
        self,
        detector_weights: list[Path],
        edge_weights: list[Path],
        device: str = "cpu",
    ) -> None:
        """Load multiple model checkpoints on the specified device."""
        self.device = device
        self.detector_weights = detector_weights
        self.edge_weights = edge_weights
        self.detectors = []
        self.edge_scorers = []
        self._load_models()

    def _load_models(self) -> None:
        """Load all specified model checkpoints."""
        # Use TemporalUNet3DAdapter and SimpleNodeTransformerAdapter
        pass

    def predict_video(
        self,
        zarr_path: Path,
        *,
        detection_threshold: float = 0.97,
        use_tta: bool = True,
        use_bidirectional: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run full ensembled prediction on a 3D+time volume, returning nodes and fused edges."""
        pass
```

---

## 📐 2. Section A: Feature-Level TTA & Centroid Pooling

Instead of image-level TTA, we operate directly on the 3D U-Net's feature-extractor representations.

1.  **Transform views:** For each input frame-pair `(B, C, Z, Y, X)` and each of the 8 XY Dihedral (D4) transforms $\mathcal{T}_i$ (defined in `biohub_tracking.baselines.royerlab.tta`):
    *   Rotate and flip the input volume: `augmented_frames = apply_spatial_transform(frames, transform)`
    *   Extract logits and intermediate features:
        $$\text{heatmap}_{k, i}, \text{features}_{k, i} = \text{Model}_k(\text{augmented\_frames})$$
2.  **Invert transforms:** Invert the transform on the spatial dimensions for both the heatmap and the channel-first feature tensor:
    *   `restored_heatmap = invert_spatial_transform(heatmap, transform)`
    *   `restored_features = invert_spatial_transform(features, transform)`
3.  **Ensembled Canonical Average:** Average all restored maps across both the 8 TTA views **and the $N$ loaded models**:
    *   `mean_heatmap = np.mean(all_restored_heatmaps, axis=0)`
    *   `mean_features = np.mean(all_restored_features, axis=0)`
4.  **Centroid Extraction & Pooling:**
    *   Detect local maxima peaks on `mean_heatmap` to extract centroid coordinates $(z, y, x)$.
    *   Pool feature vectors from `mean_features` at those $(z, y, x)$ centroids, yielding robust, spatially-invariant node embeddings.

---

## 📐 3. Section B: Temporal Bidirectional Tracking

To eliminate false-positive drift, we evaluate connection probabilities in both temporal directions.

1.  **Forward Scorer:** Predict probability of connection from source node $u$ at time $t$ to target node $v$ at time $t+1$:
    *   `forward_logits = EdgeModel.predict_edge_logits(F_t, F_{t+1})`
    *   `P_forward = activation(forward_logits)` (shape: $(|V_t|, |V_{t+1}|)$)
2.  **Reverse Scorer:** Predict connection from target node $v$ at $t+1$ back to source node $u$ at $t$:
    *   `reverse_logits = EdgeModel.predict_edge_logits(F_{t+1}, F_t)`
    *   `P_reverse = activation(reverse_logits)` (shape: $(|V_{t+1}|, |V_t|)$)
3.  **Bidirectional Fusion:** Align the reverse matrix and execute `fuse_bidirectional_probabilities` with harmonic mean:
    *   `P_reverse_aligned = P_reverse.T`
    *   `P_fused = fuse_bidirectional_probabilities(P_forward, P_reverse_aligned, mode="harmonic")`
4.  **Ensembled Averages:** For $N$ edge models, average the fused probabilities across models:
    *   `P_ensemble = np.mean([P_fused_k for k in range(N)], axis=0)`

---

## 🧪 4. Verification and Testing Strategy

To guarantee absolute numerical correctness of our implementations, we will add **`tests/test_royerlab_prediction.py`** containing synthetic, lightweight tests:

1.  **TTA Equivariance Test:** Pass a synthetic mock 3D U-Net features tensor through our TTA loop and verify that when rotating the input coordinates, the averaged canonical feature output remains exactly invariant (up to machine precision).
2.  **Harmonic Mean Veto Test:** Test `fuse_bidirectional_probabilities` with highly asymmetrical probabilities (e.g. forward $0.99$, backward $0.01$) and verify that the harmonic mean successfully vetoes the connection, while symmetrical high probabilities (e.g. $0.98$, $0.98$) pass untouched.
