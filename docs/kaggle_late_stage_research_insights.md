# Kaggle Late-Stage Research Insights & Action Plan

This document archives our late-stage research from Kaggle discussions and notebooks, capturing critical techniques that top competitors used to push scores into the `0.940+` and `0.960+` ranges.

---

## 🧭 1. Key Insights & Mathematical Rationale

### A. Spatiotemporal Velocity-Projected Gap Closing (EMA Velocity)
* **The Insight:** Standard trackers (including ILP and Hungarian-based solvers) close frame-to-frame gaps (e.g., $t \to t+2$ or $t \to t+3$) by measuring raw Euclidean distance between endpoints: $\Delta d = \|x_t - x_{t+k}\|$. However, deforming, fast-moving embryonic cells often move significantly during these gaps, leading to dropped links and tracking failures.
* **The Solution:** Top competitors track a running **Exponential Moving Average (EMA) Velocity vector** $\mathbf{v}_{\text{ema}}$ for each active cell trajectory. When evaluating a gap-closing link of length $k$, they project the expected position of the cell forward:
  $$x_{\text{proj}} = x_t + \mathbf{v}_{\text{ema}} \times k$$
  They then measure distance relative to this projected coordinate: $\Delta d_{\text{proj}} = \|x_{\text{proj}} - x_{t+k}\|$.
* **Impact:** This approach dramatically improves tracking precision and recall through fast-motion occlusions, preventing the tracker from matching cells to incorrect near-neighbors.

### B. Image-Space Veto Gating with "DeepCenter" Heatmaps
* **The Insight:** Rule-based gap-closing and division-matching heuristics are prone to "hallucinating" links across empty space. Gurobi/SCIP ILP models often generate spurious nodes or connections to optimize local graph flow.
* **The Solution:** Competitors train a separate, lightweight 3D U-Net model strictly to predict cell-center probability maps (known as **DeepCenter** or `DeepCenterUNet3D`, available in dataset `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`).
  * When the tracker proposes an interpolated node $\tilde{q}_{t+1} = \frac{q_t + q_{t+2}}{2}$ to fill a $t \to t+2$ gap:
    1. Query the pre-computed DeepCenter probability heatmap $c(\tilde{q}_{t+1})$.
    2. If $c(\tilde{q}_{t+1}) < \tau_{\text{gap}}$ (with optimal threshold $\tau_{\text{gap}} \in [0.20, 0.25]$), **veto (reject)** the proposed interpolation.
* **Impact:** This non-disruptive gatekeeper eliminates fake node insertions without altering high-confidence segments.

### C. Mitosis Sparsity & "Broken Continuation" Mitigation
* **The Insight:** Ground-truth cell divisions are extremely rare. Across all 199 training volumes, there are **only 151 total divisions** (found in 87 videos; 112 videos have zero divisions).
* **The Problem:** Trackers often treat "broken continuations" (where the cell detector missed a frame, starting a new track mid-volume) as daughter cells, linking them back to a nearby track and generating false divisions.
* **The Solution:** Enforce strict biological and physical constraints on divisions:
  1. **Sister Symmetry Gating:** Verify that the two proposed daughter branches have symmetrical migration velocities, sizes, and orientations.
  2. Reject forks that link to wide spatial deviations.

### D. Multi-Scale Difference of Gaussians (DoG) peak extraction
* **The Insight:** Pure rule-based baselines achieved a strong **0.826 LB** without deep learning.
* **The Key Lever:** Multi-scale scale-space maximum detection using Difference of Gaussians (DoG) was the single biggest contributor, boosting the baseline by **`+0.040`**.

---

## 🛠️ 2. Concrete Action Items for Our Repository

We can implement these cutting-edge insights directly into our current codebase:

| Insight | Area of Code | Action Plan |
| :--- | :--- | :--- |
| **EMA Velocity** | `src/biohub_tracking/tracking/hungarian.py` | Add an EMA velocity calculation to active trajectories and use forward projection for gap-closing evaluation. |
| **DeepCenter Veto** | `src/biohub_tracking/tracking/postprocess.py` | Create an optional DeepCenter veto wrapper that loads pre-computed heatmaps (or runs the DeepCenter UNet) to filter proposed interpolated nodes. |
| **Sister Symmetry** | `src/biohub_tracking/tracking/hungarian.py` | Calibrate `sister_symmetry_tau` and `use_sister_symmetry_gate` configurations against our local `0.90325` OOF CV. |
| **Multi-Scale DoG** | `src/biohub_tracking/detection/local_maxima.py` | Add a multi-scale SciPy `difference_of_gaussians` fallback or refinement step to improve centroid detection. |
