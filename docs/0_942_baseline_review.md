# Biohub 0.942 LB Baseline Review

Review and documentation of the public leaderboard baseline notebook `busyaprime/biohub-0-942-lb-one-knob-past-the-public-line` (LB 0.942, exploit-free).

## Core Insights & Methodology

This notebook achieves its **0.942 Public LB** score genuinely (with `metric_hack_used: False` explicitly verified and asserted). It is built upon the `TemporalUNet3D + SimpleNodeTransformer + tracksdata ILP` pipeline and leverages three powerful, legitimate modeling and post-processing improvements:

### 1. Bidirectional Harmonic-Probability Association Fusion
To establish temporally consistent cell tracks, the model executes edge scoring in both forward and backward time directions:
* **Forward Pass ($P_{fwd}$):** Predicts edge logits $P(t \to t+1)$ from source to target frame centroids.
* **Backward Pass ($P_{rev}$):** Predicts edge logits $P(t+1 \to t)$ by swapping target and source inputs.
* **Scale Alignment:** Aligns the mean and standard deviation of reverse logits with the forward logits to standardize calibration before combining.
* **Harmonic Mean Fusion:** Fuses probabilities using the **harmonic mean** in probability space. This heavily penalizes proposed edges that receive a very low score in *either* temporal direction, suppressing unidirectional tracking noise.
* **Knobs:** 
  * `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT = "0.15"`
  * `BIOHUB_BIDIRECTIONAL_FUSION_MODE = "harmonic_probability"`

### 2. Veto-Backed Post-Processing with "DeepCenter"
To clean up proposed link completions (gap closures) and cell division events, the pipeline utilizes an auxiliary 3D U-Net center-prior checkpoint (`weights/full_frame_center/best.pt`):
* **Verification:** For any gap closure or division node proposed geometrically during post-processing, the pipeline samples a 3D patch in the DeepCenter center-prior heatmap.
* **Veto:** If the maximum value in the patch is below `0.25`, the proposed node or edge is **vetoed** (rejected). This allows aggressive track repair while maintaining low false-positive rates.
* **Knobs:**
  * `BIOHUB_USE_DEEPCENTER_VETO = "1"`
  * `BIOHUB_DEEPCENTER_GAP_VETO = "1"`
  * `BIOHUB_DEEPCENTER_SAFE_DIV_VETO = "1"`
  * `BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD = "0.25"`

### 3. Precision-Gated Wide Divisions (`safe-div`)
True division events often diverge wider than the official $7\text{ }\mu\text{m}$ evaluation radius. This baseline loosens physical distance separation gates to capture these genuine splits while filtering noise using geometric constraints:
* **Loosened Gates:** 
  * `BIOHUB_SAFE_DIV_MAX_UM = "9.0"` (up from 7.0 µm; ground-truth divisions reach $10.4\text{ }\mu\text{m}$).
  * `BIOHUB_SAFE_DIV_SISTER_MAX_UM = "14.0"` (up from 12.0 µm; ground-truth sister separations reach $13.7\text{ }\mu\text{m}$).
* **Noise Filters:**
  * **Sister Symmetry Gate (`BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU = "0.6"`):** Rejects splits where daughter distances from the parent are highly asymmetric (>60% mean difference).
  * **Mutual Nearest Neighbor Gate (`SAFE_DIV_REQUIRE_MUTUAL_NN`):** Rejects divisions unless the daughters are mutual spatial nearest neighbors of the parent.
  * **Forward Divergence Gate (`BIOHUB_SAFE_DIV_DIVERGE_UM = "2.25"`):** Verifies that the daughter tracks physically separate and diverge over subsequent frames.

---

## Empirical Parametric Sweeps (Leaderboard)

The author, `busyaprime`, documented single-variable sweeps from the baseline `0.941` configuration (`analyticaobscura/biohub-lb-941`):

| Sweep Parameter (Detection Threshold) | Public LB Score | Status |
|---|---|---|
| `BIOHUB_DET_THRESHOLD = 0.94` | `0.938` | Tested |
| `BIOHUB_DET_THRESHOLD = 0.95` | `0.940` | Tested |
| `BIOHUB_DET_THRESHOLD = 0.96` | `0.942` | **Peak (Current Selected)** |
| `BIOHUB_DET_THRESHOLD = 0.965` | `0.941` | Base (Published) |

Raising the secondary edge weight independently from `0.15` to `0.25` (keeping the threshold at `0.965`) also achieved `0.942` Public LB, suggesting independent room for optimization.

---

## The Local Validator "Catch" (Anti-Correlation)

A crucial finding documented by the author is that **the in-notebook local CV proxy score is anti-correlated with LB scores for the detection threshold knob**:
* At threshold `0.965` (LB 0.941), the local 4-movie validator prints **`0.9418`**.
* At threshold `0.96` (LB 0.942), the local validator prints a **lower score**.

### Root Cause
The division Jaccard term of the local validator is pinned at `0.2000` in every sweep run, meaning the entire local validation movement is dictated solely by the edge Jaccard term across **only four training movies**. Since four training movies are not representative of the full hidden test dataset, optimizing purely against the local in-notebook proxy signal leads to sub-optimal threshold selection on the real leaderboard.

---

## Our Kaggle Submission

We copied the `.ipynb` file and created a dedicated submission directory containing the notebook and its corresponding `kernel-metadata.json`, which correctly maps the three required supporting datasets:
* `pilkwang/biohub-tracking-support-pack-50ep-v1`
* `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`
* `pilkwang/biohub-temporal-unet3d-seed314159-v1`

Pushed to Kaggle:
* **Kernel Slug:** `aleixlopez/biohub-0-942-lb-one-knob-past-the-public-line`
* **Version:** 1 (Pushed at 2026-09-09T21:40:00Z)
* **Leaderboard Status:** Executing / Pending evaluation.
