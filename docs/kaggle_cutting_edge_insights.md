# Kaggle Cutting-Edge Insights & Strategy

> **Historical strategy note, not the active endgame checklist.** Its model and CV observations are preserved for reference; its suggested next actions are not current execution instructions. In particular, do not start a new GPU run from this page or infer that the `0.90325` local CV benchmark validates the staged 0.948 EMA candidate. See [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md) for current blockers, required evidence, and the CPU/GPU split.

This document logs our research findings regarding the advanced **SGT-DFO (Spatiotemporal Graph Transformer with Differentiable Graph-Flow Optimization)** framework and **Kaiwalya Raut's 0.901 V4 baseline**, answering our core strategic questions for the Biohub Cell Tracking competition.

---

## 🧭 1. Core Answers & Strategy

### ❓ Q1: Should we submit one of those notebooks?
**No, we should not submit their raw notebooks. Here is why:**

1.  **Our Peak Baseline is Already Stronger (`0.946` vs `0.901`):** Kaiwalya Raut's baseline is actually *inferior* to our peak submission. We already achieved **`0.946` on the Public Leaderboard** with `aleixlopez/biohub-0-946-edge-feature-tta-tuned`. Pushing Raut's raw notebook (which scores `0.901`) would be a step backward.
2.  **SGT-DFO is a custom, heavy framework:** The `pomagrenate` repository is a massive custom deep learning repo. Running a raw, uncalibrated training run of a completely different model would eat up your entire GPU quota, without a verified OOF baseline or local CV.
3.  **The Winning Play:** Keep our `0.946` peak baseline. Use these new findings conceptually to improve our **custom model training and post-processing calibrations** so that we can surpass the `0.946` ceiling.

---

### ❓ Q2: Can we analyze them locally?
**Yes, but with specific resource boundaries:**

*   **What we CAN analyze locally:** We can clone their code, review their mathematical models, and write mock unit tests (exactly like we did for our overfit sanity scripts) to verify code syntax and data shapes on lightweight synthetic arrays.
*   **What we CANNOT run locally:** Because light-sheet microscopy `.zarr` volumes are massive (hundreds of gigabytes) and we run on a CPU fallback locally, we cannot run full-scale training or full-volume evaluations on our local machines. Heavy GPU execution must remain on Kaggle.

---

### ❓ Q3: How can we implement these learnings?

We can implement these cutting-edge insights directly into our current modular repository in three high-value ways:

#### A. Activate and Calibrate Sister Symmetry Gating
*   *The Concept:* Divisions must satisfy biological symmetries (the two daughter cells should have similar sizes, migration velocities, and temporal alignment). Linking deforming cells that do not share these symmetries creates massive false divisions and ruins scores.
*   *Implementation:* Our local tracker (`src/biohub_tracking/tracking/hungarian.py`) **already natively implements a `sister_symmetry_tau` gate!**
*   *Action:* We can activate `use_sister_symmetry_gate: True` in our configuration and calibrate `sister_symmetry_tau` directly against our newly secured `0.90325` OOF CV to filter out false-positive branches.

#### B. Handle the SCIP/Gurobi Solver Licensing Bottleneck
*   *The Concept:* The official tracking code uses discrete Gurobi/SCIP ILP solvers, which throw warnings like `Gurobi license is not available, trying Scip`. If SCIP is slow or unavailable, tracking fails or hangs.
*   *Implementation:* Following SGT-DFO's concept of continuous relaxations, we can write a native continuous linear assignment solver (using `scipy.optimize.linear_sum_assignment` or Sinkhorn iterations) directly inside our codebase as a robust, license-free, fast fallback for edge association on Kaggle.

---

## 📊 Comparison Matrix

| Feature | SGT-DFO (`pomagrenate`) | **Our Ensembled Codebase** |
| :--- | :---: | :---: |
| **Model Architecture** | Graph Transformer | 3D U-Net + Node Transformer |
| **Tracking Formulation** | Differentiable Network Flow (DFO) | Linear Assignment / ILP |
| **Mitosis Branching** | Flow splits | Hungarian + Sister Symmetry Gating |
| **Local OOF CV** | ⚠️ Unknown | 🌟 **`0.90325` (Fully Secured)** |
| **Integration Cost** | 🔴 Extremely High (Complete Rewrite) | 🟢 **Zero (Already Built)** |
