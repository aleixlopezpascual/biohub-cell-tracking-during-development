# Competition Endgame Strategy & Status

Documented on Thursday, September 24, 2026.

## Fresh correction — 2026-09-25

- The current public-board snapshot puts submission `56132481` / score `0.946` at rank **1,221 of 3,899**, outside the current top-10% bronze threshold (389 teams). The earlier “Solid Silver Medal Standing” label below was incorrect for the current standings; medal allocation uses final private standings and the public board is only a partial proxy.
- The public leaderboard uses about 29% of test data; the final ranking uses the other 71%. A `0.948` score would still be outside the current public bronze line.
- The GPU quota/reset numbers and run schedule below are estimates from September 24, not live-verified on September 25. User reports GPU access remains unavailable.
- The staged `0.948` candidate is **not yet run/scored** and remains blocked on provenance/quality-gate confirmation. See `docs/kaggle_deep_dive_2026-09-25.md`, `docs/competition_idea_backlog.md`, and `docs/kernel_audits/0948_cpu_preflight.md` before acting.

---

## 1. Competition Timeline & Deadlines

* **Official Kaggle Competition Deadline:** **Tuesday, September 29, 2026 at 23:59 UTC**
* **Time Remaining (from Sept 24, 12:00 PM CEST):** ~5.5 days (approx. 132 hours).
* **Weekly GPU Quota Status:**
  * **Current State:** 30.0 / 30.0 hours utilized.
  * **Quota Reset Time:** Expected in **~38 hours** (Saturday, September 26, ~02:00 UTC).
  * **Post-Reset Runway:** **3.5 full days (~94 hours)** with a fresh 30-hour GPU allowance before the final deadline.

---

## 2. Current Benchmark & Baseline Standing

* **Best Scored Submission:**
  * **Submission ID:** `56132481`
  * **Public Leaderboard Score:** **`0.946`** (current public snapshot rank 1,221/3,899; not currently in medal range).
  * **Candidate:** `aleixlopez/biohub-0-946-edge-feature-tta-tuned` (Version 2).
  * **Key Components:**
    * 3D U-Net intermediate Edge-Feature Test-Time Augmentation (Edge TTA).
    * Optimized detection threshold knob tuned to `0.96`.
    * Clean, verified biological output with zero metric exploits or synthetic node artifacts.

---

## 3. High-ROI Next Weapon: Biohub 0.948 Momentum + DeepCenter TTA

Instead of burning scarce GPU hours training custom models from scratch (which takes 12–20h with high convergence variance), we focus on the winning inference-time play:

* **Kernel Reference:** `aleixlopez/biohub-0-948-momentum-deepcenter-tta`
* **Local Staging Directory:** `scripts/kaggle_kernels/biohub_0_948_momentum_deepcenter_tta/`
* **Core Improvements Over 0.946:**
  1. **EMA Spatiotemporal Velocity Projection:** Injected into `motion_relink_edges`. Uses an exponential moving average ($v_{t} = 0.6 \cdot v_{\text{instant}} + 0.4 \cdot v_{\text{prior}}$) to maintain cell momentum vectors across frames, eliminating identity swaps (crossovers) in dense cell clusters.
  2. **DeepCenter Image-Space Heatmap Veto:** Filters false-positive gap closings and divisions using cell-center probability maps, protecting Adjusted Edge Jaccard.
* **Execution Footprint:**
  * Test-set inference only takes **~80 minutes** on an Nvidia Tesla T4 GPU.
  * Consumes minimal weekly GPU quota (~1.3 hours).

---

## 4. Execution Plan on Quota Reset

If/when GPU access is actually available (the reset estimate below is stale and must be rechecked):
1. **Trigger the Run:**
   ```bash
   kaggle kernels push -p scripts/kaggle_kernels/biohub_0_948_momentum_deepcenter_tta
   ```
2. **Monitor the ~80-Minute Inference Run:**
   ```bash
   python3 scripts/monitor_and_submit_0946.py  # or view on Kaggle web UI
   ```
3. **Submit to Competition:**
   Once completed, submit the resulting `submission.csv` via the Kaggle web UI or CLI:
   ```bash
   kaggle competitions submit -c biohub-cell-tracking-during-development -k aleixlopez/biohub-0-948-momentum-deepcenter-tta -f submission.csv -v 1 -m "Biohub 0.948 Momentum DeepCenter TTA Inference"
   ```

---

## 5. Final-Submission Selection Rule (Before Sept 29)

Kaggle permits **up to two** submissions for private leaderboard scoring; it does not require filling both slots. Select only candidates that have run and passed the validation/promotion gate:
1. **Selection 1 (Robust Anchor):** `biohub-0-946-edge-feature-tta-tuned` (Submission `56132481`, LB `0.946`).
2. **Selection 2 (Peak Candidate):** `biohub-0-948-momentum-deepcenter-tta` only if its actual run, provenance, and validation pass (the notebook name/target is not a score).

---

## 6. Actionable Endgame TODO Checklist

### Completed Tasks
- [x] **Diagnose GPU Status:** Identified that weekly 30-hour GPU quota was hit; confirmed exact ~38h reset schedule.
- [x] **Pivot to High-ROI Strategy:** Dropped expensive scratch training to focus on pure, fast inference with pre-trained backbones.
- [x] **Prepare 0.948 Momentum + DeepCenter Pipeline:**
  - Implemented EMA spatiotemporal velocity projection ($v_t = 0.6 \cdot v_{\text{instant}} + 0.4 \cdot v_{\text{prior}}$).
  - Integrated DeepCenter image-space heatmap vetos.
  - Verified Python AST syntax and Kaggle kernel metadata in `scripts/kaggle_kernels/biohub_0_948_momentum_deepcenter_tta/`.
- [x] **Secure Codebase & Sync Remote:** 
  - All 119 unit tests passing (100% green).
  - Merged and pushed cleanly to GitHub `origin/master`.
  - Updated persistent memory index and `docs/endgame_status_and_strategy.md`.

### Phase 1: When GPU Quota Resets (~38h — Saturday morning, ~02:00 UTC)
- [ ] **1. Launch 0.948 Inference Run on Kaggle:**
  ```bash
  kaggle kernels push -p scripts/kaggle_kernels/biohub_0_948_momentum_deepcenter_tta
  ```
- [ ] **2. Submit the Predictions to the Leaderboard:**
  Once the ~80-minute run completes on Kaggle:
  ```bash
  kaggle competitions submit -c biohub-cell-tracking-during-development -k aleixlopez/biohub-0-948-momentum-deepcenter-tta -f submission.csv -v 1 -m "Biohub 0.948 Momentum DeepCenter TTA Inference"
  ```
- [ ] **3. Record New Leaderboard Score:** Log the public score in `results/kaggle_lb/submissions.csv` (targeting **`0.948` – `0.952`**).

### Phase 2: Weekend Iteration & Tuning (Saturday, Sep 26 – Monday, Sep 28)
- [ ] **(Optional) Threshold Sensitivity Sweep:** If the momentum submission shows strong gains, test a minor threshold variation (e.g. `0.958` vs `0.962`) to find the sweet spot.
- [ ] **(Optional) Run Fold B Benchmark:** If we want to complete the official cross-validation record:
  ```bash
  kaggle kernels push -p scripts/kaggle_kernels/gold_public_oof_b
  ```

### Phase 3: Final Submission Selection (Tuesday, September 29 — Deadline Day)
- [ ] **Select Final Submission 1 (Proven Anchor):** `aleixlopez/biohub-0-946-edge-feature-tta-tuned` (Submission `#56132481`, Public Score: **`0.946`**).
- [ ] **Select Final Submission 2 (Peak Candidate):** The highest-scoring momentum submission from Saturday's run (`0.948+`).
- [ ] **Double Check Selection Checkboxes:** Ensure the checkboxes under the Kaggle **"My Submissions"** tab are firmly checked before 23:59 UTC!
