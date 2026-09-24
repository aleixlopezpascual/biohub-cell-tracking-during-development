# Competition Endgame Strategy & Status

Documented on Thursday, September 24, 2026.

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
  * **Public Leaderboard Score:** **`0.946`** (Solid Silver Medal Standing).
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

Once the GPU quota resets in ~38 hours (Saturday, September 26):
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

## 5. Final Two-Submission Selection Rule (Before Sept 29)

Under Kaggle rules, every team must manually select **two submissions** for private leaderboard scoring:
1. **Selection 1 (Robust Anchor):** `biohub-0-946-edge-feature-tta-tuned` (Submission `56132481`, LB `0.946`).
2. **Selection 2 (Peak Candidate):** `biohub-0-948-momentum-deepcenter-tta` (Targeting `0.95+`).
