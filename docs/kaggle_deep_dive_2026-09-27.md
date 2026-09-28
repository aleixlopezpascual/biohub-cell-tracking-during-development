# Kaggle Deep Dive & Endgame Research Synthesis (2026-09-27)

**Date:** Sunday, September 27, 2026 (Competition deadline: September 29, 2026 23:59 UTC — ~52 hours remaining)  
**Author:** AI Research Assistant  
**Context:** Comprehensive audit of public discussions, latest notebook architectures (up to 0.953–0.956), and live leaderboard dynamics prior to committing GPU quota.

---

## 1. Executive Summary & Current Standings

### Leaderboard Tiers (3,956 total teams)
- **Top 1:** `0.978` (`yu4u`)
- **Gold Threshold (Rank 10):** `0.969` (`VOXEL ART`)
- **Silver Threshold (Rank 50):** `0.963` (`[RU] Kirill Makhankov`)
- **Bronze Threshold (Rank 100):** `0.959` (`ji-ma`)
- **Rank 166:** `0.956` (`Anvith Pothula`)
- **Rank 500:** `0.953` (`Robson`, `Aman Atar` at Rank 564, `Raunak Dey` at Rank 586)
- **Rank 1,000:** `0.947` (`FWC`, `imissher` at Rank 922)
- **Our Anchor:** `0.946` (Rank 1,366, Submission ID `56132481`)

Over **800 teams** are compressed between `0.947` and `0.956`. Reaching `0.953–0.956` places an entry firmly in the top 5%–14% (comfortably inside the top 500).

---

## 2. Evolution of Top Public Notebooks (0.946 → 0.956)

A detailed code-level audit of the latest public kernels was conducted:
1. `flexonafft/biohub-lineage-forge-precision-tracking` (LB **0.946**)
2. `haideptry/biohub-0-951-sota-deepcenter-fast-ilp-19m` (LB **0.951**)
3. `raunakdey07/biohub-harmonic-fusion-v3` (LB **0.953**, submitted 2026-09-25)
4. `amanatar/optimized-biohub-max-score` (LB **0.953**, run/submitted 2026-09-27 01:39 UTC)
5. `anvithpothula/biohub-0-953-lb-original` (LB **0.956**, submitted 2026-09-27 00:03 UTC)

### Key Technical Innovations & Progression

| Technique | Origin / First Appearance | LB Impact | Description & Mechanics |
|---|---|---|---|
| **Edge-Feature TTA** | `flexonafft` (0.946) | `+0.004` over 0.942 | Flips/rotates 3D U-Net feature maps and aggregates before node feature extraction, providing spatial invariance. |
| **EMA Motion Relink (`tight55`)** | `imissher` / `Hammad Farooq` | `+0.002` (0.946 → 0.948) | Uses exponential moving average velocity to resolve ambiguous linking steps; optimal gate identified at `5.5 µm`. |
| **Density-Adaptive Gating** | `haideptry` (0.951) | `+0.003` | Categorizes movies into low (<120 cells/frame), middle (<400), and high density, adapting candidate search radii and velocity weights. |
| **DivNet Mitosis Veto** | `haideptry` / `giorgosi` | `+0.002` | Employs a lightweight 3D CNN (`biohub-divnet-v2`) on volumetric patches around putative divisions; vetoes calls if $P(\text{div}) < 0.50$. |
| **Quantile Logit Calibration** | `amanatar` (0.953) | `+0.003` | Matches dynamic range of secondary detector logits to primary ($\frac{s - \mu_s}{\sigma_s} \cdot \sigma_p + \mu_p$) to avoid destructive interference before linear blending. |
| **Neighborhood-Flow Prior** | `raunakdey07` / `amanatar` | `+0.002` | Estimates local tissue velocity field from $k=12\text{--}16$ nearest tracked neighbors within $40\text{--}48\,\mu\text{m}$ to guide relinking in dense regions. |
| **Sub-Threshold Gap-Filling** | `raunakdey07` / `amanatar` | `+0.001` | Recovers weak detections ($P \ge 0.50$, radius $3.5\,\mu\text{m}$) specifically when needed to bridge 1-to-3 frame tracking gaps. |
| **Sub-Voxel Coordinate Regression (`v1284`)** | `anvithpothula` (0.956) | `+0.003` (0.953 → 0.956) | Small MLP (`v1284_head.pt`) takes directional feature differences around integer peaks in 224-dim U-Net maps to predict sub-voxel shifts; pairs with trilinear feature interpolation. |

---

## 3. Critical Forum Findings & Pitfalls Uncovered

### A. The "All-Train" Memorization Trap (Eric, Topic 742064 & imissher, Topic 743222)
- **The Finding:** The public secondary model checkpoint (`pilkwang/biohub-temporal-unet3d-seed314159-v1`) was trained on **all 199 training videos** (`unet_transformer_alltrain_seed314159_v1` in its `split_manifest.json`). The DeepCenter model was trained on 71 videos.
- **Consequence:** Any local hold-out split created from the 199 training videos is **in-sample** for 80% of detection logits.
- **Impact on Validation:** Local CV has near-zero or negative correlation ($r \approx -0.2$) with the public leaderboard. Steps that appear to improve local CV (e.g. disabling motion relinking gave $+0.017$ locally but $-0.002$ on LB) are exploiting memorized training data. Only true cross-embryo splits or genuine out-of-sample models reflect generalization.

### B. The Notebook In-Sample Parameter Sweep Trap (Topic 741242 & 743222)
- **The Finding:** The base public notebook ran an automated sweep over 7 post-processing parameter candidates across 8 training videos at the end of the run, overwriting `submission.csv` with the winning config.
- **The Danger:** On the commit run, it drops the 4 test videos that overlap with train. On the hidden test rerun, there is zero overlap, so a completely different 8 videos are evaluated. The resulting winner may diverge from the public LB configuration!
- **The Fix:** Pin the proven best parameter set explicitly (`BIOHUB_MOTION_RELINK_TIGHT_UM = "5.5"`) and disable the internal validator sweep (`BIOHUB_VALIDATOR_ENABLE = "0"`). This eliminates config instability on hidden data and **saves ~75 minutes of GPU runtime**.

### C. Ground Truth Label Noise & Division Under-Calling (Topic 742942 & Anvith Pothula's `divqc`)
- **The Finding:** The competition dataset contains significant label noise. Human spot-checks revealed that up to 70% of apparent "false positives" produced by clean detectors are actually real unannotated cells.
- **Division Jaccard Reality:** Overall division Jaccard across all competitive pipelines is low ($\sim 0.18\text{--}0.23$). On over 70% of individual videos, division Jaccard is 0. 
- **Guidance:** Do not blindly inflate division recall; excessive division proposals heavily penalize both division and edge metrics. Strict geometric vetting (sister symmetry $\tau=0.6$, divergence thresholds) or CNN-based verification (DivNet) is essential.

### D. Feasibility of Retraining on Kaggle GPUs (Topic 743222)
- **The Finding:** On 2× T4 GPUs, the 3D U-Net trainer takes 35–40 minutes per epoch (and OOMs with batch sizes $>8$ on $64^3$ patches). An entire 8-hour session achieves only 12–25 epochs, whereas the public checkpoint received 402 epochs.
- **Verdict:** Training from scratch on Kaggle hardware is intractable with ~2 days remaining; attempts to fine-tune on Kaggle GPUs dropped public scores from `0.948` to `0.939`.

---

## 4. Strategic Recommendations for Final Submissions

1. **Do Not Burn GPU on Full Training:** All evidence confirms that fine-tuning or training on Kaggle GPUs at this stage is high-risk, slow, and tends to degrade strong public checkpoints.
2. **Anchor Strategy:** Preserve submission `56132481` (LB `0.946`) as the safe, proven baseline anchor.
3. **Primary Upgrade Target (`0.953–0.956` Frontier):**
   - The top reproducible pipeline incorporates:
     - Hardened runtime (`BIOHUB_VALIDATOR_ENABLE = "0"`, timeout guards) to guarantee completion within the 9-hour limit.
     - Quantile-calibrated dual-seed blending.
     - Fixed `tight55` EMA motion relinking with neighborhood-flow prior ($k=12$, radius $40\,\mu\text{m}$).
     - Sub-threshold gap filling ($P \ge 0.50$, max gap 3).
     - Sub-voxel coordinate refinement (`v1284_head.pt` + trilinear feature interpolation) as validated in `anvithpothula/biohub-0-953-lb-original`.
4. **Execution Plan:**
   - Resolve sidecar metadata (`enable_gpu: true`, Tesla T4).
   - Test pre-flight checks on CPU.
   - Request explicit user authorization before pushing and executing the kernel on Kaggle.
