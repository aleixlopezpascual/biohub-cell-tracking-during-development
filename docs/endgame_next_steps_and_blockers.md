# Biohub Endgame: Current Status, Blockers, and Next Actions

**Status checked:** 2026-09-28 14:00 UTC (Competition deadline: 2026-09-29 23:59 UTC — ~34 hours remaining).

**Purpose:** Single current source of truth for what has been achieved, what is currently running, and our final 2-submission portfolio strategy.

---

## 1. Executive Summary & Verified Breakthrough

- **New Verified Baseline:** Kaggle submission **`56616065`** achieved a public score of **`0.953`** (Candidate 1: `aleixlopez/biohub-0-956-subvoxel-flow-harmonic`), surging **+529 ranks from 1,366 to Rank 837**.
- **Safety Anchor Secured:** Submission **`56132481`** at **`0.946`** (Edge-Feature TTA Tuned Baseline).
- **GPU Quota Status:** **28.15 hours remaining** of weekly GPU quota on Kaggle (only 1.85h consumed across 5 full end-to-end runs).
- **Daily Submissions:** 4 of 5 submissions remaining today.

---

## 2. Key Empirical Diagnostics from Today's Runs

1. **Why Candidate 1 Scored 0.953:**
   - Strict detection threshold ($0.965$) and dual-seed retention ($0.90$).
   - Continuous sub-voxel coordinate regression head (`v1284_head.pt`) with trilinear feature interpolation in `UNetNodeTransformer`.
   - Collective neighborhood tissue flow prior ($k=12$, radius $40\,\mu\text{m}$).
   - Pinned `tight55` relinking ($5.5\,\mu\text{m}$).
   - Tightly constrained $N_{\text{pred}}$, avoiding denominator penalties.
2. **Why Candidate 2 & 3 Dropped to 0.901 (Root Cause Discovered):**
   - Log inspection revealed an unhandled `NameError: name 'SAFE_DIV_HORIZON_FRAMES' is not defined` inside Aman's `add_safe_divisions_postlink`.
   - This caused the repair loop to fail and fall back to raw un-relinked ILP graphs with 0 divisions.
   - Combined with `torch.maximum` detection inflation, $N_{\text{pred}}$ bloated and dragged Adjusted Edge Jaccard down.
   - **Fix Applied:** In all subsequent candidates, `SAFE_DIV_HORIZON_FRAMES` is properly scoped or removed, and `torch.maximum` is strictly banned.

---

## 3. Active Kaggle Pipelines & Current Status

| Pipeline | Target / Innovation | GPU Status | Submission Status |
|---|---|---|---|
| **Candidate 1** (`biohub-0-956-subvoxel-flow-harmonic`) | Subvoxel `v1284` + Tissue Flow | Complete | **`0.953` (Scored, Rank 837)** |
| **Candidate 4** (`biohub-0-956-divnet-vetted`) | Candidate 1 + DivNet 3D-CNN mitosis veto | Complete | **`56634490` (Pending Evaluation)** |
| **Candidate 5** (`biohub-consensus-ensemble-sota` v3) | In-line Bipartite Consensus (0.946 + 0.953) | Complete | **`56640103` (Pending Evaluation)** |
| **Candidate 6** (`biohub-active-mitosis-recovery`) | Loosened symmetry ($\tau \le 0.80$) + DivNet neural rescue | Complete | **`56639822` (Pending Evaluation)** |
| **Candidate 7** (`biohub-0-959-frontier-sota`) | John Taylor Topic 743929: ILP div 0.4 + Readmit 0.94 | Complete | **`56640857` (Pending Evaluation)** |
| **Candidate 8** (`biohub-density-calibrated-flow`) | B10 density-adaptive flow + B9 singleton pruning | **RUNNING** on T4 GPU | Background watcher active (PID `81892`) |

---

## 4. Final 2-Submission Portfolio Strategy (B11)

Under official Kaggle rules, teams can select exactly 2 submissions for final judging:
- **Slot 1 (Highest Ceiling — SOTA Innovation):** The highest-scoring multi-model candidate (`0.953` baseline, DivNet Vetted, or Consensus Ensemble).
- **Slot 2 (Anti-Shakeup Safety Anchor):** Our verified baseline `56132481` (**`0.946`** Edge TTA), completely decoupled from late-stage public notebook tuning.

Use the automated verification script before deadline:
```bash
python3 scripts/verify_final_selections.py
```
Navigate to the Kaggle Submissions dashboard and ensure checkboxes for the two designated submissions are checked.
