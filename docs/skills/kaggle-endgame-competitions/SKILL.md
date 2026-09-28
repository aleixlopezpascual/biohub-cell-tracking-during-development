---
name: kaggle-endgame-competitions
description: Use when preparing, hardening, auditing, or submitting notebooks to Kaggle code competitions in the final 72 hours, especially when facing hidden private-test rerun timeouts, GPU OOMs, metadata mismatches, in-sample parameter sweep traps, or designing risk-hedged 2-slot submission portfolios.
---

# Kaggle Endgame Competitions

## Overview

In Kaggle code competitions, winning models often fail or score `0.000` because the hidden private test evaluation operates under fundamentally different constraints than visible public runs. This skill provides proven hardening, validation, ensembling, and portfolio selection protocols for the final 72 hours of a competition.

## When to Use

Use this skill when:
- Deploying Kaggle code competition kernels for scoring on hidden test sets.
- Auditing public notebooks for hidden private test failure modes (OOMs, timeouts, sweep traps).
- Building multi-model consensus ensembles that execute self-contained in a single session.
- Selecting the final 2 submissions before the competition deadline.

Do NOT use for:
- Standard tabular / non-code competitions (CSV upload competitions).
- Exploratory data analysis (EDA) or early-stage feature engineering.

---

## The 5 Fatal Endgame Traps & Countermeasures

### 1. The Hidden Private Test Re-run Trap (Timeout & OOM)
- **The Reality:** Public test sets are small previews (~4–6 items); private test sets contain 70–80% of the entire dataset. A notebook that takes 30 minutes on public data will execute for 3–5 hours on private data.
- **Countermeasure:**
  - **Hard-cap memory caches:** Never leave image, tensor, or graph caches unbounded. Use LRU eviction with a strict frame cap (e.g. `MAX_FRAMES = 48`).
  - **Emergency Deadline Degrade:** Implement a wall-clock timer that checks elapsed execution time. If time exceeds $80\%$ of the platform limit (e.g., $7.5\text{ h}$ of a $9\text{ h}$ limit), immediately bypass optional post-processing and force-write the fallback submission CSV:
    ```python
    import time
    START_TIME = time.time()
    DEADLINE_S = 27000  # 7.5 hours

    def check_deadline():
        if time.time() - START_TIME > DEADLINE_S:
            print("WARNING: Approaching 9h limit. Degraded export triggered.")
            write_fallback_submission()
            sys.exit(0)
    ```

### 2. The In-Sample Parameter Sweep Trap
- **The Reality:** Many public notebooks run an automated hyperparameter sweep over a few training samples at the end of the notebook and overwrite `submission.csv` with the "winner". On the private re-run, different samples are evaluated, causing erratic parameter drift and risking timeouts.
- **Countermeasure:**
  - Identify the proven winning parameter set from validation (e.g. `TIGHT_UM = 5.5`).
  - Statically pin the parameters in global configuration.
  - Disable the internal validation sweep completely (`VALIDATOR_ENABLE = "0"`), saving 45–90 minutes of GPU time and guaranteeing deterministic private execution.

### 3. The Metric Denominator Inflation Trap ($N_{\text{pred}}$)
- **The Reality:** Metrics with asymmetric penalty structures (e.g. Adjusted Jaccard, F1 with strict FP penalties) penalize extra predicted entities directly:
  $$\text{Penalty} = \max(0, N_{\text{pred}} - N_{\text{true}})$$
- **Countermeasure:**
  - **Never use element-wise maximum ensembling** (`torch.maximum(model_a, model_b)`). Max-pooling across models forces every false-positive noise peak through, ballooning $N_{\text{pred}}$ and causing massive score drops (e.g., $0.953 \to 0.901$).
  - Maintain strict candidate retention ($0.90+$) and high detection thresholds. Re-wire edges or refine coordinates rather than adding unverified nodes.

### 4. GPU & Sidecar Metadata Synchronization
- **The Reality:** Kaggle kernels rely on two separate metadata layers: `kernel-metadata.json` (for the CLI) and internal notebook JSON metadata (`nb["metadata"]["kaggle"]`). If they conflict (e.g. CLI requests GPU but notebook declares `isGpuEnabled: false`), the worker may allocate a CPU container or fail silently.
- **Countermeasure:**
  - Ensure both files declare GPU and accelerator type identically:
    ```json
    // kernel-metadata.json
    { "enable_gpu": true, "machine_shape": "NvidiaTeslaT4" }
    ```
    ```json
    // notebook.ipynb metadata
    { "kaggle": { "isGpuEnabled": true, "accelerator": "nvidiaTeslaT4" } }
    ```

### 5. In-Line Consensus Ensembling (No Static Leak)
- **The Reality:** In code competitions, attaching a previous kernel's output as an input dataset only mounts predictions on the *public* test set. During private evaluation, that static dataset does not contain private predictions!
- **Countermeasure:**
  - Run all models sequentially inside the same notebook if total runtime fits under the limit, OR
  - Share intermediate feature extraction / backbone representations in memory and run distinct post-processing passes.
  - Blend with a bipartite matching consensus script in memory before writing the final `submission.csv`.

---

## Final 2-Submission Selection Strategy

Kaggle allows each team to designate **exactly 2 final submissions** for private scoring. If unselected, Kaggle defaults to the two highest public LB scores, which often correlate and suffer the same private shakeup.

### The Barbell Portfolio Strategy

| Slot | Strategy | Candidate Profile | Objective |
|---|---|---|---|
| **Slot 1** | **High-Ceiling SOTA** | Highest-scoring multi-model innovation (sub-voxel refinement, neural verification, multi-horizon flow). | Maximizes medal potential (Gold/Silver) if private test distribution mirrors public data. |
| **Slot 2** | **Anti-Shakeup Anchor** | Clean, proven baseline with conservative geometric thresholds, decoupled from late-stage public notebook tuning. | Guarantees top-tier standing even if late-stage public tricks overfit and shake down. |

---

## Quick Reference: Verification Checklist

Before locking in your endgame submissions:
- [ ] **Private Test Completeness:** Notebook generates predictions dynamically from `/kaggle/input/.../test/`, never relying on precomputed static test CSVs.
- [ ] **Strict Format Validation:** 0 NaNs, 0 duplicate IDs, correct column headers, sentinel values (`-1`) used appropriately.
- [ ] **Memory Bounded:** All frame caches have strict length limits and automated eviction.
- [ ] **GPU Aligned:** Both `kernel-metadata.json` and notebook JSON metadata agree on `enable_gpu: true`.
- [ ] **2 Submissions Selected:** Verify checkboxes are explicitly selected on `https://www.kaggle.com/competitions/<slug>/submissions`.
