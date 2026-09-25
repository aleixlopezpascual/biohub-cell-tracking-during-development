# Kaggle Gold-Zone Battle Plan

This document serves as your concrete, step-by-step instructions for what to run and configure the moment your weekly Kaggle GPU quota resets. 

> **Status correction (2026-09-25):** This is a historical plan, not an execution authorization. It has not established that its proposed run will beat any public notebook or secure a medal. Do not resume `temporal-pu-a` from epoch 10 until its recorded `0.0079` OOF score / `0.017` node recall is diagnosed. The current ordered queue and GPU/provenance gate are in `docs/competition_idea_backlog.md`.

The intended experiment was to combine custom checkpoints with **Feature-Level TTA**, **Bidirectional Tracking**, and **Multi-Model Ensembling**. This has not been validated as a result that outclasses public notebooks.

---

## 🚀 Step 1: Harvest the Upgraded Baseline Score (Approx. 15 mins)

Before training any custom models, we want to evaluate the clean public 50-epoch reference checkpoint with our newly integrated overlays to see our peak baseline score.

1.  Go to your **Kaggle Account** -> **Notebooks**.
2.  Open your evaluation kernel: **`aleixlopez/biohub-gold-public-oof`** (Version 10).
3.  Ensure your attached dataset inputs are:
    *   `aleixlopez/biohub-gold-training-runner` (Our latest codebase)
    *   `dariushafshar/biohub-local-cv-pack` (Local CV splits)
    *   `pilkwang/biohub-tracking-support-pack-50ep-v1` (Official weights & dependencies)
4.  Click **Run** (Verify accelerator is set to **GPU T4 x2**).
5.  **What it does:** It runs our recursive, PyTorch-native GPU-accelerated TTA and Bidirectional Tracking, saving the results under candidate `public-50ep-overlays` at threshold `0.96`.
6.  **Expected Output:** Once complete, check the notebook logs. This is expected to boost your Fold 0 score from `0.897` up toward `0.94+` on your local holdout!

---

## 🏋️‍♂️ Step 2: Custom 50-Epoch Disjoint Folds — BLOCKED

**Do not resume `temporal-pu-a` from epoch 10 using the instructions in this historical plan.** The experiment ledger records an OOF score of `0.0079` and node recall of `0.017`; diagnose that result before spending another GPU allocation. The current endgame priority is the staged inference candidate in `docs/competition_idea_backlog.md`, not scratch-training this checkpoint.

---

## 🏆 Step 3: Multi-Model Checkpoint Ensembling & Submission

Once your custom checkpoints have finished training, we can ensemble them with the public weights and apply our overlays to generate your peak leaderboard submission.

1.  **Configure your Submission Kernel:** Open your final submission notebook on Kaggle.
2.  **Attach inputs:** Attach your newly trained checkpoints dataset along with `biohub-gold-training-runner` and the support pack.
3.  **Setup Ensembled Predictor:**
    *   Load your custom Fold A & Fold B checkpoints along with the public `split_0` weights.
    *   Configure our ensembling logic in your prediction script:
        ```python
        # List of checkpoints to ensemble
        detector_weights = [
            "/kaggle/input/your-dataset/fold_a_detector.pth",
            "/kaggle/input/your-dataset/fold_b_detector.pth",
            "/kaggle/input/biohub-tracking-support-pack-50ep-v1/.../split_0/edge_predictor_best.pth"
        ]
        ```
4.  **Execute Overlay Prediction:**
    *   Run predictions with both **Edge-Feature TTA** and **Bidirectional Tracking** enabled.
    *   Use the community-optimal threshold of **`0.96`**.
5.  **Submit to Leaderboard:** Submit only after the candidate passes fold-disjoint validation and the user explicitly approves; no medal standing is guaranteed.
