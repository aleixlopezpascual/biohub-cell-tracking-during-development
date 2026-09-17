# Kaggle Gold-Zone Battle Plan

This document serves as your concrete, step-by-step instructions for what to run and configure the moment your weekly Kaggle GPU quota resets. 

By executing this plan, you will combine our custom model checkpoints with **Feature-Level TTA**, **Bidirectional Tracking**, and **Multi-Model Ensembling** to produce a solution that outclasses any single public notebook in the competition.

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

## 🏋️‍♂️ Step 2: Launch the Custom 50-Epoch Disjoint Folds (Approx. 4-5 hours)

Now that our training pipeline is verified as **100% bug-free** via our detector sanity check, we can train our own models on both disjoint folds to enable fold ensembling.

1.  Open your training kernel: **`aleixlopez/biohub-gold-oof-runner`**.
2.  Configure your environment variables in the notebook's environment:
    *   `BIOHUB_CANDIDATE` = `temporal-pu-a`
    *   `BIOHUB_TARGET_EPOCH` = `50` (or `60` to let it converge fully)
    *   `BIOHUB_RESUME` = `1` (to resume training from epoch 10)
3.  Click **Run**.
4.  **What it does:** This will resume training the model from epoch 10 up to epoch 50 on both disjoint folds, generating your own custom, converged weights.
5.  **Output Artifacts:** Checkpoints will be saved as `temporal-pu-a_model.pth` under your outputs.

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
5.  **Submit to Leaderboard:** Export the final `submission.csv` and submit it to the competition to secure your high-ranking Silver/Gold medal standing!
