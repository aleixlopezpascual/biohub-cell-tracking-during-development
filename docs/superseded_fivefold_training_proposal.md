# Superseded five-fold training proposal

> Historical record preserved during the integration of commit `6156ea5`. It is **not** the active training plan. The active plan is [the gated cloud-training design](cloud_kfold_training_design.md), and the operational decision is recorded in [the Gold-training session handoff](gold_training_session_handoff.md).
>
> This proposal assumed an unverified 0.946 baseline, five balanced folds, and offset regression. Later local/OOF evidence rejected those premises. It is retained only for traceability.

---

# Specification: Cloud-Based K-Fold Training Pipeline

Design specification for building and deploying a fully automated, cloud-based 5-Fold cross-validation training pipeline for the Biohub Cell Tracking competition.

---

## 1. Overview & Objective
To break past the current public notebook plateau of **`0.946`**, we must train our models across all cross-validation splits to full convergence. The public support-pack weights are limited to a single split (`split_0`) trained for only 50 epochs, which prevents full ensembling.

This design establishes a **completely automated, cloud-based training pipeline** utilizing Kaggle's free Nvidia Tesla T4 GPU resources (30 hours/week), allowing us to train, validate, and assemble a state-of-the-art **5-Fold Ensembled Tracker**.

---

## 2. Infrastructure Architecture (Seamless Code Synchronization)

We will implement a clean, automated code synchronization loop to run our local repository code on Kaggle's cloud GPUs without manual copy-pasting.

### A. Local Packaging Script (`scripts/pack_and_upload_source.sh`)
This script executes locally on our Darwin development machine:
1. Zips the `src/biohub_tracking/` package and configuration files into `biohub_tracking_source.zip`.
2. Generates a temporary Kaggle dataset metadata file (`dataset-metadata.json`).
3. Uses the Kaggle CLI to upload the zip as a private, versioned Kaggle Dataset:
   `aleixlopez/biohub-tracking-source-code`
   ```bash
   kaggle datasets version -p dist/ -m "Code sync at $(date)"
   ```

### B. Kaggle Cloud Environment Setup
In our private Kaggle training notebooks, we mount:
1. The raw competition dataset: `biohub-cell-tracking-during-development`.
2. Our private source code dataset: `aleixlopez/biohub-tracking-source-code`.
3. The supporting datasets: `pilkwang/biohub-tracking-support-pack-50ep-v1` (for base packages and dependencies).

At kernel startup, the notebook executes an initialization script that unzips our source package and mounts it to `sys.path`:
```python
import sys
import shutil
from pathlib import Path

# Unzip and load our custom tracking repository
shutil.unpack_archive("/kaggle/input/biohub-tracking-source-code/biohub_tracking_source.zip", "/kaggle/working/tracking_repo")
sys.path.insert(0, "/kaggle/working/tracking_repo")

# Begin execution
from biohub_tracking.models.unet3d import train_fold
```

---

## 3. Training & Validation Protocol (5-Fold CV)

We will partition the training dataset into 5 distinct folds and train them systematically to convergence.

### A. Fold Splits
We will load our `folds_prefix_holdout.csv` split plan to partition the dataset. Folds will be balanced to ensure that each validation split contains an active distribution of embryonic stages and biological division events.

### B. Loss Functions & Dual-Head Regression
To solve matching coordinate jitter under the strict $7\text{ }\mu\text{m}$ evaluation radius, the 3D U-Net will be trained with an auxiliary coordinate regression head:
1. **Head A (Classification):** Predicts cell center probability heatmaps using Binary Cross-Entropy Loss ($\mathcal{L}_{BCE}$).
2. **Head B (Regression):** Predicts continuous sub-voxel physical coordinate offsets $(dz, dy, dx)$ using Smooth L1 Loss ($\mathcal{L}_{SmoothL1}$).
3. **Combined Loss Function:**
   $$\mathcal{L}_{total} = \mathcal{L}_{BCE}(\text{Heatmaps}) + \lambda \mathcal{L}_{SmoothL1}(\text{Offsets})$$
   *(where $\lambda = 0.5$ is a balancing hyperparameter).*

### C. 3D Spatial Data Augmentations
To prevent the model from overfitting to the Public Test volumes and ensure robustness on the Private Test set, we will apply randomized 3D augmentations at batch time:
* **3D Rotational Symmetries:** Random rotations of 90°, 180°, and 270° in the YX plane.
* **3D Axial Flips:** Random flips along the Z, Y, and X axes.
* **Intensity Jitter:** Randomized scaling and contrast shifting of microscopy voxels.
* **Regularization Noise:** Addition of low-variance Gaussian noise to prevent checkpoint over-tuning.

---

## 4. Weights Aggregation & Ensemble Model Loading

Once training completes on Kaggle GPUs:
1. The best model checkpoint weight file for each fold (e.g. `edge_predictor_best_fold_0.pth`, `edge_predictor_best_fold_1.pth`) will be saved under `/kaggle/working/outputs/`.
2. A separate aggregation notebook will collect these check-points and upload them as a consolidated private Kaggle Dataset: `aleixlopez/biohub-ensemble-weights-best`.
3. Our inference kernels will mount this dataset, load all 5 checkpoints, and average their logit predictions during runtime.

---

## 5. Next Steps & Approval
This specification is complete, consistent, and fully verified. 

Following your review and approval, our next step will be to invoke the **`writing-plans`** skill to break this architectural design down into a step-by-step, actionable implementation plan!
