# Biohub 0.883 LB Baseline Review

Review and documentation of our fork of the `amanatar/biohub-v6-ultra-best` baseline (LB 0.883, clean/exploit-free).

## Core Insights & Methodology

This candidate represents a complete, clean, and exploit-free reproduction of the highly competitive **TemporalUNet3D + SimpleNodeTransformer + tracksdata ILP** pipeline. 

### Why it scores 0.883 instead of 0.942+ (Reproducibility Ceiling)
Although the original author `amanatar` sits high on the leaderboard with a `0.942` Public LB score, our clean replication of their exact code and weights scores **`0.883`** (matching the `0.882` displayed on the original author's Kaggle notebook run).

We identified the following root causes for this difference:
1. **Incomplete Shared Model Weights:** The public support-pack dataset `pilkwang/biohub-tracking-support-pack-50ep-v1` only ships with **one checkpoint split (`split_0`)**, containing 3 weight files.
2. **Ensembling Disabled:** The notebook's ensembling logic expects multiple splits (e.g., `split_0`, `split_1`, etc.) to run multi-checkpoint ensembled inference. Finding only `split_0` in the directory, the pipeline silently falls back to **single-model inference** (`Ensemble mode: False` is printed in the kernel log).
3. **Training Discrepancy:** The shared pack is a reduced "50-epoch" snapshot of the weights. The author's peak leaderboard submission utilizes their private, un-shared weights trained over more splits and epochs. 

---

## Identification & Stripping of the Division Exploit

During our review of the public notebook family under `research/kaggle_notebooks/`, we discovered that all six downloaded notebooks ended with an exploit cell (`augment_dataset` / `MAX_COMPONENTS` / `FORKS`) that injected synthetic nodes and edges with impossible out-of-bounds coordinates (e.g., `t=-1000, z=-10000`) into the final `submission.csv`. 

### The Exploit Mechanism
These synthetic forks were designed to merge separate connected components into a single massive connected component far outside the imaging volume. Against the *pre-patch* grader (weakly connected division-matching), this registered as fake division true positives, inflating the final score by `+0.08` to `+0.10` division-Jaccard points.

### The Grader Patch (July 2026)
Organizers recognized and patched the grader on July 22, 2026, requiring divisions to be strongly-connected parent-to-two-daughters structures matched within the real $7\text{ }\mu\text{m}$ volume. Leftover exploit nodes now register as false positives and **hurt** final scores. 

### Our Clean Resolution
We completely stripped the exploit-injection cells from our `biohub-v6-ultra-best-fork` codebase. The notebook originally produced a clean intermediate file called `submission_clean.csv` in its preceding cells. We replaced the exploit step with a clean no-op that promotes `submission_clean.csv` directly, adding an in-notebook assertion checking for `0 exploit rows` (no coordinates $\le -9000$ or $t \le -900$). 

* **Clean Submission rows:** 259,564 rows (132,482 nodes / 127,082 edges), **0 exploit rows**.
* **Kaggle Submission:** ID `56120358` scored **`0.883`**, demonstrating a 100% genuine deep learning tracking baseline score.
