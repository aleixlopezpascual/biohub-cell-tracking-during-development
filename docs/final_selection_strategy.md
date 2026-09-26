# Final Submission Selection Strategy

Strategic guide and risk assessment for selecting up to two final submissions for the Kaggle Biohub Cell Tracking leaderboard.

> **Current status (2026-09-26):** The live submission query still shows `56132481` at `0.946`; its rank/medal context below is a dated September 25 snapshot, not current standings. The staged `0.948` candidate is unscored, and the Candidate 2 parameter changes below are historical hypotheses—not a ready candidate. Keep only the proven anchor unless a distinct candidate passes fold-disjoint validation. Current blockers, GPU needs, and approval gates are in [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md); refresh the leaderboard before making current-rank claims.

---

## 1. The Up-to-Two-Submission Rule
Kaggle allows each team to select **up to two (2) submissions** to be graded on the private test set; it is not necessary to fill both slots. The public leaderboard is only a partial test split, so prioritize robust validation over maximizing the number of finalists.

If we select two submissions, they should not be highly correlated candidates that share the same overfitting traps. Use a two-pronged strategy only when both candidates are validated; otherwise select only the best proven submission:
* **Submission A (Aggressive):** Optimized for the Public Test set (maximizing known Public LB score).
* **Submission B (Conservative):** Optimized for robust generalization on unseen Private Test biological volumes (minimizing shakeup risk).

---

## 2. Public Leaderboard Overfitting Risks (The Traps)

Our deep research into the baseline lineage has identified several parameters that are heavily over-tuned to the noise and biological densities of the Public Test set:

### Trap A: Rigid Fractional Division Caps (High Shakeup Risk)
The highest-scoring public notebooks utilize hard fractional caps to limit the number of cell divisions inserted during post-processing:
* `BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP = "0.00375"`
* `BIOHUB_SAFE_DIV_FRAME_FRAC_CAP = "0.0076"`

#### The Risk:
These caps are calibrated strictly to match the division density of the Public Test volumes. Cell division rates can vary significantly between different embryos, stages of development, or imaging sessions. If the Private Test volumes contain embryos with slightly higher cell division rates, these rigid caps will block genuine divisions, causing a massive penalty to our Division Jaccard score.

### Trap B: Over-Tuned Detection Thresholds
Our peak baseline tunes the `BIOHUB_DET_THRESHOLD` down to `0.96` (from the default `0.965`), giving a `+0.001` Public LB boost.
#### The Risk:
While `0.96` works beautifully for the contrast and lighting conditions of the Public Test volumes, the Private Test set may have slight differences in signal-to-noise ratio, background autofluorescence, or laser power. A threshold tuned to the third decimal place lacks the safety margin required to handle raw image intensity drift.

### Trap C: Unvalidated Inherited "Knobs"
An audit of the public pipeline variables revealed that out of 36 non-default parameters (e.g., `BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU = 0.6`, `BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN = 4`), **33 of them were never systematically validated**. They were simply stacked by consecutive notebook forks to fit Public LB noise.

---

## 3. Local Validator "Proxy" Limitations

The in-notebook proxy validator (which evaluates against 4 held-out training movies using the official Royerlab formula) is **not a reliable signal for micro-parameter tuning**:
* Lowering the detection threshold from `0.965 -> 0.96` **increased** Public LB score but **decreased** the local proxy score.
* The division term of the local proxy is pinned at `0.2000` in every sweep, meaning all validation movement comes from the edge term across only four movies.
* **The Rule:** Trust local validation for major architectural changes (e.g., validating that *Edge-Feature TTA* or *Bidirectional Association* actually improves metrics), but **do not** use it to validate micro-parameters (thresholds, post-processing weights), as the signal is highly noisy and anti-correlated with the leaderboard.

---

## 4. The Recommended Selection Protocol

When the competition deadline approaches, we should select our two candidates as follows:

### Candidate 1: The Public LB Peak (Submission ID: `56132481`)
* **Configuration:** `aleixlopez/biohub-0-946-edge-feature-tta-tuned` (Version 2)
* **Score:** **`0.946`** Public LB
* **Philosophy:** Stacks both Igor's Edge-Feature TTA and our optimal `0.96` threshold knob to secure our best possible score on the known test set.

### Candidate 2: Potential Private-LB Generalist (only if validated; Submission ID: `TBD`)
* **Configuration:** A modified `0.946` pipeline with **increased safety margins** to handle biological and optical drift:
  * The following settings are unvalidated hypotheses, not a ready-to-submit recommendation.
  * Raise the detection threshold slightly back to **`0.965`** to ensure robustness against lower-contrast images.
  * Loosen or completely disable the hard fractional division caps (`GLOBAL_FRAC_CAP` and `FRAME_FRAC_CAP`) to let the geometric tracking rules (`safe-div` parent-sister symmetry and forward divergence) decide division topology naturally.
  * Optionally reduce the secondary edge feature TTA weight slightly (`0.50` instead of `0.75`) to avoid ensembling over-reliance if the secondary model weights drift on private domains.
* **Philosophy:** Built to survive biological variation in unseen embryonic volumes and secure a high standing if the leaderboard undergoes a major private shakeup.
