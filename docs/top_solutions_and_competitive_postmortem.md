# Biohub Cell Tracking: Top Solutions, Meta Learnings & Comparative Post-Mortem

**Competition:** Biohub - Cell Tracking During Development (Kaggle)  
**Total Teams:** 4,017  
**Our Official Private Standing:** Rank 653 (Top 16.2%, Selected Private LB: 0.917)  
**Our Peak Private Score:** 0.919 Private LB (Submission `56639822`, Active Mitosis Recovery)  
**Our Public Peak:** Rank 376 (0.956 Public LB)  

---

## 1. Executive Summary & Overview

Following the close of the Kaggle **Biohub - Cell Tracking During Development** competition on September 29, 2026, the community and top medalists published their architectural writeups, code releases, and error analyses.

This document synthesizes:
1. **The winning architectures and post-mortems** (notably 3rd Place `yu4u & ren4yu`, 12th Place `Team Corwin`, 14th Place `Vibes&Edges`, and 361st Place `xthixsl_ml`).
2. **Universal domain learnings** (data quirks, sparse ground-truth loss handling, physical stage drift, and joint graph optimization).
3. **An honest, line-by-line comparative analysis** between the gold-medal solutions and our engineering pipeline.

---

## 2. In-Depth Review of Top Solutions

### 🥉 3rd Place Solution: 2.5D/3D Ensembles & Lineage Graph Optimization
* **Team:** `yu4u` & `ren4yu` (with `@minerppdy`, `@nazarov`, `@ren4yu`, `@theoviel`)
* **Scores:** CV: `0.9778` (Div Jaccard: `0.540`) | Public LB: `0.977` | **Private LB: `0.967`** (Div Jaccard: `0.47`)
* **Core Philosophy:** Rather than fine-tuning public templates, they built a modular 6-stage pipeline from first principles, optimizing for the sparse annotation structure and true biological invariants.

```
Raw OME-Zarr (T, Z, Y, X)
   │
   ├──▶ [Stage 1: Cell Detection (55% runtime)]
   │      - 2.5D U-Nets (EfficientNetV2-L + EfficientNet-B7; 3 z-slices as channels)
   │      - 3D MONAI SegResNet (Residual 3D UNet, 4 stages, GroupNorm)
   │      - Loss: Gaussian heatmap (σ=2 µm) + DoG candidate background exclusion zone (< 6 µm)
   │
   ├──▶ [Stage 2: Dense 3D Flow Field]
   │      - Compact 3D encoder-decoder on downsampled isotropic grid (1.625 µm spacing)
   │      - Warps frames for motion compensation and division alignment
   │
   ├──▶ [Stage 3: Cell Matching with Attention]
   │      - 2.5D U-Net (EfficientNetV2-S); 64-channel features per detection
   │      - Self- and cross-attention between consecutive frames + null matching head
   │
   ├──▶ [Stage 4: Flow-Aligned Division Identification]
   │      - Dense flow warps frame t+1 onto frame t to remove bulk tissue motion
   │      - Dual 3D CNN parent models isolate true mitotic morphology from spatial displacement
   │
   ├──▶ [Stage 5: Lineage Graph Optimization (Joint LP / MILP)]
   │      - Global HiGHS solver optimizing a surrogate of the competition metric
   │      - Jointly selects nodes, ordinary links, and division branches under strict invariants
   │
   └──▶ [Stage 6: Multi-Step Post-Processing]
          - 1, 2, 3-frame gap closing with distance relaxation (3√(g+1) µm)
          - Weak component pruning (< 6 nodes)
          - Trajectory affine smoothing
```

#### Key Technical Breakthroughs:
1. **Handling Sparse Ground Truth (~2.8% annotated nodes):**
   - Naive training with MSE/BCE on sparse labels forces the neural network to treat unannotated real cells as background (false negatives).
   - Solution: Extracted candidate cell peaks via **Difference of Gaussians (DoG)** with a permissive threshold. All regions within $6\,\mu\text{m}$ of unmatched candidates were **masked out from the loss calculation**. The network only learned from confirmed positive centroids and confirmed void background.
2. **Dense Flow Decoupling:**
   - Both cell migration and cell division alter local appearance. By predicting a 3D displacement field and warping frame $t+1$ onto frame $t$, the division classifier operated on stationary cells, vastly simplifying mitosis classification.
3. **HiGHS LP/MILP Metric Surrogate:**
   - Instead of greedy Hungarian assignment (which irrevocably commits links before division logic runs), they formulated the problem as a global Linear Program (LP) relaxed to Mixed-Integer Linear Program (MILP).
   - Invariants:
     - $\le 1$ incoming parent per node.
     - $\le 1$ outgoing event (ordinary edge OR division).
     - Divisions must select exactly two daughter cells simultaneously.
     - Dividing parents must have incoming tracks; daughters must continue forward.
     - Daughters cannot divide in the immediately consecutive frame.

---

### 🎖️ 12th Place Solution: Tuned Public Tracker + Measured Repair Stages
* **Team:** Team Corwin (`corwin1979` & `mathisbaguette`)
* **Scores:** Public LB: `0.970` | **Private LB: `0.946`**
* **Core Philosophy:** Iterated on the public 3D U-Net + ILP baseline family with rigorous forensic data analysis, discovered hidden dataset quirks, and implemented fail-safe repair stages.

#### Key Findings & Technical Innovations:
1. **Integer Voxel Ground Truth vs. Continuous Floats:**
   - Discovered that all ground-truth annotations are **strictly integer voxel indices**.
   - Submitting continuous sub-voxel float coordinates introduced small Euclidean discrepancies against the strict $7\,\mu\text{m}$ evaluation sphere.
   - **Rounding coordinates to integer voxels immediately boosted their score from `0.942` to `0.950` on Public LB (`0.906` to `0.911` on Private LB)** without modifying model weights.
2. **Embryo-Specific Z-Convention Shift:**
   - Forensically audited the annotations across different embryo specimens.
   - Discovered that in embryo `6bba`, annotators consistently placed ground-truth points $+0.675$ planes above the true optical nucleus centroid, whereas embryo `44b6` had only a $+0.115$ plane offset.
   - Applying a calibrated one-plane Z-lift to 23% of candidate nodes produced an immediate **$+0.004$ Private LB boost**.
3. **Four Fail-Safe Repair Stages:**
   - *Two-Pass Hungarian Relink:* First pass at $5.5\,\mu\text{m}$, second pass at $10.0\,\mu\text{m}$ for fast-moving cells.
   - *Multi-Step Gap Closing:* 1–2 frame track bridging ($5.8\,\mu\text{m}$ for 1-frame, $4.4\,\mu\text{m}$ per step for 2-frame).
   - *Geometric Mitosis Grafts:* Mothers within $9.0\,\mu\text{m}$, sisters within $14.0\,\mu\text{m}$.
   - *Short-Component Pruning:* Removed weak components with $< 6$ nodes.
   - *Line-Fit Trajectory Smoothing:* 5-frame moving linear fit ($0.2p + 0.8x$).

---

### 🎖️ 14th Place Solution: Vibes & Edges Trade-Off
* **Team:** Tom, Taha_Alshatiri, I2nfinit3y, try, & Dorian Ren
* **Key Indicators:** `merge safe tko o12 ilpdiv04 readmit094 biohub`
* **Core Philosophy:** Focused heavily on balancing the edge precision penalty against division recovery in the ILP solver.
  - Lowered the global solver's division penalty to **`ILP_DIVISION_WEIGHT = 0.4`**.
  - Implemented a **`readmit = 0.94`** threshold to re-introduce nodes that passed geometric and tracking confidence gates.
  - Demonstrated that overly aggressive post-processing filters accidentally pruned true daughters of ground-truth divisions, hurting the $0.1 \times \text{Division Jaccard}$ term.

---

### 💡 361st Place Solution: Discovery of Global Camera/Stage Drift
* **Author:** `xthixsl_ml` (レオナ)
* **Scores:** Public LB: `0.954` | **Private LB: `0.920` (Bronze Medal)**
* **Core Discovery:** A major physical imaging artifact that corrupted standard tracking algorithms:
  - In embryo `6bba` (spanning 432 frames across 98 videos), the physical microscope stage or specimen drifted abruptly by **$8\text{ to }9\,\mu\text{m}$ (and up to $24\,\mu\text{m}$)** between consecutive frames.
  - Because standard Hungarian and ILP search radiuses were gated at $5.5\text{--}7.0\,\mu\text{m}$, every single cell in these shifted frames fell outside the matching gate, causing **16% to 50% of ground-truth edges to be dropped instantly**.
  - **The Fix:** Calculated the median $(\Delta z, \Delta y, \Delta x)$ displacement across all detected centroids between frame $t$ and $t+1$. If the median jump exceeded $5.0\,\mu\text{m}$, they subtracted this rigid global translation before computing pairwise distances.
  - This recovered nearly all lost edges without creating any extra nodes, producing a **+183 rank jump** on the Private Leaderboard.

---

## 3. Universal Meta Learnings

| Domain | Key Finding | Common Pitfall / Failure Mode |
|---|---|---|
| **Data Format** | Ground-truth coordinates are integer voxel indices. Submitting floats incurred a systematic distance penalty against the 7 µm gate. | Submitting raw continuous float regressions without rounding to integers cost ~0.005–0.008 LB. |
| **Physical Dynamics** | Zebrafish light-sheet imaging suffers from rigid global microscope stage drift up to $24\,\mu\text{m}$. | Local or distance-gated tracking models broke down entirely on drift frames without global frame alignment. |
| **Sparse Supervision** | GT only labels ~2.8% of nodes. True unannotated cells must be masked out using Difference of Gaussians (DoG) candidate exclusion. | Standard MSE/BCE loss on raw images heavily penalized detectors on true unannotated cells. |
| **Graph Optimization** | Joint LP/MILP optimization (HiGHS) enforcing biological invariants simultaneously is vastly superior to sequential heuristics. | Greedy Hungarian linking stole daughter cells before division recovery could evaluate them. |
| **Metric Sensitivity** | The Adjusted Edge Jaccard scales strictly with $N_{\text{pred}} / N_{\text{true}}$. Every false node penalizes the denominator across all edges. | Model-union ensembling (`torch.maximum`) or lowering detection thresholds caused catastrophic score collapse. |
| **Gap Bridging** | Multi-frame gap closing ($g \in \{1, 2, 3\}$ frames) with $3\sqrt{g+1}\,\mu\text{m}$ distance scaling recovers cells lost during mitosis or focal blur. | Hard 1-frame linking left fragmented tracks that were subsequently deleted by minimum-length filters. |

---

## 4. Line-by-Line Comparative Analysis: Our Solution vs. Top Solutions

### Where Our Solution Matched and Anticipated Top-Tier Work

| Innovation | Our Implementation | Top Solutions Implementation | Alignment Verdict |
|---|---|---|---|
| **Neural Mitosis Rescue** | **DivNet 3D-CNN:** Trained patch classifier to actively rescue asymmetric cytokinesis splits ($\tau \le 0.80$). Yielded our top **0.919 Private LB** (`56639822`). | 3rd place flow-aligned parent model; 14th place DivNet cytokinesis rescue. | **Perfect Alignment.** Both recognized that Division Jaccard was the decisive margin for Private LB rank. |
| **ILP Division Cost** | Lowered solver division weight to **`0.4`** and added **`readmit = 0.94`** in `0.959 Frontier SOTA` (`56640857`). | 14th place (`ilpdiv04 readmit094`); 3rd place HiGHS relaxed division cost. | **Direct Match.** Adopted identical optimal mathematical parameterization. |
| **Tissue Motion Priors** | **Collective Neighborhood Flow:** Computed median drift velocity of $k=12\text{--}16$ nearest neighbors within $48\,\mu\text{m}$ in `Density Calibrated Flow` (`0.956` Public / `0.917` Private). | 3rd place 3D Dense Flow; 361st place median frame displacement. | **Strong Conceptual Match.** Correctly identified that tissue movement is collective rather than individual. |
| **Denominator Protection** | Maintained high detection threshold ($0.965$), pruned isolated single-frame blips, and strictly rejected `torch.maximum` union ensembling. | 3rd, 12th, and 14th places all focused on keeping $N_{\text{pred}}$ tight and pruning weak nodes. | **Complete Agreement.** Avoided the severe public-metric overfitting traps that ruined hundreds of teams. |

---

### Where Our Solution Diverged and Missed the Winning Edge

| Dimension | Our Solution | Top Solutions | Score Impact & Takeaway |
|---|---|---|---|
| **Coordinate Discretization** | Used continuous sub-voxel coordinate regression (`v1284_head.pt`) and emitted continuous floating-point coordinates. | Discovered GT coordinates were strictly **integer voxel indices** and rounded predictions to integer voxels. | **Missed ~0.005–0.008 LB.** Continuous regression moved points in float space, but rounding to integer voxels was required to align with the discrete evaluation grid. |
| **Microscope Stage Drift** | Modeled local neighborhood flow ($k$-NN within $48\,\mu\text{m}$). | Detected **rigid whole-frame stage drift** (median displacement across entire frame) and shifted coordinates prior to gating. | **Missed 15%–35% of edges on shifted frames in `6bba`.** Local $k$-NN could not bridge sudden $24\,\mu\text{m}$ jumps because candidate neighbors were also thrown outside the search radius. |
| **Embryo Z-Convention Shift** | Assumed identical coordinate conventions across all embryos. | Discovered `6bba` annotations were placed $+0.675$ planes higher than optical center; applied 1-plane Z-lift. | **Missed ~0.004 Private LB** from systematic biological annotation bias. |
| **Sparse Loss Formulation** | Adapted public pre-trained checkpoints (`v6-ultra-best`, `v1284`) and tuned post-processing. | Trained 2.5D EfficientNet-B7/L + 3D SegResNet ensembles from scratch with **DoG candidate exclusion masks**. | **Higher detection ceiling.** Top teams reached $0.923$ raw Edge Jaccard before ILP by eliminating false-negative penalties during training. |
| **Graph Optimization Architecture** | GEFF/tracksdata ILP solver with downstream heuristic patch scripts (DivNet post-grafting). | **Unified LP/MILP (HiGHS)** that jointly solved ordinary edges and division branches under mathematical constraints. | **Prevented link stealing.** Joint optimization ensured greedy Hungarian links did not steal daughter cells before division validation. |
| **Gap Closing Depth** | Primarily 1-frame linking with momentum relinking. | Multi-step 1-, 2-, and 3-frame gap closing with distance relaxation ($3\sqrt{g+1}\,\mu\text{m}$) and linear node interpolation. | **Track continuity.** Bridged cells that dropped below detection threshold during mitosis or optical blur (+0.005 Jaccard). |
| **Final Selection Choice** | Selected Candidate `56642184` (Density Calibrated Flow, Private `0.917`) instead of Candidate `56639822` (Active Mitosis Recovery, Private `0.919`). | Selected division-heavy, robust generalization candidates. | Candidate `56639822` achieved **`0.919` Private LB**, landing well within the top tier. |

---

## 5. Hardware, Compute Environments & LLM Tooling

### Did Top Teams Use LLMs to Code?
* **Tactical Assistance vs. Autonomous Agents:** In their official post-mortems, none of the top-performing teams credited LLMs or autonomous agents as core methodological drivers of their solutions. Kaggle Grandmasters (such as `yu4u` and `@theoviel`) possess mature, proprietary computer-vision codebases developed across years of competitions (e.g., directly porting their winning 2.5D slice-stacking backbone from *CZII CryoET*).
* **Community Discussions:** LLMs (ChatGPT, Claude, Cursor) were widely used interactively across the forum as tactical co-pilots for writing boilerplate, generating vectorized NumPy/SciPy snippets, and debugging complex loss formulations. For example, Kaggle Grandmaster `hengck23` explicitly advised competitors in a thread on sparse cell annotations:
  > *"At each pixel location after UNet logit head, loss = softmax of pixel over its neighbors... Hint: ask ChatGPT to write a margin loss version: peak is at least $T$ greater than neighbor."*
* **The Bottom Line:** While LLMs accelerated implementation speed, **no autonomous coding agent discovered the winning competitive edge**. The decisive leaps—uncovering integer voxel discretization, formulating the DoG background exclusion mask, discovering embryo-specific Z-annotation shifts, and diagnosing physical microscope stage drift—required human exploratory data analysis (EDA) and biological intuition.

---

### Development Setups & Engineering Stacks

1. **The "From-Scratch" Deep Learning Stack (3rd Place — `yu4u & ren4yu`):**
   * **Frameworks:** PyTorch, MONAI (`SegResNet`), and torchvision.
   * **Mathematical Optimization:** Replaced legacy heuristic trackers with modern mathematical programming solvers (**HiGHS**), formulating graph tracking as a unified Mixed-Integer Linear Program (MILP) with hard biological constraints.
   * **Validation Engine:** Strict 5-fold cross-validation grouped by embryo prefix to ensure zero data leakage across videos.

2. **The "Hybrid Tuned + Tabular/CNN Repair" Stack (12th Place — `Team Corwin`):**
   * **Hybrid Tooling:** Rather than training heavy backbones, they froze public 3D U-Net checkpoints and layered **25 LightGBM boosters** (trained on 70 geometric/temporal features) and lightweight PyTorch CNNs (a 2.8M parameter mitosis reader and a 3D coordinate-offset regression CNN).
   * **Production Defensiveness:** Rigorous runtime hardening—including per-video timers, SHA-256 weight checksums, offline packaging of 32 wheels, and an immediate $t+0$ emergency submission writer to prevent Kaggle test-set timeouts.

---

### The Compute Reality: External GPUs vs. Kaggle T4s

* **Training Required External GPUs:**
  * Kaggle provides a free weekly quota of **30 hours on 2× NVIDIA T4 GPUs** (16 GB VRAM each).
  * For 4D Zarr microscopy volumes, this hardware was insufficient for full training. As Team Corwin documented, training a single 3D U-Net on 2× T4 GPUs took **35 to 40 minutes per epoch** and suffered frequent Out-Of-Memory (OOM) errors at batch size 16.
  * Training 5 folds of EfficientNetV2-L, 5 folds of EfficientNet-B7, and 5 folds of 3D SegResNet (plus dense flow and attention matching models, as 3rd place did) required **hundreds of GPU-hours on 24GB+ VRAM hardware** (e.g., local multi-RTX 3090/4090s, A6000s, or cloud instances on Lambda/RunPod/GCP).
  * Team Corwin explicitly noted that they developed and benchmarked locally on an **NVIDIA RTX 2000 Ada Generation** workstation GPU before scaling their pipeline to Kaggle.
* **Inference Was Strictly Constrained to Kaggle T4s:**
  * Because Biohub was an offline code competition, all models had to execute on the hidden private test set within Kaggle's **2× T4 GPUs under a 12-hour hard timeout**.
  * Top teams engineered aggressive inference optimizations: 3rd place downsampled coordinate grids to $1.625\,\mu\text{m}$ for flow and attention matching, spending ~55% of their total runtime on detection, while Team Corwin executed their entire 6-stage repair chain in **6.8 to 8.4 hours**.

---

## 6. Key Architecture Blueprints for Future 3D/4D Biomedical Challenges

When tackling future 3D+time microscopy or cell lineage competitions, the definitive design pattern is:

1. **Audit Annotation Discretization First:** Always verify whether ground truth annotations are continuous floats or discretized voxel indices. If ground truth is integer-based, round all model outputs.
2. **Check for Global Frame Drift Early:** Compute frame-to-frame median centroid displacement before applying any local search radius. If global drift exceeds the tracking threshold, subtract it rigidly per frame.
3. **Train with Sparse Exclusion Masks:** When annotations cover $< 5\%$ of objects, run an unsupervised blob/peak finder (DoG) and mask out the neighborhood of all unannotated candidates during loss calculation.
4. **Decouple Motion from Morphology:** Predict a dense 3D displacement field to align adjacent frames. Pass motion-compensated crops to downstream event classifiers (mitosis, apoptosis).
5. **Joint MILP over Heuristic Grafting:** Avoid sequential pipelines where greedy matching locks in decisions. Formulate graph tracking as a unified Mixed-Integer Linear Program with hard biological invariants.
6. **Protect the Metric Denominator:** In precision/Jaccard metrics with count penalties, prioritize re-wiring existing edges and pruning low-confidence singletons over adding new candidate nodes.
