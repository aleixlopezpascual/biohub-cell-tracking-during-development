# Agentic Competitive Machine Learning: Post-Mortem Introspection & The Winning Playbook

**Authors:** Aleix López Pascual & AI Research Assistant  
**Context:** Biohub - Cell Tracking During Development (Kaggle) & Cross-Competition Blueprint  
**Final Competition Result:** Rank 653 of 4,017 teams (0.956 Public LB, 0.919 Private LB)  
**Purpose:** An exhaustive, permanent guide analyzing the root causes of why we missed the gold tier, how to fundamentally re-architect AI agent deployment in competitive machine learning, and an actionable 5-phase execution framework for all future competitions.

---

## 1. Executive Summary: The Gap Between Top 16% and Gold

In the Biohub competition, our software engineering was world-class:
- Clean, modular `src/biohub_tracking` architecture with 100% type hints.
- 223 unit tests (100% green).
- Full provenance tracking across every Kaggle submission and sidecar.
- Custom neural innovations: **DivNet 3D-CNN mitosis recovery**, **collective neighborhood tissue flow**, and **ILP division weight tuning** that drove our score from `0.251` to **`0.956` Public LB** and **`0.919` Private LB**.

Yet, the podium finishers achieved **`0.967` Private LB** (3rd Place) and **`0.946`** (12th Place). 

The difference was **not** model tuning, ensembling, or compute limits. It was driven by:
1. **Relying on public baseline templates** rather than building a clean first-principles training pipeline.
2. **Missing basic data format forensics** (such as integer coordinate discretization and embryo-specific Z-offsets).
3. **Failing to diagnose physical sensor/imaging artifacts** (rigid global microscope stage drift).
4. **Failing to solve the mathematical loss formulation for sparse labels** (using DoG candidate exclusion masks).
5. **Using AI agents as code implementers and patchers** rather than exploratory data scientists and historical meta-researchers.

---

## 2. Root-Cause Analysis: The 5 Decisive Blind Spots

### Blind Spot 1: The "Public Baseline Tuning" Trap vs. First-Principles Training
* **What We Did:** We inherited public notebooks (`amanatar/biohub-v6-ultra-best`, `flexonafft/0.946`, `v1284_head`) as our primary experimental vehicle. We treated their pretrained model weights as fixed constants and tried to extract winning margins through post-processing, ensembling, and heuristic patches.
* **What Winners Did:** The 3rd place team (`yu4u & ren4yu`) completely ignored the public baseline family. They trained an ensemble of **2.5D U-Nets (EfficientNetV2-L, EfficientNet-B7) and 3D SegResNet** from scratch, reaching a raw **`0.923` Edge Jaccard before any ILP or post-processing**.
* **The Reality:** In dense Kaggle competitions, public notebooks converge to a crowded local optimum (the "0.946–0.956 plateau"). Winning solutions almost universally originate from clean, decoupled, first-principles modeling.

### Blind Spot 2: Missing Ground-Truth Quirks (Inadequate Label Forensics)
* **What We Did:** We assumed the ground-truth cell annotations represented continuous physical coordinates in 3D space. We spent substantial effort developing continuous sub-voxel coordinate regression heads (`v1284_head.pt`) and trilinear feature gatherers.
* **What Winners Discovered:**
  1. **Integer Voxel Grid:** Ground-truth annotations were strictly **integer voxel indices**. Submitting continuous floating-point predictions introduced sub-voxel Euclidean errors against the strict $7\,\mu\text{m}$ evaluation sphere. Simply rounding predicted coordinates to integers yielded an immediate **$+0.008$ score boost on Public LB (`0.942` to `0.950`) and $+0.005$ on Private LB**.
  2. **Embryo Z-Offset:** Annotators had an embryo-specific systematic bias: embryo `6bba` annotations sat $+0.675$ planes above the true optical nucleus centroid, while `44b6` sat $+0.115$ planes above. Applying a 1-plane Z-lift to candidate nodes produced an immediate **$+0.004$ Private LB boost**.
* **The Reality:** Sophisticated modeling cannot compensate for fundamental misalignments with the ground-truth annotation protocol. Label forensics must precede model development.

### Blind Spot 3: Failure to Diagnose Physical Imaging Artifacts (Camera Drift)
* **What We Did:** We observed that cell trajectories were noisy and modeled local motion using a $k$-NN collective tissue flow ($k=16$ within $48\,\mu\text{m}$).
* **What Was Actually Happening:** In embryo `6bba` (spanning 432 frames across 98 videos), the physical microscope stage or specimen drifted abruptly by **$8\text{ to }9\,\mu\text{m}$ (and up to $24\,\mu\text{m}$)** between consecutive frames.
* **The Consequence:** Because standard Hungarian and ILP search radiuses were gated at $5.5\text{--}7.0\,\mu\text{m}$, every single cell in those drifted frames jumped outside the matching gate simultaneously, causing **16% to 50% of ground-truth edges to be lost instantly**.
* **The Fix:** The 361st place team discovered this simply by computing the median $(\Delta z, \Delta y, \Delta x)$ displacement across all detected centroids in every frame. Subtracting this rigid global shift before tracking recovered nearly all lost edges and jumped them **+183 places** on the Private Leaderboard. Local $k$-NN could not solve this because candidate neighbors were also thrown outside the search radius.

### Blind Spot 4: Not Solving the "Sparse Annotation Loss" Problem
* **What Happened:** Annotations covered only ~2.8% of total cells. Training a standard segmentation or heatmap model with MSE or BCE on raw images heavily penalized detectors on true unannotated cells, treating them as false negatives and capping base detector recall.
* **What 3rd Place Did:** They ran an unsupervised Difference of Gaussians (DoG) peak finder with a permissive threshold. All regions within $6\,\mu\text{m}$ of unmatched candidates were **completely masked out from the loss calculation**. The network only learned from confirmed positive centroids and confirmed void background.

### Blind Spot 5: Selection Bias: Public LB vs. Biological Generalization
* **What Happened:** Our unselected **Active Mitosis Recovery** candidate (`56639822`) achieved our highest Private Score of **`0.919`**, proving the superior generalization of neural cytokinesis rescue. However, we selected Candidate `56642184` because it showed a higher Public Score of **`0.956`** (which degraded to `0.917` on Private).
* **The Reality:** The Public Leaderboard was only 29% of the test set. Active mitosis recovery was biologically sound, reduced structural division errors, and had solid OOF validation backing; we should have trusted biological invariants over public sample variance.

---

## 3. Re-Architecting AI Agent Orchestration for Competitive ML

Throughout this project, we directed our AI agents primarily as **software engineers and code executors** (writing boilerplate, scaffolding classes, refactoring imports, fixing type errors, and building offline packages). 

**In competitive machine learning, the true leverage of AI agents is not code writing; it is exploratory data analysis, historical literature synthesis, and error forensics.**

### The 4 Specialized Agent Archetypes to Deploy:

```
                  ┌─────────────────────────────────────────┐
                  │        HUMAN LEAD / ORCHESTRATOR        │
                  └───────────────────┬─────────────────────┘
                                      │
       ┌──────────────────┬───────────┴───────────┬──────────────────┐
       ▼                  ▼                       ▼                  ▼
┌──────────────┐   ┌──────────────┐        ┌──────────────┐   ┌──────────────┐
│  ARCHETYPE 1 │   │  ARCHETYPE 2 │        │  ARCHETYPE 3 │   │  ARCHETYPE 4 │
│  Historical  │   │   Forensic   │        │   Residual   │   │  Autonomous  │
│    Miner     │   │     Data     │        │  Detective   │   │   Parallel   │
│   (Day 1)    │   │   Auditor    │        │  (Mid-Game)  │   │  Subagents   │
└──────────────┘   └──────────────┘        └──────────────┘   └──────────────┘
```

#### Archetype 1: The "Historical Competition Miner" (Day 1 Mandatory Task)
* **Mission:** Scan and synthesize top-performing architectures, loss functions, and post-processing tricks from the 3 to 5 most relevant past competitions on Kaggle, CVPR, or NeurIPS.
* **Example Directive:**
  > *"Analyze the top 5 solutions from Kaggle CZII CryoET, Sartorius Cell Instance Segmentation, and the Cell Tracking Challenge. What backbones won? Did they use 2D, 2.5D, or 3D? How did they handle sparse ground-truth annotations? What optimization solvers did they use for graph linking? Synthesize a 1-page architectural matrix."*
* **What We Missed:** The 3rd place team in Biohub literally ported their winning 2.5D slice-stacking U-Net architecture directly from the **CZII CryoET** competition. If an agent had mined CryoET on Day 1, we would have had that backbone on our roadmap from week one.

#### Archetype 2: The "Forensic Data Auditor" (Days 1–3)
* **Mission:** Exhaustively interrogate raw data formats, ground-truth label distributions, sensor dynamics, and coordinate conventions before writing a single modeling line.
* **Example Directive:**
  > *"Audit the training and test labels exhaustively. Are coordinates floating-point or integer? What is the histogram of nearest-neighbor distances? Are there temporal discontinuities or sudden frame-to-frame median displacement jumps? Group the annotations by metadata/patient/embryo and check if coordinate centroids or Z-planes differ systematically."*
* **What We Missed:** The agent would have immediately surfaced: *"Ground truth is pure integer voxel indices, embryo 6bba has a +0.675 Z-bias, and 432 frames experience global translations exceeding 8µm."* That is a **+0.020 LB lift** discovered before training a single neural network.

#### Archetype 3: The "Residual & Error Detective" (Mid-Game & Validation)
* **Mission:** Perform forensic autopsies on failed validation samples. Never stop at aggregate summary metrics (e.g., "Edge Jaccard = 0.912").
* **Example Directive:**
  > *"Isolate the bottom 10% worst-performing validation videos. In those videos, plot the spatial error distribution and frame-by-frame recall curves. What exactly happens on frames where recall drops below 0.80? Are cells disappearing, moving too fast, or is the whole frame shifting?"*
* **What We Missed:** This directive would have directly exposed the microscope stage drift artifact, prompting an immediate median-shift subtraction filter.

#### Archetype 4: Autonomous Parallel Subagents (Exploration Mode)
* **Mission:** Avoid serializing speculative research in the main session loop. Spin up parallel subagents in isolated git worktrees with strict, measurable pass/fail gates.
* **Setup:**
  - **Subagent A:** Implements and tests sparse-loss DoG masking against a held-out slice.
  - **Subagent B:** Formulates a Mixed-Integer Linear Program (MILP) in HiGHS.
  - **Subagent C:** Tests integer rounding and coordinate snapping.

---

## 4. The 5-Phase End-to-End Competitive ML Playbook

Use this framework as the master standard operating procedure for every future competitive machine learning campaign.

```
PHASE 0: Domain Mining & Knowledge Synthesis
   │  ▶ Mine 3-5 similar historical competitions.
   │  ▶ Identify dominant backbones, loss functions, and post-processing paradigms.
   ▼
PHASE 1: Forensic Data & Metric Asymmetry Audit
   │  ▶ Audit label data types (integer vs float, bounding box formats, coordinate origins).
   │  ▶ Detect sensor/acquisition artifacts (stage drift, blur, aspect ratio variations).
   │  ▶ Mathematically model the evaluation metric to find penalty asymmetries (e.g. false node penalties).
   ▼
PHASE 2: Leak-Free, Ultra-Fast Local Validation Proxy
   │  ▶ Construct group/patient-disjoint CV splits (5-fold minimum).
   │  ▶ Build a 2-minute validation sub-sample for rapid hypothesis testing.
   │  ▶ Gate all candidate promotions on fold-disjoint local evaluation, never Public LB.
   ▼
PHASE 3: The "First-Principles" Modeling Spine
   │  ▶ Train at least one clean, independent multi-backbone ensemble from scratch.
   │  ▶ Implement domain-specific loss formulations (e.g. background exclusion masks for sparse data).
   │  ▶ Use public baselines strictly as external benchmarks or feature extractors, not the sole spine.
   ▼
PHASE 4: Systematic Error Decomposition & Residual Hunting
   │  ▶ Perform automated failure-case audits on the worst 5% validation samples.
   │  ▶ Address failure modes with targeted domain modules (e.g., drift correction, cytokinesis rescue).
   ▼
PHASE 5: Submission Portfolio & Anti-Overfitting Selection
   │  ▶ Strict rule: Never select two submissions that rely on the same public baseline knob.
   │  ▶ Slot 1: Highest public LB candidate (if validated locally).
   │  ▶ Slot 2: Highest CV / biologically sound candidate with proven out-of-fold generalization.
```

---

## 5. Ready-to-Use Agent Prompt Templates

Copy and adapt these prompt templates at the start of future competitions:

### Template 1: Historical Competition Mining (Day 1)
```text
@generalist You are an elite Kaggle Grandmaster research assistant. 
Our new competition is [COMPETITION_NAME], where the task is [BRIEF_DESCRIPTION] evaluated on [EVALUATION_METRIC].

Your objective is to perform a comprehensive historical literature and Kaggle solution review:
1. Identify the 3-5 most relevant past Kaggle or academic competitions with similar data modalities, label sparsity, or evaluation metrics.
2. For each competition, extract the top 3 winning solutions:
   - What model architectures (2D, 2.5D, 3D, Transformers, GNNs) were used?
   - What custom loss functions addressed class imbalance or sparse annotations?
   - What optimization algorithms or post-processing stages gave the decisive edge?
3. Synthesize your findings into a 1-page 'Winning Design Blueprint' recommending backbones, loss formulations, and validation strategies for our competition.
```

### Template 2: Forensic Data & Metric Audit (Day 1–3)
```text
@generalist You are a forensic data auditor. We are starting [COMPETITION_NAME].
Before we write any model code, you must execute a forensic audit of the dataset:
1. Examine raw ground-truth annotations:
   - Check exact data types, precision (floats vs integers), coordinate conventions, and boundaries.
   - Group labels by metadata (patient, video, scanner, embryo) and check if coordinate distributions or centroids shift systematically between groups.
2. Analyze physical and temporal dynamics:
   - Compute displacement histograms across consecutive frames/steps. Are there sudden outliers or global shifts across all objects?
3. Metric Asymmetry Analysis:
   - Write a mathematical breakdown of [EVALUATION_METRIC].
   - If we add 10 false positives vs miss 10 true positives, what is the relative score impact?
   - Deliver an executive report with explicit warnings on data quirks and metric traps.
```

### Template 3: Residual & Error Autopsy (Mid-Game)
```text
@generalist You are an error analysis specialist. 
We have generated out-of-fold predictions for candidate [CANDIDATE_NAME] on our validation split.
Run an exhaustive residual autopsy:
1. Isolate the bottom 10% worst-performing validation samples.
2. Break down the errors into discrete categories:
   - False Positives (over-detection) vs False Negatives (missed objects).
   - Identity swaps vs track fragmentations vs event misclassifications.
3. Correlate error spikes with physical/image attributes (contrast, density, blur, motion magnitude).
4. Propose 3 targeted, minimal algorithmic remedies that fix the primary error mode without degrading global precision.
```

---

## 6. Summary Conclusion

Our work in Biohub established a rock-solid engineering standard: 223 green tests, typed modularity, and reproducible submission ledgers. By integrating this retrospective into our core methodology—**auditing data formats first**, **mining historical winning playbooks on Day 1**, **training first-principles models with proper loss masks**, and **deploying AI agents as exploratory research scientists**—we enter our next competition equipped to win gold.
