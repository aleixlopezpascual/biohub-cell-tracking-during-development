# Biohub endgame experiment backlog

**Created:** 2026-09-25. **Research-snapshot deadline:** 2026-09-29 23:59 UTC (about 4 days 11 hours remained at that snapshot). This is an ordered experiment queue, not a claim that any unrun idea improves the score. No Kaggle kernel was run or submitted as part of creating it. **For live blockers, compute requirements, and the immediate action list, use [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md), the current source of truth.**

## Current constraints and target

- A read-only Kaggle submission query at `2026-09-26 11:58 CEST` still showed `0.946` (submission `56132481`) as the latest scored result; no `0.948` submission exists. The candidate kernel listing's `lastRunTime` is not proof of a successful run/output.
- GPU was previously reported unavailable; live quota/GPU availability was not verified. The metadata conflict and required T4 check are tracked in the current endgame status document.
- The project and known prior CV workspace searches found no local validation Zarrs or matching candidate OOF predictions/graphs/weights. Score comparisons and B1/B2 therefore remain blocked until valid artifacts are located or generated.
- Rank `1,221 / 3,899`, the approximate bronze boundary, and the hypothetical `0.948` rank are from the dated `2026-09-25` public-board snapshot only. Do not describe them as current standings; refresh the leaderboard before any live rank/medal claim.[1][4]
- The staged candidate's `0.948` name is not evidence of a score. A CPU structural validator and an optional Gold-model `--motion-relink` OOF path now exist, but no real EMA predictions or complete four-run evidence package have been produced. The validator checks metric arithmetic but always returns `promote=false` because schema v1 has no trusted runtime/scorer attestation; the notebook does not consume its structural report.
- Keep experiment outcomes in `results/local_cv/experiments.csv` and `results/kaggle_lb/submissions.csv`; this document is only the ordered idea backlog.

## Promotion gate for every candidate

1. One code or configuration change at a time. Do not use the public leaderboard as the inner-loop tuner.
2. Validate with fold/embryo-disjoint weights. Public support weights trained on all 199 annotated videos cannot validate on those same videos.[7]
3. Score with the current official patched metric. Log per-fold and per-video adjusted edge Jaccard, edge TP/FP/FN, division TP/FP/FN, `T_pred/T_true`, retained-node counts, and runtime. The metric combines adjusted-edge Jaccard with a 0.1-weighted division Jaccard; false forks, duplicate detections, and node-count shifts can outweigh tiny gains in one submetric.[5][6][9]
4. Proposed internal guard (not an official Kaggle rule): require a paired aggregate gain of at least `+0.001`, no held-out embryo drop beyond `0.003`, no material metric/output-integrity regression, and trusted runtime/scorer provenance before promoting. The current schema-v1 structural report cannot authorize promotion. If validation artifacts are unavailable, label the idea **untested**—do not infer a gain from its name or a notebook title.
5. Any Kaggle push/submission is a separate, consequential action and needs the user's explicit approval. Do not fill final-submission slots just because Kaggle permits up to two.[3]

## Ordered backlog

### B0 — Resolve and run the already-staged `0.948` inference candidate

**Priority:** P0 — unblock first; CPU preparation can proceed without GPU.<br>
**Status:** Static audit, read-only Kaggle metadata check, CPU EMA helper tests, the paired evidence validator, and an optional `scripts/run_oof_checkpoint.py --motion-relink` Gold-model ablation path are implemented. The new path binds per-dataset sidecars to its config, checkpoint, local code, upstream Python source tree, and GEFF hashes, but has not been run on real OOF artifacts. The relinker is one-to-one and omits the staged candidate's later division-repair step; it does not reproduce the exact `0.946` Kaggle pipeline or emit the validator's complete four-run `evidence.json`. The validator checks adjusted-edge arithmetic but always reports `promote=false` until a trusted runtime/scorer attestation path exists. The notebook names `bidirectional_blend_union13_receipt.json`, but it is not wired to the local report and no real evidence exists. This is an internal gate, not a Kaggle-issued file or rule. The private Kaggle copy last reports a run on 2026-09-24 but predates the local provenance correction; its outputs were unavailable, and there is no 0.948 submission record.
**Compute:** Provenance review and structural-report validation are CPU-only. Paired scoring is CPU-only once valid OOF outputs exist; generating missing neural predictions is likely GPU-dependent. Staged Kaggle inference requires a verified GPU/T4 runtime. See the current endgame status document for blockers and approval gates.
**Hypothesis:** EMA momentum relinking improves identity continuity on crossings and fast motion over the proven `0.946` candidate at low training cost. DeepCenter is already present in both notebooks; it is not the differentiating change.

**Work:**
- The structural report generator is implemented at `scripts/evaluate_promotion_gate.py`; it now validates adjusted-edge arithmetic but is deliberately non-promotable under schema v1 because hashes do not authenticate execution. The local OOF runner has an isolated `--motion-relink` mode that binds sidecars to checkpoint and upstream source hashes; it is documented in `docs/local_evaluation.md`. Remaining work is to design trusted runtime/scorer attestation, run both variants on real fold-excluded artifacts, assemble the complete four-run evidence package, and independently verify the report. This one-to-one Gold-model ablation omits the candidate's later division-repair stage and does not validate the exact `0.946` Kaggle pipeline. The staged notebook report only declares a receipt requirement; it does not generate, read, or validate the named file. The artifact is an internal quality gate, not a Kaggle-issued file.
- Locate or generate fold-excluded model outputs and their split/weight provenance. Public weights trained on the evaluation movies are not held-out evidence for those movies. If valid OOF predictions already exist, the comparison/scoring step can run on CPU; otherwise fresh neural inference is likely to need GPU.
- Do not substitute the separate `gold_oof_runner`'s `promotion.json`: that runner evaluates a different training candidate and does not validate this EMA change.
- Confirm the locally corrected provenance report matches the intended runtime config; reconcile sidecar/embedded GPU flags, primary/secondary weight hashes, DeepCenter checkpoint/epoch, active flags, and fail-closed fallback behavior before any Kaggle run. The Kaggle remote copy is stale and the referenced upstream source remains unverified.
- Only after validation/readiness and separate user approval, run hidden-test inference and inspect the submission schema/topology, runtime manifest, loaded artifacts, and fallback state. This GPU test-set inference is distinct from the fold-disjoint promotion evaluation.

**Pass:** The fold-disjoint paired evaluation meets the predeclared metric gate, an independent trusted attestation verifies runtime/scorer provenance, and the report is reviewed; then separately authorized GPU inference loads the intended artifacts with no fallback/exploit rows and produces a valid submission. The current schema-v1 report alone cannot pass this promotion condition. Log only a score actually read back from Kaggle. A `0.948`-named candidate is not presumed to score that value.

### B1 — Metric-aligned per-video node-count stability audit

**Priority:** P1 — best diagnostic before changing more knobs.<br>
**Status:** Blocked on fold-held-out prediction artifacts. The current notebook already contains node-ratio diagnostics; use those rather than building duplicate instrumentation.
**Compute:** CPU for paired per-video analysis when matching fold-held-out outputs and labels exist; GPU likely only if neural predictions must first be generated.
**Hypothesis:** Changes that subtly alter retained-node counts can move the score even when offline edge Jaccard improves. One competitor reported a `+0.0011` offline edge change becoming a `-0.005` public change, with node-count shifts implicated; another report says a four-node change on one video was enough to matter. These are cautionary observations, not a universal law. [Discussion #742266](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266)

**Work:** For the same held-out movies, compare `0.946` and `0.948` predictions per movie: predicted/estimated node ratio, unmatched duplicates, edge TP/FP/FN, divisions, and score. Check the official coarse `T_true` adjustment; do not force every movie to an estimated exact count.[5]

**Pass:** any adjustment improves the paired full metric under the promotion gate and remains stable across both embryo groups. If only one movie drives the gain, reject.

### B2 — Relative-rank / mutual-best edge association

**Priority:** P1 — first genuinely orthogonal association change, only after B1.<br>
**Status:** Not implemented in the staged candidate; blocked on OOF edge predictions.
**Compute:** CPU for edge rescoring/evaluation on cached OOF scores; GPU likely if those model outputs must be regenerated.
**Hypothesis:** Prefer an edge that is best in both the source row and target column, with a bounded penalty for non-mutual edges. The public [LB exploration E — mutual-best association notebook](https://www.kaggle.com/code/yudaiyamauchi/lb-exploration-e-mutual-best-association) freezes the rest of the pipeline and uses beta `0.25`; it supplies an isolated test design but no public candidate score. The separate density-rank notebook combines beta `0.12` with density adaptation and DivNet and displays `0.944`, so it is not proof that mutual-best alone beats our baseline.[17]

**Work:** Keep detections, candidate-edge sets, and post-processing fixed. Compare the current harmonic score with the isolated `mutual_best` beta `0.25` configuration on the same held-out predictions. Do not sweep extra beta values or tune thresholds in the same run.

**Pass:** paired improvement on both embryo groups, with no increase in duplicate/conflicting edges or loss of division branches. Otherwise revert.

### B3 — Neighborhood-flow relinking versus per-track EMA

**Priority:** P2 — only if the candidate still fails on crossings/fast-moving clusters.<br>
**Status:** Not implemented; no OOF graph/image cache available.
**Compute:** CPU for the flow/post-processing comparison on cached clips/graphs; GPU only if neural detections/edge predictions need regeneration.
**Hypothesis:** A neighborhood-flow estimate can predict motion through a short occlusion where an individual track's EMA velocity is stale. A public notebook exposes `seed` flow mode with `k=12`, radius `40 µm`, and additional gates; it is bundled with other changes, so its score does not isolate flow. This is complementary to—not a replacement automatically justified for—the candidate's EMA.[19]

**Work:** Evaluate a flow override on a predeclared hard subset of held-out clips. Keep detection/node selection fixed, record identity switches and edge FN/FP, and compare to EMA alone.

**Pass:** aggregate official-score gain that passes the promotion gate and no degradation on non-hard clips. Skip if it needs new training or full public-LB tuning.

### B4 — Test a separate division verifier (DivNet / conditional synthetic model)

**Priority:** P3 — optional if a pretrained artifact is already attachable and legally reusable.<br>
**Status:** Not in the staged candidate; public conditional third-model notebook reports `0.946`, so upside is uncertain.[18]
**Compute:** Do not train a new verifier for this deadline. Inference compute depends on the available model and may require GPU; first verify its license, checkpoint, and offline runtime.

**Work:** Gate only ambiguous division candidates with a learned mitosis score; do not union every predicted node from a third detector. Verify dataset/model license and that the artifact is available offline in the Kaggle notebook.

**Pass:** fold-held-out division TP rises without a false-fork increase, and the overall score improves. Do not train a fresh DivNet under the current time/GPU constraint.

### B5 — Multi-scale DoG as a recall-only detector proposal source

**Priority:** P4 — low priority against the current learned detector.<br>
**Status:** Not wired into the candidate; blocked on image/OOF validation.
**Compute:** CPU for DoG proposals and filtering; valid validation images/labels are still required to measure impact.
**Hypothesis:** DoG peaks may recover missed dim cells. A public rule-based experiment improved its own score from `0.786` to `0.826`, but that is not evidence of a gain over `0.946`.[8]

**Work:** Use DoG only to propose candidates that are then deduplicated and filtered by the learned detector/edge model. Track node-count ratio and duplicate distance.

**Pass:** recall gain survives the adjusted-node penalty and improves the paired official score. Otherwise leave it out.

### B6 — Train on the public synthetic microscopy dataset

**Priority:** P5 — high long-term upside; defer until after this competition unless GPU returns with ample time and a complete training/validation path.<br>
**Status:** Not started.
**Compute:** GPU required for practical model training; explicitly post-competition work.
**Hypothesis:** The released CC0 synthetic dataset provides substantially more mitosis labels than the competition annotations and could support a more robust division verifier or detector.[3][12]

**Work after the competition:** train fold-disjoint models, use synthetic-to-real validation, and verify external data provenance. The 18.5 GB dataset and domain gap make this unsuitable as a last-minute unvalidated training run.

### B7 — Graph-level bipartite consensus ensembling

**Priority:** P1 — high expected value; runs on CPU locally or lightweight on GPU.<br>
**Status:** Ready to implement. Candidate submissions from distinct model paradigms are available (0.946 Edge-TTA, 0.956 Sub-Voxel Flow Harmonic, Super-Fusion).
**Compute:** CPU-only for graph consensus solving.
**Hypothesis:** Ensembling distinct cell tracking graphs by retaining high-confidence consensus edges and resolving ambiguous connections with spatial distance/DeepCenter priors reduces track fragmentation and guards against private test shakeup (+0.002 to +0.004 expected gain).
**Work:** Take prediction CSVs from two or three distinct submissions. Match nodes across predictions within 3.5 µm. For temporal edges, admit edges where models agree; for contested edges, select the candidate with higher harmonic link probability and minimum spatial divergence. Ensure strict Kaggle submission format compliance.

### B8 — Pretrained DivNet 3D-CNN mitosis verification on the 0.956+ stack

**Priority:** P1 — directly targets the single lowest submetric (division Jaccard is ~0.18–0.23).<br>
**Status:** Pretrained weights exist in public dataset `giorgosi/biohub-divnet-v2` (`best_overall.pt`); validated in `haideptry/biohub-0-951` but unmerged into the 0.956 / Super-Fusion stack.
**Compute:** Kaggle GPU (~2 additional minutes of inference time per movie).
**Hypothesis:** Mitosis false positives heavily penalize both division Jaccard and adjusted edge Jaccard. While our current 0.956 / Super-Fusion stacks use geometric divergence rules (sister symmetry, 3-frame horizon), a neural patch classifier inspecting the raw volumetric intensity around putative divisions will veto false splits that geometry alone cannot detect.
**Work:** Attach `giorgosi/biohub-divnet-v2` to the Super-Fusion kernel. Crop 3D patches centered on candidate division mother-daughter centroids and run forward inference through `DivNetMitosisClassifier`. Veto candidate divisions where $P(\text{div}) < 0.50$.

### B9 — Adaptive global node-count calibration (N_pred penalty control)

**Priority:** P2 — addresses the adjusted edge Jaccard denominator penalty.<br>
**Status:** Concept defined; inspired by competitor feedback (Topic #741749).
**Compute:** CPU / config-level threshold logic.
**Hypothesis:** The official metric penalizes over-detection directly: $J_{\text{adj}} = \frac{|E \cap \hat{E}|}{|E \cup \hat{E}| + \max(0, N_{\text{pred}} - N_{\text{true}})}$. Deeper or dual models tend to inflate $N_{\text{pred}}$, causing a penalty that outweighs small recall gains. Calibrating the total predicted node count to match expected developmental cell density per timepoint maximizes the denominator efficiency.
**Work:** Calculate average cell count per developmental stage from training metadata. If a test movie produces $N_{\text{pred}} > 1.05 \times N_{\text{expected}}$, dynamically raise the detection threshold from 0.960 to 0.965 or prune short, isolated single-node components.

### B10 — Density-adaptive tissue flow radius

**Priority:** P2 — enhances neighborhood-flow motion relinking across heterogeneous movies.<br>
**Status:** Flow prior currently uses fixed $R = 40\text{--}48\,\mu\text{m}$ and $k = 12\text{--}16$.
**Compute:** Inline CPU post-processing within kernel execution.
**Hypothesis:** Early-stage embryos have widely spaced cells ($>60\,\mu\text{m}$ apart), while late-stage embryos have hundreds of tightly packed cells. A fixed $40\,\mu\text{m}$ radius pools irrelevant cells in dense movies and too few cells in sparse movies. Adapting the radius based on local density ($R_{\text{flow}} \propto 1/\sqrt{\rho}$) improves trajectory coherence.
**Work:** Estimate local centroid density $\rho$ in each frame. Dynamically scale the neighbor radius between $25\,\mu\text{m}$ (dense clusters) and $60\,\mu\text{m}$ (early stages).

### B11 — Final two-submission portfolio selection strategy

**Priority:** P0 — mandatory before competition close (September 29, 23:59 UTC).<br>
**Status:** Strategy drafted.
**Compute:** Manual selection on Kaggle competition dashboard.
**Hypothesis:** Selecting two correlated high-scoring public variants maximizes shakeup risk. The optimal portfolio pairs the highest-ceiling innovation with an orthogonal, proven safety anchor.
**Work:** 
- **Slot 1 (Highest Ceiling):** The highest-scoring multi-model SOTA candidate (Super-Fusion / DivNet candidate, targeting 0.958–0.962).
- **Slot 2 (Safety Anchor):** Our verified baseline `56132481` (0.946 Edge TTA) or the B7 consensus graph ensemble, completely insulated from hyperparameter overfitting on the public test set.

## Explicit no-go / avoid duplicate work

- **Never include `augment_dataset` / `MAX_COMPONENTS` / `FORKS` metric-exploit injection.** The old notebook audit found out-of-volume synthetic graph artifacts; the organizers patched the scorer and rescored submissions. Keep the clean graph output only.[6]
- **Do not copy the published five-node short-track rescue settings.** That notebook's own fixed-8 official-spec score was `0.000920` below its reference; our current candidate already has a different short-track rescue.[19]
- **Do not resume `temporal-pu-a` from epoch 10 without diagnosing it first.** The project ledger records `0.0079` OOF and `0.017` node recall; a new scratch-training run is not the endgame path.
- **Do not run blind threshold/short-track/division-cap sweeps against the public board.** The board is 29% of test, and forum reports show small, unstable public deltas and node-count sensitivity.[1] [Discussion #742266](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266)
- **Do not duplicate features already in the candidate:** edge-feature TTA, dual seed, harmonic bidirectional association, DeepCenter veto/TTA, physical safe-division rules, density-aware gap logic, EMA relinking, and short-track recovery.

## Endgame order of operations

1. Start with the CPU-side paired-OOF structural report and provenance/config reconciliation; do not treat schema v1 as promotion authorization or assume GPU reset/quota availability.
2. Locate valid fold-excluded predictions/weights or determine what GPU work is needed to generate them. Run the paired official-metric evaluation before relying on the candidate.
3. Only after that evidence and readiness pass, ask for separate authorization for T4 inference. The hidden-test run is not the fold-disjoint quality gate.
4. If OOF predictions are available, do B1, then one B2 ablation, then B3 only if the failures justify it. Defer B4/B5 unless time and artifacts make a focused test feasible; B6 is post-competition.
5. Submit only after separate explicit user approval. Keep the `0.946` scored submission as an anchor; use the second final-submission slot only for a validated, materially distinct candidate. No submission has been made as part of this backlog.

## Related records

- Research synthesis: `docs/kaggle_deep_dive_2026-09-25.md`
- `0.948` audit: `docs/kernel_audits/0948_cpu_preflight.md`
- Experiment ledger guidance: `docs/experiment_tracking.md`
- Kaggle leaderboard ledger: `results/kaggle_lb/submissions.csv`
- Local CV ledger: `results/local_cv/experiments.csv`

## Sources

[1] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard — Biohub current public leaderboard
    > "This leaderboard is calculated with approximately 29% of the test data. The final results will be based on the other 71%, so the final standings may be different."
    > "Sergio Alvarez 0.975"
[3] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules — Biohub official competition rules
    > "You may select up to two (2) Final Submissions for judging."
    > "External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants"
[4] https://www.kaggle.com/progression/competitions — Kaggle competition medal thresholds
    > "| Bronze | Top 40% | Top 40% | Top 100 | Top 10% |"
    > "| Silver | Top 20% | Top 20% | Top 50 | Top 5% |"
    > "| Gold | Top 10% | Top 10 | Top 10 + 0.2% | Top 10 + 0.2% |"
[5] https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md — Official patched Biohub metric
    > "The final combined score is score = adjusted_edge_jaccard + w · division_jaccard"
[6] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154 — Metric exploit patch discussion
    > "The patch (now public in the repo) requires divisions to be genuine strongly-connected parent→two-daughter structures matched within the 7 µm distance radius."
[7] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160 — CV versus public leaderboard and leakage
    > "The public weights were trained on all 199 annotated videos."
[8] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716952 — Rule-based baseline discussion
    > "Just taking DoG at multiple scales and using the scale-space max jumped LB from 0.786 → 0.826 (+0.040)."
[9] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/733877 — One-to-one linker and division metric discussion
    > "One successor per cell means no node ever has two outgoing edges, so division Jaccard is exactly 0.000"
[12] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103 — Synthetic fully labelled data discussion
    > "18.5 GB where every nucleus is labelled."
    > "I got tired of trying to train a division model on ~304 events, so I built a synthetic dataset and I'm releasing it free (CC0)."
[17] https://www.kaggle.com/code/haideptry/biohub-0-948-sota-density-rank-2xt4-fast — Biohub density-rank notebook
    > "Public Score 0.944"
[18] https://www.kaggle.com/code/andnyu/biohub-synthetic-conditional-third-model — Biohub conditional synthetic third-model notebook
    > "Public Score 0.946"
[19] https://www.kaggle.com/code/amanatar/optimized-biohub-max-score — Biohub short-track rescue notebook
    > "Delta vs Notebook 124: -0.000919500"
