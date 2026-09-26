# Kaggle notebooks and discussions: endgame deep dive

**Research snapshot:** 2026-09-25, 12:26–12:42 UTC. Competition deadline recorded by the project: 2026-09-29 23:59 UTC. This is a read-only research report; no Kaggle kernel was run or submitted during this review.

> **Current execution status:** this file preserves the dated public-notebook/leaderboard research. Its leaderboard ranks and medal bands are not live. The current execution plan is [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md): the 0.948 candidate remains unscored, paired fold-disjoint predictions and a real receipt path are missing, and a Kaggle run/submission needs separate approval.

## Executive answer

At the September 25 snapshot, the best verified submission was not in the public medal range, and a hypothetical `0.948` score would still have been outside that snapshot's bronze cutoff. The staged candidate was then considered a low-cost hypothesis because it changed motion relinking, but the later CPU preflight found it was not ready to run or promote: provenance/configuration and the missing fold-disjoint evidence/receipt path must be resolved first. The candidate name remains unscored, not a result.

The public notebooks at approximately `0.947–0.948` are mostly variations of the same learned 3D detector + learned edge scorer + graph solver family. Our `0.946`/staged `0.948` path already contains most of the strongest shared ingredients: edge-feature TTA, dual-seed support, bidirectional harmonic association, DeepCenter veto, physically constrained division repair, density-aware gap logic, and short-track rescue. Re-copying those components is unlikely to create a meaningful new edge. The clearest untested ideas in the inspected newer code are **relative-rank / mutual-best association**, **neighborhood-flow motion**, and a stronger **per-video node-count stability audit**; none has yet shown a verified gain over our baseline in the public notebooks we checked.

A medal is possible only if the final private ranking moves substantially. On the captured public snapshot, the best submission was rank **1,221 / 3,899** at **0.946** (31.32% of teams). Kaggle medals for competitions with at least 1,000 teams go to the top 10% (bronze), top 5% (silver), and top 10 plus 0.2% increments (gold); for 3,899 teams that is 389 bronze, 194 silver, and 17 gold places. The displayed public bronze boundary was in the `0.953` band. A public score of `0.948` would have landed around ranks 504–601, still outside bronze. Kaggle says this board uses about 29% of the test data and the final ranking uses the other 71%, so public rank is a noisy proxy, not a medal guarantee.[1][4]

> **Correction:** older project text describing `0.946` as a “Solid Silver Medal Standing” was not supported by the September 25 standings snapshot. The linked rank figures are dated evidence, not current standings. See the archived correction in `docs/endgame_status_and_strategy.md` and the current status document before making a live claim.

## 1. What the current leaderboard means

The official public leaderboard export captured at `2026-09-25T12:26:30` contained 3,899 teams. It placed our `aleixlopez` team at rank 1,221 with the recorded `0.946` submission. The displayed `0.946` band spans ranks 1,203–1,358. The first displayed `0.953` score is around the bronze cutoff, but that rounded score band crosses the cutoff; treat `0.953` as the *boundary*, not as a guaranteed medal score. A more realistic public-buffer target would be rank ≤170 (about `0.954` on this snapshot), safely inside today's top-5% public band, while recognizing that the final 71% can reorder teams substantially. These are snapshot ranks, not final results.[1][4]

The project’s `biohub-0-948-momentum-deepcenter-tta` had **not** been run or scored in this research review. Its `0.948` name is a target, not evidence of a result. In the September 25 leaderboard snapshot, a hypothetical score of `0.948` would have improved our position but still missed the public bronze line. That rank estimate is historical; refresh the public leaderboard before any current-rank or medal claim. The remaining medal gap is therefore not “one more tiny threshold sweep.”

Kaggle's [official overview](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview) labels this a Points & Medals competition. The rules permit up to two final submissions, not a requirement to choose two.[3] Reserve the second slot for a distinct candidate only if it has credible validation.[3][4]

## 2. Notebook families and what is actually reproducible

### A. Older high-scoring notebook lineage: useful model ideas, unsafe output cell

The local research archive contains early `amanatar`, `kaiwalyaatulraut`, and `ravi123a321at` notebook versions. The existing audit in `docs/public_notebook_metric_hack_finding.md` found a synthetic-node/fork injection step in the examined older notebooks (`augment_dataset` / `MAX_COMPONENTS` / `FORKS`, with out-of-volume coordinates). One cleaned fork was inspected before submission and its exploit-free result was retained; the synthetic injection was not submitted. The organizers patched the division metric and rescored submissions. Treat those output-generation cells as stale and never copy them into a submission. The learned detection, edge scoring, TTA, candidate-edge generation, and ILP ideas remain legitimate prior art; the fake graph construction does not.[6]

The clean fork also exposed a reproducibility ceiling: the public support pack it used shipped only one checkpoint split, so code written to ensemble multiple checkpoints fell back to a single model. That explains why a publicly pulled notebook plus its publicly attached weights need not reproduce a leaderboard entry associated with the same general family. See `docs/public_notebook_metric_hack_finding.md` for the run evidence and the corrected row count.

### B. Current clean leader family: strong, but our pipeline already overlaps heavily

The newer clean notebooks build on a common stack:

- Temporal 3D U-Net cell-center proposals plus learned node/edge features and a graph optimizer (ILP/tracksdata).
- Spatial TTA (often 8-way D4 flips/rotations) applied to edge features, not just heatmaps.
- Two temporal seeds/models and forward/backward association; the harmonic-probability variant suppresses edges that are weak in either direction.
- DeepCenter center-prior vetoes for uncertain gap/division events.
- Physical-distance and sister-symmetry rules for divisions; adaptive gap closing; short-track recovery/pruning; small validator-gated parameter sweeps.

The inspected `Biohub Geometric Fusion` notebook reports a public score of `0.948`; it is the closest clean public comparator to our planned candidate. The `Biohub Harmonic Fusion V3` page's body says “0.953 Record Edition” but its displayed best score is `0.947 V4`. Do not promote the higher narrative number over the score widget without a matching submission record.[14][15]

Other recent titles also overstate what their page verifies: the notebook titled `0.951 SOTA DeepCenter Fast ILP` displays `0.944`, and the `0.948 Density Rank` notebook also displays `0.944`. The conditional synthetic third-model notebook displays `0.946`. These are useful hypothesis sources, not proof of beating our `0.946` result.[16][17][18]

The highest public-board score visible in the snapshot was `0.975`, but I did not find a public notebook in the inspected code list that reproducibly explains that top entry. That absence is not evidence of misconduct: the code or the best weights may simply not be shared. It does mean we should not claim to have reverse-engineered the leaders from public code alone.[1]

### C. What is new relative to our staged `0.948` notebook

Static comparison of the staged `0.946` and `0.948` notebooks found the latter's meaningful change to be EMA momentum-based motion relinking; DeepCenter is already in both. The staged candidate also has dual-seed/harmonic association, density-aware gap logic, safe-division constraints, and short-track recovery. A string-level check found no relative-rank/mutual-best association, neighborhood-flow mode, DivNet division verifier, or subthreshold readmission/gapfill path in the staged notebook. It already records node-count-related diagnostics, so the question is whether those diagnostics are used to reject unstable changes, not whether to add a duplicate “node count” idea. See `docs/kernel_audits/0948_cpu_preflight.md` and the two staged notebook folders.

The staging audit remains **blocked on provenance confirmation**: the expected quality-gate receipt was absent and model/score provenance needed confirmation. That gate comes before spending GPU time or treating the candidate name as a scored submission.

## 3. High-value metric and validation findings from discussions

### Exact metric alignment matters

The current official metric matches predicted nodes to ground truth within 7 µm, scores edges with Jaccard, applies an adjusted-edge penalty based on predicted node count versus a coarse provided true-node estimate, and adds a division-Jaccard term weighted by 0.1. Extra unmatched nodes are not direct node false positives, but too many predictions still reduce adjusted edge Jaccard. The division scorer requires a genuine directed local parent-to-two-daughter topology; weak connectivity or a one-to-one assignment alone cannot recover the division term.[5][6][9]

This gives two practical rules:

1. Do not maximize detections by blindly unioning models. Deduplicate/merge first and inspect per-video `T_pred/T_true` alongside edge TP/FP/FN.
2. Do not tune only one aggregate edge score. Preserve the division topology and check whether a small node-count shift, one difficult embryo, or a set of false forks caused the change.

A recent participant stuck at `0.947` reported that ten single-knob changes all lost on the public board; the offline edge-J improvement of `+0.0011` corresponded to a public drop of `-0.005`, and the failure correlated with predicted node-count changes, including a four-node difference on one video. This is one participant's report, not a universal law, but it is a strong warning against optimizing a small local proxy or chasing tiny public-LB deltas. [Discussion #742266](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266)

Another metric-focused discussion measured that overprediction is relatively cheap compared with missed edges, but duplicate detections are not: node matching is one-to-one, so an unmerged duplicate can add count penalty without helping edge matches. It also showed that a strict one-successor Hungarian linker gives up the division term by construction. Use these as metric diagnostics, not as permission to emit extra or artificial nodes.[9]

### Avoid leaked validation and public-LB tuning

A Kaggle discussion documents that the public support weights were trained on all 199 annotated videos; validating on those same videos with those checkpoints leaks training data. The same discussion reports a participant whose post-processing ablation improved a memorizing training-set proxy but flipped sign on the leaderboard. Any local comparison must use weights that excluded the held-out embryo/video group.[7]

The Kaggle leaderboard explicitly says the public board is only approximately 29% of test data, with 71% reserved for the final ranking. Participants also describe substantial movie-to-movie variance. Use the public board as a final diagnostic, not as the model-selection loop.[1][7]

A separate forum report claims a model-agnostic “universal plugin” gave `+0.030–0.050` on several public models, but it does not provide a reproducible public implementation in the inspected material. I did not find an actionable public Biohub notebook for it. Treat this as an unverified lead, not a backlog item to implement from hearsay. [Discussion #735352](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352)

## 4. New ideas worth testing—and what not to copy

### Relative-rank / mutual-best edge association

Two public branches should not be conflated. The [`LB exploration E — mutual-best association`](https://www.kaggle.com/code/yudaiyamauchi/lb-exploration-e-mutual-best-association) notebook freezes its detector/checkpoints/post-processing and changes only edge association: prefer edges that are best in both the source row and target column, with `mutual_best` beta `0.25`. The notebook provides an isolated experiment setup but no candidate score, so this is an actionable A/B—not a proven gain. The separate density-rank notebook combines relative-rank/mutual-best beta `0.12` with density adaptation and DivNet; its displayed public score is `0.944`, so that multi-change result does not establish which component helped or hurt.[17]

### Neighborhood-flow motion prior

The discussion describes EMA projection of an individual track's velocity, which is already represented by the staged candidate.[11] A separate public notebook implements a neighborhood-flow mode (`seed`, `k=12`, radius `40 µm`, with additional gates) in its relinker; this is distinct from per-track EMA, but the notebook bundles other changes and its displayed score history is not an isolated flow ablation. Treat it as a hypothesis and test only with detections/node retention frozen.[19]

### Division-specific model / synthetic supervision

A competitor released an 18.5 GB CC0 synthetic microscopy dataset with 165,267 labelled divisions across 4,056,226 nodes. This is a potentially valuable long-term way to learn division evidence, because the competition annotations are sparse. A public conditional synthetic third-model notebook reported `0.946` at the time reviewed, so the existence of synthetic data does not by itself demonstrate a score gain. At the time of the September 25 research snapshot, GPU access was reported unavailable and only a few days remained; training from scratch was a post-competition direction, not an endgame action. External data must remain public/equally accessible under the competition rules.[3][12][18]

### Multi-scale DoG detection

A simple rule-based notebook reports that multi-scale Difference-of-Gaussians increased its own score from `0.786` to `0.826` (+0.040). That is a useful low-cost detector baseline and sanity check, but it is far below the verified `0.946` baseline; `0.948` remains only the name of an unscored candidate. Only revisit DoG as a high-recall candidate source if local held-out data shows cells our neural detector misses and the node-count penalty remains controlled.[8]

### Short-track rescue is not a free win

One clean notebook tested a conservative rescue of exactly five-node tracks, requiring mean edge probability ≥0.90, mean physical edge distance ≤2.75 µm, and a small node budget. Its own fixed-8 official-spec validation ended at `0.878300` versus a `0.879220` reference (delta `-0.000920`). Our staged candidate already has its own short-track rescue. Do not transplant this public setting; keep only the variant that wins paired held-out validation.[19]

## 5. What we should do now

This section is retained as the research-time recommendation, not a current run checklist. The current ordering is: (1) implement a fail-closed paired-OOF evaluator/receipt path and resolve provenance; (2) locate or generate genuine fold-excluded parent/EMA outputs; (3) score them on CPU; and (4) only after the evidence/readiness gates and separate user approval, perform GPU/T4 hidden-test inference. B1 then B2 require retained OOF artifacts; B3 is conditional. The exact current blocker/equipment matrix is maintained in [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md). A test-set run is not a substitute for independent held-out validation.

**Bottom line (research snapshot):** the review identified plausible ideas, not measured local gains. No score improvement is supported for the staged candidate. Use the canonical current status document for today's next action; do not infer a medal path from a candidate name or stale public-board snapshot.

## Internal project references

- `docs/kernel_audits/0948_cpu_preflight.md`
- `docs/public_notebook_metric_hack_finding.md`
- `docs/0_946_baseline_review.md`
- `docs/local_evaluation.md`
- `docs/local_cv/README.md`
- `docs/experiment_tracking.md`
- `results/kaggle_lb/submissions.csv`

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
[11] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739570 — Velocity-projected tracking discussion
    > "EMA (Exponential Moving Average) velocity and project its position forward by velocity * gap_frames."
[12] https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103 — Synthetic fully labelled data discussion
    > "18.5 GB where every nucleus is labelled."
    > "I got tired of trying to train a division model on ~304 events, so I built a synthetic dataset and I'm releasing it free (CC0)."
[14] https://www.kaggle.com/code/amanatar/biohub-geometric-fusion — Biohub Geometric Fusion notebook
    > "Public Score 0.948"
[15] https://www.kaggle.com/code/raunakdey07/biohub-harmonic-fusion-v3 — Biohub Harmonic Fusion V3 notebook
    > "Best Score [0.947 V4]"
[16] https://www.kaggle.com/code/haideptry/biohub-0-951-sota-deepcenter-fast-ilp-19m — Biohub fast ILP claimed 0.951 notebook
    > "Public Score 0.944"
[17] https://www.kaggle.com/code/haideptry/biohub-0-948-sota-density-rank-2xt4-fast — Biohub density-rank notebook
    > "Public Score 0.944"
[18] https://www.kaggle.com/code/andnyu/biohub-synthetic-conditional-third-model — Biohub conditional synthetic third-model notebook
    > "Public Score 0.946"
[19] https://www.kaggle.com/code/amanatar/optimized-biohub-max-score — Biohub short-track rescue notebook
    > "Delta vs Notebook 124: -0.000919500"
