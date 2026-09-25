# Biohub endgame experiment backlog

**Created:** 2026-09-25. **Competition deadline:** 2026-09-29 23:59 UTC (about 4 days 11 hours remained at the research snapshot). This is an ordered experiment queue, not a claim that any unrun idea improves the score. No Kaggle kernel was run or submitted as part of creating it.

## Current constraints and target

- User reports GPU access/quota is not currently available; I did not query Kaggle's live quota state.
- The project workspace has no `.zarr` directories or cached prediction graphs, so we cannot honestly score new inference ideas in this CPU-only state.
- Current best public score: `0.946`, submission `56132481`, public rank `1,221 / 3,899` in the 2026-09-25 leaderboard export. Current bronze cutoff is approximately rank 389; the displayed boundary score is `0.953`. Aim for rank ≤170 / about `0.954` on the current public board for a buffer, but final medals depend on the private 71% split.[1][4]
- A candidate scoring exactly `0.948` would currently rank around 504–601, so the staged `0.948` run is useful progress but not a medal result.
- Keep experiment outcomes in `results/local_cv/experiments.csv` and `results/kaggle_lb/submissions.csv`; this document is only the ordered idea backlog.

## Promotion gate for every candidate

1. One code or configuration change at a time. Do not use the public leaderboard as the inner-loop tuner.
2. Validate with fold/embryo-disjoint weights. Public support weights trained on all 199 annotated videos cannot validate on those same videos.[7]
3. Score with the current official patched metric. Log per-fold and per-video adjusted edge Jaccard, edge TP/FP/FN, division TP/FP/FN, `T_pred/T_true`, retained-node counts, and runtime. The metric combines adjusted-edge Jaccard with a 0.1-weighted division Jaccard; false forks, duplicate detections, and node-count shifts can outweigh tiny gains in one submetric.[5][6][9]
4. Proposed internal guard (not an official Kaggle rule): promote only if the paired aggregate gain is at least `+0.001`, no held-out embryo drops by more than `0.003`, and no metric component or output-integrity check regresses materially. If the validation artifacts are unavailable, label the idea **untested**—do not infer a gain from its name or a notebook title.
5. Any Kaggle push/submission is a separate, consequential action and needs the user's explicit approval. Do not fill final-submission slots just because Kaggle permits up to two.[3]

## Ordered backlog

### B0 — Resolve and run the already-staged `0.948` inference candidate

**Priority:** P0 — first if compute becomes available.<br>
**Status:** Static audit complete; blocked on provenance/quality-gate receipt and GPU.
**Hypothesis:** EMA momentum relinking improves identity continuity on crossings and fast motion over the proven `0.946` candidate at low training cost. DeepCenter is already present in both notebooks; it is not the differentiating change.

**Work:**
- Resolve the missing receipt/provenance blocker in `docs/kernel_audits/0948_cpu_preflight.md`.
- Before a run, confirm the exact primary/secondary weight files and hashes, DeepCenter checkpoint/epoch, active flags, and that the notebook fails rather than silently falling back to one seed or an unloaded veto.
- If GPU is available, run hidden-test inference and inspect the produced file and runtime manifest. Compare the clean candidate against the baseline; do not assume the score from the folder name.

**Pass:** correct artifacts loaded, no exploit rows or fallback, valid submission schema/coordinates, runtime safely under the deadline, and an actual Kaggle score returned after the user approves submission. This is not expected to reach bronze if it scores only `0.948`.

### B1 — Metric-aligned per-video node-count stability audit

**Priority:** P1 — best diagnostic before changing more knobs.<br>
**Status:** Blocked on fold-held-out prediction artifacts. The current notebook already contains node-ratio diagnostics; use those rather than building duplicate instrumentation.
**Hypothesis:** Changes that subtly alter retained-node counts can move the score even when offline edge Jaccard improves. One competitor reported a `+0.0011` offline edge change becoming a `-0.005` public change, with node-count shifts implicated; another report says a four-node change on one video was enough to matter. These are cautionary observations, not a universal law. [Discussion #742266](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266)

**Work:** For the same held-out movies, compare `0.946` and `0.948` predictions per movie: predicted/estimated node ratio, unmatched duplicates, edge TP/FP/FN, divisions, and score. Check the official coarse `T_true` adjustment; do not force every movie to an estimated exact count.[5]

**Pass:** any adjustment improves the paired full metric under the promotion gate and remains stable across both embryo groups. If only one movie drives the gain, reject.

### B2 — Relative-rank / mutual-best edge association

**Priority:** P1 — first genuinely orthogonal association change, only after B1.<br>
**Status:** Not implemented in the staged candidate; blocked on OOF edge predictions.
**Hypothesis:** Prefer an edge that is best in both the source row and target column, with a bounded penalty for non-mutual edges. The public [LB exploration E — mutual-best association notebook](https://www.kaggle.com/code/yudaiyamauchi/lb-exploration-e-mutual-best-association) freezes the rest of the pipeline and uses beta `0.25`; it supplies an isolated test design but no public candidate score. The separate density-rank notebook combines beta `0.12` with density adaptation and DivNet and displays `0.944`, so it is not proof that mutual-best alone beats our baseline.[17]

**Work:** Keep detections, candidate-edge sets, and post-processing fixed. Compare the current harmonic score with the isolated `mutual_best` beta `0.25` configuration on the same held-out predictions. Do not sweep extra beta values or tune thresholds in the same run.

**Pass:** paired improvement on both embryo groups, with no increase in duplicate/conflicting edges or loss of division branches. Otherwise revert.

### B3 — Neighborhood-flow relinking versus per-track EMA

**Priority:** P2 — only if the candidate still fails on crossings/fast-moving clusters.<br>
**Status:** Not implemented; no OOF graph/image cache available.
**Hypothesis:** A neighborhood-flow estimate can predict motion through a short occlusion where an individual track's EMA velocity is stale. A public notebook exposes `seed` flow mode with `k=12`, radius `40 µm`, and additional gates; it is bundled with other changes, so its score does not isolate flow. This is complementary to—not a replacement automatically justified for—the candidate's EMA.[19]

**Work:** Evaluate a flow override on a predeclared hard subset of held-out clips. Keep detection/node selection fixed, record identity switches and edge FN/FP, and compare to EMA alone.

**Pass:** aggregate official-score gain that passes the promotion gate and no degradation on non-hard clips. Skip if it needs new training or full public-LB tuning.

### B4 — Test a separate division verifier (DivNet / conditional synthetic model)

**Priority:** P3 — optional if a pretrained artifact is already attachable and legally reusable.<br>
**Status:** Not in the staged candidate; public conditional third-model notebook reports `0.946`, so upside is uncertain.[18]

**Work:** Gate only ambiguous division candidates with a learned mitosis score; do not union every predicted node from a third detector. Verify dataset/model license and that the artifact is available offline in the Kaggle notebook.

**Pass:** fold-held-out division TP rises without a false-fork increase, and the overall score improves. Do not train a fresh DivNet under the current time/GPU constraint.

### B5 — Multi-scale DoG as a recall-only detector proposal source

**Priority:** P4 — low priority against the current learned detector.<br>
**Status:** Not wired into the candidate; blocked on image/OOF validation.
**Hypothesis:** DoG peaks may recover missed dim cells. A public rule-based experiment improved its own score from `0.786` to `0.826`, but that is not evidence of a gain over `0.946`.[8]

**Work:** Use DoG only to propose candidates that are then deduplicated and filtered by the learned detector/edge model. Track node-count ratio and duplicate distance.

**Pass:** recall gain survives the adjusted-node penalty and improves the paired official score. Otherwise leave it out.

### B6 — Train on the public synthetic microscopy dataset

**Priority:** P5 — high long-term upside; defer until after this competition unless GPU returns with ample time and a complete training/validation path.<br>
**Status:** Not started.
**Hypothesis:** The released CC0 synthetic dataset provides substantially more mitosis labels than the competition annotations and could support a more robust division verifier or detector.[3][12]

**Work after the competition:** train fold-disjoint models, use synthetic-to-real validation, and verify external data provenance. The 18.5 GB dataset and domain gap make this unsuitable as a last-minute unvalidated training run.

## Explicit no-go / avoid duplicate work

- **Never include `augment_dataset` / `MAX_COMPONENTS` / `FORKS` metric-exploit injection.** The old notebook audit found out-of-volume synthetic graph artifacts; the organizers patched the scorer and rescored submissions. Keep the clean graph output only.[6]
- **Do not copy the published five-node short-track rescue settings.** That notebook's own fixed-8 official-spec score was `0.000920` below its reference; our current candidate already has a different short-track rescue.[19]
- **Do not resume `temporal-pu-a` from epoch 10 without diagnosing it first.** The project ledger records `0.0079` OOF and `0.017` node recall; a new scratch-training run is not the endgame path.
- **Do not run blind threshold/short-track/division-cap sweeps against the public board.** The board is 29% of test, and forum reports show small, unstable public deltas and node-count sensitivity.[1] [Discussion #742266](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266)
- **Do not duplicate features already in the candidate:** edge-feature TTA, dual seed, harmonic bidirectional association, DeepCenter veto/TTA, physical safe-division rules, density-aware gap logic, EMA relinking, and short-track recovery.

## Endgame order of operations

1. With no GPU: finish provenance/receipt checks, audit local score logs, and make no score claim for an unrun idea.
2. When GPU is genuinely available: clear B0 and run only the staged inference candidate first.
3. If local OOF predictions become available: B1, then one B2 ablation, then B3 only if the failure cases justify it.
4. Submit only after explicit user approval. Keep the `0.946` scored submission as an anchor; use the second final-submission slot only for a validated, materially distinct candidate. No submission has been made as part of this backlog.

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
